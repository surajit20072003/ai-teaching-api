"""
core/ollama_lifecycle.py
────────────────────────
Manage Ollama model VRAM lifecycle using Ollama's native REST API.

Strategy:
  - DON'T stop the Ollama service — just evict models from GPU VRAM.
  - Before text/Manim generation: ensure the model is loaded (/api/ps check + warm-up).
  - During image+audio phase: evict ALL loaded models to free VRAM for Wan2GP + VoxCPM.

Ollama API endpoints used:
  GET  /api/ps         → list models currently loaded in VRAM
  POST /api/generate   → with keep_alive=0 to evict, or keep_alive="10m" to load

Environment variables:
  OLLAMA_URL           → base URL  (default: http://host.docker.internal:11434)
  OLLAMA_MODEL         → model name (default: qwen3-coder:latest)
  OLLAMA_EVICT_TIMEOUT → seconds to wait for all models to evict (default: 30)
  OLLAMA_LOAD_TIMEOUT  → seconds to wait for model to appear in /api/ps (default: 120)
"""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_URL           = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL", "qwen3-coder:latest")
OLLAMA_EVICT_TIMEOUT = int(os.getenv("OLLAMA_EVICT_TIMEOUT", "30"))
OLLAMA_LOAD_TIMEOUT  = int(os.getenv("OLLAMA_LOAD_TIMEOUT", "120"))

_HTTP_TIMEOUT = 15  # seconds for individual API calls


# ── Low-level helpers ──────────────────────────────────────────────────────────

async def get_loaded_models() -> list[str]:
    """
    GET /api/ps
    Returns list of model names currently loaded in VRAM.
    Returns [] on any error (treat as nothing loaded).
    """
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/ps")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            # Each entry has a "name" or "model" key depending on Ollama version
            names = [
                m.get("name") or m.get("model", "")
                for m in models
                if isinstance(m, dict)
            ]
            return [n for n in names if n]
    except Exception as e:
        logger.warning(f"[OllamaLC] get_loaded_models failed: {e}")
        return []


async def is_model_loaded(model_name: str) -> bool:
    """
    Returns True if model_name appears in /api/ps (currently in VRAM).
    Matches on exact name or base name (ignoring tag comparison edge cases).
    """
    loaded = await get_loaded_models()
    # Normalize: "qwen3-coder:latest" matches "qwen3-coder:latest" or "qwen3-coder"
    base_name = model_name.split(":")[0].lower()
    for name in loaded:
        if name.lower() == model_name.lower():
            return True
        if name.lower().split(":")[0] == base_name:
            return True
    return False


async def evict_model(model_name: str) -> bool:
    """
    POST /api/generate {"model": model_name, "keep_alive": 0}
    Releases the model from GPU VRAM. Ollama service stays running.
    Returns True on success or if model was not loaded.
    """
    logger.info(f"[OllamaLC] Evicting model from VRAM: {model_name}")
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model_name, "keep_alive": 0, "prompt": ""},
            )
            # 404 means model not loaded — that's fine
            if resp.status_code == 404:
                logger.info(f"[OllamaLC] {model_name} was not loaded — nothing to evict")
                return True
            resp.raise_for_status()
            logger.info(f"[OllamaLC] ✓ Evicted: {model_name}")
            return True
    except Exception as e:
        logger.error(f"[OllamaLC] evict_model({model_name}) failed: {e}")
        return False


async def evict_all_models() -> int:
    """
    Evict ALL currently loaded models from VRAM.
    Polls /api/ps, calls evict_model() for each, then verifies empty.
    Returns count of models evicted.
    """
    loaded = await get_loaded_models()
    if not loaded:
        logger.info("[OllamaLC] No models in VRAM — nothing to evict")
        return 0

    logger.info(f"[OllamaLC] Evicting {len(loaded)} model(s) from VRAM: {loaded}")
    for name in loaded:
        await evict_model(name)

    # Verify VRAM is now empty
    ok = await wait_for_eviction(timeout=OLLAMA_EVICT_TIMEOUT)
    if ok:
        logger.info(f"[OllamaLC] ✓ All {len(loaded)} model(s) evicted from VRAM")
    else:
        remaining = await get_loaded_models()
        logger.warning(f"[OllamaLC] ⚠ Eviction timeout — still loaded: {remaining}")

    return len(loaded)


async def wait_for_eviction(timeout: int = OLLAMA_EVICT_TIMEOUT) -> bool:
    """
    Poll /api/ps until no models remain in VRAM (or timeout expires).
    Returns True if VRAM cleared, False if timeout.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        loaded = await get_loaded_models()
        if not loaded:
            return True
        await asyncio.sleep(1)
    return False


async def ensure_model_loaded(
    model_name: str = OLLAMA_MODEL,
    keep_alive: int = -1,      # -1 = keep loaded indefinitely (never auto-evict)
    timeout: int = OLLAMA_LOAD_TIMEOUT,
) -> bool:
    """
    Ensure model_name is loaded in VRAM before use.
    1. Check /api/ps — if already loaded, return True immediately.
    2. If not loaded: POST /api/generate with keep_alive to trigger load.
    3. Poll /api/ps until model appears (max timeout seconds).
    Returns True if loaded successfully, False if timeout.
    """
    if await is_model_loaded(model_name):
        logger.info(f"[OllamaLC] {model_name} already in VRAM ✓")
        return True

    logger.info(f"[OllamaLC] Loading {model_name} into VRAM (keep_alive={keep_alive})...")
    try:
        # Fire a no-op generate call to trigger model load
        async with httpx.AsyncClient(timeout=OLLAMA_LOAD_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": keep_alive,
                    "stream": False,
                },
            )
            resp.raise_for_status()
    except Exception as e:
        logger.error(f"[OllamaLC] Failed to trigger load for {model_name}: {e}")
        return False

    # Poll until model appears in /api/ps
    ok = await _poll_until_loaded(model_name, timeout=timeout)
    if ok:
        logger.info(f"[OllamaLC] ✓ {model_name} is now in VRAM")
    else:
        logger.error(f"[OllamaLC] ✗ {model_name} did not appear in VRAM within {timeout}s")
    return ok


async def _poll_until_loaded(model_name: str, timeout: int) -> bool:
    """Poll /api/ps every 3 seconds until model_name appears or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await is_model_loaded(model_name):
            return True
        await asyncio.sleep(3)
    return False


# ── High-level phase helpers (called from pregen.py) ──────────────────────────

async def prepare_for_text_generation() -> bool:
    """
    Phase 1 entry: ensure Ollama model is in VRAM before text generation.
    Returns True if ready.
    """
    logger.info(f"[OllamaLC] Phase 1: ensuring {OLLAMA_MODEL} is loaded for text generation")
    return await ensure_model_loaded(OLLAMA_MODEL)


async def prepare_for_media_generation() -> int:
    """
    Phase 2 entry: evict all Ollama models from VRAM before image/audio generation.
    Frees GPU memory for Wan2GP and VoxCPM.
    Returns count of models evicted.
    """
    logger.info("[OllamaLC] Phase 2: evicting all Ollama models for media generation")
    return await evict_all_models()


async def prepare_for_manim_generation() -> bool:
    """
    Phase 4 entry: ensure Ollama model is back in VRAM before Manim code generation.
    Returns True if ready.
    """
    logger.info(f"[OllamaLC] Phase 4: ensuring {OLLAMA_MODEL} is loaded for Manim generation")
    return await ensure_model_loaded(OLLAMA_MODEL)
