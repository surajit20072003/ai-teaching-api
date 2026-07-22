"""
core/embeddings.py
──────────────────
Local sentence-transformer embedding module.
Model: all-MiniLM-L6-v2 (22MB, 384 dimensions)
Cost: $0 — runs entirely on CPU using your server hardware.

The model loads ONCE per worker process and stays in memory.
A threading lock (double-checked pattern) ensures that even if multiple
threads hit embed_text() at the same time on the first call, only ONE
thread actually calls SentenceTransformer() — the rest wait and reuse it.
embed_async() runs in a thread pool so it never blocks the async event loop.
"""
import asyncio
import threading

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_model       = None
_model_lock  = threading.Lock()   # guards concurrent model loading


def get_model():
    """
    Thread-safe lazy-load. Uses double-checked locking so the lock is only
    acquired when the model is not yet loaded, avoiding contention after warmup.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        # Second check inside the lock — another thread may have loaded it
        # while we were waiting for the lock.
        if _model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Embeddings] Loading model '{EMBED_MODEL_NAME}' on CPU (first time only)...")
            # NOTE: do NOT pass low_cpu_mem_usage or device_map here.
            # Those flags trigger HuggingFace's lazy "meta" device allocation
            # which crashes with "Cannot copy out of meta tensor" in multi-worker
            # environments where workers try to load simultaneously.
            _model = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
            print("[Embeddings] ✓ Model loaded and cached in memory.")
    return _model


def embed_text(text: str) -> list[float]:
    """
    Encode text → 384-dim normalized float list.
    Runs on CPU, typically ~5ms per call.
    normalize_embeddings=True ensures cosine similarity == dot product.
    """
    model = get_model()
    vec   = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


async def embed_async(text: str) -> list[float]:
    """
    Async wrapper: runs embed_text() in the default ThreadPoolExecutor
    so it never blocks the FastAPI event loop during embedding.
    Safe to call from any async endpoint.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_text, text)


def vec_to_pg_str(vec: list[float]) -> str:
    """Convert Python list to PostgreSQL vector literal string '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


# ── Eager startup preload ─────────────────────────────────────────────────────
# Load the model at import time (when the worker process starts) so the first
# real request never triggers a cold load. This also means all workers load
# the model during startup — before any traffic hits them — eliminating the
# race condition entirely.
try:
    get_model()
except Exception as _preload_err:
    print(f"[Embeddings] ⚠ Startup preload failed (will retry on first request): {_preload_err}")
