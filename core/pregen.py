"""
core/pregen.py — Offline Pre-Generation Engine
===============================================

Pipeline per question (strictly sequential, one question at a time):
  Step 1: Ollama → generate presentation_slides JSON (local GPU, free)
  Step 2: For EACH slide:
            a. Wan2GP → generate infographic image  → upload to B2
            b. VoxCPM → generate narration audio    → upload to B2
  Step 3: Save all URLs + slides back to DB → mark pregen_status = 'done'

Design decisions:
  - NO fallback: if Ollama/Wan2GP/VoxCPM fails → row stays 'failed' for manual retry
  - Sequential image+audio per slide → avoids overloading local GPU
  - Run batch in background task via FastAPI BackgroundTasks
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.slide_generator import generate_slides
from core.b2_client import upload_to_b2
from core.embeddings import embed_async, vec_to_pg_str
from core.tts_utils import prepare_for_tts
from core.local_storage import write_image, write_audio, write_slide_cache
from core.cache import delete_from_cache
from core.ollama_lifecycle import (
    prepare_for_text_generation,
    prepare_for_media_generation,
    prepare_for_manim_generation,
)
from core.manim_generator import generate_and_render_slide_manim

# ── Prompt file loader ─────────────────────────────────────────────────────────
_PROMPT_DIR = Path(__file__).parent / "prompts"

def _load_prompt_file(filename: str) -> str:
    """Load a prompt from core/prompts/ — no Docker restart needed to update."""
    try:
        return (_PROMPT_DIR / filename).read_text(encoding="utf-8")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[Pregen] Could not load {filename}: {e}")
        return ""


def _build_image_prompt(slide: dict) -> str:
    """
    Build a rich, context-aware Wan2GP image prompt based on slide type.
    Uses the image_system_prompt.txt as a style guide to pick the correct
    visual style, then constructs a detailed prompt from slide fields.
    """
    title       = slide.get("title", "Concept")
    content     = slide.get("infographic") or slide.get("content") or ""
    key_points  = slide.get("keyPoints") or []
    formula     = slide.get("formula", "") or ""
    visual_type = slide.get("visual_type", "") or ""
    is_story    = slide.get("isStory", False)
    is_tips     = slide.get("isTips", False)
    kp_str      = ", ".join(key_points[:4]) if key_points else ""
    title_lower = title.lower()

    # ── Pick style based on slide type ────────────────────────────────────────
    if is_story:
        style = (
            "warm illustrated story scene, storybook art style, soft warm lighting, "
            "narrative illustration with characters, Indian cultural context if suitable, "
            "hand-drawn watercolor feel, educational storytelling visual"
        )
        composition = "characters centered, scene background fills frame, warm color palette"
        bg = "warm cream background"

    elif is_tips:
        style = (
            "bright educational infographic, lightbulb and brain icons, "
            "mnemonic memory aid layout, bold yellow and blue accents, "
            "numbered memory trick visual, clean icon-based design"
        )
        composition = "numbered points arranged vertically, icons on left, text on right"
        bg = "clean white background with yellow accent strip"

    elif visual_type == "manim" or formula:
        style = (
            "clean academic dark background #1a1a2e, glowing white LaTeX equation on dark surface, "
            "chalkboard or paper texture with formula, colored highlights on key variables (yellow, teal), "
            "professional mathematical illustration, deep blue and gold accents, "
            "classroom-ready formula visualization"
        )
        composition = "formula large and centered, variable labels beside with arrows, dark academic feel"
        bg = "dark navy #1a1a2e background"

    elif any(w in title_lower for w in ["cell", "organ", "dna", "molecule", "anatomy", "biology",
                                         "physiology", "tissue", "membrane", "protein", "nerve"]):
        style = (
            "detailed scientific illustration, cross-section biological diagram, "
            "labeled anatomical parts with arrows, textbook medical style, "
            "semi-transparent layers showing internal structure, "
            "teal and rose color palette, white annotation labels"
        )
        composition = "cross-section or diagram centered, labeled parts with leader lines"
        bg = "clean white scientific background"

    elif any(w in title_lower for w in ["how", "step", "process", "stage", "phase", "cycle", "pathway"]):
        style = (
            "numbered step flowchart infographic, connected boxes with arrows showing direction, "
            "timeline or process layout, blue boxes white text, professional process diagram, "
            "clear sequential flow visualization"
        )
        composition = "steps arranged left-to-right or top-to-bottom, arrows between each step"
        bg = "clean white background"

    elif any(w in title_lower for w in ["summary", "recap", "review", "conclusion", "overview"]):
        style = (
            "mind map layout infographic, central topic in circle, radiating branches with icons, "
            "color-coded concept branches, clean professional hierarchy, "
            "educational summary visual"
        )
        composition = "central node in middle, branches radiating outward, clean white background"
        bg = "clean white background"

    elif any(w in title_lower for w in ["what is", "definition", "meaning", "introduction", "concept"]):
        style = (
            "flat design educational infographic, labeled diagram with arrows, "
            "blue and orange color palette, icon-based layout, "
            "textbook-style conceptual illustration, clear hierarchy"
        )
        composition = "concept label at top, main visual center, key points listed on sides"
        bg = "clean white background"

    else:
        style = (
            "educational diagram, textbook illustration, labeled arrows, "
            "clear section boxes, blue and white color scheme, "
            "professional academic infographic"
        )
        composition = "title at top, main visual center, key labels at sides"
        bg = "clean white background"

    # ── Build the prompt ──────────────────────────────────────────────────────
    formula_part = f" Mathematical formula shown: {formula}." if formula else ""
    kp_part      = f" Key concepts: {kp_str}." if kp_str else ""

    prompt = (
        f"{style}. "
        f"Topic: {title}. "
        f"{content[:200] if content else ''}{formula_part}{kp_part} "
        f"Composition: {composition}. "
        f"Background: {bg}. "
        f"high quality, sharp focus, 4K resolution, detailed, "
        f"professional educational illustration, suitable for classroom projection, "
        f"no watermarks, no text artifacts, crisp edges, well-balanced composition. "
        f"Avoid: blurry, low quality, pixelated, ugly, distorted, watermark, "
        f"dark muddy colors, overexposed, cluttered layout, overlapping elements."
    )
    return prompt

# ── Environment ────────────────────────────────────────────────────────────────
VOXCPM_URL     = os.getenv("VOXCPM_URL",    "http://host.docker.internal:7861")
VOXCPM_API_KEY = os.getenv("VOXCPM_API_KEY", "spassword")

WAN2GP_URL     = os.getenv("WAN2GP_URL",    "http://host.docker.internal:9090")
WAN2GP_API_KEY = os.getenv("WAN2GP_API_KEY", "mypassword1234")

# ── Shared in-memory state (single background job) ─────────────────────────────
@dataclass
class PregenState:
    running:         bool  = False
    stop_requested:  bool  = False
    subject_id:      str   = ""
    total:           int   = 0
    done:            int   = 0
    failed:          int   = 0
    current_question: str  = ""
    current_step:    str   = ""      # "ollama" | "image:N" | "audio:N" | "manim:N" | "saving"
    phase:           str   = ""      # "text" | "evicting" | "media" | "loading" | "manim" | "saving"
    started_at:      float = 0.0
    last_error:      str   = ""
    log:             List[str] = field(default_factory=list)


_state = PregenState()


# ── Shared in-memory state for media-retry job ───────────────────────────────
@dataclass
class RetryState:
    running:          bool  = False
    subject_id:       str   = ""
    total:            int   = 0
    done:             int   = 0
    failed:           int   = 0
    current_cache_id: str   = ""
    current_step:     str   = ""    # e.g. "image:2" | "audio:3"
    started_at:       float = 0.0
    last_error:       str   = ""
    log:              List[str] = field(default_factory=list)


_retry_state = RetryState()


# ── Retry helpers ────────────────────────────────────────────────────────────
def _needs_image(slide: dict) -> bool:
    url = (slide.get("infographicUrl") or "").strip()
    return not url.startswith("http")


def _needs_audio(slide: dict) -> bool:
    url = (slide.get("audioUrl") or "").strip()
    return not url.startswith("http")


def get_retry_state() -> Dict[str, Any]:
    elapsed = round(time.time() - _retry_state.started_at, 1) if _retry_state.started_at else 0
    return {
        "running":          _retry_state.running,
        "subject_id":       _retry_state.subject_id,
        "total":            _retry_state.total,
        "done":             _retry_state.done,
        "failed":           _retry_state.failed,
        "current_cache_id": _retry_state.current_cache_id,
        "current_step":     _retry_state.current_step,
        "elapsed_seconds":  elapsed,
        "last_error":       _retry_state.last_error,
        "recent_log":       _retry_state.log[-30:],
    }


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _state.log.append(line)
    if len(_state.log) > 200:
        _state.log = _state.log[-200:]


def get_state() -> Dict[str, Any]:
    elapsed = round(time.time() - _state.started_at, 1) if _state.started_at else 0
    return {
        "running":          _state.running,
        "stop_requested":   _state.stop_requested,
        "subject_id":       _state.subject_id,
        "total":            _state.total,
        "done":             _state.done,
        "failed":           _state.failed,
        "current_question": _state.current_question,
        "current_step":     _state.current_step,
        "elapsed_seconds":  elapsed,
        "last_error":       _state.last_error,
        "recent_log":       _state.log[-30:],
    }


def request_stop() -> None:
    _state.stop_requested = True
    _log("[Pregen] Stop requested by user — will halt after current row.")


# ── VoxCPM TTS (local) ─────────────────────────────────────────────────────────
async def _voxcpm_tts(text: str, language: str = "hi-IN") -> Optional[bytes]:
    """
    Call VoxCPM local TTS server (Custom REST API on :7861).
    Returns WAV bytes or None on failure.
    """
    import httpx, asyncio
    try:
        headers = {}
        if VOXCPM_API_KEY:
            headers["X-API-Key"] = VOXCPM_API_KEY

        # 1. Submit Job
        payload = {"text": text, "emotion": "neutral"}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{VOXCPM_URL}/generate",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            job_data = resp.json()
            job_id = job_data.get("job_id")
            if not job_id:
                _log(f"[VoxCPM] No job_id returned: {job_data}")
                return None

        # 2. Poll Status
        _log(f"[VoxCPM] Job {job_id} submitted. Polling status...")
        async with httpx.AsyncClient(timeout=1820) as client:
            for _ in range(360):  # Poll up to 360 * 5s = 1800s (30 min)
                await asyncio.sleep(5)
                stat_resp = await client.get(f"{VOXCPM_URL}/status/{job_id}", headers=headers)
                if not stat_resp.is_success:
                    continue
                stat_data = stat_resp.json()
                status = stat_data.get("status")
                if status == "completed":
                    break
                elif status == "failed":
                    _log(f"[VoxCPM] Generation failed on GPU: {stat_data.get('error')}")
                    return None
            else:
                _log(f"[VoxCPM] Job {job_id} timed out after 30 min.")
                return None

        # 3. Download Audio
        async with httpx.AsyncClient(timeout=30) as client:
            dl_resp = await client.get(f"{VOXCPM_URL}/download/{job_id}", headers=headers)
            dl_resp.raise_for_status()
            return dl_resp.content

    except Exception as e:
        _log(f"[VoxCPM] Error: {e}")
        return None


# ── Wan2GP Image (local) ───────────────────────────────────────────────────────
async def _wan2gp_image(prompt: str) -> Optional[bytes]:
    """
    Call Wan2GP local image-gen server (Custom REST API on :9090).
    Returns PNG bytes or None on failure.
    """
    import httpx, asyncio
    try:
        headers = {"Content-Type": "application/json"}
        if WAN2GP_API_KEY:
            headers["X-API-Key"] = WAN2GP_API_KEY

        # 1. Submit Job
        payload = {
            "prompt":       prompt,
            "model":        "flux_dev",
            "resolution":   "1024x1024",
            "steps":        50,          # Flux Dev default max — sharpest output, fine for background pre-gen
            "guidance_scale": 7.5,       # stronger prompt adherence (default ~3.5 is too loose)
            "seed":         -1
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{WAN2GP_URL}/generate-image",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            job_data = resp.json()
            job_id = job_data.get("job_id")
            if not job_id:
                _log(f"[Wan2GP] No job_id returned: {job_data}")
                return None

        # 2. Poll Status
        _log(f"[Wan2GP] Job {job_id} submitted. Polling status...")
        async with httpx.AsyncClient(timeout=1820) as client:
            for _ in range(360):  # Poll up to 360 * 5s = 1800s (30 min)
                await asyncio.sleep(5)
                stat_resp = await client.get(f"{WAN2GP_URL}/status/{job_id}", headers=headers)
                if not stat_resp.is_success:
                    continue
                stat_data = stat_resp.json()
                status = stat_data.get("status")
                if status == "completed":
                    break
                elif status == "failed":
                    _log(f"[Wan2GP] Generation failed: {stat_data.get('error')}")
                    return None
            else:
                _log(f"[Wan2GP] Job {job_id} timed out after 30 min.")
                return None

        # 3. Download Image
        async with httpx.AsyncClient(timeout=30) as client:
            dl_resp = await client.get(f"{WAN2GP_URL}/download-image/{job_id}", headers=headers)
            dl_resp.raise_for_status()
            return dl_resp.content

    except Exception as e:
        _log(f"[Wan2GP] Error: {e}")
        return None


# ── WAV duration helper ────────────────────────────────────────────────────────
def _wav_duration(wav_bytes: bytes) -> float:
    """Extract duration in seconds from WAV header bytes."""
    try:
        import wave, io
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            return round(wf.getnframes() / wf.getframerate(), 2)
    except Exception:
        return 0.0


# ── Core: process ONE question row ────────────────────────────────────────────
async def _process_slide(idx: int, slide: dict, cache_id: str, language: str, subject_id: str = ""):
    """
    Generate image + audio for ONE slide concurrently, upload both to B2.
    Returns (enriched_slide_dict, audio_duration_float).
    Called in parallel for all slides via asyncio.gather().
    """
    slide_title    = slide.get("title", "Concept")
    slide_content  = slide.get("infographic") or slide.get("content") or ""
    key_points     = slide.get("keyPoints") or []
    is_story       = slide.get("isStory", False)
    is_tips        = slide.get("isTips", False)

    # ── Build a rich, context-aware image prompt from slide type ─────────────
    # Uses _build_image_prompt() which picks style based on slide type
    # (formula/biology/process/story/tips/summary/concept/default).
    img_prompt = _build_image_prompt(slide)

    narration = slide.get("narration") or slide.get("content") or ""

    # Clean greetings + normalize dashes/math/markdown before VoxCPM TTS
    narration = prepare_for_tts(narration)

    # ── Layer 1: Inline retry — run image+audio, retry up to 2× on failure ──
    # Each attempt: run both concurrently. If one fails, the 15s wait gives the
    # GPU time to recover before the next attempt. Catches ~90% of GPU timeouts.
    _INLINE_RETRIES = 2
    _RETRY_WAIT_S   = 15

    for attempt in range(1, _INLINE_RETRIES + 2):   # attempts: 1, 2, 3
        needs_img = _needs_image(slide)
        needs_aud = _needs_audio(slide)

        if not needs_img and not needs_aud:
            break   # both already filled from a previous attempt

        if attempt > 1:
            _log(f"[Pregen] slide {idx+1}: retry attempt {attempt} (waiting {_RETRY_WAIT_S}s)...")
            await asyncio.sleep(_RETRY_WAIT_S)

        # Run only what's still missing
        if needs_img and needs_aud:
            img_result, wav_result = await asyncio.gather(
                _wan2gp_image(img_prompt),
                _voxcpm_tts(narration, language),
                return_exceptions=True,
            )
        elif needs_img:
            img_result = await _wan2gp_image(img_prompt)
            wav_result = None
        else:  # needs_aud only
            img_result = None
            wav_result = await _voxcpm_tts(narration, language)

        # ── Image ────────────────────────────────────────────────────────────
        if needs_img and isinstance(img_result, bytes) and img_result:
            try:
                if subject_id:
                    try:
                        local_path = await write_image(subject_id, cache_id, idx, img_result)
                        slide["infographicLocalPath"] = local_path   # persist for future use
                        _log(f"[Pregen] slide {idx+1}: image ✓ local → {local_path}")
                    except Exception as e:
                        _log(f"[Pregen] slide {idx+1}: image local write failed (non-fatal) — {e}")
                b2_path = f"ai-teaching/{cache_id}/slide_{idx}.png"
                slide["infographicUrl"] = await upload_to_b2(img_result, b2_path, "image/png")
                _log(f"[Pregen] slide {idx+1}: image ✓ B2 → {slide['infographicUrl']}")
            except Exception as e:
                _log(f"[Pregen] slide {idx+1}: image B2 upload failed (attempt {attempt}) — {e}")
        elif needs_img:
            err = img_result if isinstance(img_result, Exception) else "no output"
            _log(f"[Pregen] slide {idx+1}: image attempt {attempt} failed ({err})")

        # ── Audio ────────────────────────────────────────────────────────────
        if needs_aud and isinstance(wav_result, bytes) and wav_result:
            try:
                duration = _wav_duration(wav_result)
                if subject_id:
                    try:
                        local_path = await write_audio(subject_id, cache_id, language, idx, wav_result)
                        slide["audioLocalPath"] = local_path   # persist for future use
                        _log(f"[Pregen] slide {idx+1}: audio ✓ local → {local_path}")
                    except Exception as e:
                        _log(f"[Pregen] slide {idx+1}: audio local write failed (non-fatal) — {e}")
                b2_path = f"ai-teaching/{cache_id}/audio_{idx}.wav"
                slide["audioUrl"] = await upload_to_b2(wav_result, b2_path, "audio/wav")
                slide["duration"] = duration
                _log(f"[Pregen] slide {idx+1}: audio ✓ {round(duration, 1)}s B2 → {slide['audioUrl']}")
            except Exception as e:
                _log(f"[Pregen] slide {idx+1}: audio B2 upload failed (attempt {attempt}) — {e}")
        elif needs_aud:
            err = wav_result if isinstance(wav_result, Exception) else "no output"
            _log(f"[Pregen] slide {idx+1}: audio attempt {attempt} failed ({err})")

    # Log if still missing after all retries (Layer 2 will catch these)
    if _needs_image(slide):
        _log(f"[Pregen] slide {idx+1}: image STILL missing after {_INLINE_RETRIES+1} attempts — will be caught by auto-retry")
    if _needs_audio(slide):
        _log(f"[Pregen] slide {idx+1}: audio STILL missing after {_INLINE_RETRIES+1} attempts — will be caught by auto-retry")

    duration = slide.get("duration", 0.0)
    return slide, duration


# ─────────────────────────────────────────────────────────────────────────────
# Phase helpers — called by run_pregen_batch's 3-phase loop
# ─────────────────────────────────────────────────────────────────────────────

async def _pregen_text_only(
    row: Dict[str, Any],
    db: AsyncSession,
    subject_name: str = "",
) -> Dict[str, Any]:
    """
    Phase A: generate slides via Ollama (or skip if slides already in DB).
    Returns the row dict updated with 'presentation_slides'.
    Raises on failure — caller marks row as 'failed'.

    subject_name: human-readable display name for the LLM prompt (optional).
    The UUID in row['subject_id'] is NEVER overwritten here.
    """
    cache_id = str(row["id"])
    question = row.get("question_text") or ""
    doc_id   = row.get("document_id")
    # Use display name for LLM prompt; fall back to UUID if name not provided
    subject_for_prompt = subject_name or row.get("subject_id") or "General"
    slides   = row.get("presentation_slides") or []

    if slides:
        _log(f"[Pregen-A] {cache_id[:8]}: slides already in DB ({len(slides)}) — skipping Ollama")
        return row

    _log(f"[Pregen-A] Ollama → generating slides for: {question[:80]}")

    # Fetch RAG context if doc-linked
    context = ""
    if doc_id:
        r = await db.execute(
            text("""
                SELECT chunk_text FROM document_chunks
                WHERE document_id = CAST(:doc_id AS uuid)
                ORDER BY chunk_index LIMIT 8
            """),
            {"doc_id": str(doc_id)},
        )
        context = "\n\n".join(r.scalars().all())

    slide_data = await generate_slides(
        question  = question,
        subject   = subject_for_prompt,   # display name for prompt quality
        context   = context,
        use_local = True,
    )
    slides = slide_data.get("presentation_slides", [])
    if not slides:
        raise ValueError("Ollama returned 0 slides")

    _log(f"[Pregen-A] ✓ {len(slides)} slides — saving to DB")
    await db.execute(
        text("""
            UPDATE teaching_qa_cache
            SET presentation_slides = CAST(:slides AS jsonb)
            WHERE id = CAST(:id AS uuid)
        """),
        {"slides": json.dumps(slides), "id": cache_id},
    )
    await db.commit()

    row = dict(row)
    row["presentation_slides"] = slides
    return row


async def _pregen_media_only(
    row: Dict[str, Any],
    slides: list,
) -> tuple:
    """
    Phase B: generate images + audio for every slide in parallel.
    Returns (enriched_slides, audio_url_list, total_duration, audio_durations, image_urls_map).
    Ollama MUST be evicted before calling this.
    """
    cache_id = str(row["id"])
    language = row.get("language") or "en-IN"
    subject  = row.get("subject_id") or "General"

    _log(f"[Pregen-B] {cache_id[:8]}: launching image+audio for {len(slides)} slides...")

    slide_results = await asyncio.gather(
        *[_process_slide(idx, slide, cache_id, language, subject)
          for idx, slide in enumerate(slides)],
        return_exceptions=True,
    )

    enriched_slides  = list(slides)  # copy
    audio_url_list:  list  = []
    audio_durations: dict  = {}
    image_urls_map:  dict  = {}
    total_duration   = 0.0

    for idx, result in enumerate(slide_results):
        if isinstance(result, Exception):
            _log(f"[Pregen-B] {cache_id[:8]} slide {idx+1}: _process_slide error — {result}")
            continue
        enriched_slide, duration = result
        enriched_slides[idx] = enriched_slide
        total_duration += duration
        if duration > 0:
            audio_durations[idx] = duration
        if enriched_slide.get("audioUrl"):
            audio_url_list.append({
                "slideIndex": idx,
                "audioUrl":   enriched_slide["audioUrl"],
                "duration":   duration,
            })
        if enriched_slide.get("infographicUrl"):
            image_urls_map[str(idx)] = {"url": enriched_slide["infographicUrl"]}

    return enriched_slides, audio_url_list, total_duration, audio_durations, image_urls_map


async def _pregen_manim_only(
    row: Dict[str, Any],
    slides: list,
    audio_durations: dict,
) -> dict:
    """
    Phase C: generate Manim videos for slides with visual_type='manim'.
    Returns manim_video_urls dict {slide_idx_str: {url, local_mp4, duration_seconds}}.
    Ollama MUST be loaded before calling this.
    """
    cache_id = str(row["id"])
    subject  = row.get("subject_id") or "General"

    manim_slides = [
        (idx, slides[idx])
        for idx in range(len(slides))
        if slides[idx].get("visual_type") == "manim" and idx in audio_durations
    ]
    if not manim_slides:
        return {}

    _log(f"[Pregen-C] {cache_id[:8]}: Manim for {len(manim_slides)} formula slides...")
    manim_video_urls: dict = {}

    for idx, slide in manim_slides:
        try:
            result = await generate_and_render_slide_manim(
                slide          = slide,
                slide_index    = idx,
                cache_id       = cache_id,
                subject_id     = subject,
                audio_duration = audio_durations[idx],
            )
            if result:
                local_mp4 = result.get("local_mp4") or ""
                b2_url    = result.get("b2_url") or ""
                # Upload to B2 if not already uploaded
                if local_mp4 and not b2_url:
                    try:
                        with open(local_mp4, "rb") as f:
                            mp4_bytes = f.read()
                        b2_path = f"ai-teaching/{cache_id}/manim_{idx}.mp4"
                        b2_url  = await upload_to_b2(mp4_bytes, b2_path, "video/mp4")
                        _log(f"[Pregen-C] {cache_id[:8]} slide {idx+1}: Manim B2 → {b2_url}")
                    except Exception as _ue:
                        _log(f"[Pregen-C] {cache_id[:8]} slide {idx+1}: Manim B2 upload failed — {_ue}")
                manim_video_urls[str(idx)] = {
                    "url":              b2_url,
                    "local_mp4":        local_mp4,
                    "duration_seconds": result.get("duration_seconds", 0.0),
                }
                _log(f"[Pregen-C] {cache_id[:8]} slide {idx+1}: Manim ✓")
            else:
                _log(f"[Pregen-C] {cache_id[:8]} slide {idx+1}: Manim failed — keeping static image")
        except Exception as e:
            _log(f"[Pregen-C] {cache_id[:8]} slide {idx+1}: Manim exception (non-fatal): {e}")

    return manim_video_urls


async def _save_row_result(
    db:               AsyncSession,
    row:              Dict[str, Any],
    slides:           list,
    audio_url_list:   list,
    total_duration:   float,
    manim_video_urls: dict,
    image_urls_map:   dict,
) -> None:
    """
    Save final results to DB and mark row as 'done'.
    Also writes the local slide-cache JSON file.

    Design: embedding is DECOUPLED from the core save.
    Step 1 always commits slides/audio/status='done'.
    Step 2 stores the embedding as a best-effort update.
    This guarantees a row is never stuck as 'failed' due to an embedding crash.
    """
    cache_id = str(row["id"])
    question = row.get("question_text") or ""
    language = row.get("language") or "en-IN"
    subject  = row.get("subject_id") or "General"  # always UUID (never display name)
    q_hash   = row.get("question_hash") or ""

    # ── Step 1: Core save — slides, audio URLs, status='done' ─────────────────
    # This MUST succeed before anything else.  No embedding here.
    await db.execute(
        text("""
            UPDATE teaching_qa_cache
            SET presentation_slides    = CAST(:slides AS jsonb),
                slide_audio_urls       = CAST(:audio  AS jsonb),
                total_duration_seconds = :dur,
                manim_video_urls       = CAST(:manim AS jsonb),
                image_urls             = CAST(:imgs  AS jsonb),
                pregen_status          = 'done',
                pregen_completed_at    = NOW()
            WHERE id = CAST(:id AS uuid)
        """),
        {
            "slides": json.dumps(slides),
            "audio":  json.dumps({"language": language, "urls": audio_url_list}),
            "dur":    total_duration,
            "manim":  json.dumps(manim_video_urls),
            "imgs":   json.dumps(image_urls_map),
            "id":     cache_id,
        },
    )
    await db.commit()
    _log(f"[Pregen] ✓ Saved: {cache_id[:8]} — pregen_status=done, "
         f"{len(slides)} slides, {round(total_duration)}s audio")

    # ── Step 2: Embedding — best-effort, never blocks save ────────────────────
    try:
        q_vec   = await embed_async(question)
        vec_str = vec_to_pg_str(q_vec)
        await db.execute(
            text("""
                UPDATE teaching_qa_cache
                SET question_embedding = CAST(:vec AS vector)
                WHERE id = CAST(:id AS uuid)
            """),
            {"vec": vec_str, "id": cache_id},
        )
        await db.commit()
        _log(f"[Pregen] ✓ Embedding stored: {cache_id[:8]}")
    except Exception as e:
        _log(f"[Pregen] Embedding failed (non-fatal — run backfill_embeddings.py later): {e}")

    # ── Step 3: Write local slide cache JSON ──────────────────────────────────
    if q_hash and subject:
        try:
            await write_slide_cache(subject, q_hash, {
                "cache_id":             cache_id,
                "presentationSlides":   slides,
                "slideAudioUrls":       {"language": language, "urls": audio_url_list},
                "totalDurationSeconds": total_duration,
                "manimVideoUrls":       manim_video_urls,
                "imageUrls":            image_urls_map,
            })
        except Exception as e:
            _log(f"[Pregen] local cache write failed (non-fatal): {e}")


async def _pregen_one(row: Dict[str, Any], db: AsyncSession) -> None:
    """
    Strictly sequential pipeline for one question:
      1. Ollama → slides JSON
      2. For each slide: Wan2GP image → VoxCPM audio
      3. Save results to DB
    """
    cache_id   = str(row["id"])
    language   = row.get("language") or "en-IN"
    question   = row.get("question_text") or ""
    doc_id     = row.get("document_id")
    subject    = row.get("subject_id") or "General"
    slides     = row.get("presentation_slides") or []

    _log(f"[Pregen] ── Starting: {question[:80]}")

    # ── Step 1: Generate slides via Ollama ────────────────────────────────────
    if not slides:
        _state.current_step = "ollama"
        _log(f"[Pregen] Step 1: Ollama → generating slides...")
        try:
            # Fetch RAG context if doc-linked
            context = ""
            if doc_id:
                r = await db.execute(
                    text("""
                        SELECT chunk_text FROM document_chunks
                        WHERE document_id = CAST(:doc_id AS uuid)
                        ORDER BY chunk_index LIMIT 8  -- 8 chunks ≈ full chapter context for deep narrations
                    """),
                    {"doc_id": str(doc_id)},
                )
                chunks = r.scalars().all()
                context = "\n\n".join(chunks)

            slide_data = await generate_slides(
                question  = question,
                subject   = subject,
                context   = context,
                use_local = True,       # Ollama — free, no API cost
            )
            slides = slide_data.get("presentation_slides", [])
            if not slides:
                raise ValueError("Ollama returned 0 slides")
            _log(f"[Pregen] Step 1: Ollama ✓ — {len(slides)} slides generated")

        except Exception as e:
            _log(f"[Pregen] Step 1: Ollama ✗ — {e}")
            raise   # bubble up → mark row as failed

        # Persist slides immediately so they aren't re-generated on retry
        await db.execute(
            text("""
                UPDATE teaching_qa_cache
                SET presentation_slides = CAST(:slides AS jsonb)
                WHERE id = CAST(:id AS uuid)
            """),
            {"slides": json.dumps(slides), "id": cache_id},
        )
        await db.commit()
    else:
        _log(f"[Pregen] Step 1: Slides already exist ({len(slides)} slides) — skipping Ollama")

    # ── Steps 2a+2b: ALL slides image+audio IN PARALLEL ─────────────────────
    # Note: Ollama VRAM already evicted before this batch loop starts (Phase 2).
    # Wan2GP + VoxCPM now have full GPU VRAM available.
    _state.current_step = "parallel_media"
    _log(f"[Pregen] Step 2a+2b: Launching image+audio for all {len(slides)} slides in parallel...")

    slide_results = await asyncio.gather(
        *[_process_slide(idx, slide, cache_id, language, subject) for idx, slide in enumerate(slides)],
        return_exceptions=True,
    )

    audio_url_list: list = []
    total_duration = 0.0

    for idx, result in enumerate(slide_results):
        if isinstance(result, Exception):
            _log(f"[Pregen] slide {idx+1}: exception in _process_slide — {result}")
            continue
        enriched_slide, duration = result
        slides[idx] = enriched_slide
        total_duration += duration
        if enriched_slide.get("audioUrl"):
            audio_url_list.append({
                "slideIndex": idx,
                "audioUrl":   enriched_slide["audioUrl"],
                "localPath":  enriched_slide.get("audioLocalPath", ""),
                "duration":   duration,
            })

    _log(f"[Pregen] Step 2a+2b: all slides done — total audio {round(total_duration, 1)}s")

    # ── Phase 4 gate: re-load Ollama before Manim code generation ────────────
    manim_slides_needed = any(
        slides[i].get("visual_type") == "manim" for i in range(len(slides))
    )
    if manim_slides_needed:
        _state.current_step = "loading_ollama"
        _log("[Pregen] Phase 4: re-loading Ollama for Manim code generation...")
        try:
            ready = await prepare_for_manim_generation()
            if ready:
                _log("[Pregen] Phase 4: Ollama model loaded ✓")
            else:
                _log("[Pregen] Phase 4: Ollama load timed out — Manim will be skipped")
        except Exception as e:
            _log(f"[Pregen] Phase 4 load failed (non-fatal): {e}")
            ready = False

    # ── Step 2c: Manim generation for formula slides ─────────────────────────
    # audio_durations collected from slide_results above; used for timing sync
    manim_video_urls: dict = {}  # {"slide_idx_str": {url, local_mp4, duration_seconds}}
    image_urls_map: dict = {}    # {"slide_idx_str": {url}}

    # Build per-slide audio duration map from results
    audio_durations: dict = {}
    for idx, result in enumerate(slide_results):
        if not isinstance(result, Exception):
            enriched_slide, duration = result
            if duration > 0:
                audio_durations[idx] = duration
            if enriched_slide.get("infographicUrl"):
                image_urls_map[str(idx)] = {
                    "url":       enriched_slide["infographicUrl"],
                    "localPath": enriched_slide.get("infographicLocalPath", ""),
                }

    # Manim phase: generate for slides flagged visual_type="manim"
    manim_slides = [
        (idx, slides[idx])
        for idx in range(len(slides))
        if slides[idx].get("visual_type") == "manim" and idx in audio_durations
    ]
    if manim_slides:
        _log(f"[Pregen] Step 2c: Manim for {len(manim_slides)} formula slides...")
        for idx, slide in manim_slides:
            _state.current_step = f"manim:{idx+1}"
            try:
                result = await generate_and_render_slide_manim(
                    slide=slide,
                    slide_index=idx,
                    cache_id=cache_id,
                    subject_id=subject,
                    audio_duration=audio_durations[idx],
                )
                if result:
                    local_mp4 = result.get("local_mp4") or ""
                    b2_url    = result.get("b2_url") or ""
                    # Upload to B2 if not already uploaded
                    if local_mp4 and not b2_url:
                        try:
                            with open(local_mp4, "rb") as f:
                                mp4_bytes = f.read()
                            b2_path = f"ai-teaching/{cache_id}/manim_{idx}.mp4"
                            b2_url  = await upload_to_b2(mp4_bytes, b2_path, "video/mp4")
                            _log(f"[Pregen] slide {idx+1}: Manim B2 → {b2_url}")
                        except Exception as _ue:
                            _log(f"[Pregen] slide {idx+1}: Manim B2 upload failed — {_ue}")
                    manim_video_urls[str(idx)] = {
                        "url":              b2_url,
                        "local_mp4":        local_mp4,
                        "duration_seconds": result.get("duration_seconds", 0.0),
                    }
                    _log(f"[Pregen] slide {idx+1}: Manim ✓")
                else:
                    _log(f"[Pregen] slide {idx+1}: Manim failed — keeping static image")
            except Exception as e:
                _log(f"[Pregen] slide {idx+1}: Manim exception (non-fatal): {e}")

    # ── Step 3: Save all results → mark done ─────────────────────────────────
    _state.current_step = "saving"
    _log(f"[Pregen] Step 3: Saving results to DB...")

    # Step 3a: Core save — slides/audio/status='done' (always runs)
    await db.execute(
        text("""
            UPDATE teaching_qa_cache
            SET presentation_slides    = CAST(:slides AS jsonb),
                slide_audio_urls       = CAST(:audio AS jsonb),
                total_duration_seconds = :dur,
                manim_video_urls       = CAST(:manim AS jsonb),
                image_urls             = CAST(:imgs  AS jsonb),
                pregen_status          = 'done',
                pregen_completed_at    = NOW()
            WHERE id = CAST(:id AS uuid)
        """),
        {
            "slides": json.dumps(slides),
            "audio":  json.dumps({"language": language, "urls": audio_url_list}),
            "dur":    total_duration,
            "manim":  json.dumps(manim_video_urls),
            "imgs":   json.dumps(image_urls_map),
            "id":     cache_id,
        },
    )
    await db.commit()
    _log(f"[Pregen] ✓ Done: {question[:60]} — {len(slides)} slides, {round(total_duration)}s audio, {len(manim_video_urls)} manim")

    # Step 3b: Embedding — best-effort, never blocks status='done'
    try:
        q_vec   = await embed_async(question)
        vec_str = vec_to_pg_str(q_vec)
        await db.execute(
            text("""
                UPDATE teaching_qa_cache
                SET question_embedding = CAST(:vec AS vector)
                WHERE id = CAST(:id AS uuid)
            """),
            {"vec": vec_str, "id": cache_id},
        )
        await db.commit()
        _log(f"[Pregen] ✓ Embedding stored: {cache_id[:8]}")
    except Exception as e:
        _log(f"[Pregen] Embedding failed (non-fatal — run backfill_embeddings.py later): {e}")

    # Step 3c: Write local slide cache JSON
    q_hash = row.get("question_hash") or ""
    if q_hash and subject:
        slide_cache_data = {
            "cache_id":            cache_id,
            "presentationSlides":  slides,
            "slideAudioUrls":      {"language": language, "urls": audio_url_list},
            "totalDurationSeconds": total_duration,
            "manimVideoUrls":      manim_video_urls,
            "imageUrls":           image_urls_map,
        }
        try:
            await write_slide_cache(subject, q_hash, slide_cache_data)
            _log(f"[Pregen] ✓ Local slide cache written: {subject}/{q_hash}")
        except Exception as e:
            _log(f"[Pregen] Local slide cache write failed (non-fatal): {e}")


# _pregen_one_question() removed — questions are synced into teaching_qa_cache
# at the start of run_pregen_batch() and processed through the single unified loop.


# ── Batch runner ──────────────────────────────────────────────────────────────
async def run_pregen_batch(
    subject_id:  str,
    db_factory,
    limit:       int = 500,
    topic_id:    str = None,
    chapter_id:  str = None,
) -> None:
    """
    Background batch job — single unified queue.

    On every start:
      1. Reset any stuck 'processing' rows → 'pending'  (handles container restarts)
      2. Sync unfinished rows from questions table → teaching_qa_cache
         using ON CONFLICT DO NOTHING  (never overwrites 'done' rows)
      3. Process all 'pending' rows in one simple loop
    """
    global _state
    _state = PregenState(
        running        = True,
        stop_requested = False,
        subject_id     = subject_id,
        started_at     = time.time(),
        phase          = "text",
    )
    _log(f"[Pregen] ══ Batch started: subject={subject_id} topic={topic_id} chapter={chapter_id} limit={limit} ══")

    # ── Phase 1 gate: ensure Ollama model is loaded before text generation ────
    _state.phase = "text"
    _log("[Pregen] Phase 1: ensuring Ollama model is in VRAM...")
    try:
        model_ready = await prepare_for_text_generation()
        if not model_ready:
            _log("[Pregen] ⚠ Ollama model did not load — proceeding anyway (may fail per-row)")
    except Exception as e:
        _log(f"[Pregen] Ollama model load check failed (non-fatal): {e}")

    try:
        from core.cache import hash_question

        async with db_factory() as db:
            # ── Step 1: Resolve subject name for Ollama prompts ───────────────
            sub_row = (await db.execute(
                text("SELECT name FROM subjects WHERE subject_id=:sid"),
                {"sid": subject_id},
            )).fetchone()
            subject_name = sub_row[0] if sub_row else subject_id

            # ── Step 2: Reset stuck 'processing' rows → 'pending' ─────────────
            # These got stuck because the container restarted mid-generation.
            stuck = await db.execute(
                text("""
                    UPDATE teaching_qa_cache
                    SET pregen_status = 'pending'
                    WHERE subject_id = :subj
                      AND pregen_status = 'processing'
                """),
                {"subj": subject_id},
            )
            await db.commit()
            if stuck.rowcount:
                _log(f"[Pregen] Reset {stuck.rowcount} stuck 'processing' rows → 'pending'")

            # ── Step 3: Sync questions table → teaching_qa_cache ──────────────
            # ON CONFLICT DO NOTHING: never touch rows that are already 'done'.
            q_filters = ["is_pregen_done = FALSE", "subject_id = :sid"]
            q_params: Dict[str, Any] = {"sid": subject_id}
            if topic_id:
                q_filters.append("topic_id = CAST(:tid AS uuid)")
                q_params["tid"] = topic_id
            if chapter_id:
                q_filters.append("chapter_id = CAST(:cid AS uuid)")
                q_params["cid"] = chapter_id

            pending_questions = (await db.execute(
                text(f"""
                    SELECT id, question_text, subject_id, chapter_id, topic_id
                    FROM questions
                    WHERE {" AND ".join(q_filters)}
                    ORDER BY created_at ASC
                    LIMIT :lim
                """),
                {**q_params, "lim": limit},
            )).fetchall()

            synced = 0
            for q in pending_questions:
                q_hash = hash_question(q.question_text)
                ch_id  = str(q.chapter_id) if q.chapter_id else None
                t_id   = str(q.topic_id)   if q.topic_id   else None
                res = await db.execute(
                    text("""
                        INSERT INTO teaching_qa_cache
                            (id, subject_id, chapter_id, topic_id,
                             question_hash, question_text, variation_number, pregen_status)
                        VALUES
                            (gen_random_uuid(), :sid, :cid, :tid,
                             :qhash, :qtext, 1, 'pending')
                        ON CONFLICT (question_hash, subject_id, variation_number)
                        DO NOTHING
                    """),
                    {"sid": q.subject_id, "cid": ch_id, "tid": t_id,
                     "qhash": q_hash, "qtext": q.question_text},
                )
                if res.rowcount:
                    synced += 1
            await db.commit()
            _log(f"[Pregen] Synced {synced} new questions into cache queue")

            # ── Count total pending for progress tracking ──────────────────────
            cache_filter = "subject_id = :subj AND pregen_status = 'pending'"
            cache_params: Dict[str, Any] = {"subj": subject_id}
            if topic_id:
                cache_filter += " AND topic_id = CAST(:tid AS uuid)"
                cache_params["tid"] = topic_id

            total_pending = (await db.execute(
                text(f"SELECT COUNT(*) FROM teaching_qa_cache WHERE {cache_filter}"),
                cache_params,
            )).scalar()
            _state.total = int(total_pending or 0)

        _log(f"[Pregen] {_state.total} rows pending — starting 3-phase batch")

        # ════════════════════════════════════════════════════════════════════
        # PHASE A — Text generation (Ollama loaded, no media models)
        #   Generate slides for EVERY pending row before touching any GPU media.
        #   Ollama is already loaded from Phase 1 above.
        # ════════════════════════════════════════════════════════════════════
        _state.phase = "text"
        _log("[Pregen] ══ Phase A: text generation for all rows ══")

        phase_a_rows: list[dict] = []   # rows that completed Phase A
        processed_a = 0

        while not _state.stop_requested and processed_a < limit:
            async with db_factory() as db:
                rows = (await db.execute(
                    text("""
                        SELECT id, question_text, presentation_slides,
                               slide_audio_urls, language, subject_id,
                               document_id, question_hash
                        FROM teaching_qa_cache
                        WHERE subject_id = :subj
                          AND pregen_status = 'pending'
                        ORDER BY usage_count DESC NULLS LAST, created_at ASC
                        LIMIT 1
                    """),
                    {"subj": subject_id},
                )).fetchall()

            if not rows:
                break

            row_dict = dict(rows[0]._mapping)
            # ⚠️ IMPORTANT: never overwrite row_dict["subject_id"] with subject_name.
            # row_dict["subject_id"] must stay as the UUID so that local storage
            # paths (write_image / write_audio) are written to the correct folder.
            # subject_name is passed separately to _pregen_text_only for LLM prompts.
            cache_id = str(row_dict["id"])
            _state.current_question = (row_dict.get("question_text") or "")[:80]

            # Mark processing
            async with db_factory() as db:
                await db.execute(
                    text("UPDATE teaching_qa_cache SET pregen_status='processing' "
                         "WHERE id=CAST(:id AS uuid) AND pregen_status='pending'"),
                    {"id": cache_id},
                )
                await db.commit()

            try:
                async with db_factory() as db:
                    row_dict = await _pregen_text_only(
                        row_dict, db, subject_name=subject_name
                    )
                phase_a_rows.append(row_dict)
                _log(f"[Pregen-A] ✓ text: {cache_id[:8]} ({len(row_dict.get('presentation_slides') or [])} slides)")
            except Exception as e:
                _state.failed += 1
                _state.last_error = str(e)
                _log(f"[Pregen-A] ✗ text failed id={cache_id[:8]}: {e}")
                async with db_factory() as db:
                    await db.execute(
                        text("UPDATE teaching_qa_cache SET pregen_status='failed' "
                             "WHERE id=CAST(:id AS uuid)"),
                        {"id": cache_id},
                    )
                    await db.commit()

            processed_a += 1

        _log(f"[Pregen] Phase A complete — {len(phase_a_rows)} rows have text, "
             f"{_state.failed} failed")

        # ════════════════════════════════════════════════════════════════════
        # PHASE B — Media generation (Ollama EVICTED, Wan2GP + VoxCPM free)
        #   Evict Ollama ONCE here, after ALL text is done.
        #   Then process each row's image + audio (parallel within row).
        # ════════════════════════════════════════════════════════════════════
        _state.phase = "media"
        _log("[Pregen] ══ Phase B: evicting Ollama → media generation ══")
        try:
            n_evicted = await prepare_for_media_generation()
            _log(f"[Pregen] Phase B: {n_evicted} model(s) evicted from VRAM ✓")
        except Exception as e:
            _log(f"[Pregen] Phase B eviction failed (non-fatal): {e}")

        phase_b_rows: list[dict] = []   # rows that completed Phase B

        for row_dict in phase_a_rows:
            if _state.stop_requested:
                break
            cache_id = str(row_dict["id"])
            slides   = row_dict.get("presentation_slides") or []
            language = row_dict.get("language") or "en-IN"
            _state.current_question = (row_dict.get("question_text") or "")[:80]

            try:
                enriched_slides, audio_url_list, total_duration, audio_durations, image_urls_map = \
                    await _pregen_media_only(row_dict, slides)
                row_dict["_enriched_slides"]  = enriched_slides
                row_dict["_audio_url_list"]   = audio_url_list
                row_dict["_total_duration"]   = total_duration
                row_dict["_audio_durations"]  = audio_durations
                row_dict["_image_urls_map"]   = image_urls_map
                phase_b_rows.append(row_dict)
                _log(f"[Pregen-B] ✓ media: {cache_id[:8]} "
                     f"({len(enriched_slides)} slides, {round(total_duration,1)}s)")
            except Exception as e:
                _state.failed += 1
                _state.last_error = str(e)
                _log(f"[Pregen-B] ✗ media failed id={cache_id[:8]}: {e}")
                async with db_factory() as db:
                    await db.execute(
                        text("UPDATE teaching_qa_cache SET pregen_status='failed' "
                             "WHERE id=CAST(:id AS uuid)"),
                        {"id": cache_id},
                    )
                    await db.commit()

        _log(f"[Pregen] Phase B complete — {len(phase_b_rows)} rows have media")

        # ════════════════════════════════════════════════════════════════════
        # PHASE C — Manim generation (reload Ollama only if needed)
        #   Only for slides flagged visual_type="manim".
        #   Save all results to DB as 'done' at the end of this phase.
        # ════════════════════════════════════════════════════════════════════
        _state.phase = "manim"
        manim_rows_needed = any(
            any(s.get("visual_type") == "manim"
                for s in (r.get("_enriched_slides") or []))
            for r in phase_b_rows
        )

        if manim_rows_needed:
            _log("[Pregen] ══ Phase C: reloading Ollama for Manim generation ══")
            try:
                ready = await prepare_for_manim_generation()
                _log(f"[Pregen] Phase C: Ollama {'loaded ✓' if ready else 'load timed out — Manim skipped'}")
            except Exception as e:
                _log(f"[Pregen] Phase C Ollama load failed (non-fatal): {e}")
                ready = False
        else:
            _log("[Pregen] ══ Phase C: no manim slides — skipping ══")
            ready = False

        for row_dict in phase_b_rows:
            if _state.stop_requested:
                break
            cache_id       = str(row_dict["id"])
            enriched_slides = row_dict["_enriched_slides"]
            audio_durations = row_dict["_audio_durations"]
            image_urls_map  = row_dict["_image_urls_map"]
            audio_url_list  = row_dict["_audio_url_list"]
            total_duration  = row_dict["_total_duration"]
            subject_name_   = row_dict.get("subject_id") or subject_name

            # Manim per-row
            manim_video_urls: dict = {}
            if ready:
                try:
                    manim_video_urls = await _pregen_manim_only(
                        row_dict, enriched_slides, audio_durations
                    )
                    if manim_video_urls:
                        _log(f"[Pregen-C] ✓ manim: {cache_id[:8]} "
                             f"({len(manim_video_urls)} videos)")
                except Exception as e:
                    _log(f"[Pregen-C] manim failed id={cache_id[:8]} (non-fatal): {e}")

            # Save everything → 'done'
            try:
                async with db_factory() as db:
                    await _save_row_result(
                        db           = db,
                        row          = row_dict,
                        slides       = enriched_slides,
                        audio_url_list = audio_url_list,
                        total_duration = total_duration,
                        manim_video_urls = manim_video_urls,
                        image_urls_map   = image_urls_map,
                    )
                # Link back to questions table
                async with db_factory() as db:
                    await db.execute(
                        text("""
                            UPDATE questions
                            SET is_pregen_done = TRUE,
                                cache_id = CAST(:cid AS uuid)
                            WHERE question_text = :qtext
                              AND subject_id = :sid
                              AND is_pregen_done = FALSE
                        """),
                        {"cid": cache_id,
                         "qtext": row_dict.get("question_text", ""),
                         "sid": subject_id},
                    )
                    await db.commit()

                _state.done += 1
                _log(f"[Pregen] ✓ Done: {cache_id[:8]} — "
                     f"{len(enriched_slides)} slides, "
                     f"{round(total_duration)}s audio, "
                     f"{len(manim_video_urls)} manim")
            except Exception as e:
                _state.failed += 1
                _state.last_error = str(e)
                _log(f"[Pregen] ✗ Save failed id={cache_id[:8]}: {e}")
                async with db_factory() as db:
                    await db.execute(
                        text("UPDATE teaching_qa_cache SET pregen_status='failed' "
                             "WHERE id=CAST(:id AS uuid)"),
                        {"id": cache_id},
                    )
                    await db.commit()

    except Exception as e:
        _log(f"[Pregen] Fatal batch error: {e}")
        _state.last_error = str(e)

    finally:
        _state.running          = False
        _state.current_step     = ""
        _state.current_question = ""
        elapsed = round(time.time() - _state.started_at, 1)
        _log(
            f"[Pregen] ══════════════ Batch ended: "
            f"done={_state.done} failed={_state.failed} "
            f"elapsed={elapsed}s ══════════════"
        )
        # ── Layer 2: Auto-trigger media retry after every batch ───────────────
        # Catches any slides where inline retries (Layer 1) were exhausted.
        # Runs only if there are actually missing media rows — safe no-op otherwise.
        _log("[Pregen] Auto-triggering media retry pass (Layer 2)...")
        try:
            await retry_media_for_rows(subject_id, db_factory)
        except Exception as e:
            _log(f"[Pregen] Auto-retry pass failed (non-fatal): {e}")


# ── Layer 2: Smart Media Retry (3-phase, mirrors run_pregen_batch) ─────────────
async def retry_media_for_rows(subject_id: str, db_factory) -> None:
    """
    3-phase retry that mirrors run_pregen_batch:
      Retry-A  Ollama ON  → re-generate slide TEXT for rows with no slides
      Retry-B  Ollama OFF → image + audio for ALL rows missing media
      Retry-C  Ollama ON  → Manim for rows missing manim video urls
    Reuses the same _pregen_text_only / _pregen_media_only /
    _pregen_manim_only / _save_row_result helpers as the main batch.

    Called automatically after every batch (auto safety-net).
    Can also be triggered manually via POST /pregen/retry-media.
    """
    global _retry_state
    _retry_state = RetryState(
        running    = True,
        subject_id = subject_id,
        started_at = time.time(),
    )
    _log(f"[Retry] ══ 3-phase retry started: subject={subject_id} ══")

    try:
        # ── Resolve subject name ──────────────────────────────────────────────
        async with db_factory() as db:
            name_row = (await db.execute(
                text("SELECT name FROM subjects WHERE id = CAST(:subj AS uuid) LIMIT 1"),
                {"subj": subject_id},
            )).fetchone()
        subject_name = name_row.name if name_row else subject_id

        # ── Fetch ALL incomplete rows ─────────────────────────────────────────
        # "incomplete" = pending/failed/done but missing slides, image, audio, or manim
        async with db_factory() as db:
            all_rows = (await db.execute(
                text("""
                    SELECT id, question_text, question_hash, language,
                           subject_id, document_id, presentation_slides,
                           slide_audio_urls, image_urls, manim_video_urls,
                           pregen_status
                    FROM teaching_qa_cache
                    WHERE subject_id = :subj
                      AND (
                        pregen_status IN ('failed', 'pending')
                        OR (
                          pregen_status = 'done'
                          AND (
                            jsonb_array_length(COALESCE(presentation_slides,'[]'::jsonb)) = 0
                            OR jsonb_array_length(COALESCE(slide_audio_urls->'urls','[]'::jsonb)) = 0
                            OR EXISTS(
                              SELECT 1 FROM jsonb_array_elements(
                                COALESCE(presentation_slides,'[]'::jsonb)) s
                              WHERE (s->>'infographicUrl') IS NULL
                                 OR (s->>'infographicUrl') = ''
                            )
                            OR EXISTS(
                              SELECT 1 FROM jsonb_array_elements(
                                COALESCE(presentation_slides,'[]'::jsonb)) s
                              WHERE (s->>'audioUrl') IS NULL
                                 OR (s->>'audioUrl') = ''
                            )
                          )
                        )
                      )
                    ORDER BY created_at ASC
                """),
                {"subj": subject_id},
            )).fetchall()

        _retry_state.total = len(all_rows)
        if not all_rows:
            _log("[Retry] No incomplete rows — nothing to do.")
            return

        _log(f"[Retry] Found {len(all_rows)} incomplete rows")

        # ═══════════════════════════════════════════════════════════════════
        # RETRY PHASE A — Ollama: regenerate slide text for rows missing it
        # ═══════════════════════════════════════════════════════════════════
        _retry_state.current_step = "retry-text"
        _log("[Retry] ══ Phase A: text generation for rows with no slides ══")

        rows_need_text = [
            dict(r._mapping)
            for r in all_rows
            if not (r.presentation_slides and len(r.presentation_slides) > 0)
        ]
        rows_have_text = [
            dict(r._mapping)
            for r in all_rows
            if r.presentation_slides and len(r.presentation_slides) > 0
        ]

        if rows_need_text:
            # Load Ollama
            try:
                loaded = await prepare_for_text_generation()
                _log(f"[Retry-A] Ollama {'loaded ✓' if loaded else 'load failed — text retry skipped'}")
            except Exception as e:
                loaded = False
                _log(f"[Retry-A] Ollama load error: {e}")

            if loaded:
                for row_dict in rows_need_text:
                    cache_id = str(row_dict["id"])
                    row_dict["subject_id"] = subject_name
                    try:
                        async with db_factory() as db:
                            row_dict = await _pregen_text_only(row_dict, db)
                        rows_have_text.append(row_dict)
                        _log(f"[Retry-A] ✓ text: {cache_id[:8]} "
                             f"({len(row_dict.get('presentation_slides') or [])} slides)")
                    except Exception as e:
                        _retry_state.failed += 1
                        _log(f"[Retry-A] ✗ text failed {cache_id[:8]}: {e}")
            else:
                _log("[Retry-A] Ollama not ready — skipping text-missing rows")
        else:
            _log("[Retry-A] All rows already have slide text — skipping Ollama load")

        _log(f"[Retry] Phase A complete — {len(rows_have_text)} rows have text")

        # ═══════════════════════════════════════════════════════════════════
        # RETRY PHASE B — evict Ollama → image + audio for rows missing media
        # ═══════════════════════════════════════════════════════════════════
        _retry_state.current_step = "retry-media"
        _log("[Retry] ══ Phase B: evicting Ollama → image + audio ══")

        try:
            n_evicted = await prepare_for_media_generation()
            _log(f"[Retry] Phase B: {n_evicted} model(s) evicted ✓")
        except Exception as e:
            _log(f"[Retry] Phase B eviction failed (non-fatal): {e}")

        phase_b_rows: list[dict] = []

        for row_dict in rows_have_text:
            cache_id = str(row_dict["id"])
            slides   = list(row_dict.get("presentation_slides") or [])
            language = row_dict.get("language") or "en-IN"
            row_dict["subject_id"] = subject_name

            # Check whether image/audio is actually missing
            needs_img = any(
                not (s.get("infographicUrl") or "").startswith("http")
                for s in slides
            )
            needs_aud = any(
                not (s.get("audioUrl") or "").startswith("http")
                for s in slides
            )

            if not needs_img and not needs_aud:
                # Media complete — still push to Phase C for Manim check
                row_dict["_enriched_slides"]  = slides
                row_dict["_audio_url_list"]   = [
                    {"slideIndex": i, "audioUrl": s["audioUrl"],
                     "duration": s.get("duration", 0)}
                    for i, s in enumerate(slides)
                    if (s.get("audioUrl") or "").startswith("http")
                ]
                row_dict["_total_duration"]   = sum(
                    s.get("duration", 0) for s in slides
                    if (s.get("audioUrl") or "").startswith("http")
                )
                row_dict["_audio_durations"]  = {
                    i: s.get("duration", 0)
                    for i, s in enumerate(slides)
                    if s.get("duration", 0) > 0
                }
                row_dict["_image_urls_map"]   = {
                    str(i): {"url": s["infographicUrl"]}
                    for i, s in enumerate(slides)
                    if (s.get("infographicUrl") or "").startswith("http")
                }
                phase_b_rows.append(row_dict)
                _log(f"[Retry-B] {cache_id[:8]}: media already complete — forwarding to Phase C")
                continue

            try:
                enriched_slides, audio_url_list, total_duration, audio_durations, image_urls_map = \
                    await _pregen_media_only(row_dict, slides)
                row_dict["_enriched_slides"]  = enriched_slides
                row_dict["_audio_url_list"]   = audio_url_list
                row_dict["_total_duration"]   = total_duration
                row_dict["_audio_durations"]  = audio_durations
                row_dict["_image_urls_map"]   = image_urls_map
                phase_b_rows.append(row_dict)
                _log(f"[Retry-B] ✓ media: {cache_id[:8]} "
                     f"({len(enriched_slides)} slides, {round(total_duration,1)}s)")
            except Exception as e:
                _retry_state.failed += 1
                _log(f"[Retry-B] ✗ media failed {cache_id[:8]}: {e}")

        _log(f"[Retry] Phase B complete — {len(phase_b_rows)} rows have media")

        # ═══════════════════════════════════════════════════════════════════
        # RETRY PHASE C — reload Ollama → Manim for formula slides
        # ═══════════════════════════════════════════════════════════════════
        _retry_state.current_step = "retry-manim"

        manim_needed = any(
            any(s.get("visual_type") == "manim"
                for s in (r.get("_enriched_slides") or []))
            for r in phase_b_rows
        )

        if manim_needed:
            _log("[Retry] ══ Phase C: reloading Ollama for Manim ══")
            try:
                ready = await prepare_for_manim_generation()
                _log(f"[Retry] Phase C: Ollama {'loaded ✓' if ready else 'timed out — Manim skipped'}")
            except Exception as e:
                ready = False
                _log(f"[Retry] Phase C Ollama load failed (non-fatal): {e}")
        else:
            _log("[Retry] ══ Phase C: no manim slides — skipping ══")
            ready = False

        # ── Save all Phase B rows ─────────────────────────────────────────
        for row_dict in phase_b_rows:
            cache_id        = str(row_dict["id"])
            enriched_slides = row_dict["_enriched_slides"]
            audio_durations = row_dict["_audio_durations"]
            audio_url_list  = row_dict["_audio_url_list"]
            total_duration  = row_dict["_total_duration"]
            image_urls_map  = row_dict["_image_urls_map"]

            # Manim
            manim_video_urls: dict = {}
            if ready:
                has_manim_slides = any(
                    s.get("visual_type") == "manim"
                    for s in enriched_slides
                )
                if has_manim_slides:
                    try:
                        manim_video_urls = await _pregen_manim_only(
                            row_dict, enriched_slides, audio_durations
                        )
                        if manim_video_urls:
                            _log(f"[Retry-C] ✓ manim: {cache_id[:8]} "
                                 f"({len(manim_video_urls)} videos)")
                    except Exception as e:
                        _log(f"[Retry-C] manim failed {cache_id[:8]} (non-fatal): {e}")

            # Save
            try:
                async with db_factory() as db:
                    await _save_row_result(
                        db               = db,
                        row              = row_dict,
                        slides           = enriched_slides,
                        audio_url_list   = audio_url_list,
                        total_duration   = total_duration,
                        manim_video_urls = manim_video_urls,
                        image_urls_map   = image_urls_map,
                    )
                # Invalidate Redis L1 so fresh data is served next time
                q_hash = row_dict.get("question_hash") or ""
                if q_hash:
                    try:
                        await delete_from_cache(q_hash, subject_id)
                    except Exception as e:
                        _log(f"[Retry] Redis invalidate failed (non-fatal): {e}")

                _retry_state.done += 1
                slides_with_img = sum(
                    1 for s in enriched_slides
                    if (s.get("infographicUrl") or "").startswith("http")
                )
                slides_with_aud = sum(
                    1 for s in enriched_slides
                    if (s.get("audioUrl") or "").startswith("http")
                )
                _log(f"[Retry] ✓ Done: {cache_id[:8]} — "
                     f"img={slides_with_img}/{len(enriched_slides)} "
                     f"aud={slides_with_aud}/{len(enriched_slides)} "
                     f"manim={len(manim_video_urls)}")
            except Exception as e:
                _retry_state.failed += 1
                _retry_state.last_error = str(e)
                _log(f"[Retry] ✗ Save failed {cache_id[:8]}: {e}")

    except Exception as e:
        _log(f"[Retry] Fatal error: {e}")
        _retry_state.last_error = str(e)

    finally:
        _retry_state.running      = False
        _retry_state.current_step = ""
        elapsed = round(time.time() - _retry_state.started_at, 1)
        _log(
            f"[Retry] ══ 3-phase retry complete: "
            f"done={_retry_state.done} failed={_retry_state.failed} "
            f"elapsed={elapsed}s ══"
        )



async def predict_questions(doc_id: str, subject_id: str, chunk_texts: list[str], async_session_maker) -> int:
    """
    Generate up to 20 AI-predicted questions from document text.
    Insert them into teaching_qa_cache with status='pending'.
    Returns the number of questions generated.
    """
    import hashlib
    from core.slide_generator import OLLAMA_URL, OLLAMA_MODEL
    
    context = "\n".join(chunk_texts)[:8000]

    prompt = f"""You are an expert teacher. Based on the following document content, generate 20 important questions that students might ask about this material.
Return ONLY a valid JSON array of strings, like this: ["Question 1", "Question 2"]
Do not output any markdown formatting, backticks, or introductory text. JUST the JSON array.

DOCUMENT CONTENT:
{context}
"""

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
            )
            resp.raise_for_status()
            result_text = resp.json().get("response", "[]").strip()
            
            questions = json.loads(result_text)
            if not isinstance(questions, list):
                _log(f"[predict_questions] Ollama returned non-list: {questions}")
                return 0
                
            questions = [str(q).strip() for q in questions if str(q).strip()]
    except Exception as e:
        _log(f"[predict_questions] Ollama generation failed: {e}")
        return 0

    count = 0
    async with async_session_maker() as db:
        for q in questions[:20]:
            q_hash = hashlib.md5(q.lower().strip().encode()).hexdigest()
            new_id = str(uuid.uuid4())
            try:
                await db.execute(
                    text("""
                        INSERT INTO teaching_qa_cache 
                        (id, subject_id, question_hash, question_text, variation_number, pregen_status)
                        VALUES (:id, :subject_id, :question_hash, :question, 1, 'pending')
                        ON CONFLICT (question_hash, subject_id, variation_number) DO UPDATE SET pregen_status = 'pending'
                    """),
                    {"id": new_id, "subject_id": subject_id, "question_hash": q_hash, "question": q}
                )
                count += 1
            except Exception as e:
                _log(f"[predict_questions] DB insert failed for '{q}': {e}")
        await db.commit()
        
    return count
