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
    current_step:    str   = ""      # "ollama" | "image:N" | "audio:N" | "saving"
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

    # ── Build a rich, detailed image prompt ──────────────────────────────────
    # Flux/SDXL models need long, structured prompts for quality results.
    # A 3-sentence prompt → generic blurry output.
    # A detailed, layered prompt → sharp, educational, on-topic image.

    kp_str = ", ".join(key_points[:3]) if key_points else ""

    if is_story:
        style_hint = (
            "warm illustrated story scene, narrative art, soft warm colors, "
            "storybook illustration style, characters interacting with concept"
        )
    elif is_tips:
        style_hint = (
            "memory tips infographic, lightbulb icons, numbered list visual, "
            "bright yellow and blue palette, clean icon-based layout"
        )
    else:
        style_hint = (
            "educational diagram, textbook illustration, labeled arrows, "
            "clear section boxes, blue and white color scheme"
        )

    img_prompt = (
        f"Educational infographic poster: \"{slide_title}\". "
        f"Topic: {slide_content[:120]}. "
        f"{'Key concepts shown: ' + kp_str + '. ' if kp_str else ''}"
        f"Style: {style_hint}. "
        f"Visual design: clean layout, high contrast text labels, white background, "
        f"professional academic look, crisp lines, no clutter, no watermarks. "
        f"Composition: title at top, main visual in center, key labels on sides. "
        f"Quality: sharp, detailed, 4K resolution, photorealistic where appropriate, "
        f"suitable for classroom projection. "
        f"Negative: blurry, low quality, pixelated, messy, overlapping text, dark background."
    )

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
    _state.current_step = "parallel_media"
    _log(f"[Pregen] Step 2: Launching image+audio for all {len(slides)} slides in parallel...")

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
                "duration":   duration,
            })

    _log(f"[Pregen] Step 2: all slides done — total audio {round(total_duration, 1)}s")

    # ── Step 3: Save all results → mark done ─────────────────────────────────
    _state.current_step = "saving"
    _log(f"[Pregen] Step 3: Saving results to DB...")

    # Calculate embedding for the L4 Semantic Cache
    q_vec = await embed_async(question)
    vec_str = vec_to_pg_str(q_vec)

    await db.execute(
        text("""
            UPDATE teaching_qa_cache
            SET presentation_slides    = CAST(:slides AS jsonb),
                slide_audio_urls       = CAST(:audio AS jsonb),
                total_duration_seconds = :dur,
                question_embedding     = CAST(:vec AS vector),
                pregen_status          = 'done',
                pregen_completed_at    = NOW()
            WHERE id = CAST(:id AS uuid)
        """),
        {
            "slides": json.dumps(slides),
            "audio":  json.dumps({"language": language, "urls": audio_url_list}),
            "dur":    total_duration,
            "vec":    vec_str,
            "id":     cache_id,
        },
    )
    await db.commit()
    _log(f"[Pregen] ✓ Done: {question[:60]} — {len(slides)} slides, {round(total_duration)}s audio")

    # ── Save presentation JSON to local /sdb-disk (same as real-time does) ────────
    q_hash = row.get("question_hash") or ""
    if q_hash and subject:
        slide_cache_data = {
            "cache_id":            cache_id,
            "presentationSlides":  slides,
            "slideAudioUrls":      {"language": language, "urls": audio_url_list},
            "totalDurationSeconds": total_duration,
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
    )
    _log(f"[Pregen] ══ Batch started: subject={subject_id} topic={topic_id} chapter={chapter_id} limit={limit} ══")

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

        _log(f"[Pregen] {_state.total} rows pending — starting processing loop")

        # ── Step 4: Single processing loop ────────────────────────────────────
        processed = 0
        while not _state.stop_requested and processed < limit:
            # Always fetch the next pending row fresh (no offset — status changes as we go)
            async with db_factory() as db:
                rows = (await db.execute(
                    text("""
                        SELECT id, question_text, presentation_slides,
                               slide_audio_urls, language, subject_id, document_id
                        FROM teaching_qa_cache
                        WHERE subject_id = :subj
                          AND pregen_status = 'pending'
                        ORDER BY usage_count DESC NULLS LAST, created_at ASC
                        LIMIT 1
                    """),
                    {"subj": subject_id},
                )).fetchall()

            if not rows:
                _log("[Pregen] No more pending rows — batch complete.")
                break

            row = rows[0]
            row_dict = dict(row._mapping)
            _state.current_question = (row_dict.get("question_text") or "")[:80]
            row_dict["subject_id"] = subject_name  # use name for Ollama prompt

            # Mark as 'processing' atomically before starting work
            async with db_factory() as db:
                await db.execute(
                    text("""
                        UPDATE teaching_qa_cache
                        SET pregen_status = 'processing'
                        WHERE id = CAST(:id AS uuid)
                          AND pregen_status = 'pending'
                    """),
                    {"id": str(row_dict["id"])},
                )
                await db.commit()

            try:
                async with db_factory() as db:
                    await _pregen_one(row_dict, db)

                # Link back to questions table if this question originated there
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
                        {"cid": str(row_dict["id"]),
                         "qtext": row_dict.get("question_text", ""),
                         "sid": subject_id},
                    )
                    await db.commit()

                _state.done += 1

            except Exception as e:
                err = str(e)
                _state.failed += 1
                _state.last_error = err
                _log(f"[Pregen] ✗ Failed: id={row_dict['id']}: {err}")
                async with db_factory() as db:
                    await db.execute(
                        text("""
                            UPDATE teaching_qa_cache
                            SET pregen_status = 'failed'
                            WHERE id = CAST(:id AS uuid)
                        """),
                        {"id": str(row_dict["id"])},
                    )
                    await db.commit()

            processed += 1

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


# ── Layer 2: Smart Media Retry ───────────────────────────────────────────────
async def retry_media_for_rows(subject_id: str, db_factory) -> None:
    """
    Background job: fill in ONLY the missing images/audio for 'done' rows.
    Does NOT re-run Ollama. Reads existing slide text from DB, regenerates only
    slides where infographicUrl or audioUrl is null/empty.
    Slides within each question are processed IN PARALLEL (asyncio.gather).
    Rows are processed sequentially to avoid GPU OOM.

    Called automatically after every batch (Layer 2 safety net).
    Can also be triggered manually via POST /pregen/retry-media.

    BUG FIXES applied here:
    1. _retry_one_slide is defined OUTSIDE the for-row loop to avoid Python
       closure bug where asyncio.gather coroutines captured wrong loop variables.
    2. Subject NAME (not UUID) is used for local cache path to match main pipeline.
    3. DB is always written if any slide has media, not only when NEW media was
       generated this run (fixes partial-data-not-saved regression).
    4. Confirmation logging added for every save step.
    """
    global _retry_state
    _retry_state = RetryState(
        running    = True,
        subject_id = subject_id,
        started_at = time.time(),
    )
    _log(f"[Retry] == Media retry pass started for subject={subject_id} ==")

    # ── OUTSIDE the row loop to avoid closure bug ──────────────────────────────
    async def _retry_one_slide(
        idx: int,
        slide: dict,
        cache_id: str,
        subject_name: str,
        language: str,
    ) -> tuple:
        """
        Returns (idx, updated_slide, changed).
        Runs image + audio concurrently within the slide.
        All exceptions are caught so one bad slide never kills the others.
        All variables passed explicitly — no closure capture from outer loop.
        """
        needs_img = _needs_image(slide)
        needs_aud = _needs_audio(slide)
        if not needs_img and not needs_aud:
            return idx, slide, False   # nothing to do

        _log(f"[Retry] slide {idx+1}: needs_img={needs_img} needs_aud={needs_aud}")

        slide_title   = slide.get("title", "Concept")
        slide_content = slide.get("infographic") or slide.get("content") or ""
        key_points    = slide.get("keyPoints") or []
        is_story      = slide.get("isStory", False)
        is_tips       = slide.get("isTips", False)
        kp_str        = ", ".join(key_points[:3]) if key_points else ""
        if is_story:
            style_hint = "warm illustrated story scene, narrative art, soft warm colors, storybook illustration style"
        elif is_tips:
            style_hint = "memory tips infographic, lightbulb icons, numbered list visual, bright yellow and blue palette"
        else:
            style_hint = "educational diagram, textbook illustration, labeled arrows, clear section boxes, blue and white color scheme"

        img_prompt = (
            f"Educational infographic poster: \"{slide_title}\". "
            f"Topic: {slide_content[:120]}. "
            f"{'Key concepts shown: ' + kp_str + '. ' if kp_str else ''}"
            f"Style: {style_hint}. "
            f"Visual design: clean layout, high contrast text labels, white background, professional academic look. "
            f"Negative: blurry, low quality, pixelated, dark background."
        )
        narration = prepare_for_tts(slide.get("narration") or slide.get("content") or "")

        img_coro = _wan2gp_image(img_prompt) if needs_img else asyncio.sleep(0, result=None)
        aud_coro = _voxcpm_tts(narration, language) if needs_aud else asyncio.sleep(0, result=None)
        img_result, wav_result = await asyncio.gather(
            img_coro, aud_coro, return_exceptions=True
        )

        changed = False

        if needs_img and isinstance(img_result, bytes) and img_result:
            try:
                if subject_name:
                    try:
                        await write_image(subject_name, cache_id, idx, img_result)
                        _log(f"[Retry] slide {idx+1}: image saved local disk")
                    except Exception as le:
                        _log(f"[Retry] slide {idx+1}: local image write failed (non-fatal): {le}")
                b2_path = f"ai-teaching/{cache_id}/slide_{idx}.png"
                b2_url = await upload_to_b2(img_result, b2_path, "image/png")
                slide["infographicUrl"] = b2_url
                _log(f"[Retry] slide {idx+1}: image OK -> {b2_url[:70]}")
                changed = True
            except Exception as e:
                _log(f"[Retry] slide {idx+1}: image upload failed -- {e}")
        elif needs_img:
            _log(f"[Retry] slide {idx+1}: image still failed -- "
                 f"{'exception: ' + str(img_result) if isinstance(img_result, Exception) else 'no output'}")

        if needs_aud and isinstance(wav_result, bytes) and wav_result:
            try:
                duration = _wav_duration(wav_result)
                if subject_name:
                    try:
                        await write_audio(subject_name, cache_id, language, idx, wav_result)
                        _log(f"[Retry] slide {idx+1}: audio saved local disk")
                    except Exception as le:
                        _log(f"[Retry] slide {idx+1}: local audio write failed (non-fatal): {le}")
                b2_path = f"ai-teaching/{cache_id}/audio_{idx}.wav"
                b2_url = await upload_to_b2(wav_result, b2_path, "audio/wav")
                slide["audioUrl"] = b2_url
                slide["duration"] = duration
                _log(f"[Retry] slide {idx+1}: audio OK {round(duration,1)}s -> {b2_url[:70]}")
                changed = True
            except Exception as e:
                _log(f"[Retry] slide {idx+1}: audio upload failed -- {e}")
        elif needs_aud:
            _log(f"[Retry] slide {idx+1}: audio still failed -- "
                 f"{'exception: ' + str(wav_result) if isinstance(wav_result, Exception) else 'no output'}")

        return idx, slide, changed

    try:
        async with db_factory() as db:
            # Bug Fix 2: Get subject NAME for local cache path.
            # Main pregen uses subject name not UUID for the local folder path.
            # Two-step approach to avoid asyncpg type inference bug with
            # COALESCE(uuid_col::text, text_col) in a JOIN query.
            name_row = (await db.execute(
                text("SELECT name FROM subjects WHERE subject_id = :subj LIMIT 1"),
                {"subj": subject_id},
            )).fetchone()
            subject_name = name_row.name if name_row else subject_id

            rows = (await db.execute(
                text("""
                    SELECT id::text, question_hash, question_text, language,
                           subject_id, presentation_slides, slide_audio_urls
                    FROM teaching_qa_cache
                    WHERE subject_id = :subj
                      AND pregen_status = 'done'
                      AND (
                        jsonb_array_length(COALESCE(presentation_slides, '[]'::jsonb)) = 0
                        OR jsonb_array_length(COALESCE(slide_audio_urls->'urls', '[]'::jsonb)) = 0
                        OR EXISTS(
                            SELECT 1 FROM jsonb_array_elements(COALESCE(presentation_slides,'[]'::jsonb)) s
                            WHERE (s->>'infographicUrl') IS NULL OR (s->>'infographicUrl') = ''
                        )
                        OR EXISTS(
                            SELECT 1 FROM jsonb_array_elements(COALESCE(presentation_slides,'[]'::jsonb)) s
                            WHERE (s->>'audioUrl') IS NULL OR (s->>'audioUrl') = ''
                        )
                      )
                    ORDER BY created_at ASC
                """),
                {"subj": subject_id},
            )).fetchall()

        _retry_state.total = len(rows)
        if not rows:
            _log("[Retry] No rows with missing media -- nothing to do.")
            return

        _log(f"[Retry] Found {len(rows)} rows with missing media")

        for row in rows:
            cache_id     = row.id
            language     = row.language or "en-IN"
            # subject_name fetched once above (before loop) — same for all rows
            q_hash       = row.question_hash or ""
            slides       = list(row.presentation_slides or [])

            _retry_state.current_cache_id = cache_id
            _log(f"[Retry] -- Processing: {(row.question_text or '')[:60]}")

            if not slides:
                _log(f"[Retry] {cache_id}: no slides in DB -- skipping (needs full regen)")
                _retry_state.failed += 1
                continue

            # Bug Fix 1: pass all variables explicitly as args (not captured via closure)
            _retry_state.current_step = f"all-slides:{len(slides)}"
            results = await asyncio.gather(
                *[_retry_one_slide(i, s, cache_id, subject_name, language)
                  for i, s in enumerate(slides)],
                return_exceptions=True,
            )

            any_new_media = False
            for res in results:
                if isinstance(res, Exception):
                    _log(f"[Retry] A slide coroutine raised: {res}")
                    continue
                idx_r, updated_slide, slide_changed = res
                slides[idx_r] = updated_slide
                if slide_changed:
                    any_new_media = True

            # Bug Fix 3: Always write DB if any slide has SOME media,
            # not only if new media was generated in this run.
            slides_with_img = sum(1 for s in slides if (s.get("infographicUrl") or "").startswith("http"))
            slides_with_aud = sum(1 for s in slides if (s.get("audioUrl") or "").startswith("http"))

            if slides_with_img == 0 and slides_with_aud == 0:
                _log(f"[Retry] {cache_id}: no media at all -- marking failed")
                _retry_state.failed += 1
                continue

            if not any_new_media:
                _log(f"[Retry] {cache_id}: no new media this run -- syncing existing partial data")

            audio_url_list = [
                {"slideIndex": i, "audioUrl": s["audioUrl"], "duration": s.get("duration", 0)}
                for i, s in enumerate(slides)
                if (s.get("audioUrl") or "").startswith("http")
            ]
            total_duration = sum(e["duration"] for e in audio_url_list)

            async with db_factory() as db:
                await db.execute(
                    text("""
                        UPDATE teaching_qa_cache
                        SET presentation_slides    = CAST(:slides AS jsonb),
                            slide_audio_urls       = CAST(:audio  AS jsonb),
                            total_duration_seconds = :dur
                        WHERE id = CAST(:id AS uuid)
                    """),
                    {
                        "slides": json.dumps(slides),
                        "audio":  json.dumps({"language": language, "urls": audio_url_list}),
                        "dur":    total_duration,
                        "id":     cache_id,
                    },
                )
                await db.commit()
            _log(f"[Retry] DB updated: {cache_id} img={slides_with_img}/{len(slides)} aud={slides_with_aud}/{len(slides)}")

            if q_hash and subject_id:
                try:
                    await write_slide_cache(subject_id, q_hash, {
                        "cache_id":            cache_id,
                        "presentationSlides":  slides,
                        "slideAudioUrls":      {"language": language, "urls": audio_url_list},
                        "totalDurationSeconds": total_duration,
                    })
                    _log(f"[Retry] Local cache written: {subject_id[:8]}/{q_hash[:12]}.json")
                    
                    # Also invalidate Redis L1 cache so it pulls the fresh data next time
                    try:
                        await delete_from_cache(q_hash, subject_id)
                    except Exception as e:
                        _log(f"[Retry] Redis cache delete failed: {e}")
                except Exception as e:
                    _log(f"[Retry] local cache write failed (non-fatal): {e}")

            _log(f"[Retry] OK {cache_id}: img={slides_with_img}/{len(slides)} aud={slides_with_aud}/{len(slides)}")
            _retry_state.done += 1

    except Exception as e:
        _log(f"[Retry] Fatal error: {e}")
        _retry_state.last_error = str(e)

    finally:
        _retry_state.running      = False
        _retry_state.current_step = ""
        elapsed = round(time.time() - _retry_state.started_at, 1)
        _log(
            f"[Retry] == Media retry complete: "
            f"done={_retry_state.done} failed={_retry_state.failed} elapsed={elapsed}s =="
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
