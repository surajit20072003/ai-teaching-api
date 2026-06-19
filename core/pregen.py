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
            "prompt": prompt,
            "model": "flux_dev",
            "resolution": "1024x1024",
            "steps": 20,
            "seed": -1
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
async def _process_slide(idx: int, slide: dict, cache_id: str, language: str):
    """
    Generate image + audio for ONE slide concurrently, upload both to B2.
    Returns (enriched_slide_dict, audio_duration_float).
    Called in parallel for all slides via asyncio.gather().
    """
    img_prompt = (
        f"Educational infographic: {slide.get('title', 'Concept')}\n"
        f"{slide.get('infographic', slide.get('content', ''))[:200]}\n"
        "Style: clean, colorful, labeled, white background, no clutter"
    )
    narration = slide.get("narration") or slide.get("content") or ""

    # Run image AND audio for this slide at the same time
    img_result, wav_result = await asyncio.gather(
        _wan2gp_image(img_prompt),
        _voxcpm_tts(narration, language),
        return_exceptions=True,
    )

    # ── Image ────────────────────────────────────────────────────────────────
    if isinstance(img_result, bytes) and img_result:
        try:
            b2_path = f"ai-teaching/{cache_id}/slide_{idx}.png"
            slide["infographicUrl"] = await upload_to_b2(img_result, b2_path, "image/png")
            _log(f"[Pregen] slide {idx+1}: image ✓ → {slide['infographicUrl']}")
        except Exception as e:
            _log(f"[Pregen] slide {idx+1}: image B2 upload failed — {e}")
    else:
        err = img_result if isinstance(img_result, Exception) else "no output"
        _log(f"[Pregen] slide {idx+1}: image skipped ({err})")

    # ── Audio ────────────────────────────────────────────────────────────────
    duration = 0.0
    if isinstance(wav_result, bytes) and wav_result:
        try:
            duration = _wav_duration(wav_result)
            b2_path  = f"ai-teaching/{cache_id}/audio_{idx}.wav"
            slide["audioUrl"] = await upload_to_b2(wav_result, b2_path, "audio/wav")
            slide["duration"] = duration
            _log(f"[Pregen] slide {idx+1}: audio ✓ {round(duration, 1)}s → {slide['audioUrl']}")
        except Exception as e:
            _log(f"[Pregen] slide {idx+1}: audio B2 upload failed — {e}")
    else:
        err = wav_result if isinstance(wav_result, Exception) else "no output"
        _log(f"[Pregen] slide {idx+1}: audio skipped ({err})")

    return slide, duration


async def _pregen_one(row: Dict[str, Any], db: AsyncSession) -> None:
    """
    Strictly sequential pipeline for one question:
      1. Ollama → slides JSON
      2. For each slide: Wan2GP image → VoxCPM audio
      3. Save results to DB
    """
    cache_id   = str(row["id"])
    language   = row.get("language") or "hi-IN"
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
                        ORDER BY chunk_index LIMIT 3
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
        *[_process_slide(idx, slide, cache_id, language) for idx, slide in enumerate(slides)],
        return_exceptions=True,
    )

    audio_urls: Dict[str, Any] = {}
    total_duration = 0.0

    for idx, result in enumerate(slide_results):
        if isinstance(result, Exception):
            _log(f"[Pregen] slide {idx+1}: exception in _process_slide — {result}")
            continue
        enriched_slide, duration = result
        slides[idx] = enriched_slide
        total_duration += duration
        if enriched_slide.get("audioUrl"):
            audio_urls[str(idx)] = {
                "audioUrl": enriched_slide["audioUrl"],
                "duration": duration,
            }

    _log(f"[Pregen] Step 2: all slides done — total audio {round(total_duration, 1)}s")

    # ── Step 3: Save all results → mark done ─────────────────────────────────
    _state.current_step = "saving"
    _log(f"[Pregen] Step 3: Saving results to DB...")
    await db.execute(
        text("""
            UPDATE teaching_qa_cache
            SET presentation_slides    = CAST(:slides AS jsonb),
                slide_audio_urls       = CAST(:audio AS jsonb),
                total_duration_seconds = :dur,
                pregen_status          = 'done',
                pregen_completed_at    = NOW()
            WHERE id = CAST(:id AS uuid)
        """),
        {
            "slides": json.dumps(slides),
            "audio":  json.dumps(audio_urls),
            "dur":    total_duration,
            "id":     cache_id,
        },
    )
    await db.commit()
    _log(f"[Pregen] ✓ Done: {question[:60]} — {len(slides)} slides, {round(total_duration)}s audio")


# ── Batch runner ──────────────────────────────────────────────────────────────
async def run_pregen_batch(
    subject_id:  str,
    db_factory,
    limit:       int = 500,
) -> None:
    """
    Background batch job. Processes pending rows one at a time.
    Each row goes through: Ollama → [Image → Audio per slide] → Save.
    """
    global _state
    _state = PregenState(
        running      = True,
        stop_requested = False,
        subject_id   = subject_id,
        started_at   = time.time(),
    )
    _log(f"[Pregen] ══════════════ Batch started: subject={subject_id} limit={limit} ══════════════")

    try:
        # Count total pending rows
        async with db_factory() as db:
            row = (await db.execute(
                text("""
                    SELECT COUNT(*) FROM (
                        SELECT 1 FROM teaching_qa_cache
                        WHERE subject_id = :subj
                          AND (pregen_status IS NULL OR pregen_status = 'pending')
                        LIMIT :lim
                    ) sub
                """),
                {"subj": subject_id, "lim": limit},
            )).scalar()
            _state.total = int(row or 0)

        _log(f"[Pregen] {_state.total} rows to process")

        batch_size = 5
        offset     = 0
        processed  = 0

        while not _state.stop_requested and processed < limit:
            # Fetch next batch of pending rows
            async with db_factory() as db:
                rows = (await db.execute(
                    text("""
                        SELECT id, question_text, presentation_slides,
                               slide_audio_urls, language, subject_id, document_id
                        FROM teaching_qa_cache
                        WHERE subject_id = :subj
                          AND (pregen_status IS NULL OR pregen_status = 'pending')
                        ORDER BY usage_count DESC NULLS LAST, created_at ASC
                        LIMIT :batch OFFSET :offset
                    """),
                    {"subj": subject_id, "batch": batch_size, "offset": offset},
                )).fetchall()

            if not rows:
                _log("[Pregen] No more pending rows — batch complete.")
                break

            for row in rows:
                if _state.stop_requested:
                    break
                if processed >= limit:
                    break

                row_dict = dict(row._mapping)
                _state.current_question = (row_dict.get("question_text") or "")[:80]

                # Mark as 'processing' in DB
                async with db_factory() as db:
                    await db.execute(
                        text("""
                            UPDATE teaching_qa_cache
                            SET pregen_status = 'processing'
                            WHERE id = CAST(:id AS uuid)
                        """),
                        {"id": str(row_dict["id"])},
                    )
                    await db.commit()

                try:
                    async with db_factory() as db:
                        await _pregen_one(row_dict, db)
                    _state.done += 1

                except Exception as e:
                    err = str(e)
                    _state.failed += 1
                    _state.last_error = err
                    _log(f"[Pregen] ✗ Failed: id={row_dict['id']}: {err}")
                    # Mark as failed in DB
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

            offset += batch_size

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
