"""
routers/pregen.py — Pre-Generation Control API
===============================================

Endpoints:
  POST /pregen/start               — start batch pre-gen for a subject
  POST /pregen/stop                — request graceful stop
  GET  /pregen/status              — live progress
  POST /pregen/retry-failed        — re-queue all 'failed' rows for a subject
  GET  /pregen/pending-count       — count rows not yet pre-generated
  POST /pregen/retry-media         — fill in missing images/audio for 'done' rows
  GET  /pregen/retry-status        — live progress of the media retry job
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import get_db, AsyncSessionLocal
from core.pregen import (
    run_pregen_batch, get_state, request_stop,
    retry_media_for_rows, get_retry_state,
)

router = APIRouter(prefix="/pregen", tags=["Pre-Generation"])


# ── POST /pregen/start ────────────────────────────────────
@router.post("/start")
async def start_pregen(
    body: dict,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Start the offline pre-generation engine for a given subject.
    Runs in the background — returns immediately with job state.

    Body:
        subjectId  (required) — subject to pre-generate
        limit      (optional, default 500) — max rows to process
    """
    subject_id = body.get("subjectId", "").strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subjectId is required")

    state = get_state()
    if state["running"]:
        return JSONResponse(
            status_code=409,
            content={
                "error": "Pre-gen already running",
                "subject_id": state["subject_id"],
                "done": state["done"],
                "total": state["total"],
            },
        )

    limit      = int(body.get("limit", 500))
    topic_id   = body.get("topicId", "").strip() or None
    chapter_id = body.get("chapterId", "").strip() or None

    # Count all actionable rows from the single source of truth.
    # 'processing' rows will be reset to 'pending' at batch start (handles restarts).
    pending = (await db.execute(
        text("""
            SELECT COUNT(*) FROM teaching_qa_cache
            WHERE subject_id = :subj
              AND pregen_status IN ('pending', 'processing')
        """),
        {"subj": subject_id},
    )).scalar()

    # Also count questions not yet synced into the cache
    unsynced = (await db.execute(
        text("""
            SELECT COUNT(*) FROM questions q
            WHERE q.subject_id = :subj
              AND q.is_pregen_done = FALSE
              AND NOT EXISTS (
                SELECT 1 FROM teaching_qa_cache c
                WHERE c.subject_id = q.subject_id
                  AND c.question_text = q.question_text
              )
        """),
        {"subj": subject_id},
    )).scalar()

    total_pending = int(pending or 0) + int(unsynced or 0)

    if total_pending == 0:
        return {"message": "Nothing to pre-generate", "pending": 0, "subject_id": subject_id}

    # Kick off in background
    background_tasks.add_task(
        run_pregen_batch, subject_id, AsyncSessionLocal, limit, topic_id, chapter_id
    )

    return {
        "message": f"Pre-generation started for {subject_id!r}",
        "subject_id": subject_id,
        "topic_id":   topic_id,
        "chapter_id": chapter_id,
        "pending_rows": total_pending,
        "limit": limit,
        "tip": "Poll GET /pregen/status for live progress",
    }


# ── POST /pregen/stop ─────────────────────────────────────
@router.post("/stop")
async def stop_pregen():
    """Request a graceful stop. Current row finishes before stopping."""
    state = get_state()
    if not state["running"]:
        return {"message": "Pre-gen is not running", "state": state}
    request_stop()
    return {"message": "Stop requested — will halt after current row", "state": state}


# ── GET /pregen/status ────────────────────────────────────
@router.get("/status")
async def pregen_status():
    """Return live progress of the running (or last completed) pre-gen job."""
    return get_state()


# ── GET /pregen/pending-count ─────────────────────────────
@router.get("/pending-count")
async def pending_count(subject_id: str = "", db: AsyncSession = Depends(get_db)):
    """
    Count how many rows still need pre-generation.
    If subject_id is provided, scoped to that subject only.
    """
    if subject_id:
        sql = text("""
            SELECT COUNT(*) FROM teaching_qa_cache
            WHERE subject_id = :subj
              AND pregen_status IN ('pending', 'processing')
        """)
        total = (await db.execute(sql, {"subj": subject_id})).scalar()
    else:
        sql = text("""
            SELECT subject_id, COUNT(*) AS cnt
            FROM teaching_qa_cache
            WHERE pregen_status IN ('pending', 'processing')
            GROUP BY subject_id ORDER BY cnt DESC
        """)
        rows = (await db.execute(sql)).fetchall()
        return {
            "total": sum(r.cnt for r in rows),
            "by_subject": [{"subject_id": r.subject_id, "pending": r.cnt} for r in rows],
        }

    return {"subject_id": subject_id, "pending": int(total or 0)}


# ── POST /pregen/retry-failed ─────────────────────────────
@router.post("/retry-failed")
async def retry_failed(body: dict, db: AsyncSession = Depends(get_db)):
    """
    Reset all 'failed' rows back to 'pending' so the next /pregen/start
    will re-process them.
    """
    subject_id = body.get("subjectId", "").strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subjectId is required")

    result = await db.execute(
        text("""
            UPDATE teaching_qa_cache
            SET pregen_status = 'pending'
            WHERE subject_id = :subj AND pregen_status = 'failed'
        """),
        {"subj": subject_id},
    )
    await db.commit()

    return {
        "message": f"Reset {result.rowcount} failed rows to 'pending'",
        "subject_id": subject_id,
        "rows_reset": result.rowcount,
    }


# ── POST /pregen/add-question ─────────────────────────────
@router.post("/add-question")
async def add_question(body: dict, db: AsyncSession = Depends(get_db)):
    import uuid, hashlib
    subject_id = body.get("subjectId", "").strip()
    question = body.get("question", "").strip()

    if not subject_id:
        raise HTTPException(status_code=400, detail="subjectId is required")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    new_id = str(uuid.uuid4())
    q_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()

    await db.execute(
        text("""
            INSERT INTO teaching_qa_cache 
            (id, subject_id, question_hash, question_text, variation_number, pregen_status)
            VALUES (:id, :subject_id, :question_hash, :question, 1, 'pending')
            ON CONFLICT (question_hash, subject_id, variation_number) DO UPDATE SET pregen_status = 'pending'
        """),
        {"id": new_id, "subject_id": subject_id, "question_hash": q_hash, "question": question}
    )
    await db.commit()

    return {"message": "Question added successfully", "id": new_id, "question_hash": q_hash}


# ── POST /pregen/retry-media ──────────────────────────────
@router.post("/retry-media")
async def trigger_retry_media(
    body: dict,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Smart media retry — re-generates ONLY the missing images and audio
    for rows that are already 'done' but have incomplete media.

    Does NOT re-run Ollama. Reads existing slide text from DB.
    Safe to call manually at any time. Also runs automatically after every batch.

    Body:
        subjectId (required) — subject to audit and fix
    """
    subject_id = body.get("subjectId", "").strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subjectId is required")

    retry_st = get_retry_state()
    if retry_st["running"]:
        return JSONResponse(
            status_code=409,
            content={
                "error": "Media retry already running",
                "subject_id": retry_st["subject_id"],
                "done":       retry_st["done"],
                "total":      retry_st["total"],
                "tip":        "Poll GET /pregen/retry-status for live progress",
            },
        )

    # Count problem rows before starting so we can give a useful response
    problem_rows = (await db.execute(
        text("""
            SELECT COUNT(*) FROM teaching_qa_cache
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
        """),
        {"subj": subject_id},
    )).scalar()

    problem_rows = int(problem_rows or 0)
    if problem_rows == 0:
        return {
            "message": "All media is complete — nothing to retry",
            "subject_id": subject_id,
            "problem_rows": 0,
        }

    background_tasks.add_task(retry_media_for_rows, subject_id, AsyncSessionLocal)

    return {
        "message": f"Media retry started for {subject_id!r}",
        "subject_id":   subject_id,
        "problem_rows": problem_rows,
        "tip": "Poll GET /pregen/retry-status for live progress",
    }


# ── GET /pregen/retry-status ──────────────────────────────
@router.get("/retry-status")
async def retry_status():
    """
    Return live progress of the running (or last completed) media retry job.

    Response:
        running         — true if retry is currently active
        total           — total rows with missing media found
        done            — rows successfully fixed
        failed          — rows that could not be fixed (still missing media)
        current_step    — e.g. "image:3" or "audio:1" or "both:0"
        elapsed_seconds — seconds since retry started
    """
    return get_retry_state()
