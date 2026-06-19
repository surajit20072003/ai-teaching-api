"""
routers/pregen.py — Pre-Generation Control API
===============================================

Endpoints:
  POST /pregen/start               — start batch pre-gen for a subject
  POST /pregen/stop                — request graceful stop
  GET  /pregen/status              — live progress
  POST /pregen/retry-failed        — re-queue all 'failed' rows for a subject
  GET  /pregen/pending-count       — count rows not yet pre-generated
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import get_db, AsyncSessionLocal
from core.pregen import run_pregen_batch, get_state, request_stop

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

    limit = int(body.get("limit", 500))

    # Count pending rows first
    count_row = (await db.execute(
        text("""
            SELECT COUNT(*) FROM teaching_qa_cache
            WHERE subject_id = :subj
              AND (pregen_status IS NULL OR pregen_status = 'pending')
        """),
        {"subj": subject_id},
    )).scalar()
    pending = int(count_row or 0)

    if pending == 0:
        return {"message": "Nothing to pre-generate", "pending": 0, "subject_id": subject_id}

    # Kick off in background
    background_tasks.add_task(run_pregen_batch, subject_id, AsyncSessionLocal, limit)

    return {
        "message": f"Pre-generation started for {subject_id!r}",
        "subject_id": subject_id,
        "pending_rows": pending,
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
              AND (pregen_status IS NULL OR pregen_status = 'pending')
        """)
        total = (await db.execute(sql, {"subj": subject_id})).scalar()
    else:
        sql = text("""
            SELECT subject_id, COUNT(*) AS cnt
            FROM teaching_qa_cache
            WHERE (pregen_status IS NULL OR pregen_status = 'pending')
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
