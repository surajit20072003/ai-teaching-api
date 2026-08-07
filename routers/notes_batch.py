"""
routers/notes_batch.py — Server-side Controlled Notes Batch Queue
=================================================================

Uses a DB-backed batch_jobs table so state is shared across all uvicorn workers.

Endpoints:
  POST /notes/batch/start   — Start processing all pending json_import docs
  POST /notes/batch/stop    — Gracefully stop after current doc
  GET  /notes/batch/status  — Current batch state (DB-backed, works across workers)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import get_db, AsyncSessionLocal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/notes", tags=["Notes Batch"])

BATCH_ID = "notes_batch"


# ── DB helpers ─────────────────────────────────────────────────────────────────
async def _db_update(fields: dict):
    """Update batch_jobs row in its own session."""
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    fields["bid"] = BATCH_ID
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(f"UPDATE batch_jobs SET {sets}, updated_at=NOW() WHERE id = :bid"),
            fields,
        )
        await db.commit()


async def _db_get(db: AsyncSession) -> dict:
    r = await db.execute(
        text("SELECT running, stop_flag, current_doc, current_title, "
             "done, failed, total, subject_id, delay_ms, last_update, started_at "
             "FROM batch_jobs WHERE id = :bid"),
        {"bid": BATCH_ID},
    )
    row = r.fetchone()
    if not row:
        return {}
    keys = ["running","stop_flag","current_doc","current_title",
            "done","failed","total","subject_id","delay_ms","last_update","started_at"]
    return dict(zip(keys, row))


async def _is_stop_requested() -> bool:
    async with AsyncSessionLocal() as db:
        r = await db.execute(
            text("SELECT stop_flag FROM batch_jobs WHERE id = :bid"),
            {"bid": BATCH_ID},
        )
        row = r.fetchone()
        return bool(row and row[0])


# ── Request model ──────────────────────────────────────────────────────────────
class BatchStartRequest(BaseModel):
    subject_id:  Optional[str] = None
    delay_ms:    int = 2000
    reset_stuck: bool = True


# ── Background worker ──────────────────────────────────────────────────────────
async def _batch_worker(subject_id: str, delay_ms: int, reset_stuck: bool):
    from core.notes.note_service import generate_notes_for_document

    logger.info(f"[NotesBatch] Worker started | subject={subject_id or 'ALL'} | delay={delay_ms}ms")

    try:
        # Step 0: Reset stuck docs (generating → notes_pending) in note_documents
        if reset_stuck:
            async with AsyncSessionLocal() as db:
                stuck = await db.execute(
                    text("""
                        UPDATE note_documents
                        SET notes_status='notes_pending', updated_at=NOW()
                        WHERE notes_status='generating'
                          AND (:sid='' OR CAST(subject_id AS text)=:sid)
                        RETURNING id
                    """),
                    {"sid": subject_id or ""},
                )
                n = len(stuck.fetchall())
                if n:
                    await db.commit()
                    logger.info(f"[NotesBatch] Reset {n} stuck docs to 'notes_pending'")

        done = 0
        failed = 0

        while True:
            if await _is_stop_requested():
                await _db_update({"last_update": "Stopped by user request."})
                break

            async with AsyncSessionLocal() as db:
                q = await db.execute(
                    text("""
                        SELECT id, title, content_markdown
                        FROM note_documents
                        WHERE notes_status='notes_pending'
                          AND (:sid='' OR CAST(subject_id AS text)=:sid)
                        ORDER BY created_at ASC
                        LIMIT 1
                    """),
                    {"sid": subject_id or ""},
                )
                row = q.fetchone()

            if not row:
                await _db_update({"last_update": "All pending documents processed."})
                logger.info("[NotesBatch] Complete")
                break

            doc_id    = str(row[0])
            doc_title = row[1] or doc_id[:8]
            content   = row[2]

            if not content or not content.strip():
                async with AsyncSessionLocal() as db:
                    await db.execute(
                        text("UPDATE note_documents SET notes_status='failed', updated_at=NOW() "
                             "WHERE id=CAST(:did AS uuid)"),
                        {"did": doc_id},
                    )
                    await db.commit()
                failed += 1
                await _db_update({"failed": failed, "last_update": f"Skipped (no content): {doc_title}"})
                continue

            async with AsyncSessionLocal() as db:
                await db.execute(
                    text("UPDATE note_documents SET notes_status='generating', updated_at=NOW() "
                         "WHERE id=CAST(:did AS uuid)"),
                    {"did": doc_id},
                )
                await db.commit()

            await _db_update({
                "current_doc":   doc_id,
                "current_title": doc_title,
                "last_update":   f"Generating: {doc_title}",
            })
            logger.info(f"[NotesBatch] -> {doc_title}")

            try:
                async with AsyncSessionLocal() as db:
                    await generate_notes_for_document(doc_id, db)
                done += 1
                await _db_update({"done": done, "last_update": f"Done: {doc_title}"})
                logger.info(f"[NotesBatch] OK {doc_title}")
            except Exception as e:
                logger.error(f"[NotesBatch] FAIL {doc_title}: {e}")
                failed += 1
                await _db_update({"failed": failed, "last_update": f"Failed: {doc_title}"})
                try:
                    async with AsyncSessionLocal() as db:
                        await db.execute(
                            text("UPDATE note_documents SET notes_status='failed', updated_at=NOW() "
                                 "WHERE id=CAST(:did AS uuid)"),
                            {"did": doc_id},
                        )
                        await db.commit()
                except Exception:
                    pass

            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)

    except Exception as e:
        logger.error(f"[NotesBatch] Worker crashed: {e}")
    finally:
        await _db_update({
            "running":       False,
            "stop_flag":     False,
            "current_doc":   "",
            "current_title": "",
        })
        logger.info("[NotesBatch] Worker finished")


# ── POST /notes/batch/start ────────────────────────────────────────────────────
@router.post("/batch/start")
async def start_notes_batch(
    body: BatchStartRequest = BatchStartRequest(),
    db: AsyncSession = Depends(get_db),
):
    """Start serial notes batch. Processes all ready json_import docs one by one."""
    state = await _db_get(db)

    if state.get("running"):
        return {
            "success": False,
            "message": "Batch already running. Call /notes/batch/stop first.",
            "done":  state.get("done", 0),
            "total": state.get("total", 0),
        }

    sid_filter = body.subject_id or ""
    cnt = await db.execute(
        text("""
            SELECT COUNT(*) FROM note_documents
            WHERE notes_status IN ('notes_pending', 'generating')
              AND (:sid='' OR CAST(subject_id AS text)=:sid)
        """),
        {"sid": sid_filter},
    )
    total = int(cnt.scalar() or 0)

    # UPSERT — works even if row was deleted/never existed
    await db.execute(
        text("""
            INSERT INTO batch_jobs (id, running, stop_flag, current_doc, current_title,
                                    done, failed, total, subject_id, delay_ms,
                                    last_update, started_at, updated_at)
            VALUES (:bid, true, false, '', '', 0, 0, :total, :sid, :delay,
                    'Starting...', NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                running=true, stop_flag=false,
                current_doc='', current_title='',
                done=0, failed=0, total=:total,
                subject_id=:sid, delay_ms=:delay,
                last_update='Starting...', started_at=NOW(), updated_at=NOW()
        """),
        {"total": total, "sid": sid_filter, "delay": body.delay_ms, "bid": BATCH_ID},
    )
    await db.commit()

    asyncio.create_task(_batch_worker(sid_filter, body.delay_ms, body.reset_stuck))

    return {
        "success":    True,
        "message":    f"Batch started. {total} documents queued.",
        "total":      total,
        "subject_id": sid_filter or "ALL",
        "delay_ms":   body.delay_ms,
    }


# ── POST /notes/batch/stop ─────────────────────────────────────────────────────
@router.post("/batch/stop")
async def stop_notes_batch(db: AsyncSession = Depends(get_db)):
    """Gracefully stop the batch after current document finishes."""
    state = await _db_get(db)
    if not state.get("running"):
        return {"success": False, "message": "No batch is running."}

    await db.execute(
        text("UPDATE batch_jobs SET stop_flag=true, updated_at=NOW() WHERE id=:bid"),
        {"bid": BATCH_ID},
    )
    await db.commit()
    return {
        "success": True,
        "message": "Stop signal sent. Batch will stop after current document.",
        "current": state.get("current_title", ""),
    }


# ── GET /notes/batch/status ────────────────────────────────────────────────────
@router.get("/batch/status")
async def notes_batch_status(db: AsyncSession = Depends(get_db)):
    """Get batch state from DB (works across all uvicorn workers)."""
    state = await _db_get(db)
    if not state:
        return {"running": False, "error": "batch_jobs table not initialized"}

    done    = state.get("done", 0)
    failed  = state.get("failed", 0)
    total   = state.get("total", 0)
    elapsed = 0
    if state.get("started_at") and state.get("running"):
        import datetime
        from datetime import timezone
        now = datetime.datetime.now(timezone.utc)
        sa  = state["started_at"]
        if sa.tzinfo is None:
            sa = sa.replace(tzinfo=timezone.utc)
        elapsed = round((now - sa).total_seconds(), 1)

    processed = done + failed
    pct = round(processed / total * 100, 1) if total > 0 else 0

    return {
        "running":        state.get("running", False),
        "stop_requested": state.get("stop_flag", False),
        "current_doc_id": state.get("current_doc", ""),
        "current_title":  state.get("current_title", ""),
        "done":           done,
        "failed":         failed,
        "total":          total,
        "pending":        max(0, total - processed),
        "progress_pct":   pct,
        "elapsed_sec":    elapsed,
        "subject_filter": state.get("subject_id") or "ALL",
        "last_update":    state.get("last_update", ""),
    }
