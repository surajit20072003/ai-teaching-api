"""
routers/notes.py — Textbook Notes API (CPU Server)
===================================================

Endpoints:
  POST /notes/generate/{document_id}  — queue notes generation for a document
  GET  /notes/status/{document_id}    — check generation status
  GET  /notes/topic/{topic_id}        — get notes for a specific topic
  GET  /notes/document/{document_id}  — get all notes for a document
  POST /notes/retry/{document_id}     — retry failed note generation
  POST /notes/reset/{document_id}     — admin: reset document + clear topic_notes
  GET  /notes/logs                    — tail server logs for debugging
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import get_db
from core.notes.note_service import generate_notes_for_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/notes", tags=["Textbook Notes"])


# ── POST /notes/generate/{document_id} ────────────────────────────────────────
@router.post("/generate/{document_id}")
async def generate_notes(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Queue textbook notes + question answers generation for a document.
    Allowed statuses: 'ready' or 'failed' (auto-resets to ready).
    """
    # Validate document exists
    result = await db.execute(
        text("SELECT id, title, notes_status, content_markdown FROM note_documents WHERE id = CAST(:did AS uuid)"),
        {"did": document_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "Document not found")

    _, title, status, content_md = row

    # Allow 'ready', 'notes_pending' (json_import), or 'failed' (retry)
    if status not in ("ready", "notes_pending", "failed"):
        raise HTTPException(
            400,
            f"Document status is '{status}'. Must be 'ready' or 'failed'. "
            "Use POST /notes/reset/{doc_id} to unblock stuck documents."
        )

    if not content_md or not content_md.strip():
        raise HTTPException(400, "Document has no content_markdown. Cannot generate notes.")

    # Update document status to indicate notes are being generated
    await db.execute(
        text("UPDATE note_documents SET notes_status = 'generating', updated_at = NOW() "
             "WHERE id = CAST(:did AS uuid)"),
        {"did": document_id},
    )
    await db.commit()

    # Queue background task
    background_tasks.add_task(_run_notes_generation, document_id)

    return {
        "success": True,
        "document_id": document_id,
        "title": title,
        "message": "Notes generation queued in background.",
    }


async def _run_notes_generation(document_id: str):
    """Background task wrapper — opens its own DB session."""
    from db.models import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            await generate_notes_for_document(document_id, db)
        except Exception as e:
            logger.error(f"[Notes] Background task failed for doc={document_id}: {e}")
            # Mark as failed in DB
            await db.execute(
                text("UPDATE note_documents SET notes_status = 'failed', updated_at = NOW() "
                     "WHERE id = CAST(:did AS uuid)"),
                {"did": document_id},
            )
            await db.commit()


# ── GET /notes/status/{document_id} ───────────────────────────────────────────
@router.get("/status/{document_id}")
async def notes_status(document_id: str, db: AsyncSession = Depends(get_db)):
    """Get notes generation status for a document."""
    result = await db.execute(
        text("""
            SELECT d.notes_status, d.title,
                   COUNT(tn.id) FILTER (WHERE tn.notes_status = 'done')   AS done_count,
                   COUNT(tn.id) FILTER (WHERE tn.notes_status = 'pending') AS pending_count,
                   COUNT(tn.id) FILTER (WHERE tn.notes_status = 'failed')  AS failed_count
            FROM note_documents d
            LEFT JOIN topic_notes tn ON tn.document_id = d.id
            WHERE d.id = CAST(:did AS uuid)
            GROUP BY d.id, d.title, d.notes_status
        """),
        {"did": document_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "Document not found")

    return {
        "document_id": document_id,
        "document_status": row[0],
        "title": row[1],
        "topics_done": row[2] or 0,
        "topics_pending": row[3] or 0,
        "topics_failed": row[4] or 0,
    }


# ── GET /notes/{topic_id} ─────────────────────────────────────────────────────
@router.get("/topic/{topic_id}")
async def get_topic_notes(topic_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get all notes + answers for a specific topic.

    The path param {topic_id} is used as topic_note_id (topic_notes.id)
    because the frontend selects from the chapter dropdown which renders
    topic_note_id as the option value.
    """
    result = await db.execute(
        text("""
            SELECT tn.id, tn.document_id, nd.title, nd.subject_id,
                   tn.note_sections, tn.note_latex_formulas, tn.note_image_urls,
                   tn.note_image_local_paths,
                   tn.question_answers, tn.answer_image_urls,
                   tn.notes_status, tn.generated_at
            FROM topic_notes tn
            JOIN note_documents nd ON nd.id = tn.document_id
            WHERE tn.id = CAST(:tid AS uuid)
        """),
        {"tid": topic_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "No notes found for this topic")

    # Build paired {url, local_path, local_url} objects for note images
    note_imgs = []
    for b2_url, local_path in zip(
        (row[6] or []),  # note_image_urls
        (row[7] or []),  # note_image_local_paths
    ):
        public_local_url = ""
        if local_path:
            rel = local_path.replace("/app/storage/subjects/", "")
            public_local_url = f"/local-images/{rel}"
        note_imgs.append({
            "url": b2_url,
            "local_url": public_local_url,
            "local_path": local_path
        })

    return {
        "topic_note_id": str(row[0]),
        "document_id": str(row[1]),
        "document_title": row[2],
        "subject_id": row[3],
        "status": row[10],
        "generated_at": row[11].isoformat() if row[11] else None,
        "note_sections": row[4] or [],
        "latex_formulas": row[5] or [],
        "note_images": note_imgs,
        "question_answers": row[8] or [],
        "answer_images": row[9] or [],
    }


# ── GET /notes/document/{document_id} ─────────────────────────────────────────
@router.get("/document/{document_id}")
async def get_document_notes(document_id: str, db: AsyncSession = Depends(get_db)):
    """Get all topic notes for a document (grouped by topic)."""
    result = await db.execute(
        text("""
            SELECT tn.id, tn.topic_id, tn.chapter_id, t.title AS topic_title,
                   tn.note_sections, tn.note_image_urls, tn.note_image_local_paths,
                   tn.question_answers, tn.notes_status, tn.generated_at
            FROM topic_notes tn
            LEFT JOIN topics t ON t.id = tn.topic_id
            WHERE tn.document_id = CAST(:did AS uuid)
            ORDER BY t.topic_number NULLS LAST
        """),
        {"did": document_id},
    )
    rows = result.fetchall()

    return {
        "document_id": document_id,
        "total_topics": len(rows),
        "topics": [
            {
                "topic_note_id": str(r[0]),
                "topic_id": str(r[1]) if r[1] else None,
                "chapter_id": str(r[2]) if r[2] else None,
                "topic_title": r[3],
                "status": r[8],
                "generated_at": r[9].isoformat() if r[9] else None,
                "note_sections": r[4] or [],
                "note_images": [
                    {"url": u, "local_path": p}
                    for u, p in zip((r[5] or []), (r[6] or []))
                ],
                "question_answers_count": len(r[7] or []),
            }
            for r in rows
        ],
    }


# ── GET /notes/chapters ───────────────────────────────────────────────────────
@router.get("/chapters")
async def list_chapters_with_notes(
    subject_id: str = "",
    db: AsyncSession = Depends(get_db),
):
    """
    Returns only chapters that have at least one generated topic_note.
    Used by the frontend dropdown — avoids showing all 60 slide chapters
    when only a few have notes.
    """
    params: dict = {}
    sid_clause = ""
    if subject_id:
        sid_clause = "AND c.subject_id = :sid"
        params["sid"] = subject_id

    rows = await db.execute(
        text(f"""
            SELECT
                c.id::text          AS chapter_id,
                c.chapter_number,
                c.title,
                COUNT(tn.id)        AS topics_with_notes,
                COUNT(tn.id) FILTER (WHERE tn.notes_status = 'done')    AS topics_done,
                COUNT(tn.id) FILTER (WHERE tn.notes_status = 'pending') AS topics_pending,
                COUNT(tn.id) FILTER (WHERE tn.notes_status = 'failed')  AS topics_failed
            FROM chapters c
            JOIN topic_notes tn ON tn.chapter_id = c.id
            WHERE 1=1 {sid_clause}
            GROUP BY c.id, c.chapter_number, c.title
            HAVING COUNT(tn.id) > 0
            ORDER BY c.chapter_number
        """),
        params,
    )
    chapters = rows.fetchall()

    return {
        "subject_id": subject_id or "ALL",
        "chapters": [
            {
                "chapter_id":       r[0],
                "chapter_number":   int(r[1] or 0),
                "title":            r[2],
                "topics_with_notes": int(r[3] or 0),
                "topics_done":      int(r[4] or 0),
                "topics_pending":   int(r[5] or 0),
                "topics_failed":    int(r[6] or 0),
            }
            for r in chapters
        ],
    }


# ── GET /notes/chapter/{chapter_id} ───────────────────────────────────────────
@router.get("/chapter/{chapter_id}")
async def get_chapter_notes(chapter_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get ALL topic notes for a chapter, with topic metadata, note content,
    images, and associated questions with answers. Used by the frontend
    textbook view: Subject → Topic → render inline textbook.
    """
    # 1. Fetch all topic_notes for this chapter, joined with topics + document
    notes_result = await db.execute(
        text("""
            SELECT
                tn.id AS topic_note_id,
                tn.topic_id,
                tn.document_id,
                nd.title AS document_title,
                t.topic_number,
                t.title AS topic_title,
                tn.note_sections,
                tn.note_latex_formulas,
                tn.note_image_urls,
                tn.note_image_local_paths,
                tn.question_answers,
                tn.answer_image_urls,
                tn.notes_status,
                tn.generated_at,
                tn.error_message
            FROM topic_notes tn
            JOIN note_documents nd ON nd.id = tn.document_id
            LEFT JOIN topics t ON t.id = tn.topic_id
            WHERE tn.chapter_id = CAST(:cid AS uuid)
            ORDER BY t.topic_number NULLS LAST
        """),
        {"cid": chapter_id},
    )
    notes_rows = notes_result.fetchall()

    if not notes_rows:
        raise HTTPException(404, "No notes found for this chapter")

    # 2. Fetch questions for this chapter (grouped by topic_id)
    questions_result = await db.execute(
        text("""
            SELECT id, topic_id, question_text, question_format, question_type,
                   options, correct_answer, difficulty, marks
            FROM questions
            WHERE chapter_id = CAST(:cid AS uuid)
            ORDER BY created_at ASC
        """),
        {"cid": chapter_id},
    )
    q_rows = questions_result.fetchall()

    # Group questions by topic_id
    questions_by_topic: Dict[str, List[dict]] = {}
    for q in q_rows:
        tid = str(q[1]) if q[1] else "__untopic__"
        questions_by_topic.setdefault(tid, []).append({
            "id": str(q[0]),
            "topic_id": str(q[1]) if q[1] else None,
            "question_text": q[2],
            "question_format": q[3],
            "question_type": q[4],
            "options": q[5] or {},
            "correct_answer": q[6] or "",
            "difficulty": q[7] or "Medium",
            "marks": q[8] or 4,
        })

    # 3. Build response
    subject_id = None
    topics_out = []
    for r in notes_rows:
        # Fetch subject_id from document (only once)
        if subject_id is None:
            doc_subj = await db.execute(
                text("SELECT subject_id FROM note_documents WHERE id = CAST(:did AS uuid)"),
                {"did": str(r[2])},
            )
            subj_row = doc_subj.fetchone()
            if subj_row:
                subject_id = subj_row[0]

        tid = str(r[1]) if r[1] else "__untopic__"
        topic_questions = questions_by_topic.get(tid, [])

        # Pair note image URLs with local paths and public URLs
        note_imgs = []
        for b2_url, local_path in zip(
            (r[8] or []),
            (r[9] or []),
        ):
            public_local_url = ""
            if local_path:
                rel = local_path.replace("/app/storage/subjects/", "")
                public_local_url = f"/local-images/{rel}"
            note_imgs.append({
                "url": b2_url,
                "local_url": public_local_url,
                "local_path": local_path
            })

        # Parse question_answers JSON
        q_answers = r[10] or []
        if isinstance(q_answers, str):
            try:
                q_answers = json.loads(q_answers)
            except Exception:
                q_answers = []

        topics_out.append({
            "topic_note_id": str(r[0]),
            "topic_id": str(r[1]) if r[1] else None,
            "topic_number": r[4] or "",
            "topic_title": r[5],
            "document_id": str(r[2]),
            "document_title": r[3],
            "notes_status": r[12],
            "generated_at": r[13].isoformat() if r[13] else None,
            "error_message": r[14],
            "note_sections": r[6] or [],
            "latex_formulas": r[7] or [],
            "note_images": note_imgs,
            "questions": topic_questions,
            "question_answers": q_answers,
            "answer_images": r[11] or [],
        })

    return {
        "chapter_id": chapter_id,
        "subject_id": subject_id,
        "total_topics": len(topics_out),
        "topics": topics_out,
    }


# ── GET /notes/documents ────────────────────────────────────────────────────────
@router.get("/documents")
async def list_documents_with_notes(
    subject_id: str = "",
    import_source: str = "",
    db: AsyncSession = Depends(get_db),
):
    """
    List note documents that have or need notes generation, with aggregate topic counts.
    Query params:
      subject_id    (optional) filter by subject
    """
    # Build dynamic WHERE conditions
    conditions = []
    params: dict = {}
    if subject_id:
        conditions.append("subject_id = :sid")
        params["sid"] = subject_id

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    doc_rows = await db.execute(
        text(f"""
            SELECT id, subject_id, title, notes_status,
                   language, created_at
            FROM note_documents
            {where_clause}
            ORDER BY created_at DESC
            LIMIT 200
        """),
        params,
    )
    docs = [dict(r._mapping) for r in doc_rows.fetchall()]

    doc_ids = [str(d["id"]) for d in docs]
    if not doc_ids:
        return {"documents": []}

    # Aggregate topic counts per document in one query
    agg = await db.execute(
        text(f"""
            SELECT document_id,
                   COUNT(*)                                 AS topics_total,
                   COUNT(*) FILTER (WHERE notes_status = 'done')        AS topics_done,
                   COUNT(*) FILTER (WHERE notes_status = 'pending')     AS topics_pending,
                   COUNT(*) FILTER (WHERE notes_status = 'generating')  AS topics_generating,
                   COUNT(*) FILTER (WHERE notes_status = 'failed')      AS topics_failed
            FROM topic_notes
            WHERE document_id = ANY(CAST(:ids AS uuid[]))
            GROUP BY document_id
        """),
        {"ids": doc_ids},
    )
    agg_map = {str(r[0]): dict(r._mapping) for r in agg.fetchall()}

    result = []
    for d in docs:
        did = str(d["id"])
        a = agg_map.get(did, {})
        result.append({
            "id": did,
            "subject_id": d.get("subject_id"),
            "title": d.get("title"),
            "status": d.get("notes_status"),
            "language": d.get("language"),
            "created_at": d["created_at"].isoformat() if d.get("created_at") else None,
            "topics_total": a.get("topics_total", 0),
            "topics_done": a.get("topics_done", 0),
            "topics_pending": a.get("topics_pending", 0),
            "topics_generating": a.get("topics_generating", 0),
            "topics_failed": a.get("topics_failed", 0),
        })

    return {"documents": result}


# ── POST /notes/retry/{document_id} ────────────────────────────────────────────────
@router.post("/retry/{document_id}")
async def retry_failed_notes(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Retry failed topic notes. Also unsticks docs stuck at 'notes_generating'."""
    # 1. Reset failed topic_notes rows back to pending
    result = await db.execute(
        text("UPDATE topic_notes SET notes_status = 'pending', error_message = NULL, updated_at = NOW() "
             "WHERE document_id = CAST(:did AS uuid) AND notes_status IN ('failed', 'generating')"),
        {"did": document_id},
    )
    failed_count = result.rowcount

    # 2. Reset stuck document status back to ready
    await db.execute(
        text("UPDATE note_documents SET notes_status = 'ready', updated_at = NOW() "
             "WHERE id = CAST(:did AS uuid) AND notes_status IN ('failed', 'generating')"),
        {"did": document_id},
    )
    await db.commit()

    background_tasks.add_task(_run_notes_generation, document_id)
    return {
        "success": True,
        "retried": failed_count,
        "message": f"Reset {failed_count} failed topics. Re-queued notes generation.",
    }


# ── POST /notes/reset/{document_id} ───────────────────────────────────────────────
@router.post("/reset/{document_id}")
async def reset_document_notes(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Admin: Force-reset document to 'ready' and clear all topic_notes for it."""
    # Delete all existing topic_notes for this doc
    del_result = await db.execute(
        text("DELETE FROM topic_notes WHERE document_id = CAST(:did AS uuid)"),
        {"did": document_id},
    )
    deleted = del_result.rowcount

    # Reset document status to ready
    await db.execute(
        text("UPDATE note_documents SET notes_status = 'ready', updated_at = NOW() "
             "WHERE id = CAST(:did AS uuid)"),
        {"did": document_id},
    )
    await db.commit()

    return {
        "success": True,
        "document_id": document_id,
        "topic_notes_deleted": deleted,
        "message": "Document reset to 'ready'. You can now re-trigger notes generation.",
    }


# ── GET /notes/logs ────────────────────────────────────────────────────────────────
@router.get("/logs")
async def get_notes_logs(
    name: str = Query("uploads", regex="^(uploads|notes|pregen|errors)$"),
    tail: int = Query(200, ge=1, le=2000),
):
    """
    Tail the notes-system log files for debugging.
    Log files live in /sdb-disk/ai-teaching/logs/{name}.log by default.
    """
    log_base = os.getenv("LOCAL_STORAGE_BASE", "/sdb-disk/ai-teaching")
    log_path = os.path.join(log_base, "logs", f"{name}.log")
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        lines = [l.rstrip("\n") for l in lines[-tail:]]
        return {"name": name, "path": log_path, "lines": lines, "count": len(lines)}
    except FileNotFoundError:
        raise HTTPException(404, f"Log file not found: {log_path}")
    except Exception as e:
        raise HTTPException(500, f"Failed to read log: {e}")
