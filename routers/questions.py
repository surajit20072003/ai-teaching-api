"""
routers/questions.py
────────────────────
Endpoints for importing and querying the rich question bank.

POST /questions/import  — bulk import questions with subject/chapter/topic hierarchy
GET  /questions         — list questions with optional filters
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import get_db
from core.cache import hash_question

router = APIRouter(prefix="/questions", tags=["questions"])


# ─────────────────────────────────────────────────────────────────────────────
# POST /questions/import
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/import")
async def import_questions(payload: Dict[str, Any], db: AsyncSession = Depends(get_db)):
    """
    Bulk import questions from the external system.

    Expected body:
    {
      "subject":   {"id": "...", "name": "Maths", "slug": "maths"},
      "chapter":   {"id": "...", "chapter_number": 1, "title": "Real Numbers"},
      "topic":     {"id": "...", "topic_number": "1.2", "title": "Fundamental Theorem..."},
      "questions": [{
          "id": "...", "question_text": "...", "question_format": "mcq",
          "options": {"A": {"text": "..."}, ...}, "correct_answer": "B",
          "difficulty": "Medium", "marks": 4
      }]
    }

    Idempotent — safe to call multiple times. Uses ON CONFLICT DO NOTHING.
    """
    subject_data  = payload.get("subject", {})
    chapter_data  = payload.get("chapter", {})
    topic_data    = payload.get("topic", {})
    questions_raw = payload.get("questions", [])

    subject_id = subject_data.get("id", "")
    if not subject_id or not questions_raw:
        return {"error": "subject.id and questions[] are required", "imported": 0}

    # ── 1. Upsert subject (deduplicate by name, case-insensitive) ───────────
    subject_name = subject_data.get("name", "")

    # First: try to find an existing subject by name (prevents duplicates when
    # the same subject is imported from two different external systems with
    # different IDs but the same display name e.g. "Science" vs "science").
    name_match = (await db.execute(
        text("SELECT subject_id FROM subjects WHERE LOWER(name) = LOWER(:n)"),
        {"n": subject_name},
    )).fetchone()

    if name_match:
        # Reuse existing subject — ignore the external id completely
        subject_id = name_match.subject_id
    else:
        # Insert using the external id as subject_id; update name if it changes
        await db.execute(
            text("""
                INSERT INTO subjects (subject_id, name, slug)
                VALUES (:sid, :name, :slug)
                ON CONFLICT (subject_id) DO UPDATE
                  SET name = EXCLUDED.name,
                      slug = COALESCE(EXCLUDED.slug, subjects.slug)
            """),
            {
                "sid":  subject_id,
                "name": subject_name,
                "slug": subject_data.get("slug"),
            },
        )


    # ── 2. Upsert chapter ────────────────────────────────────────────────────
    chapter_id = None
    if chapter_data.get("id"):
        ext_chapter_id  = chapter_data["id"]
        chapter_number  = chapter_data.get("chapter_number", 0)
        chapter_title   = chapter_data.get("title", "")

        # Try to use external id as PK; fall back to lookup if conflict
        await db.execute(
            text("""
                INSERT INTO chapters (id, subject_id, chapter_number, title)
                VALUES (CAST(:id AS uuid), :sid, :num, :title)
                ON CONFLICT (subject_id, chapter_number) DO UPDATE
                  SET title = EXCLUDED.title
            """),
            {
                "id":    ext_chapter_id,
                "sid":   subject_id,
                "num":   chapter_number,
                "title": chapter_title,
            },
        )
        # Resolve the actual chapter UUID (may differ if conflict hit an existing row)
        row = (await db.execute(
            text("SELECT id FROM chapters WHERE subject_id=:sid AND chapter_number=:num"),
            {"sid": subject_id, "num": chapter_number},
        )).fetchone()
        chapter_id = str(row[0]) if row else None

    # ── 3. Upsert topic ──────────────────────────────────────────────────────
    topic_id = None
    if topic_data.get("id") and chapter_id:
        ext_topic_id = topic_data["id"]
        topic_number = topic_data.get("topic_number", "")
        topic_title  = topic_data.get("title", "")

        await db.execute(
            text("""
                INSERT INTO topics (id, chapter_id, subject_id, topic_number, title)
                VALUES (CAST(:id AS uuid), CAST(:cid AS uuid), :sid, :num, :title)
                ON CONFLICT (chapter_id, topic_number) DO UPDATE
                  SET title = EXCLUDED.title
            """),
            {
                "id":    ext_topic_id,
                "cid":   chapter_id,
                "sid":   subject_id,
                "num":   topic_number,
                "title": topic_title,
            },
        )
        row = (await db.execute(
            text("SELECT id FROM topics WHERE chapter_id=CAST(:cid AS uuid) AND topic_number=:num"),
            {"cid": chapter_id, "num": topic_number},
        )).fetchone()
        topic_id = str(row[0]) if row else None

    # ── 3.5 Ingest document + content_markdown ───────────────────────────────
    document_data    = payload.get("document") or {}
    document_id      = document_data.get("id")
    content_markdown = (document_data.get("parsed_json") or {}).get("content_markdown", "")
    doc_chunk_result = None  # populated below if markdown is present

    if document_id:
        doc_display_name = document_data.get("display_name", "imported_doc")

        if content_markdown and content_markdown.strip():
            # ── Full pipeline: chunk + embed + save to disk ───────────────────
            from core.document_processor import chunk_markdown_text
            from core.local_storage import write_doc_meta
            import logging as _logging
            _doc_log = _logging.getLogger(__name__)

            # Use a SAVEPOINT so a failure in chunking/disk I/O doesn't abort
            # the outer transaction — we can rollback to savepoint and continue
            await db.execute(text("SAVEPOINT doc_ingest"))
            try:
                doc_chunk_result = await chunk_markdown_text(
                    subject_id, document_id, content_markdown
                )

                # Upsert documents row with real paths and chunk count
                await db.execute(
                    text("""
                        INSERT INTO documents (
                            id, subject_id, chapter_id, topic_id,
                            title, filename, local_raw_path,
                            local_processed_path, total_chunks, status
                        )
                        VALUES (
                            CAST(:id AS uuid), :sid, :cid, :tid,
                            :title, :filename, :raw_path,
                            :proc_path, :n_chunks, 'ready'
                        )
                        ON CONFLICT (id) DO UPDATE
                          SET chapter_id           = EXCLUDED.chapter_id,
                              topic_id             = EXCLUDED.topic_id,
                              title                = EXCLUDED.title,
                              local_processed_path = EXCLUDED.local_processed_path,
                              total_chunks         = EXCLUDED.total_chunks,
                              status               = 'ready'
                    """),
                    {
                        "id":        document_id,
                        "sid":       subject_id,
                        "cid":       chapter_id,
                        "tid":       topic_id,
                        "title":     doc_display_name,
                        "filename":  doc_display_name,
                        "raw_path":  doc_chunk_result["local_raw_path"],
                        "proc_path": doc_chunk_result["local_processed_path"],
                        "n_chunks":  doc_chunk_result["total_chunks"],
                    },
                )

                # Insert document_chunks (with embeddings for RAG search)
                import json as _json
                for chunk in doc_chunk_result["chunks"]:
                    emb = chunk.get("embedding")
                    await db.execute(
                        text("""
                            INSERT INTO document_chunks (
                                document_id, subject_id, chunk_index,
                                section_title, chunk_text, chunk_embedding
                            )
                            VALUES (
                                CAST(:doc_id AS uuid), :sid, :idx,
                                :sec_title, :chunk_text,
                                CAST(:emb AS vector)
                            )
                            ON CONFLICT DO NOTHING
                        """),
                        {
                            "doc_id":     document_id,
                            "sid":        subject_id,
                            "idx":        chunk["chunk_index"],
                            "sec_title":  chunk.get("section_title"),
                            "chunk_text": chunk["chunk_text"],
                            "emb":        _json.dumps(emb) if emb else None,
                        },
                    )

                # Save meta.json on disk for easy inspection
                await write_doc_meta(subject_id, document_id, {
                    "id": document_id, "subject_id": subject_id,
                    "chapter_id": chapter_id, "topic_id": topic_id,
                    "title": doc_display_name, "source": "server_import",
                    "total_chunks": doc_chunk_result["total_chunks"],
                    "total_chars": doc_chunk_result["total_chars"],
                })

                await db.execute(text("RELEASE SAVEPOINT doc_ingest"))

            except Exception as _doc_exc:
                # Roll back only the document ingest — outer tx stays alive
                await db.execute(text("ROLLBACK TO SAVEPOINT doc_ingest"))
                await db.execute(text("RELEASE SAVEPOINT doc_ingest"))
                _doc_log.warning(
                    f"[import] Document markdown ingestion failed (non-fatal): {_doc_exc}"
                )
                # Bare-minimum row so FK constraints on questions don't break
                await db.execute(
                    text("""
                        INSERT INTO documents (
                            id, subject_id, chapter_id, topic_id,
                            title, filename, local_raw_path, status
                        )
                        VALUES (
                            CAST(:id AS uuid), :sid, :cid, :tid,
                            :title, :filename, :raw_path, 'failed'
                        )
                        ON CONFLICT (id) DO NOTHING
                    """),
                    {
                        "id":       document_id,
                        "sid":      subject_id,
                        "cid":      chapter_id,
                        "tid":      topic_id,
                        "title":    doc_display_name,
                        "filename": doc_display_name,
                        "raw_path": f"imported/{doc_display_name}",
                    },
                )

        else:
            # No markdown provided — just register the document row
            await db.execute(
                text("""
                    INSERT INTO documents (
                        id, subject_id, chapter_id, topic_id,
                        title, filename, local_raw_path, status
                    )
                    VALUES (
                        CAST(:id AS uuid), :sid, :cid, :tid,
                        :title, :filename, :raw_path, 'ready'
                    )
                    ON CONFLICT (id) DO UPDATE
                      SET chapter_id = EXCLUDED.chapter_id,
                          topic_id   = EXCLUDED.topic_id,
                          title      = EXCLUDED.title
                """),
                {
                    "id":       document_id,
                    "sid":      subject_id,
                    "cid":      chapter_id,
                    "tid":      topic_id,
                    "title":    doc_display_name,
                    "filename": doc_display_name,
                    "raw_path": f"imported/{doc_display_name}",
                },
            )


    # ── 4. Insert questions + queue in teaching_qa_cache ─────────────────────
    imported = 0
    skipped  = 0
    queued   = 0

    for q in questions_raw:
        q_id   = q.get("id", str(uuid.uuid4()))
        q_text = (q.get("question_text") or "").strip()
        if not q_text:
            skipped += 1
            continue

        # Insert question (skip duplicates by external id)
        result = await db.execute(
            text("""
                INSERT INTO questions (
                    id, subject_id, chapter_id, topic_id,
                    source_document_id, source_document_purpose,
                    question_text, question_type, question_format,
                    options, option_images, question_image_url,
                    correct_answer, explanation, difficulty, marks,
                    is_verified, is_ai_generated
                )
                VALUES (
                    CAST(:id AS uuid), :sid,
                    CAST(:cid AS uuid), CAST(:tid AS uuid),
                    CAST(:doc_id AS uuid), :doc_purpose,
                    :text, :qtype, :qfmt,
                    CAST(:options AS jsonb), CAST(:opt_img AS jsonb), :q_img,
                    :correct, :explanation, :difficulty, :marks,
                    :verified, :ai_gen
                )
                ON CONFLICT (id) DO NOTHING
            """),
            {
                "id":          q_id,
                "sid":         subject_id,
                "cid":         chapter_id,
                "tid":         topic_id,
                "doc_id":      document_id or (q.get("source") or {}).get("source_document_id"),
                "doc_purpose": (q.get("source") or {}).get("source_document_purpose", "general"),
                "text":        q_text,
                "qtype":       q.get("question_type", "subjective"),
                "qfmt":        q.get("question_format", "subjective"),
                "options":     __import__("json").dumps(q.get("options") or {}),
                "opt_img":     __import__("json").dumps(q.get("option_images") or {}),
                "q_img":       q.get("question_image_url"),
                "correct":     q.get("correct_answer"),
                "explanation": q.get("explanation"),
                "difficulty":  q.get("difficulty", "Medium"),
                "marks":       q.get("marks", 4),
                "verified":    q.get("is_verified", False),
                "ai_gen":      q.get("is_ai_generated", True),
            },
        )

        if result.rowcount == 0:
            skipped += 1
            continue

        imported += 1

        # Queue in teaching_qa_cache for pre-generation
        q_hash    = hash_question(q_text)
        cache_row = await db.execute(
            text("""
                INSERT INTO teaching_qa_cache
                    (id, subject_id, chapter_id, topic_id,
                     question_hash, question_text, variation_number, pregen_status)
                VALUES
                    (gen_random_uuid(), :sid, :cid, :tid,
                     :qhash, :qtext, 1, 'pending')
                ON CONFLICT (question_hash, subject_id, variation_number) DO NOTHING
            """),
            {
                "sid":   subject_id,
                "cid":   chapter_id,
                "tid":   topic_id,
                "qhash": q_hash,
                "qtext": q_text,
            },
        )
        if cache_row.rowcount > 0:
            queued += 1

    await db.commit()

    # ── Build document status block ──────────────────────────────────────────
    doc_status = None
    if document_id:
        doc_status = {
            "id":     document_id,
            "status": "complete" if doc_chunk_result else "no_markdown",
            "chunks": doc_chunk_result["total_chunks"] if doc_chunk_result else 0,
            "chars":  doc_chunk_result["total_chars"]  if doc_chunk_result else 0,
        }

    return {
        # ── Question import counts ───────────────────────────────────────────
        "imported":           imported,
        "skipped_duplicates": skipped,
        "queued_for_pregen":  queued,

        # ── Hierarchy IDs ────────────────────────────────────────────────────
        "subject_id": subject_id,
        "chapter_id": chapter_id,
        "topic_id":   topic_id,

        # ── Document RAG ingestion result ────────────────────────────────────
        "document": doc_status,

        # ── Verify — use these endpoints to confirm everything worked ────────
        "verify": {
            "questions_in_db":  f"GET /questions?subject_id={subject_id}&limit=50",
            "pregen_progress":  f"GET /questions/status/{subject_id}",
            "start_pregen":     "POST /pregen/start  { \"subjectId\": \"...\" }",
            "live_pregen_poll": "GET  /pregen/status",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /questions
# ─────────────────────────────────────────────────────────────────────────────
@router.get("")
async def list_questions(
    subject_id:    Optional[str] = Query(None),
    chapter_id:    Optional[str] = Query(None),
    topic_id:      Optional[str] = Query(None),
    is_pregen_done: Optional[bool] = Query(None),
    limit:         int = Query(50, le=200),
    offset:        int = Query(0),
    db: AsyncSession = Depends(get_db),
):
    """List questions with optional filters."""
    filters = ["1=1"]
    params: Dict[str, Any] = {"limit": limit, "offset": offset}

    if subject_id:
        filters.append("q.subject_id = :subject_id")
        params["subject_id"] = subject_id
    if chapter_id:
        filters.append("q.chapter_id = CAST(:chapter_id AS uuid)")
        params["chapter_id"] = chapter_id
    if topic_id:
        filters.append("q.topic_id = CAST(:topic_id AS uuid)")
        params["topic_id"] = topic_id
    if is_pregen_done is not None:
        filters.append("q.is_pregen_done = :is_pregen_done")
        params["is_pregen_done"] = is_pregen_done

    where = " AND ".join(filters)
    rows = (await db.execute(
        text(f"""
            SELECT q.id, q.subject_id, q.chapter_id, q.topic_id,
                   q.question_text, q.question_format, q.correct_answer,
                   q.difficulty, q.marks, q.is_pregen_done, q.is_verified,
                   q.created_at,
                   ch.title AS chapter_title, ch.chapter_number,
                   t.title  AS topic_title,   t.topic_number
            FROM questions q
            LEFT JOIN chapters ch ON ch.id = q.chapter_id
            LEFT JOIN topics   t  ON t.id  = q.topic_id
            WHERE {where}
            ORDER BY q.created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )).fetchall()

    total = (await db.execute(
        text(f"SELECT COUNT(*) FROM questions q WHERE {where}"),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )).scalar()

    return {
        "total": total,
        "questions": [
            {
                "id":             str(r.id),
                "subject_id":     r.subject_id,
                "chapter_id":     str(r.chapter_id) if r.chapter_id else None,
                "topic_id":       str(r.topic_id) if r.topic_id else None,
                "chapter_title":  r.chapter_title,
                "chapter_number": r.chapter_number,
                "topic_title":    r.topic_title,
                "topic_number":   r.topic_number,
                "question_text":  r.question_text,
                "question_format": r.question_format,
                "correct_answer": r.correct_answer,
                "difficulty":     r.difficulty,
                "marks":          r.marks,
                 "is_pregen_done": r.is_pregen_done,
                "is_verified":    r.is_verified,
                "created_at":     r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /questions/status  — all-subjects overview
# NOTE: must be declared BEFORE /questions/{question_id} to avoid route clash
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/status", summary="Question bank status — all subjects")
async def questions_status_all(
    subject_id: Optional[str] = Query(None, description="Filter to a single subject"),
    db: AsyncSession = Depends(get_db),
):
    """
    Overall summary of question bank + pregen progress, grouped by subject.
    Optionally filter to one subject with ?subject_id=...
    """
    where = "WHERE q.subject_id = :sid" if subject_id else ""
    params: Dict[str, Any] = {"sid": subject_id} if subject_id else {}

    rows = (await db.execute(
        text(f"""
            SELECT
                q.subject_id,
                s.name                                                      AS subject_name,
                COUNT(q.id)                                                 AS total,
                COUNT(q.id) FILTER (WHERE q.is_pregen_done = TRUE)         AS pregen_done,
                COUNT(q.id) FILTER (WHERE q.is_pregen_done = FALSE)        AS pregen_pending,
                COUNT(DISTINCT q.chapter_id)                               AS chapters,
                COUNT(DISTINCT q.topic_id)                                 AS topics
            FROM questions q
            LEFT JOIN subjects s ON s.subject_id = q.subject_id
            {where}
            GROUP BY q.subject_id, s.name
            ORDER BY total DESC
        """),
        params,
    )).fetchall()

    # Also get pregen_status='failed' counts from teaching_qa_cache
    failed_rows = (await db.execute(
        text(f"""
            SELECT subject_id, COUNT(*) AS cnt
            FROM teaching_qa_cache
            WHERE pregen_status = 'failed'
            {"AND subject_id = :sid" if subject_id else ""}
            GROUP BY subject_id
        """),
        params,
    )).fetchall()
    failed_map = {r.subject_id: r.cnt for r in failed_rows}

    # Difficulty + format breakdown per subject
    breakdown_rows = (await db.execute(
        text(f"""
            SELECT q.subject_id,
                   q.difficulty,
                   q.question_format,
                   COUNT(*) AS cnt
            FROM questions q
            {where}
            GROUP BY q.subject_id, q.difficulty, q.question_format
        """),
        params,
    )).fetchall()

    breakdown: Dict[str, Dict] = {}
    for b in breakdown_rows:
        sid = b.subject_id
        if sid not in breakdown:
            breakdown[sid] = {"difficulty": {}, "format": {}}
        diff = b.difficulty or "Unknown"
        fmt  = b.question_format or "unknown"
        breakdown[sid]["difficulty"][diff] = breakdown[sid]["difficulty"].get(diff, 0) + b.cnt
        breakdown[sid]["format"][fmt]      = breakdown[sid]["format"].get(fmt, 0) + b.cnt

    subjects = []
    total_all = 0
    for r in rows:
        sid      = r.subject_id
        total    = int(r.total or 0)
        done     = int(r.pregen_done or 0)
        pending  = int(r.pregen_pending or 0)
        failed   = int(failed_map.get(sid, 0))
        total_all += total
        subjects.append({
            "subject_id":     sid,
            "subject_name":   r.subject_name or sid,
            "total":          total,
            "pregen_done":    done,
            "pregen_pending": pending,
            "pregen_failed":  failed,
            "pregen_pct":     round(done / total * 100) if total else 0,
            "chapters":       int(r.chapters or 0),
            "topics":         int(r.topics or 0),
            "by_difficulty":  breakdown.get(sid, {}).get("difficulty", {}),
            "by_format":      breakdown.get(sid, {}).get("format", {}),
        })

    return {
        "total_questions": total_all,
        "subjects":        subjects,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /questions/status/{subject_id}  — chapter + topic tree with pregen %
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/status/{subject_id}", summary="Question bank status — per chapter/topic tree")
async def questions_status_subject(
    subject_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Detailed pregen progress broken down by chapter → topic for a single subject.
    Use this to verify an import worked and track pregen completion per chapter.
    """
    # ── Subject totals ────────────────────────────────────────────────────────
    totals = (await db.execute(
        text("""
            SELECT
                COUNT(*)                                              AS total,
                COUNT(*) FILTER (WHERE is_pregen_done = TRUE)        AS pregen_done,
                COUNT(*) FILTER (WHERE is_pregen_done = FALSE)       AS pregen_pending
            FROM questions WHERE subject_id = :sid
        """),
        {"sid": subject_id},
    )).fetchone()

    failed_count = (await db.execute(
        text("""
            SELECT COUNT(*) FROM teaching_qa_cache
            WHERE subject_id = :sid AND pregen_status = 'failed'
        """),
        {"sid": subject_id},
    )).scalar() or 0

    subject_name = (await db.execute(
        text("SELECT name FROM subjects WHERE subject_id = :sid LIMIT 1"),
        {"sid": subject_id},
    )).scalar() or subject_id

    # ── Chapter rollup ────────────────────────────────────────────────────────
    chapter_rows = (await db.execute(
        text("""
            SELECT
                ch.id::text                                             AS chapter_id,
                ch.chapter_number,
                ch.title,
                COUNT(q.id)                                             AS total,
                COUNT(q.id) FILTER (WHERE q.is_pregen_done = TRUE)     AS pregen_done,
                COUNT(q.id) FILTER (WHERE q.is_pregen_done = FALSE)    AS pregen_pending
            FROM chapters ch
            LEFT JOIN questions q ON q.chapter_id = ch.id
            WHERE ch.subject_id = :sid
            GROUP BY ch.id, ch.chapter_number, ch.title
            ORDER BY ch.chapter_number
        """),
        {"sid": subject_id},
    )).fetchall()

    # ── Topic rollup ──────────────────────────────────────────────────────────
    topic_rows = (await db.execute(
        text("""
            SELECT
                t.id::text                                             AS topic_id,
                t.chapter_id::text,
                t.topic_number,
                t.title,
                COUNT(q.id)                                            AS total,
                COUNT(q.id) FILTER (WHERE q.is_pregen_done = TRUE)    AS pregen_done,
                COUNT(q.id) FILTER (WHERE q.is_pregen_done = FALSE)   AS pregen_pending
            FROM topics t
            LEFT JOIN questions q ON q.topic_id = t.id
            WHERE t.subject_id = :sid
            GROUP BY t.id, t.chapter_id, t.topic_number, t.title
            ORDER BY t.topic_number
        """),
        {"sid": subject_id},
    )).fetchall()

    # Group topics by chapter
    topics_by_chapter: Dict[str, list] = {}
    for t in topic_rows:
        cid = t.chapter_id
        if cid not in topics_by_chapter:
            topics_by_chapter[cid] = []
        tot = int(t.total or 0)
        don = int(t.pregen_done or 0)
        topics_by_chapter[cid].append({
            "topic_id":       t.topic_id,
            "topic_number":   t.topic_number,
            "title":          t.title,
            "total":          tot,
            "pregen_done":    don,
            "pregen_pending": int(t.pregen_pending or 0),
            "pregen_pct":     round(don / tot * 100) if tot else 0,
        })

    chapters = []
    for ch in chapter_rows:
        tot = int(ch.total or 0)
        don = int(ch.pregen_done or 0)
        chapters.append({
            "chapter_id":     ch.chapter_id,
            "chapter_number": ch.chapter_number,
            "title":          ch.title,
            "total":          tot,
            "pregen_done":    don,
            "pregen_pending": int(ch.pregen_pending or 0),
            "pregen_pct":     round(don / tot * 100) if tot else 0,
            "topics":         topics_by_chapter.get(ch.chapter_id, []),
        })

    total = int(totals.total or 0)
    done  = int(totals.pregen_done or 0)
    return {
        "subject_id":     subject_id,
        "subject_name":   subject_name,
        "total":          total,
        "pregen_done":    done,
        "pregen_pending": int(totals.pregen_pending or 0),
        "pregen_failed":  int(failed_count),
        "pregen_pct":     round(done / total * 100) if total else 0,
        "chapters":       chapters,
        "next_step":      "POST /pregen/start" if done < total else "All pre-generated ✓",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /questions/{question_id}  — single question + full pipeline state
# NOTE: must be declared AFTER /questions/status to avoid route clash
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/{question_id}", summary="Single question detail + cache/pregen state")
async def get_question(
    question_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Full detail for one question including its pre-generation cache state.
    Use this to confirm a specific question was imported + pre-generated correctly.
    """
    row = (await db.execute(
        text("""
            SELECT q.id, q.subject_id, q.chapter_id::text, q.topic_id::text,
                   q.question_text, q.question_format, q.question_type,
                   q.options, q.correct_answer, q.explanation,
                   q.difficulty, q.marks,
                   q.is_pregen_done, q.is_verified, q.is_ai_generated,
                   q.created_at,
                   ch.title  AS chapter_title,  ch.chapter_number,
                   t.title   AS topic_title,    t.topic_number,
                   s.name    AS subject_name
            FROM questions q
            LEFT JOIN chapters ch ON ch.id = q.chapter_id
            LEFT JOIN topics   t  ON t.id  = q.topic_id
            LEFT JOIN subjects s  ON s.subject_id = q.subject_id
            WHERE q.id = CAST(:qid AS uuid)
        """),
        {"qid": question_id},
    )).fetchone()

    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found")

    # ── Look up the cache entry for this question ─────────────────────────────
    cache_row = (await db.execute(
        text("""
            SELECT id::text, pregen_status, presentation_slides,
                   slide_audio_urls, total_duration_seconds, is_doc_grounded,
                   created_at
            FROM teaching_qa_cache
            WHERE question_text = :qtext AND subject_id = :sid
            ORDER BY created_at DESC NULLS LAST LIMIT 1
        """),
        {"qtext": row.question_text, "sid": row.subject_id},
    )).fetchone()

    cache = None
    if cache_row:
        slides     = cache_row.presentation_slides or []
        audio_urls = cache_row.slide_audio_urls or {}
        audio_list = audio_urls.get("urls", []) if isinstance(audio_urls, dict) else []
        cache = {
            "cache_id":              cache_row.id,
            "pregen_status":         cache_row.pregen_status,
            "slides_count":          len(slides),
            "has_audio":             len(audio_list) > 0,
            "has_images":            any(s.get("infographicUrl") for s in slides) if slides else False,
            "total_duration_seconds": float(cache_row.total_duration_seconds or 0),
            "is_doc_grounded":       cache_row.is_doc_grounded,
            "created_at":            cache_row.created_at.isoformat() if cache_row.created_at else None
        }

    return {
        "id":              str(row.id),
        "subject_id":      row.subject_id,
        "subject_name":    row.subject_name,
        "chapter_id":      row.chapter_id,
        "chapter_title":   row.chapter_title,
        "chapter_number":  row.chapter_number,
        "topic_id":        row.topic_id,
        "topic_title":     row.topic_title,
        "topic_number":    row.topic_number,
        "question_text":   row.question_text,
        "question_format": row.question_format,
        "question_type":   row.question_type,
        "options":         row.options,
        "correct_answer":  row.correct_answer,
        "explanation":     row.explanation,
        "difficulty":      row.difficulty,
        "marks":           row.marks,
        "is_pregen_done":  row.is_pregen_done,
        "is_verified":     row.is_verified,
        "is_ai_generated": row.is_ai_generated,
        "created_at":      row.created_at.isoformat() if row.created_at else None,
        "cache":           cache,
    }

