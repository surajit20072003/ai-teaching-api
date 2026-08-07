"""
routers/import_json.py — JSON Document Import Endpoint
=======================================================

Accepts a rich JSON payload from an external system (Supabase-backed CMS)
containing subject, chapter, topic, document (with pre-parsed markdown + images),
and questions. Stores everything correctly into the database and queues
textbook notes generation in the background.

Endpoint:
  POST /documents/import-json

Does NOT touch teaching_qa_cache — this is a notes-only pipeline,
completely separate from slide-wise pregen.
"""
from __future__ import annotations

import json as _json
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


# ── Pydantic request models ────────────────────────────────────────────────────

class SubjectIn(BaseModel):
    id: str
    name: str
    slug: Optional[str] = None


class ChapterIn(BaseModel):
    id: str
    chapter_number: int
    title: str


class TopicIn(BaseModel):
    id: str
    topic_number: Optional[str] = None
    title: str


class ParsedJsonIn(BaseModel):
    content_markdown: Optional[str] = ""
    images: Optional[Dict[str, str]] = {}   # {filename: supabase_url}


class DocumentIn(BaseModel):
    id: str
    display_name: str
    source_type: Optional[str] = "docx"
    source_url: Optional[str] = None
    parsed_json: Optional[ParsedJsonIn] = None


class QuestionIn(BaseModel):
    id: str
    question_text: str
    question_format: Optional[str] = "long_answer"
    options: Optional[Dict[str, Any]] = {}
    correct_answer: Optional[str] = None
    difficulty: Optional[str] = "Medium"
    marks: Optional[int] = 4


class ImportJsonPayload(BaseModel):
    subject: SubjectIn
    chapter: ChapterIn
    topic: TopicIn
    document: DocumentIn
    questions: Optional[List[QuestionIn]] = []
    important_questions: Optional[List[Any]] = []   # stored for reference, not processed


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_image_urls(markdown: str, images: Dict[str, str]) -> str:
    """
    Replace local image references like ![](hash_img.jpg) or
    ![alt text](hash_img.jpg) in markdown with the actual hosted URLs.
    """
    if not markdown or not images:
        return markdown

    for filename, url in images.items():
        # Matches both ![](filename) and ![any alt text](filename)
        pattern = r'!\[([^\]]*)\]\(' + re.escape(filename) + r'\)'
        replacement = r'![\1](' + url + r')'
        markdown = re.sub(pattern, replacement, markdown)

    return markdown


# ── Main endpoint ──────────────────────────────────────────────────────────────

@router.post("/import-json", summary="Import a structured JSON document (notes pipeline)")
async def import_json_document(
    payload: ImportJsonPayload,
    db: AsyncSession = Depends(get_db),
):
    """
    Import a structured JSON document payload from an external CMS.

    Pipeline (6 steps — notes generation is NOT auto-started):
      1. UPSERT subject
      2. UPSERT chapter
      3. UPSERT topic
      4. Resolve image URLs in markdown (replace local filenames with hosted URLs)
      5. UPSERT document  (status='ready', content_markdown stored directly)
      6. BULK INSERT questions (preserve external UUIDs, skip duplicates)

    To generate notes: use the Notes Batch in the dashboard or POST /notes/batch/start
    Does NOT touch teaching_qa_cache — this is a notes-only pipeline.
    """
    subject_id  = payload.subject.id
    chapter_id  = payload.chapter.id
    topic_id    = payload.topic.id
    document_id = payload.document.id
    log         = f"[import-json/doc={document_id[:8]}]"

    logger.info(
        f"{log} Import start | subject={payload.subject.name} "
        f"| chapter='{payload.chapter.title}' | topic='{payload.topic.title}'"
    )

    # ── Step 1: UPSERT subject ────────────────────────────────────────────────
    # subjects table: PK is subject_id (text), no separate id column
    await db.execute(text("""
        INSERT INTO subjects (subject_id, name, slug)
        VALUES (:subject_id, :name, :slug)
        ON CONFLICT (subject_id) DO UPDATE SET
            name = EXCLUDED.name,
            slug = EXCLUDED.slug
    """), {
        "subject_id": subject_id,
        "name":       payload.subject.name,
        "slug":       payload.subject.slug or payload.subject.name.lower().replace(" ", "-"),
    })
    logger.info(f"{log} ✓ [1/7] Subject upserted")

    # ── Step 2: UPSERT chapter ────────────────────────────────────────────────
    # chapters unique constraint: (subject_id, chapter_number)
    await db.execute(text("""
        INSERT INTO chapters (id, subject_id, chapter_number, title)
        VALUES (CAST(:id AS uuid), :subject_id, :chapter_number, :title)
        ON CONFLICT (subject_id, chapter_number) DO UPDATE SET
            title          = EXCLUDED.title,
            chapter_number = EXCLUDED.chapter_number
    """), {
        "id":             chapter_id,
        "subject_id":     subject_id,
        "chapter_number": payload.chapter.chapter_number,
        "title":          payload.chapter.title,
    })
    logger.info(f"{log} ✓ [2/7] Chapter upserted")

    # ── Step 3: UPSERT topic ──────────────────────────────────────────────────
    # topics unique constraint: (chapter_id, topic_number)
    await db.execute(text("""
        INSERT INTO topics (id, chapter_id, subject_id, topic_number, title)
        VALUES (
            CAST(:id AS uuid), CAST(:chapter_id AS uuid),
            :subject_id, :topic_number, :title
        )
        ON CONFLICT (chapter_id, topic_number) DO UPDATE SET
            title        = EXCLUDED.title,
            topic_number = EXCLUDED.topic_number
    """), {
        "id":           topic_id,
        "chapter_id":   chapter_id,
        "subject_id":   subject_id,
        "topic_number": payload.topic.topic_number or "0",
        "title":        payload.topic.title,
    })
    logger.info(f"{log} ✓ [3/7] Topic upserted")

    # ── Step 4: Resolve image URLs in markdown ────────────────────────────────
    parsed        = payload.document.parsed_json or ParsedJsonIn()
    raw_markdown  = parsed.content_markdown or ""
    images_map    = parsed.images or {}
    resolved_md   = _resolve_image_urls(raw_markdown, images_map)
    logger.info(
        f"{log} ✓ [4/7] Markdown resolved "
        f"({len(images_map)} images → {len(resolved_md)} chars)"
    )

    # ── Step 5: UPSERT note_document ────────────────────────────────────────────
    await db.execute(text("""
        INSERT INTO note_documents
            (id, subject_id, chapter_id, topic_id,
             title, filename, local_raw_path,
             content_markdown, parsed_images,
             notes_status, language, access_tier,
             created_at, updated_at)
        VALUES
            (CAST(:id AS uuid), :subject_id, CAST(:chapter_id AS uuid), CAST(:topic_id AS uuid),
             :title, :filename, 'imported.json',
             :content_markdown, CAST(:parsed_images AS jsonb),
             'notes_pending', 'hi-IN', 'pro',
             NOW(), NOW())
        ON CONFLICT (id) DO UPDATE SET
            title            = EXCLUDED.title,
            content_markdown = EXCLUDED.content_markdown,
            parsed_images    = EXCLUDED.parsed_images,
            local_raw_path   = EXCLUDED.local_raw_path,
            notes_status     = 'notes_pending',
            updated_at       = NOW()
    """), {
        "id":               document_id,
        "subject_id":       subject_id,
        "chapter_id":       chapter_id,
        "topic_id":         topic_id,
        "title":            payload.document.display_name,
        "filename":         payload.document.display_name,
        "content_markdown": resolved_md,
        "parsed_images":    _json.dumps(images_map),
    })
    logger.info(f"{log} ✓ [5/7] Document upserted")

    # ── Step 6: BULK INSERT questions ─────────────────────────────────────────
    questions_imported = 0
    questions_skipped  = 0
    for q in (payload.questions or []):
        try:
            # Normalize options: accept {A: {text: "..."}} or {A: "..."}
            normalized: Dict[str, Any] = {}
            for k, v in (q.options or {}).items():
                normalized[k] = v if isinstance(v, dict) else {"text": str(v)}

            await db.execute(text("""
                INSERT INTO questions
                    (id, subject_id, chapter_id, topic_id,
                     source_document_id,
                     question_text, question_format, question_type,
                     options, correct_answer,
                     difficulty, marks,
                     is_ai_generated, is_verified)
                VALUES
                    (CAST(:id AS uuid),
                     :subject_id, CAST(:chapter_id AS uuid), CAST(:topic_id AS uuid),
                     CAST(:doc_id AS uuid),
                     :question_text, :question_format, :question_format,
                     CAST(:options AS jsonb), :correct_answer,
                     :difficulty, :marks,
                     false, true)
                ON CONFLICT (id) DO NOTHING
            """), {
                "id":              q.id,
                "subject_id":      subject_id,
                "chapter_id":      chapter_id,
                "topic_id":        topic_id,
                "doc_id":          document_id,
                "question_text":   q.question_text,
                "question_format": q.question_format or "long_answer",
                "options":         _json.dumps(normalized),
                "correct_answer":  q.correct_answer or "",
                "difficulty":      q.difficulty or "Medium",
                "marks":           q.marks or 4,
            })
            questions_imported += 1
        except Exception as e:
            logger.warning(f"{log} Question {q.id} skipped: {e}")
            questions_skipped += 1

    logger.info(f"{log} ✓ [6/6] Questions: {questions_imported} imported, {questions_skipped} skipped")

    # Commit all DB writes
    await db.commit()
    logger.info(f"{log} Import complete — document ready for batch notes generation")

    return {
        "success":            True,
        "document_id":        document_id,
        "subject_id":         subject_id,
        "chapter_id":         chapter_id,
        "topic_id":           topic_id,
        "title":              payload.document.display_name,
        "questions_imported": questions_imported,
        "questions_skipped":  questions_skipped,
        "images_resolved":    len(images_map),
        "markdown_chars":     len(resolved_md),
        "notes_queued":       False,
        "message": (
            f"Document imported. {questions_imported} questions stored. "
            f"Use the Notes Batch to generate notes."
        ),
    }
