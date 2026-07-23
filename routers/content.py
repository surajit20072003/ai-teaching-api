"""
routers/content.py — Content Library Browser API
=================================================

Endpoints:
  GET /content/subjects                          — all subjects with chapter + question counts
  GET /content/subjects/{subject_id}/chapters    — chapters for a subject with question counts
  GET /content/chapters/{chapter_id}/questions   — paginated list of done questions
  GET /content/questions/{cache_id}/slides       — full slide payload for one question
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import get_db

router = APIRouter(prefix="/content", tags=["Content Library"])


# ── GET /content/subjects ─────────────────────────────────────────────────────
@router.get("/subjects")
async def list_subjects(db: AsyncSession = Depends(get_db)):
    """
    Returns all subjects with chapter count and fully-done question count.
    """
    rows = (await db.execute(text("""
        SELECT
            s.subject_id::text AS subject_id,
            s.name,
            (SELECT COUNT(*) FROM chapters WHERE subject_id = s.subject_id) AS chapter_count,
            (SELECT COALESCE(SUM(CASE
                WHEN pregen_status = 'done'
                 AND presentation_slides->0->>'infographicUrl' LIKE 'http%'
                 AND presentation_slides->0->>'audioUrl'       LIKE 'http%'
                THEN 1 ELSE 0
            END), 0) FROM teaching_qa_cache WHERE subject_id = s.subject_id) AS questions_done,
            (SELECT COUNT(*) FROM teaching_qa_cache WHERE subject_id = s.subject_id) AS questions_total
        FROM subjects s
        ORDER BY s.name
    """))).fetchall()

    return [
        {
            "subject_id":      r.subject_id,
            "name":            r.name,
            "chapter_count":   int(r.chapter_count or 0),
            "questions_done":  int(r.questions_done or 0),
            "questions_total": int(r.questions_total or 0),
        }
        for r in rows
    ]


# ── GET /content/subjects/{subject_id}/chapters ───────────────────────────────
@router.get("/subjects/{subject_id}/chapters")
async def list_chapters(
    subject_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns all chapters for a subject with per-chapter done question counts.
    """
    rows = (await db.execute(text("""
        SELECT
            c.id::text           AS chapter_id,
            c.chapter_number,
            c.title,
            COUNT(tqc.id)        AS questions_total,
            COALESCE(SUM(CASE
                WHEN tqc.pregen_status = 'done'
                 AND tqc.presentation_slides->0->>'infographicUrl' LIKE 'http%'
                 AND tqc.presentation_slides->0->>'audioUrl'       LIKE 'http%'
                THEN 1 ELSE 0
            END), 0) AS questions_done,
            COALESCE(SUM(CASE
                WHEN tqc.manim_video_urls IS NOT NULL
                 AND tqc.manim_video_urls != '{}'::jsonb
                THEN 1 ELSE 0
            END), 0) AS questions_with_manim
        FROM chapters c
        LEFT JOIN teaching_qa_cache tqc
            ON tqc.chapter_id = c.id::text
            AND tqc.subject_id = :sid
        WHERE c.subject_id = :sid
        GROUP BY c.id, c.chapter_number, c.title
        ORDER BY c.chapter_number
    """), {"sid": subject_id})).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="Subject not found or has no chapters")

    return [
        {
            "chapter_id":           r.chapter_id,
            "chapter_number":       int(r.chapter_number or 0),
            "title":                r.title,
            "questions_total":      int(r.questions_total or 0),
            "questions_done":       int(r.questions_done or 0),
            "questions_with_manim": int(r.questions_with_manim or 0),
        }
        for r in rows
    ]


# ── GET /content/chapters/{chapter_id}/questions ──────────────────────────────
@router.get("/chapters/{chapter_id}/questions")
async def list_questions(
    chapter_id: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=100, ge=1, le=500),
    status: str = Query(default="done", description="Filter: done | all"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns paginated list of questions for a chapter.
    Default: only fully done rows (with images + audio).
    """
    offset = (page - 1) * limit
    status_filter = """
        pregen_status = 'done'
        AND presentation_slides->0->>'infographicUrl' LIKE 'http%'
        AND presentation_slides->0->>'audioUrl'       LIKE 'http%'
    """ if status == "done" else "1=1"

    total_row = (await db.execute(text(f"""
        SELECT COUNT(*) as n FROM teaching_qa_cache
        WHERE chapter_id = :cid AND ({status_filter})
    """), {"cid": chapter_id})).fetchone()
    total = int(total_row.n or 0)

    rows = (await db.execute(text(f"""
        SELECT
            id::text              AS cache_id,
            question_text,
            access_tier,
            pregen_status,
            jsonb_array_length(COALESCE(presentation_slides, '[]'::jsonb)) AS slide_count,
            (presentation_slides->0->>'infographicUrl' LIKE 'http%')       AS has_image,
            (presentation_slides->0->>'audioUrl'       LIKE 'http%')       AS has_audio,
            (manim_video_urls IS NOT NULL
             AND manim_video_urls != '{{}}'::jsonb
             AND manim_video_urls != 'null'::jsonb)                         AS has_manim,
            created_at
        FROM teaching_qa_cache
        WHERE chapter_id = :cid AND ({status_filter})
        ORDER BY created_at ASC
        LIMIT :lim OFFSET :off
    """), {"cid": chapter_id, "lim": limit, "off": offset})).fetchall()

    return {
        "total":   total,
        "page":    page,
        "limit":   limit,
        "pages":   max(1, (total + limit - 1) // limit),
        "questions": [
            {
                "cache_id":      r.cache_id,
                "question_text": r.question_text,
                "access_tier":   r.access_tier,
                "pregen_status": r.pregen_status,
                "slide_count":   int(r.slide_count or 0),
                "has_image":     bool(r.has_image),
                "has_audio":     bool(r.has_audio),
                "has_manim":     bool(r.has_manim),
            }
            for r in rows
        ],
    }


# ── GET /content/questions/{cache_id}/slides ──────────────────────────────────
@router.get("/questions/{cache_id}/slides")
async def get_question_slides(
    cache_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the full slide payload for one question — ready for the viewer.
    Each slide is enriched with its manimVideoUrl if available.
    """
    row = (await db.execute(text("""
        SELECT
            id::text              AS cache_id,
            question_text,
            access_tier,
            pregen_status,
            language,
            presentation_slides,
            manim_video_urls,
            slide_audio_urls,
            image_urls,
            total_duration_seconds
        FROM teaching_qa_cache
        WHERE id = CAST(:cid AS uuid)
    """), {"cid": cache_id})).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Question not found")

    slides = row.presentation_slides or []
    manim  = row.manim_video_urls    or {}
    audio  = row.slide_audio_urls    or {}
    images = row.image_urls          or {}

    # Enrich each slide with its Manim video URL if available
    enriched = []
    for i, slide in enumerate(slides):
        s = dict(slide)
        manim_entry = manim.get(str(i))
        if manim_entry and isinstance(manim_entry, dict):
            s["manimVideoUrl"]        = manim_entry.get("url", "")
            s["manimDurationSeconds"] = manim_entry.get("duration_seconds", 0)
        else:
            s["manimVideoUrl"] = ""
            s["manimDurationSeconds"] = 0
        enriched.append(s)

    return {
        "cache_id":             row.cache_id,
        "question_text":        row.question_text,
        "access_tier":          row.access_tier,
        "language":             row.language,
        "slide_count":          len(enriched),
        "presentationSlides":   enriched,
        "manimVideoUrls":       manim,
        "slideAudioUrls":       audio,
        "imageUrls":            images,
        "totalDurationSeconds": float(row.total_duration_seconds or 0),
    }
