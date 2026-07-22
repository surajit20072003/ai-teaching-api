"""
core/cross_system_enrichment.py
────────────────────────────────
Runs two parallel lookups after any text answer is resolved:
1. slide_preview  — fully-completed teaching_qa_cache entry (slides + audio)
2. suggestions    — other fully-completed questions ordered by similarity
Only returns entries where both slides AND audio exist.
"""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sql_text
from core.embeddings import vec_to_pg_str


def _is_fully_complete(row: dict) -> bool:
    slides = row.get("presentation_slides") or []
    audio  = row.get("slide_audio_urls") or {}
    if not slides:
        return False
    audio_urls = audio.get("urls", audio) if isinstance(audio, dict) else audio
    return bool(audio_urls)


async def enrich_with_slide_preview(question_embedding, subject_id, db, current_hash, min_similarity=0.65):
    if not question_embedding:
        return {"found": False}
    vec_str = vec_to_pg_str(question_embedding)
    try:
        rows = await db.execute(sql_text("""
            SELECT id::text AS cache_id, question_text, question_hash,
                   presentation_slides, slide_audio_urls, total_duration_seconds,
                   1 - (question_embedding <=> CAST(:vec AS vector)) AS similarity
            FROM teaching_qa_cache
            WHERE subject_id = :subject_id
              AND question_hash != :skip_hash
              AND question_embedding IS NOT NULL
            ORDER BY question_embedding <=> CAST(:vec AS vector)
            LIMIT 5
        """), {"vec": vec_str, "subject_id": subject_id, "skip_hash": current_hash})
        for row in rows.mappings().all():
            sim = float(row["similarity"])
            if sim < min_similarity:
                break
            row_dict = dict(row)
            if not _is_fully_complete(row_dict):
                continue
            return {"found": True, "cache_id": row_dict["cache_id"],
                    "similarity": round(sim, 3),
                    "matched_question": row_dict["question_text"],
                    "presentation_slides": row_dict["presentation_slides"],
                    "slide_audio_urls": row_dict["slide_audio_urls"],
                    "total_duration_seconds": row_dict.get("total_duration_seconds", 0)}
    except Exception as e:
        print(f"[Enrichment] slide_preview failed: {e}")
    return {"found": False}


async def fetch_suggestion_questions(question_embedding, subject_id, db, skip_hash, limit=5):
    if not question_embedding:
        return []
    vec_str = vec_to_pg_str(question_embedding)
    try:
        rows = await db.execute(sql_text("""
            SELECT id::text AS cache_id, question_text, slide_audio_urls, presentation_slides,
                   1 - (question_embedding <=> CAST(:vec AS vector)) AS similarity
            FROM teaching_qa_cache
            WHERE subject_id = :subject_id
              AND question_hash != :skip_hash
              AND question_embedding IS NOT NULL
            ORDER BY question_embedding <=> CAST(:vec AS vector)
            LIMIT 20
        """), {"vec": vec_str, "subject_id": subject_id, "skip_hash": skip_hash})
        results = []
        for row in rows.mappings().all():
            if len(results) >= limit:
                break
            row_dict = dict(row)
            if not _is_fully_complete(row_dict):
                continue
            results.append({"question": row_dict["question_text"],
                             "cache_id": row_dict["cache_id"],
                             "similarity": round(float(row_dict["similarity"]), 3),
                             "has_slides": True})
        return results
    except Exception as e:
        print(f"[Enrichment] suggestions failed: {e}")
        return []


async def enrich_response(question_embedding, subject_id, db, current_hash):
    slide_preview = await enrich_with_slide_preview(question_embedding, subject_id, db, current_hash)
    suggestions   = await fetch_suggestion_questions(question_embedding, subject_id, db, current_hash)
    return slide_preview, suggestions
