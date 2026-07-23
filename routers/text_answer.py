"""
routers/text_answer.py
───────────────────────
POST /ai-text-answer

Returns a rich, educationally-formatted text answer for a student question.
Also returns (when available):
  - slide_preview: a fully-completed entry from teaching_qa_cache
  - suggestions:   other fully-completed questions from teaching_qa_cache

Cache layers (L1 → L2 → generation):
  L1  Redis exact hash          ~0.5ms   (only exact same question)
  L2  Postgres exact hash       ~3ms     (only exact same question)
  L3  Semantic cache            REMOVED  (was causing similar-question cross-contamination)
  L4  RAG: document_chunks      ~30ms    (retrieves relevant course material)
  L5  LLM generation            ~3-8s    (OpenRouter free models)
"""
import asyncio
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError

from db.models import get_db
from core.cache import (
    hash_question,
    get_text_from_cache, set_text_to_cache,
    acquire_text_lock, release_text_lock, wait_for_text_cache,
)
from core.embeddings import embed_async, vec_to_pg_str
from core.text_answer_generator import generate_text_answer
from core.cross_system_enrichment import enrich_response

router = APIRouter()

async def _fetch_doc_context(db, subject_id, vec_str, user_tier="pro"):
    tier_filter = "AND dc.access_tier = 'free'" if user_tier == "free" else ""
    rows = await db.execute(sql_text(f"""
        SELECT
            dc.chunk_text,
            dc.section_title,
            d.title AS doc_title, dc.access_tier,
            1 - (dc.chunk_embedding <=> CAST(:vec AS vector)) AS sim
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE dc.subject_id = :subject_id
          AND dc.chunk_embedding IS NOT NULL
          {tier_filter}
        ORDER BY dc.chunk_embedding <=> CAST(:vec AS vector)
        LIMIT 3
    """), {"vec": vec_str, "subject_id": subject_id})
    results = rows.mappings().all()
    chunks  = [r["chunk_text"] for r in results]
    sources = [{"doc_title": r["doc_title"], "section_title": r.get("section_title", "")} for r in results]
    cache_tier = "free"
    for r in results:
        if r["access_tier"] == "pro":
            cache_tier = "pro"
            break
    return chunks, sources, cache_tier


async def _save_to_db(db, q_hash, question, subject_id, language,
                      answer_text, key_points, sources, is_doc_grounded,
                      question_embedding, cache_tier, example=None, quick_tip=None):
    import json as _json
    cache_id = str(uuid.uuid4())
    vec_str = vec_to_pg_str(question_embedding) if question_embedding else None
    try:
        await db.execute(sql_text("""
            INSERT INTO text_answer_cache
                (id, question_hash, question_text, subject_id, language,
                 answer_text, key_points, sources, is_doc_grounded, question_embedding, access_tier)
            VALUES
                (:id, :hash, :question, :subject_id, :language,
                 :answer, cast(:key_points as jsonb), cast(:sources as jsonb),
                 :grounded, cast(:vec as vector), :access_tier)
            ON CONFLICT (question_hash, subject_id) DO NOTHING
        """), {
            "id": cache_id, "hash": q_hash, "question": question,
            "subject_id": subject_id, "language": language,
            "answer": answer_text,
            "key_points": _json.dumps(key_points),
            "sources": _json.dumps(sources),
            "grounded": is_doc_grounded,
            "vec": vec_str, "access_tier": cache_tier,
        })
        await db.commit()
    except IntegrityError:
        await db.rollback()
    except Exception as e:
        await db.rollback()
        print(f"[TextAnswer] DB save error (non-fatal): {e}")
    return cache_id


@router.post("/ai-text-answer")
async def ai_text_answer(request: Request, body: dict, db: AsyncSession = Depends(get_db)):
    question     = (body.get("question") or "").strip()
    subject_id   = (body.get("subjectId") or "").strip()
    language     = body.get("language", "en")
    user_tier    = (body.get("userTier") or "pro").strip().lower()

    if user_tier not in ("free", "pro"):
        user_tier = "pro"

    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    if not subject_id:
        return JSONResponse({"error": "subjectId is required"}, status_code=400)

    q_hash = hash_question(question)
    print(f"[TextAnswer] '{question[:60]}' subj={subject_id} hash={q_hash[:8]}")

    # ── L1: Redis exact match ─────────────────────────────────────────────────
    cached = await get_text_from_cache(q_hash, subject_id)
    if cached:
        if cached.get("access_tier", "free") == "pro" and user_tier == "free":
            return {"blocked": True, "reason": "subscription_required",
                    "message": "This topic is part of the Pro course. Upgrade your plan to access full AI-generated presentations, audio explanations, and more."}
        print(f"[L1] HIT")
        embedding = cached.get("_embedding")
        slide_preview, suggestions = await enrich_response(embedding, subject_id, db, q_hash)
        return {**{k: v for k, v in cached.items() if k != "_embedding"},
                "cached": True, "cache_layer": "L1_redis",
                "slide_preview": slide_preview, "suggestions": suggestions}

    # ── Embed (needed for both L2 check and RAG) ──────────────────────────────
    question_embedding = await embed_async(question)
    vec_str = vec_to_pg_str(question_embedding)

    # ── L2: Postgres exact match ──────────────────────────────────────────────
    pg_rows = await db.execute(sql_text("""
        SELECT id::text AS cache_id, answer_text, key_points, sources,
               is_doc_grounded, access_tier
        FROM text_answer_cache
        WHERE question_hash = :hash AND subject_id = :subject_id LIMIT 1
    """), {"hash": q_hash, "subject_id": subject_id})
    pg_row = pg_rows.mappings().first()
    if pg_row:
        if pg_row["access_tier"] == "pro" and user_tier == "free":
            return {"blocked": True, "reason": "subscription_required",
                    "message": "This topic is part of the Pro course. Upgrade your plan to access full AI-generated presentations, audio explanations, and more."}
        print(f"[L2] HIT")
        result = {"cached": True, "cache_layer": "L2_postgres",
                  "cache_id": pg_row["cache_id"], "answer": pg_row["answer_text"],
                  "key_points": pg_row["key_points"] or [], "sources": pg_row["sources"] or [],
                  "is_doc_grounded": pg_row["is_doc_grounded"], "access_tier": pg_row["access_tier"]}
        await set_text_to_cache(q_hash, subject_id, {**result, "_embedding": question_embedding})
        slide_preview, suggestions = await enrich_response(question_embedding, subject_id, db, q_hash)
        return {**result, "slide_preview": slide_preview, "suggestions": suggestions}

    # ── Concurrency lock: prevent stampede for the same question ──────────────
    lock_acquired = await acquire_text_lock(q_hash, subject_id, ttl=90)
    if not lock_acquired:
        print(f"[Lock] Waiting for other worker…")
        waited = await wait_for_text_cache(q_hash, subject_id, max_wait=85)
        if waited:
            if waited.get("access_tier") == "pro" and user_tier == "free":
                return {"blocked": True, "reason": "subscription_required",
                        "message": "This topic is part of the Pro course. Upgrade your plan to access full AI-generated presentations, audio explanations, and more."}
            slide_preview, suggestions = await enrich_response(question_embedding, subject_id, db, q_hash)
            return {**{k: v for k, v in waited.items() if k != "_embedding"},
                    "cached": True, "cache_layer": "L_lock_wait",
                    "slide_preview": slide_preview, "suggestions": suggestions}

    # ── L4: RAG — fetch relevant document chunks ──────────────────────────────
    try:
        chunks, sources, cache_tier = await _fetch_doc_context(db, subject_id, vec_str, user_tier)
    except Exception as e:
        await release_text_lock(q_hash, subject_id)
        return JSONResponse({"error": "RAG lookup failed", "detail": str(e)}, status_code=500)

    if not chunks:
        await release_text_lock(q_hash, subject_id)
        if user_tier == "free":
            pro_check = await db.execute(sql_text("""
                SELECT COUNT(*) FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.subject_id = :subject_id
                  AND dc.chunk_embedding IS NOT NULL
                  AND dc.access_tier = 'pro'
                  AND 1 - (dc.chunk_embedding <=> CAST(:vec AS vector)) > 0.50
            """), {"vec": vec_str, "subject_id": subject_id})
            pro_cnt = pro_check.scalar() or 0
            if pro_cnt > 0:
                return {"blocked": True, "reason": "subscription_required",
                        "message": "This topic is part of the Pro course. Upgrade your plan to access full AI-generated presentations, audio explanations, and more."}

        print(f"[L4] No docs found")
        return {"no_content": True, "message": "No relevant course material found for this question."}

    context = "\n\n".join(chunks)
    print(f"[L4] {len(chunks)} chunks found")

    # ── L5: LLM generation ────────────────────────────────────────────────────
    try:
        llm_result = await generate_text_answer(question, context)
    except RuntimeError as e:
        await release_text_lock(q_hash, subject_id)
        print(f"[L5] LLM failed: {e}")
        return JSONResponse({"error": "LLM unavailable", "detail": str(e)}, status_code=503)

    answer_text = llm_result.get("answer", "")
    key_points  = llm_result.get("key_points", [])
    example     = llm_result.get("example")      # optional
    quick_tip   = llm_result.get("quick_tip")    # optional

    cache_id = await _save_to_db(
        db, q_hash, question, subject_id, language,
        answer_text, key_points, sources, True, question_embedding, cache_tier,
        example=example, quick_tip=quick_tip
    )
    await release_text_lock(q_hash, subject_id)

    result = {
        "cached": False, "cache_layer": "GENERATED", "cache_id": cache_id,
        "answer": answer_text,
        "key_points": key_points,
        "sources": sources,
        "is_doc_grounded": True,
        "access_tier": cache_tier,
    }
    if example:
        result["example"] = example
    if quick_tip:
        result["quick_tip"] = quick_tip

    await set_text_to_cache(q_hash, subject_id, {**result, "_embedding": question_embedding})

    slide_preview, suggestions = await enrich_response(question_embedding, subject_id, db, q_hash)
    print(f"[L5] GENERATED cache_id={cache_id[:8]}")
    return {**result, "slide_preview": slide_preview, "suggestions": suggestions}
