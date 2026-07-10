import os, uuid, base64, asyncio
from sqlalchemy import text, update, select
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv

load_dotenv()

from db.models import get_db, TeachingCache, AsyncSessionLocal
from core.cache import hash_question, get_from_cache, set_to_cache, increment_usage, \
                       acquire_lock, release_lock, wait_for_cache
from core.subject_gate import gate_subject, detect_topic
from core.slide_generator import generate_slides
from core.image_generator import generate_all_images
from core.tts_client import synthesize
from core.b2_client import upload_to_b2
from core.embeddings import embed_async, vec_to_pg_str
from core.semantic_check import llm_same_topic
from core.local_storage import ensure_base_dirs, read_slide_cache, write_slide_cache, write_audio
from core.llm_judge import judge_and_pick
from routers.documents import router as documents_router
from routers.pregen import router as pregen_router
from routers.questions import router as questions_router


# ─────────────────────────────────────────────────────────
# Audio helper (used by /ai-teaching-assistant)
# ─────────────────────────────────────────────────────────
async def generate_all_audios(
    slides: list, cache_id: str, language: str, subject_id: str = ""
) -> list[dict]:
    """
    Generate audio for all slides in parallel.
    Dual-writes: local /sdb-disk (primary) + B2 (cloud backup).
    """
    import io, wave

    async def _gen_and_up(idx: int, narration: str):
        if not narration:
            return None
        try:
            print(f"[Audio] Slide {idx} → generating audio")
            audio_b64 = await synthesize(narration, language)
            wav_data  = base64.b64decode(audio_b64)

            # Duration
            try:
                with wave.open(io.BytesIO(wav_data), "rb") as wf:
                    duration = wf.getnframes() / float(wf.getframerate() or 1)
            except Exception as e:
                print(f"[Audio] Warning: WAV parse failed: {e}")
                duration = 10.0

            # ── Dual write ────────────────────────────────────────────────────
            # 1. Local disk — primary fast storage
            if subject_id:
                try:
                    local_path = await write_audio(subject_id, cache_id, language, idx, wav_data)
                    print(f"[Audio] Slide {idx} → local disk: {local_path}")
                except Exception as e:
                    print(f"[Audio] Slide {idx} → local disk failed (non-fatal): {e}")

            # 2. B2 — cloud backup / CDN
            b2_path = f"ai-teaching/{cache_id}/audio_{idx}.wav"
            url     = await upload_to_b2(wav_data, b2_path, "audio/wav")
            print(f"[Audio] Slide {idx} → B2: {url}")

            return {"slideIndex": idx, "audioUrl": url, "duration": round(duration, 2)}
        except Exception as e:
            print(f"[Audio] Slide {idx} failed: {e}")
            return None

    tasks   = [_gen_and_up(i, s.get("narration", "")) for i, s in enumerate(slides)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


# ─────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────
app = FastAPI(title="AI Teaching Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(pregen_router)
app.include_router(questions_router)

@app.on_event("startup")
async def startup_event():
    ensure_base_dirs()


# ─────────────────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    return FileResponse("/app/test_frontend.html", media_type="text/html")

@app.get("/pregen-dashboard", include_in_schema=False)
async def serve_pregen_dashboard():
    """Pre-Generation monitoring & control dashboard."""
    return FileResponse("/app/pregen_dashboard.html", media_type="text/html")

@app.get("/health")
async def health():
    _is_cpu = os.getenv("IS_CPU_SERVER", "false").lower() == "true"
    return {
        "status": "ok",
        "version": "2.0.0",
        "role": "cpu_student_server" if _is_cpu else "gpu_pregen_factory",
    }


# ─────────────────────────────────────────────────────────
# GET /search-questions  — two-source semantic search
# ─────────────────────────────────────────────────────────
@app.get("/search-questions")
async def search_questions(
    q: str = "",
    subject_id: str = "",
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
):
    """
    Two-source semantic search:
      Source 1: teaching_qa_cache   (past answered Q&As)
      Source 2: document_chunks     (uploaded lecture documents)
    subject_id is REQUIRED — prevents cross-subject results.
    """
    if not subject_id:
        return JSONResponse(
            status_code=400,
            content={"error": "subject_id is required", "hint": "Pass ?subject_id=physics-101"}
        )
    if not q or len(q.strip()) < 2:
        return {"results": [], "document_chunks": []}

    try:
        query_vec = await embed_async(q.strip())
        vec_str   = vec_to_pg_str(query_vec)

        # Source 1: Q&A cache
        cache_sql = text("""
            SELECT question_text, subject_id, usage_count,
                   1 - (question_embedding <=> CAST(:vec AS vector)) AS similarity
            FROM teaching_qa_cache
            WHERE question_embedding IS NOT NULL
              AND subject_id = :subj
              AND 1 - (question_embedding <=> CAST(:vec AS vector)) > 0.45
            ORDER BY similarity DESC LIMIT :lim
        """)
        cache_rows = (await db.execute(
            cache_sql, {"vec": vec_str, "subj": subject_id, "lim": limit}
        )).fetchall()

        # Source 2: Document chunks
        chunk_sql = text("""
            SELECT dc.chunk_text, dc.section_title, dc.document_id,
                   d.title AS document_title,
                   1 - (dc.chunk_embedding <=> CAST(:vec AS vector)) AS similarity
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.chunk_embedding IS NOT NULL
              AND dc.subject_id = :subj
              AND d.status = 'ready'
              AND 1 - (dc.chunk_embedding <=> CAST(:vec AS vector)) > 0.45
            ORDER BY similarity DESC LIMIT :lim
        """)
        chunk_rows = (await db.execute(
            chunk_sql, {"vec": vec_str, "subj": subject_id, "lim": limit}
        )).fetchall()

        return {
            "query": q,
            "subject_id": subject_id,
            "qa_cache_results": [
                {
                    "source": "qa_cache",
                    "question": r.question_text,
                    "subject_id": r.subject_id,
                    "usage_count": r.usage_count,
                    "similarity": round(float(r.similarity), 3),
                }
                for r in cache_rows
            ],
            "document_chunk_results": [
                {
                    "source": "document_chunk",
                    "chunk_text": r.chunk_text[:300] + "..." if len(r.chunk_text) > 300 else r.chunk_text,
                    "section_title": r.section_title,
                    "document_id": str(r.document_id),
                    "document_title": r.document_title,
                    "similarity": round(float(r.similarity), 3),
                }
                for r in chunk_rows
            ],
        }
    except Exception as e:
        print(f"[Search] Error: {e}")
        return {"results": [], "document_chunks": [], "error": str(e)}


# ─────────────────────────────────────────────────────────
# GET /subjects  — list all subjects that have content
# ─────────────────────────────────────────────────────────
@app.get("/subjects")
async def list_subjects(db: AsyncSession = Depends(get_db)):
    """List all unique subject_ids from both Q&A cache and documents."""
    sql = text("""
        SELECT subject_id, 'qa_cache' AS source, COUNT(*) AS count
        FROM teaching_qa_cache
        WHERE subject_id IS NOT NULL GROUP BY subject_id
        UNION
        SELECT subject_id, 'documents' AS source, COUNT(*) AS count
        FROM documents
        GROUP BY subject_id
        ORDER BY subject_id, source
    """)
    rows = (await db.execute(sql)).fetchall()

    subjects: dict[str, dict] = {}
    for r in rows:
        sid = r.subject_id
        if sid not in subjects:
            subjects[sid] = {"subject_id": sid, "name": sid[:20], "qa_count": 0, "document_count": 0}
        if r.source == "qa_cache":
            subjects[sid]["qa_count"] = r.count
        else:
            subjects[sid]["document_count"] = r.count

    # Also include subjects that were explicitly created but have no content yet
    named = (await db.execute(text("SELECT subject_id, name FROM subjects ORDER BY created_at DESC"))).fetchall()
    for r in named:
        if r.subject_id not in subjects:
            subjects[r.subject_id] = {"subject_id": r.subject_id, "name": r.name, "qa_count": 0, "document_count": 0}
        else:
            subjects[r.subject_id]["name"] = r.name  # enrich existing entry with name

    return {"total": len(subjects), "subjects": list(subjects.values())}


# POST /subjects — create and persist a new named subject
@app.post("/subjects")
async def create_subject(body: dict, db: AsyncSession = Depends(get_db)):
    """Create a new subject — deduplicates by name (case-insensitive).
    If a subject with the same name already exists, returns it instead of creating a duplicate.
    """
    import uuid as _uuid
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    # ── Look up by name first (case-insensitive) ─────────────────────────────
    existing = (await db.execute(
        text("SELECT subject_id, name, description FROM subjects WHERE LOWER(name) = LOWER(:n)"),
        {"n": name},
    )).fetchone()

    if existing:
        return {
            "subject_id":    existing.subject_id,
            "name":          existing.name,
            "description":   existing.description or "",
            "already_existed": True,
        }

    # ── Not found — create new ───────────────────────────────────────────────
    subject_id = str(_uuid.uuid4())
    slug = name.lower().replace(" ", "-")
    await db.execute(
        text("""
            INSERT INTO subjects (subject_id, name, slug, description)
            VALUES (:sid, :name, :slug, :desc)
            ON CONFLICT (subject_id) DO NOTHING
        """),
        {"sid": subject_id, "name": name, "slug": slug, "desc": body.get("description", "")},
    )
    await db.commit()
    return {
        "subject_id":    subject_id,
        "name":          name,
        "slug":          slug,
        "description":   body.get("description", ""),
        "already_existed": False,
    }


# ─────────────────────────────────────────────────────────
# POST /sarvam-tts
# ─────────────────────────────────────────────────────────
@app.post("/sarvam-tts")
async def sarvam_tts(body: dict):
    """Convert narration text to speech using Sarvam AI TTS."""
    text_input    = body.get("text", "")
    language_code = body.get("languageCode", "hi-IN")
    gender        = body.get("gender", "male")
    if not text_input:
        return JSONResponse(status_code=400, content={"error": "text is required"})

    audio_b64 = await synthesize(text_input, language_code, gender)
    return {
        "audioContent": audio_b64,
        "languageCode": language_code,
        "voice": "bulbul:v2",
        "format": "wav",
    }


# ─────────────────────────────────────────────────────────
# POST /ai-generate-image
# ─────────────────────────────────────────────────────────
@app.post("/ai-generate-image")
async def ai_generate_image(body: dict):
    """Generate a single educational infographic image."""
    from core.image_generator import generate_one_image
    cache_id    = body.get("cacheId", str(uuid.uuid4()))
    slide_index = body.get("slideIndex", 0)
    slide       = {"title": body.get("prompt", ""), "infographic": body.get("prompt", "")}
    url = await generate_one_image(slide, cache_id, slide_index)
    return {"imageUrl": url}


# ─────────────────────────────────────────────────────────
# POST /save-presentation-audio
# ─────────────────────────────────────────────────────────
@app.post("/save-presentation-audio")
async def save_presentation_audio(body: dict, db: AsyncSession = Depends(get_db)):
    """Upload WAV audio chunks to B2 and update cache row."""
    cache_id = body.get("cache_id", "")
    language = body.get("language", "hi-IN")
    slides   = body.get("slides", [])

    audio_urls = []
    total_dur  = 0.0

    for slide in slides:
        idx    = slide.get("slideIndex", 0)
        chunks = slide.get("base64Chunks", [])
        if not chunks:
            continue

        wav_data = b"".join(base64.b64decode(c) for c in chunks)
        try:
            import io, wave
            with wave.open(io.BytesIO(wav_data), "rb") as wf:
                duration = wf.getnframes() / float(wf.getframerate() or 1)
        except Exception as e:
            print(f"[Audio Save] WAV parse failed: {e}")
            duration = 10.0

        path = f"ai-presentations/{cache_id}/{language}/slide_{idx}.wav"
        url  = await upload_to_b2(wav_data, path, "audio/wav")
        audio_urls.append({"slideIndex": idx, "audioUrl": url, "duration": round(duration, 2)})
        total_dur += duration

    try:
        await db.execute(
            update(TeachingCache)
            .where(TeachingCache.id == cache_id)
            .values(
                slide_audio_urls={"language": language, "urls": audio_urls},
                total_duration_seconds=round(total_dur, 2),
            )
        )
        await db.commit()
    except Exception as e:
        print(f"[Audio Save] DB update failed: {e}")

    return {"success": True, "audioUrls": audio_urls, "totalDuration": round(total_dur, 2)}


# ─────────────────────────────────────────────────────────
# POST /ai-teaching-assistant  (MAIN ENDPOINT)
# ─────────────────────────────────────────────────────────
@app.post("/ai-teaching-assistant")
async def teaching_assistant(body: dict, db: AsyncSession = Depends(get_db)):
    mode         = body.get("mode", "full")
    question     = body.get("question", "")
    subject_name = body.get("subjectName", "")
    subject_id   = body.get("subjectId", "")
    language     = body.get("language", "hi-IN")

    # ── MODE: detect_topic ───────────────────────────────
    if mode == "detect_topic":
        return await detect_topic(question, subject_name)

    # ── MODE: doubt ──────────────────────────────────────
    if mode == "doubt":
        question_text = body.get("questionText", question)
        correct       = body.get("correctAnswer", "")
        student_ans   = body.get("studentAnswer", "")
        doubt_q       = f"Why is '{correct}' correct and '{student_ans}' wrong for: {question_text}"
        slides_data   = await generate_slides(doubt_q, subject_name)
        slides        = slides_data.get("presentation_slides", [])[:1]
        cache_id      = str(uuid.uuid4())
        images        = await generate_all_images(slides, cache_id)
        for i, s in enumerate(slides):
            s["infographicUrl"] = images[i] if i < len(images) else ""
        return {"presentationSlides": slides, "isDoubtExplanation": True, "cache_id": cache_id}

    # ── MODE: full ───────────────────────────────────────
    if not question:
        return {"error": "question is required"}
    if not subject_id:
        return {"error": "subjectId is required", "hint": "Pass subjectId in the request body"}

    q_hash = hash_question(question)

    # [L1] Redis exact hash (~0.5ms)
    cached = await get_from_cache(q_hash, subject_id)
    if cached:
        await increment_usage(q_hash, subject_id)
        return {**cached, "cached": True, "cache_layer": "L1_redis"}

    # [L2] Local disk JSON (~1ms, avoids DB round-trip)
    local_cached = await read_slide_cache(subject_id, q_hash)
    if local_cached:
        print(f"[L2] Local cache HIT for '{question[:60]}'")
        await set_to_cache(q_hash, subject_id, local_cached)
        return {**local_cached, "cached": True, "cache_layer": "L2_local_disk"}

    # [L3] Postgres exact hash (~3ms)
    print(f"[L3] hash={q_hash[:8]}… subject={subject_id[:8]}… question='{question[:50]}'")
    row = (await db.execute(
        select(TeachingCache)
        .where(TeachingCache.question_hash == q_hash)
        .where(TeachingCache.subject_id == subject_id)
        .order_by(TeachingCache.usage_count.desc())
        .limit(1)
    )).scalar_one_or_none()

    if row and row.presentation_slides:
        print(f"[L3] ✓ HIT — returning pre-generated slides")
        data = {
            "cached": True, "cache_layer": "L3_postgres",
            "cache_id": str(row.id),
            "presentationSlides": row.presentation_slides,
            "latexFormulas": row.latex_formulas,
            "slideAudioUrls": row.slide_audio_urls.get("urls", []) if row.slide_audio_urls else [],
            "totalDurationSeconds": row.total_duration_seconds,
            "manimVideoUrls": row.manim_video_urls or {},
            "imageUrls": row.image_urls or {},
        }
        await set_to_cache(q_hash, subject_id, data)
        await write_slide_cache(subject_id, q_hash, data)
        return data
    print(f"[L3] ✗ MISS")

    # [L4] Semantic search + LLM-as-Judge (single LLM call for up to 5 candidates)
    print(f"[L4] Starting semantic search for '{question[:50]}'")
    try:
        query_vec = await embed_async(question)
        print(f"[L4] ✓ Embedding generated ({len(query_vec)} dims)")
        vec_str   = vec_to_pg_str(query_vec)

        sem_sql = text("""
            SELECT id, question_text, presentation_slides, latex_formulas,
                   slide_audio_urls, total_duration_seconds,
                   manim_video_urls, image_urls,
                   1 - (question_embedding <=> CAST(:vec AS vector)) AS sim_score
            FROM teaching_qa_cache
            WHERE question_embedding IS NOT NULL
              AND subject_id = :subj
              AND presentation_slides IS NOT NULL
              AND 1 - (question_embedding <=> CAST(:vec AS vector)) > 0.60
            ORDER BY sim_score DESC LIMIT 5
        """)
        rows = (await db.execute(sem_sql, {"vec": vec_str, "subj": subject_id})).fetchall()
        print(f"[L4] Semantic candidates found: {len(rows)}")
        for r in rows:
            print(f"[L4]   score={float(r.sim_score):.3f} | '{r.question_text[:60]}'")

        if rows:
            judge_candidates = [
                {
                    "question": r.question_text,
                    "score": float(r.sim_score),
                    "answer_data": {
                        "cached": True, "cache_layer": "L4_llm_judge",
                        "cache_id": str(r.id),
                        "presentationSlides": r.presentation_slides,
                        "latexFormulas": r.latex_formulas,
                        "slideAudioUrls": r.slide_audio_urls.get("urls", []) if r.slide_audio_urls else [],
                        "totalDurationSeconds": r.total_duration_seconds,
                        "manimVideoUrls": (r.manim_video_urls or {}) if hasattr(r, "manim_video_urls") else {},
                        "imageUrls": (r.image_urls or {}) if hasattr(r, "image_urls") else {},
                    }
                }
                for r in rows
            ]
            winner = await judge_and_pick(question, judge_candidates, subject_id)
            if winner:
                print(f"[L4] ✓ LLM Judge selected a cache hit")
                data = winner["answer_data"]
                await set_to_cache(q_hash, subject_id, data)
                await write_slide_cache(subject_id, q_hash, data)
                return data
            print(f"[L4] LLM Judge said NEW — no close match")
        else:
            print(f"[L4] ✗ No semantic candidates above 0.60 threshold")

    except Exception as e:
        print(f"[L4 LLM Judge] failed (non-fatal): {e}")

    # ── Subject gating (only when subject_name given) ────
    if subject_name:
        gate = await gate_subject(question, subject_name)
        if not gate.get("allowed", True):
            return {
                "blocked": True,
                "reason": "off_topic",
                "currentSubject": subject_name,
                "detectedSubject": gate.get("detected_subject", ""),
                "message": (
                    f"This question is about {gate.get('detected_subject','')}. "
                    f"Please ask a {subject_name} question."
                ),
            }

    # ── GENERATE (all cache layers exhausted) ──────────
    # [L5] Document RAG — find top 3 chunks from subject's docs
    # STRICT: if no document context found, do NOT generate via general LLM
    rag_context     = ""
    is_doc_grounded = False

    _NO_CONTENT_RESPONSE = {
        "no_content":  True,
        "blocked":     True,
        "reason":      "no_document_context",
        "message": (
            f"No study material found for this question"
            f"{' in ' + subject_name if subject_name else ''}. "
            "Please ask your teacher to upload relevant content, "
            "or try searching for this topic."
        ),
        "suggestion": "search_topic",
    }

    try:
        if not query_vec:  # may already be computed from L4
            query_vec = await embed_async(question)
            vec_str   = vec_to_pg_str(query_vec)

        rag_sql = text("""
            SELECT dc.chunk_text, dc.section_title, d.title AS doc_title,
                   1 - (dc.chunk_embedding <=> CAST(:vec AS vector)) AS sim
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.chunk_embedding IS NOT NULL
              AND dc.subject_id = :subj
              AND d.status = 'ready'
              AND 1 - (dc.chunk_embedding <=> CAST(:vec AS vector)) > 0.50
            ORDER BY sim DESC LIMIT 3
        """)
        rag_rows = (await db.execute(rag_sql, {"vec": vec_str, "subj": subject_id})).fetchall()

        if not rag_rows:
            # No document content — block generation, tell frontend to search
            print(f"[L5 RAG] 0 chunks found for '{question[:60]}' — blocking generation")
            return _NO_CONTENT_RESPONSE

        # Document context found — build grounded prompt
        rag_context = "\n\n".join(
            f"[{r.doc_title} / {r.section_title or 'Section'}]\n{r.chunk_text}"
            for r in rag_rows
        )
        is_doc_grounded = True
        print(f"[L5 RAG] {len(rag_rows)} chunks found → grounded generation for '{question[:60]}'")

    except Exception as e:
        # RAG lookup failed — still block, don't hallucinate
        print(f"[L5 RAG] lookup failed: {e} — blocking generation (no fallback)")
        return _NO_CONTENT_RESPONSE

    slides_data = await generate_slides(question, subject_name, context=rag_context)

    slides      = slides_data.get("presentation_slides", [])
    cache_id    = str(uuid.uuid4())

    images_task = generate_all_images(slides, cache_id, subject_id)
    audios_task = generate_all_audios(slides, cache_id, language, subject_id)
    images, audios = await asyncio.gather(images_task, audios_task)

    for i, s in enumerate(slides):
        s["infographicUrl"] = images[i] if i < len(images) else ""
        audio_info = next((a for a in audios if a and a.get("slideIndex") == i), None)
        if audio_info:
            s["audioUrl"] = audio_info.get("audioUrl")
            s["duration"] = audio_info.get("duration")

    total_duration = sum(a.get("duration", 0) for a in audios if a)

    # Save to Postgres — set is_doc_grounded if generated from document RAG
    # Tag as realtime so GPU enrichment queue can pick this up for Wan2GP/VoxCPM upgrade
    new_row = TeachingCache(
        id=uuid.UUID(cache_id),
        question_hash=q_hash,
        question_text=question,
        subject_id=subject_id or None,
        language=language,
        variation_number=1,
        presentation_slides=slides,
        latex_formulas=slides_data.get("latex_formulas", []),
        slide_audio_urls={"language": language, "urls": audios},
        total_duration_seconds=total_duration,
        is_doc_grounded=is_doc_grounded,
        pregen_status="done",
        realtime_generated=True,   # CPU real-time path
        realtime_tier=2,           # tier 2 = OpenRouter + Sarvam (CPU cloud APIs)
    )
    db.add(new_row)
    await db.commit()

    result_data = {
        "cached": False,
        "cache_layer": "GENERATED",
        "cache_id": cache_id,
        "isDocGrounded": is_doc_grounded,
        "presentationSlides": slides,
        "latexFormulas": slides_data.get("latex_formulas", []),
        "keyPoints": slides_data.get("key_points", []),
        "followUpQuestions": slides_data.get("follow_up_questions", []),
        "slideAudioUrls": audios,
        "totalDurationSeconds": total_duration,
        "manimVideoUrls": {},   # empty for real-time; pre-gen fills this via pregen pipeline
        "imageUrls": {},
    }

    # Warm L1 + L2 caches
    await set_to_cache(q_hash, subject_id, result_data)
    await write_slide_cache(subject_id, q_hash, result_data)

    # Store embedding (background, non-blocking)
    async def _store_embedding():
        try:
            vec     = await embed_async(question)
            v_str   = vec_to_pg_str(vec)
            async with AsyncSessionLocal() as bg_db:
                await bg_db.execute(
                    text("""
                        UPDATE teaching_qa_cache
                        SET question_embedding = CAST(:vec AS vector)
                        WHERE id = CAST(:id AS uuid)
                    """),
                    {"vec": v_str, "id": cache_id},
                )
                await bg_db.commit()
            print(f"[Embeddings] ✓ Stored for cache_id={cache_id}")
        except Exception as e:
            print(f"[Embeddings] Failed: {e}")

    asyncio.create_task(_store_embedding())
    return result_data
