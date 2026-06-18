import os, uuid, base64, asyncio
from sqlalchemy import text
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
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

# ─────────────────────────────────────────────────────────
# Global concurrency limiter
# Max 16 AI generations at a time — others WAIT IN QUEUE
# (no errors, no timeouts — just a longer wait)
# ─────────────────────────────────────────────────────────
GENERATION_SEMAPHORE = asyncio.Semaphore(16)

async def generate_all_audios(slides: list, cache_id: str, language: str) -> list[dict]:
    """Generate audio for all slides in parallel and upload to B2."""
    async def _gen_and_up(idx: int, narration: str):
        if not narration:
            return None
        try:
            print(f"[Audio] Slide {idx} → generating audio")
            audio_b64 = await synthesize(narration, language)
            wav_data = base64.b64decode(audio_b64)
            path = f"ai-presentations/{cache_id}/{language}/slide_{idx}.wav"
            url = await upload_to_b2(wav_data, path, "audio/wav")
            print(f"[Audio] Slide {idx} → uploaded to {url}")
            
            # Estimate duration robustly using built-in wave module
            try:
                import io, wave
                with wave.open(io.BytesIO(wav_data), 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate) if rate else 0.0
            except Exception as e:
                print(f"[Audio] Warning: Failed to parse WAV duration: {e}")
                duration = 10.0
                
            return {"slideIndex": idx, "audioUrl": url, "duration": round(duration, 2)}
        except Exception as e:
            print(f"[Audio] Slide {idx} failed: {e}")
            return None

    tasks = [_gen_and_up(i, s.get("narration", "")) for i, s in enumerate(slides)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]

app = FastAPI(title="AI Teaching Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────
from fastapi.responses import FileResponse

@app.get("/")
async def serve_frontend():
    return FileResponse("/app/test_frontend.html", media_type="text/html")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}



# ─────────────────────────────────────────────────────────
# GET /search-questions  (autocomplete + semantic search)
# ─────────────────────────────────────────────────────────
@app.get("/search-questions")
async def search_questions(
    q: str = "",
    subject_id: str = "",
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Semantic autocomplete: returns top matching past questions.
    Uses pgvector cosine similarity on all-MiniLM-L6-v2 embeddings.
    Call this on every keystroke (debounce 300ms on frontend).
    """
    if not q or len(q.strip()) < 2:
        return {"results": []}

    try:
        query_vec = await embed_async(q.strip())
        vec_str   = vec_to_pg_str(query_vec)

        sql = text("""
            SELECT
                question_text,
                subject_id,
                usage_count,
                1 - (question_embedding <=> CAST(:vec AS vector)) AS similarity
            FROM teaching_qa_cache
            WHERE
                question_embedding IS NOT NULL
                AND (:subj = '' OR subject_id = :subj)
                AND 1 - (question_embedding <=> CAST(:vec AS vector)) > 0.45
            ORDER BY similarity DESC
            LIMIT :lim
        """)

        rows = (await db.execute(sql, {"vec": vec_str, "subj": subject_id, "lim": limit})).fetchall()

        return {
            "results": [
                {
                    "question":    r.question_text,
                    "subject_id":  r.subject_id,
                    "usage_count": r.usage_count,
                    "similarity":  round(float(r.similarity), 3)
                }
                for r in rows
            ]
        }
    except Exception as e:
        print(f"[Search] Error: {e}")
        return {"results": []}


# ─────────────────────────────────────────────────────────
# POST /sarvam-tts
# ─────────────────────────────────────────────────────────
@app.post("/sarvam-tts")
async def sarvam_tts(body: dict):
    """Convert narration text to speech using Sarvam AI TTS."""
    text          = body.get("text", "")
    language_code = body.get("languageCode", "hi-IN")
    gender        = body.get("gender", "male")

    if not text:
        return {"error": "text is required"}, 400

    audio_b64 = await synthesize(text, language_code, gender)
    return {
        "audioContent": audio_b64,
        "languageCode": language_code,
        "voice": "bulbul:v2",
        "format": "wav"
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

    audio_urls  = []
    total_dur   = 0.0

    for slide in slides:
        idx    = slide.get("slideIndex", 0)
        chunks = slide.get("base64Chunks", [])
        if not chunks:
            continue

        # Decode and concatenate WAV chunks
        wav_data = b"".join(base64.b64decode(c) for c in chunks)

        # Estimate duration robustly using built-in wave module
        try:
            import io, wave
            with wave.open(io.BytesIO(wav_data), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate) if rate else 0.0
        except Exception as e:
            print(f"[Audio Save] Warning: Failed to parse WAV duration: {e}")
            duration = 10.0

        path = f"ai-presentations/{cache_id}/{language}/slide_{idx}.wav"
        url  = await upload_to_b2(wav_data, path, "audio/wav")

        audio_urls.append({"slideIndex": idx, "audioUrl": url, "duration": round(duration, 2)})
        total_dur += duration

    # Update DB
    try:
        await db.execute(
            update(TeachingCache)
            .where(TeachingCache.id == cache_id)
            .values(
                slide_audio_urls={"language": language, "urls": audio_urls},
                total_duration_seconds=round(total_dur, 2)
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

    # ── MODE: detect_topic (fast, no slides) ─────────────
    if mode == "detect_topic":
        return await detect_topic(question, subject_name)

    # ── MODE: doubt (1 slide explanation) ────────────────
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

    # ── MODE: full ────────────────────────────────────────
    if not question:
        return {"error": "question is required"}

    q_hash = hash_question(question)

    # Step 1: Redis L1 cache
    cached = await get_from_cache(q_hash, subject_id)
    if cached:
        await increment_usage(q_hash, subject_id)
        return {**cached, "cached": True}

    # Step 2A: Postgres exact hash cache
    result = await db.execute(
        select(TeachingCache)
        .where(TeachingCache.question_hash == q_hash)
        .order_by(TeachingCache.usage_count.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row and row.presentation_slides:
        data = {
            "cached": True,
            "cache_id": str(row.id),
            "presentationSlides": row.presentation_slides,
            "latexFormulas": row.latex_formulas,
            "slideAudioUrls": row.slide_audio_urls.get("urls", []) if row.slide_audio_urls else [],
            "totalDurationSeconds": row.total_duration_seconds,
        }
        await set_to_cache(q_hash, subject_id, data)
        return data

    # Step 2B: Semantic + LLM equivalence cache
    # Three zones based on cosine similarity:
    #   > 0.97  → near-identical phrasing, trust vector alone (no LLM needed)
    #   0.70–0.97 → gray zone: ask LLM "same topic?"
    #   < 0.70  → too different, skip (will generate fresh)
    GRAY_LOW  = 0.70
    EXACT_HIT = 0.97
    try:
        query_vec = await embed_async(question)
        vec_str   = vec_to_pg_str(query_vec)

        sem_sql = text("""
            SELECT *,
                   1 - (question_embedding <=> CAST(:vec AS vector)) AS sim_score
            FROM teaching_qa_cache
            WHERE
                question_embedding IS NOT NULL
                AND (:subj = '' OR subject_id = :subj)
                AND 1 - (question_embedding <=> CAST(:vec AS vector)) > :low
            ORDER BY sim_score DESC
            LIMIT 5
        """)

        candidates = (await db.execute(
            sem_sql, {"vec": vec_str, "subj": subject_id or "", "low": GRAY_LOW}
        )).fetchall()

        for candidate in candidates:
            if not candidate.presentation_slides:
                continue

            score = float(candidate.sim_score)

            if score >= EXACT_HIT:
                # Near-identical text — trust vector, no LLM needed
                print(f"[Semantic Cache] EXACT HIT ({score:.3f}) — '{question}' = '{candidate.question_text}'")
                is_same = True
            else:
                # Gray zone — ask LLM to verify (costs ~$0.000001)
                print(f"[Semantic Cache] Gray zone ({score:.3f}) — asking LLM: '{question}' vs '{candidate.question_text}'")
                is_same = await llm_same_topic(question, candidate.question_text)

            if is_same:
                data = {
                    "cached": True,
                    "cache_id": str(candidate.id),
                    "presentationSlides": candidate.presentation_slides,
                    "latexFormulas": candidate.latex_formulas,
                    "slideAudioUrls": candidate.slide_audio_urls.get("urls", []) if candidate.slide_audio_urls else [],
                    "totalDurationSeconds": candidate.total_duration_seconds,
                }
                await set_to_cache(q_hash, subject_id, data)
                return data
            # else: try next candidate

    except Exception as e:
        print(f"[Semantic Cache] Check failed (non-fatal): {e}")

    # Step 3: Subject gating (only when subject is explicitly provided)
    if subject_name:
        gate = await gate_subject(question, subject_name)
        if not gate.get("allowed", True):
            return {
                "blocked": True,
                "reason": "off_topic",
                "currentSubject": subject_name,
                "detectedSubject": gate.get("detected_subject", ""),
                "message": f"This question is about {gate.get('detected_subject','')}. Please ask a {subject_name} question."
            }

    # ── Distributed lock: prevent thundering herd ─────────────────────────
    # If 100 users ask same question at once → only 1 fires AI call,
    # the other 99 wait here and get the cached result when it's ready.
    lock_acquired = await acquire_lock(q_hash, subject_id, ttl=120)
    if not lock_acquired:
        print(f"[Lock] Another worker is generating '{question}' — waiting for cache…")
        cached = await wait_for_cache(q_hash, subject_id, max_wait=110)
        if cached:
            print(f"[Lock] Got result from waiting worker for '{question}'")
            return {**cached, "cached": True}
        # Timeout — the other worker may have crashed, fall through and generate ourselves
        print(f"[Lock] Wait timeout for '{question}' — generating as fallback")
        await acquire_lock(q_hash, subject_id, ttl=120)  # acquire for ourselves now

    # ── Generation semaphore: queue if system is busy ─────────────────────
    # Max 16 concurrent AI generations. Others WAIT here (no error returned).
    # This is the queue — user just sees a longer spinner, never an error.
    print(f"[Queue] Waiting for generation slot… ({GENERATION_SEMAPHORE._value} slots free)")
    async with GENERATION_SEMAPHORE:
        print(f"[Queue] Got slot — starting generation for '{question}'")
        try:
            # Step 4: Generate slides
            slides_data = await generate_slides(question, subject_name)
            slides      = slides_data.get("presentation_slides", [])

            # Step 5: Generate images and audio in PARALLEL
            cache_id = str(uuid.uuid4())

            images_task = generate_all_images(slides, cache_id)
            audios_task = generate_all_audios(slides, cache_id, language)

            images, audios = await asyncio.gather(images_task, audios_task)

            for i, s in enumerate(slides):
                s["infographicUrl"] = images[i] if i < len(images) else ""
                audio_info = next((a for a in audios if a and a.get("slideIndex") == i), None)
                if audio_info:
                    s["audioUrl"] = audio_info.get("audioUrl")
                    s["duration"] = audio_info.get("duration")

            total_duration = sum(a.get("duration", 0) for a in audios if a)

            # Step 6: Save to Postgres
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
                total_duration_seconds=total_duration
            )
            db.add(new_row)
            await db.commit()

            result_data = {
                "cached": False,
                "cache_id": cache_id,
                "presentationSlides": slides,
                "latexFormulas": slides_data.get("latex_formulas", []),
                "keyPoints": slides_data.get("key_points", []),
                "followUpQuestions": slides_data.get("follow_up_questions", []),
                "slideAudioUrls": audios,
                "totalDurationSeconds": total_duration,
            }

            # Step 7: Warm Redis — other waiting workers will now find this
            await set_to_cache(q_hash, subject_id, result_data)

        finally:
            # ALWAYS release the lock — even if generation failed
            await release_lock(q_hash, subject_id)

    # Step 8: Store embedding (background, non-blocking)
    async def _store_embedding():
        try:
            vec = await embed_async(question)
            vec_str = vec_to_pg_str(vec)
            async with AsyncSessionLocal() as bg_db:
                await bg_db.execute(
                    text("""
                        UPDATE teaching_qa_cache
                        SET question_embedding = CAST(:vec AS vector)
                        WHERE id = CAST(:id AS uuid)
                    """),
                    {"vec": vec_str, "id": cache_id}
                )
                await bg_db.commit()
            print(f"[Embeddings] ✓ Stored for cache_id={cache_id}")
        except Exception as e:
            print(f"[Embeddings] Warning: failed to store embedding: {e}")

    asyncio.create_task(_store_embedding())

    return result_data
