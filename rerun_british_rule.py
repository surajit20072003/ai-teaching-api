"""
rerun_british_rule.py
─────────────────────
Regenerates slides + images + audio for 8 hand-picked questions from
Social Science → Chapter 1, Topic 4 (The British Rule in India)
using the NEW improved prompts.

Run inside the container:
  docker exec ai-teaching-api python3 rerun_british_rule.py

Options (edit at top of file):
  SUBJECT_ID   — UUID of the Social Science subject
  LANGUAGE     — TTS language (default hi-IN)
  DRY_RUN      — True = generate but don't update DB or clear Redis
"""

import asyncio, json, time, uuid as _uuid
from sqlalchemy import text, update
from db.models import AsyncSessionLocal, TeachingCache
from core.slide_generator import generate_slides
from core.pregen import _process_slide, _enhance_image_prompt_llm   # Wan2GP + VoxCPM (local GPU)
from core.cache import get_redis, hash_question
from core.ollama_lifecycle import prepare_for_text_generation, prepare_for_media_generation

# ── Config ────────────────────────────────────────────────────────────────────
SUBJECT_ID   = "b4b83f9b-bc1f-433c-9400-234e50ac1b70"
SUBJECT_NAME = "Social Science"
LANGUAGE     = "hi-IN"
DRY_RUN      = False   # Set True to generate but skip DB write + Redis clear

# ── 8 Questions to regenerate ─────────────────────────────────────────────────
# Set LIMIT = 1 to test one first. Set to 8 (or None) to run all.
LIMIT = 1

QUESTIONS = [
    "The Battle of Plassey was fought in",
    "The Battle of Buxar was fought in",
    "The Battle of Plassey established British control over",
    "After the Battle of Buxar, the British gained",
    "The foundation of British rule in India was laid after",
    "What was the importance of the Battle of Plassey?",
    "Why is the Battle of Buxar considered significant?",
    "Explain how the British established their rule in India.",
]

# Apply limit
if LIMIT:
    QUESTIONS = QUESTIONS[:LIMIT]

# ── RAG helper — same query as main.py L5 ────────────────────────────────────
from core.embeddings import embed_async, vec_to_pg_str

async def _fetch_rag_context(db, question: str) -> str:
    embedding = await embed_async(question)
    vec_str   = vec_to_pg_str(embedding)
    rows = (await db.execute(text("""
        SELECT dc.chunk_text, dc.section_title, d.title AS doc_title,
               1 - (dc.chunk_embedding <=> CAST(:vec AS vector)) AS sim
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE dc.chunk_embedding IS NOT NULL
          AND dc.subject_id = :subj
          AND d.status = 'ready'
          AND 1 - (dc.chunk_embedding <=> CAST(:vec AS vector)) > 0.50
        ORDER BY sim DESC LIMIT 3
    """), {"vec": vec_str, "subj": SUBJECT_ID})).fetchall()

    if not rows:
        return ""
    return "\n\n".join(
        f"[{r.doc_title} / {r.section_title or 'Section'}]\n{r.chunk_text}"
        for r in rows
    )


# ── Find existing DB row for a question hash ──────────────────────────────────
async def _get_existing_row(db, q_hash: str):
    row = (await db.execute(text("""
        SELECT id::text, question_text, pregen_status,
               jsonb_array_length(COALESCE(presentation_slides,'[]'::jsonb)) AS slide_count
        FROM teaching_qa_cache
        WHERE question_hash = :h AND subject_id = :s
        LIMIT 1
    """), {"h": q_hash, "s": SUBJECT_ID})).first()
    return row


# ── Clear Redis cache for this question ───────────────────────────────────────
async def _clear_redis(q_hash: str):
    r = get_redis()
    key = f"teaching:{q_hash}:{SUBJECT_ID}"
    deleted = await r.delete(key)
    if deleted:
        print(f"    🗑  Redis: cleared {key}")


# ── Save / update DB row ─────────────────────────────────────────────────────
async def _save_to_db(db, question: str, q_hash: str, slides: list,
                      total_dur: float, slides_data: dict):
    """
    UPDATE existing row if question_hash matches, else INSERT new row.
    Audio list is built from slide objects (audioUrl + duration fields)
    since _process_slide embeds them directly into each slide dict.
    """
    existing = await _get_existing_row(db, q_hash)
    new_id   = str(_uuid.uuid4())

    # Build audio url list from slide objects (same format as pregen pipeline saves)
    audio_urls = [
        {"slideIndex": i, "audioUrl": s.get("audioUrl", ""), "duration": s.get("duration", 0)}
        for i, s in enumerate(slides)
    ]
    audio_payload = json.dumps({"language": LANGUAGE, "urls": audio_urls})

    if existing:
        row_id = existing[0]
        await db.execute(text("""
            UPDATE teaching_qa_cache SET
                presentation_slides    = CAST(:slides AS jsonb),
                slide_audio_urls       = CAST(:audio  AS jsonb),
                total_duration_seconds = :dur,
                latex_formulas         = CAST(:latex  AS jsonb),
                pregen_status          = 'done',
                pregen_completed_at    = NOW()
            WHERE id = CAST(:id AS uuid)
        """), {
            "slides": json.dumps(slides),
            "audio":  audio_payload,
            "dur":    total_dur,
            "latex":  json.dumps(slides_data.get("latex_formulas", [])),
            "id":     row_id,
        })
        await db.commit()
        return row_id, "UPDATED"
    else:
        # INSERT new row
        embedding = await embed_async(question)
        vec_str   = vec_to_pg_str(embedding)
        await db.execute(text("""
            INSERT INTO teaching_qa_cache
                (id, question_hash, question_text, subject_id, language,
                 variation_number, presentation_slides, slide_audio_urls,
                 total_duration_seconds, latex_formulas, is_doc_grounded,
                 pregen_status, pregen_completed_at, question_embedding)
            VALUES
                (CAST(:id AS uuid), :hash, :question, :subj, :lang,
                 1, CAST(:slides AS jsonb), CAST(:audio AS jsonb),
                 :dur, CAST(:latex AS jsonb), true,
                 'done', NOW(), CAST(:vec AS vector))
            ON CONFLICT (question_hash, subject_id, variation_number) DO NOTHING
        """), {
            "id":     new_id,
            "hash":   q_hash,
            "question": question,
            "subj":   SUBJECT_ID,
            "lang":   LANGUAGE,
            "slides": json.dumps(slides),
            "audio":  audio_payload,
            "dur":    total_dur,
            "latex":  json.dumps(slides_data.get("latex_formulas", [])),
            "vec":    vec_str,
        })
        await db.commit()
        return new_id, "INSERTED"


# ── Print quality report for one question ────────────────────────────────────
def _print_report(idx: int, question: str, slides: list,
                  total_dur: float, row_id: str, db_action: str, elapsed: float):
    print()
    print(f"{'━'*65}")
    print(f"[Q{idx}] {question[:85]}{'…' if len(question)>85 else ''}")
    print(f"  Slides  : {len(slides)}")
    imgs_ok  = sum(1 for s in slides if s.get("infographicUrl","").startswith("http"))
    audio_ok = sum(1 for s in slides if s.get("audioUrl","").startswith("http"))
    print(f"  Images  : {'✅' if imgs_ok==len(slides) else '⚠️ '} {imgs_ok}/{len(slides)} generated")
    print(f"  Audio   : {'✅' if audio_ok==len(slides) else '⚠️ '} {audio_ok}/{len(slides)} generated")
    print(f"  Duration: {int(total_dur//60)}m {int(total_dur%60)}s total | elapsed={elapsed:.1f}s")
    print(f"  DB      : {db_action} (id={row_id[:12]}…)")
    print()
    for i, s in enumerate(slides):
        slide_type = "📖"
        if s.get("isStory"): slide_type = "📚"
        if s.get("isTips"):  slide_type = "💡"
        img_ok  = "✅" if s.get("infographicUrl","").startswith("http") else "❌"
        aud_ok  = "✅" if s.get("audioUrl","").startswith("http") else "❌"
        print(f"  Slide {i+1} {slide_type} \"{s.get('title','?')[:50]}\"")
        print(f"    img={img_ok}  audio={aud_ok}")
        narr = s.get("narration","")
        if narr:
            print(f"    narration: \"{narr[:120]}…\"")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print()
    print("=" * 65)
    print("RERUN SCRIPT — Social Science Ch.1 Topic 4: British Rule")
    print(f"Subject: {SUBJECT_ID}")
    print(f"Language: {LANGUAGE}")
    print(f"DryRun: {DRY_RUN}")
    print(f"Questions: {len(QUESTIONS)}")
    print("=" * 65)

    results = {"success": 0, "failed": 0, "skipped": 0}

    async with AsyncSessionLocal() as db:
        for idx, question in enumerate(QUESTIONS, 1):
            print(f"\n▶ [{idx}/{len(QUESTIONS)}] Generating: \"{question[:70]}…\"")
            q_hash = hash_question(question)
            t_start = time.time()

            try:
                # 1. RAG lookup
                print(f"  [RAG] Fetching document context…")
                rag_context = await _fetch_rag_context(db, question)
                if rag_context:
                    print(f"  [RAG] ✅ {len(rag_context)} chars of context found")
                else:
                    print(f"  [RAG] ⚠️  No document chunks found — using pure LLM")

                # 2. Generate slides
                print(f"  [Model] Loading Ollama into VRAM (unloading media models)...")
                await prepare_for_text_generation()
                
                print(f"  [Slides] Calling Ollama (qwen2.5-coder:32b) for local pregen generation…")
                slides_data = await generate_slides(
                    question=question,
                    subject=SUBJECT_NAME,
                    context=rag_context,
                    use_local=True,    # Ollama on GPU server (pregen path)
                )
                slides = slides_data.get("presentation_slides", [])
                print(f"  [Slides] ✅ {len(slides)} slides generated")

                # 2.5 Phase A.5 — LLM image prompt enhancement (Ollama still in VRAM)
                # Call _enhance_image_prompt_llm() for every slide BEFORE evicting Ollama.
                # This writes a rich, rule-following image prompt into each slide dict.
                print(f"  [Prompts] Enhancing image prompts for {len(slides)} slides...")
                for i, slide in enumerate(slides):
                    try:
                        enhanced = await _enhance_image_prompt_llm(slide)
                        slides[i]["enhanced_image_prompt"] = enhanced
                        print(f"    slide {i+1}: prompt enhanced ({len(enhanced)} chars)")
                    except Exception as e:
                        print(f"    slide {i+1}: prompt enhance failed (non-fatal): {e}")
                print(f"  [Prompts] ✅ image prompts enhanced")

                # 3. Images + Audio via Wan2GP + VoxCPM (pregen local path)
                # Sequential per slide to avoid overloading GPU (same as pregen pipeline)
                print(f"  [Model] Loading Wan2GP + VoxCPM into VRAM (unloading Ollama)...")
                await prepare_for_media_generation()
                
                print(f"  [Media] Processing {len(slides)} slides via Wan2GP + VoxCPM…")
                cache_id = str(_uuid.uuid4())
                total_dur = 0.0
                for i, slide in enumerate(slides):
                    print(f"    slide {i+1}/{len(slides)}: generating image + audio…")
                    try:
                        enriched, duration = await _process_slide(i, slide, cache_id, LANGUAGE, SUBJECT_ID)
                        slides[i] = enriched
                        total_dur += duration
                        img_ok  = "✅" if enriched.get("infographicUrl","").startswith("http") else "❌"
                        aud_ok  = "✅" if enriched.get("audioUrl","").startswith("http") else "❌"
                        print(f"    slide {i+1}: img={img_ok} audio={aud_ok} ({round(duration,1)}s)")
                    except Exception as e:
                        print(f"    slide {i+1}: ⚠️ media failed (non-fatal): {e}")

                # 5. Save to DB + clear Redis
                elapsed = time.time() - t_start
                if not DRY_RUN:
                    row_id, db_action = await _save_to_db(
                        db, question, q_hash, slides, total_dur, slides_data
                    )
                    await _clear_redis(q_hash)
                else:
                    row_id, db_action = "DRY_RUN", "SKIPPED"

                _print_report(idx, question, slides, total_dur, row_id, db_action, elapsed)
                results["success"] += 1

            except Exception as e:
                elapsed = time.time() - t_start
                print(f"  ❌ FAILED after {elapsed:.1f}s: {e}")
                import traceback; traceback.print_exc()
                results["failed"] += 1

    # Final summary
    print()
    print("═" * 65)
    print("RERUN COMPLETE")
    print("═" * 65)
    print(f"  Successful : {results['success']}")
    print(f"  Failed     : {results['failed']}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
