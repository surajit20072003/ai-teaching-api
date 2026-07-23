#!/usr/bin/env python3
"""
run_science_ecosystem.py
Run the FULL normal pregen pipeline for 2 specific Science questions.

Phases (identical to run_pregen_batch):
  Phase A   -> Ollama generates slides text
  Phase A.5 -> Ollama enhances image prompts (image_system_prompt.txt)
  Phase B   -> Wan2GP + VoxCPM generate images + audio
  Phase D   -> Save results to DB, mark done, clear Redis

Usage:
  docker exec ai-teaching-api python3 -u run_science_ecosystem.py
"""

import asyncio, json, time, uuid as _uuid, sys, os
sys.path.insert(0, "/home2/ai-teaching-api")
os.chdir("/home2/ai-teaching-api")

from sqlalchemy import text
from db.models import AsyncSessionLocal
from core.slide_generator import generate_slides
from core.pregen import (
    _pregen_text_only,
    _pregen_media_only,
    _enhance_image_prompt_llm,
    _save_row_result,
)
from core.cache import hash_question, get_redis
from core.ollama_lifecycle import prepare_for_text_generation, prepare_for_media_generation

# ── Config ─────────────────────────────────────────────────────────────────────
SUBJECT_ID   = "ceaf73fb-528a-4d4a-947c-4a7be304db2b"
SUBJECT_NAME = "Science"
LANGUAGE     = "hi-IN"
DRY_RUN      = False

QUESTIONS = [
    "Explain the structure and functioning of an ecosystem. (4 Marks)",
    "Explain the role of producers, consumers, and decomposers in an ecosystem",
]


async def _ensure_row(db, question: str) -> dict:
    """Get existing row or insert new pending row from questions table."""
    q_hash = hash_question(question)

    existing = (await db.execute(text("""
        SELECT id, question_text, language, subject_id,
               presentation_slides, slide_audio_urls,
               question_hash, document_id, pregen_status
        FROM teaching_qa_cache
        WHERE question_hash = :h AND subject_id = :s
        LIMIT 1
    """), {"h": q_hash, "s": SUBJECT_ID})).first()

    if existing:
        row = dict(existing._mapping)
        print(f"    Existing row — status: {row['pregen_status']}")
        return row

    # Look up from questions table
    q_src = (await db.execute(text("""
        SELECT chapter_id, topic_id FROM questions
        WHERE question_text = :qt AND subject_id = :s LIMIT 1
    """), {"qt": question, "s": SUBJECT_ID})).first()

    new_id = str(_uuid.uuid4())
    ch_id  = str(q_src.chapter_id) if q_src and q_src.chapter_id else None
    t_id   = str(q_src.topic_id)   if q_src and q_src.topic_id   else None

    await db.execute(text("""
        INSERT INTO teaching_qa_cache
            (id, subject_id, chapter_id, topic_id,
             question_hash, question_text, language, variation_number, pregen_status)
        VALUES
            (CAST(:id AS uuid), :sid, :cid, :tid,
             :qhash, :qtext, :lang, 1, 'pending')
        ON CONFLICT (question_hash, subject_id, variation_number) DO NOTHING
    """), {"id": new_id, "sid": SUBJECT_ID, "cid": ch_id, "tid": t_id,
           "qhash": q_hash, "qtext": question, "lang": LANGUAGE})
    await db.commit()

    row = dict((await db.execute(text("""
        SELECT id, question_text, language, subject_id,
               presentation_slides, slide_audio_urls,
               question_hash, document_id, pregen_status
        FROM teaching_qa_cache
        WHERE question_hash = :h AND subject_id = :s LIMIT 1
    """), {"h": q_hash, "s": SUBJECT_ID})).first()._mapping)
    print(f"    New row inserted")
    return row


async def _clear_redis(q_hash: str):
    r = get_redis()
    key = f"teaching:{q_hash}:{SUBJECT_ID}"
    deleted = await r.delete(key)
    if deleted:
        print(f"    Redis cleared: {key}")


async def main():
    print()
    print("=" * 65)
    print(f"  PREGEN: {SUBJECT_NAME} — Ecosystem Questions")
    print(f"  Subject: {SUBJECT_ID}")
    print(f"  Language: {LANGUAGE}  |  DryRun: {DRY_RUN}")
    print("=" * 65)

    # ── Phase A gate ──────────────────────────────────────────────────────────
    print("\n[Phase A] Ensuring Ollama is in VRAM...")
    try:
        await prepare_for_text_generation()
        print("[Phase A] Ollama ready")
    except Exception as e:
        print(f"[Phase A] Warning (proceeding): {e}")

    # ── Phase A: text generation ──────────────────────────────────────────────
    phase_a_rows = []
    print()
    for idx, question in enumerate(QUESTIONS, 1):
        print(f"[{idx}/{len(QUESTIONS)}] {question[:70]}")
        t = time.time()

        async with AsyncSessionLocal() as db:
            row_dict = await _ensure_row(db, question)
            cid = str(row_dict["id"])

            # Mark processing
            await db.execute(text(
                "UPDATE teaching_qa_cache SET pregen_status='processing' "
                "WHERE id=CAST(:id AS uuid) AND pregen_status IN ('pending','failed')"
            ), {"id": cid})
            await db.commit()

            try:
                row_dict = await _pregen_text_only(row_dict, db, subject_name=SUBJECT_NAME)
                slides = row_dict.get("presentation_slides") or []
                print(f"    [Phase A] OK — {len(slides)} slides ({round(time.time()-t,1)}s)")
                phase_a_rows.append(row_dict)
            except Exception as e:
                print(f"    [Phase A] FAILED: {e}")
                await db.execute(text(
                    "UPDATE teaching_qa_cache SET pregen_status='failed' WHERE id=CAST(:id AS uuid)"
                ), {"id": cid})
                await db.commit()

    print(f"\n[Phase A] Complete: {len(phase_a_rows)}/{len(QUESTIONS)} succeeded")

    if not phase_a_rows:
        print("Nothing to continue. Exiting.")
        return

    # ── Phase A.5: image prompt enhancement ──────────────────────────────────
    print("\n[Phase A.5] Enhancing image prompts (Ollama still in VRAM)...")
    for row_dict in phase_a_rows:
        slides   = row_dict.get("presentation_slides") or []
        cache_id = str(row_dict["id"])
        print(f"  {row_dict['question_text'][:60]}...")
        enhanced_any = False
        for i, slide in enumerate(slides):
            try:
                enhanced = await _enhance_image_prompt_llm(slide)
                slides[i]["enhanced_image_prompt"] = enhanced
                enhanced_any = True
                print(f"    slide {i+1}: OK ({len(enhanced)} chars)")
            except Exception as e:
                print(f"    slide {i+1}: failed (non-fatal): {e}")

        if enhanced_any and not DRY_RUN:
            async with AsyncSessionLocal() as db:
                await db.execute(text(
                    "UPDATE teaching_qa_cache "
                    "SET presentation_slides = CAST(:s AS jsonb) "
                    "WHERE id = CAST(:id AS uuid)"
                ), {"s": json.dumps(slides), "id": cache_id})
                await db.commit()
            row_dict["presentation_slides"] = slides
            print(f"    Enhanced prompts saved to DB")

    print("[Phase A.5] Complete")

    # ── Phase B: media generation ─────────────────────────────────────────────
    print("\n[Phase B] Evicting Ollama, loading Wan2GP + VoxCPM...")
    try:
        await prepare_for_media_generation()
        print("[Phase B] Media models ready")
    except Exception as e:
        print(f"[Phase B] Warning: {e}")

    phase_b_rows = []
    print()
    for row_dict in phase_a_rows:
        slides   = row_dict.get("presentation_slides") or []
        cache_id = str(row_dict["id"])
        print(f"[Media] {row_dict['question_text'][:65]}")
        try:
            async with AsyncSessionLocal() as db:
                enriched_slides, audio_url_list, total_duration, audio_durations, image_urls_map = \
                    await _pregen_media_only(row_dict, slides, db=db)

            row_dict["_enriched_slides"]  = enriched_slides
            row_dict["_audio_url_list"]   = audio_url_list
            row_dict["_total_duration"]   = total_duration
            row_dict["_audio_durations"]  = audio_durations
            row_dict["_image_urls_map"]   = image_urls_map
            phase_b_rows.append(row_dict)

            img_ok = sum(1 for s in enriched_slides if (s.get("infographicUrl","")).startswith("http"))
            aud_ok = sum(1 for s in enriched_slides if (s.get("audioUrl","")).startswith("http"))
            print(f"    OK — {img_ok}/{len(slides)} images, {aud_ok}/{len(slides)} audio, {round(total_duration,1)}s")
        except Exception as e:
            print(f"    FAILED: {e}")
            async with AsyncSessionLocal() as db:
                await db.execute(text(
                    "UPDATE teaching_qa_cache SET pregen_status='failed' WHERE id=CAST(:id AS uuid)"
                ), {"id": cache_id})
                await db.commit()

    print(f"\n[Phase B] Complete: {len(phase_b_rows)}/{len(phase_a_rows)} succeeded")

    # ── Phase D: save results ─────────────────────────────────────────────────
    if not DRY_RUN:
        print("\n[Phase D] Saving results to DB...")
        for row_dict in phase_b_rows:
            try:
                async with AsyncSessionLocal() as db:
                    await _save_row_result(
                        db               = db,
                        row              = row_dict,
                        slides           = row_dict["_enriched_slides"],
                        audio_url_list   = row_dict["_audio_url_list"],
                        total_duration   = row_dict["_total_duration"],
                        manim_video_urls = {},
                        image_urls_map   = row_dict["_image_urls_map"],
                        access_tier      = "pro",
                    )
                await _clear_redis(row_dict.get("question_hash", ""))
                print(f"    Saved: {row_dict['question_text'][:60]}...")
            except Exception as e:
                print(f"    Save failed: {e}")

    print()
    print("=" * 65)
    print(f"  DONE — {len(phase_b_rows)}/{len(QUESTIONS)} completed")
    print("=" * 65)
    print()


if __name__ == "__main__":
    asyncio.run(main())
