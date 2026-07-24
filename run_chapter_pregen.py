"""
run_chapter_pregen.py — Generate content for any chapter
=========================================================

Generalized version of ch1_pregen.py. Works for any chapter number.

Pipeline per question:
  Phase A:   Ollama → slides JSON (with STEP 0 + semantic RAG)
  Phase A.5: Ollama → enhanced image prompts
  Phase B:   Wan2GP images + VoxCPM audio
  Phase C:   Manim animations (formula slides only)
  Phase D:   Save to DB + L2 local disk cache

Usage:
  # Generate Science Chapter 1 (free tier)
  python3 run_chapter_pregen.py --subject science --chapter 1

  # Generate Social Science Chapter 3 (pro tier)
  python3 run_chapter_pregen.py --subject social --chapter 3 --tier pro

  # Dry run to preview
  python3 run_chapter_pregen.py --subject science --chapter 2 --dry-run

  # Only retry missing Manim animations for a chapter
  python3 run_chapter_pregen.py --subject science --chapter 1 --retry-manim
"""

from __future__ import annotations

import argparse, asyncio, json, logging, os, signal, sys, time, uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# ── Logging ────────────────────────────────────────────────────────────────────
_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FILE = _REPO_ROOT / f"chapter_pregen_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger("chapter_pregen")

# ── Subject aliases ────────────────────────────────────────────────────────────
SUBJECT_ALIASES = {
    "social":      "Social Science",
    "science":     "Science",
    "math":        "Maths",
    "maths":       "Maths",
    "mathematics": "Maths",
}

# ── Graceful stop ──────────────────────────────────────────────────────────────
_stop = False
def _handle_sigint(sig, frame):
    global _stop
    log.warning("Ctrl+C — finishing current question then stopping cleanly...")
    _stop = True


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def _resolve_target(db, subject_alias: str, chapter_number: int) -> Dict:
    """Return {subject_id, subject_name, chapter_id, chapter_title}."""
    from sqlalchemy import text

    name = SUBJECT_ALIASES.get(subject_alias.lower())
    if not name:
        log.error(f"Unknown subject '{subject_alias}'. Valid: {list(SUBJECT_ALIASES)}")
        sys.exit(1)

    subj = (await db.execute(text(
        "SELECT subject_id FROM subjects WHERE name = :n"
    ), {"n": name})).fetchone()
    if not subj:
        log.error(f"Subject '{name}' not found in DB"); sys.exit(1)

    sid = subj.subject_id
    ch = (await db.execute(text(
        "SELECT id, title FROM chapters WHERE subject_id = :sid AND chapter_number = :n"
    ), {"sid": sid, "n": chapter_number})).fetchone()
    if not ch:
        log.error(f"Chapter {chapter_number} not found for '{name}'"); sys.exit(1)

    return {"subject_id": sid, "subject_name": name,
            "chapter_id": str(ch.id), "chapter_title": ch.title}


async def _ensure_cache_rows(db, target: Dict, tier: str, language: str) -> List[Dict]:
    """
    Ensure every question in the chapter has a teaching_qa_cache row.
    Creates 'pending' rows for questions that don't have one yet.
    Returns all rows (existing + newly created) that need processing.
    """
    from sqlalchemy import text
    from core.cache import hash_question

    sid        = target["subject_id"]
    chapter_id = target["chapter_id"]

    # All questions for this chapter
    questions = (await db.execute(text("""
        SELECT id, question_text, source_document_id
        FROM questions
        WHERE chapter_id = CAST(:cid AS uuid)
        ORDER BY created_at ASC
    """), {"cid": chapter_id})).fetchall()

    log.info(f"  {len(questions)} questions in chapter")
    created = 0

    for q in questions:
        q_hash = hash_question(q.question_text or "")
        # Check if row exists
        existing = (await db.execute(text("""
            SELECT id FROM teaching_qa_cache
            WHERE question_hash = :h AND subject_id = :sid
            LIMIT 1
        """), {"h": q_hash, "sid": sid})).fetchone()

        if not existing:
            new_id = str(uuid.uuid4())
            await db.execute(text("""
                INSERT INTO teaching_qa_cache
                    (id, question_hash, question_text, subject_id, chapter_id,
                     language, access_tier, pregen_status, document_id, created_at)
                VALUES
                    (CAST(:id AS uuid), :hash, :text, :sid, :cid,
                     :lang, :tier, 'pending', CAST(:doc_id AS uuid), NOW())
            """), {
                "id":     new_id,
                "hash":   q_hash,
                "text":   q.question_text,
                "sid":    sid,
                "cid":    chapter_id,
                "lang":   language,
                "tier":   tier,
                "doc_id": str(q.source_document_id) if q.source_document_id else None,
            })
            created += 1

    if created:
        await db.commit()

        log.info(f"  Created {created} new pending cache rows")

    # Fetch all rows that need processing (pending/failed/processing + done with missing media)
    rows = (await db.execute(text("""
        SELECT
            id, question_text, question_hash, subject_id, chapter_id,
            document_id, language, pregen_status, access_tier,
            presentation_slides, image_urls, slide_audio_urls, manim_video_urls
        FROM teaching_qa_cache
        WHERE subject_id = :sid
          AND chapter_id = :cid
          AND (
              pregen_status IN ('pending', 'failed', 'processing')
              OR (
                  pregen_status = 'done'
                  AND presentation_slides IS NOT NULL
                  AND (
                      presentation_slides->0->>'infographicUrl' NOT LIKE 'http%'
                      OR presentation_slides->0->>'audioUrl' NOT LIKE 'http%'
                  )
              )
          )
        ORDER BY created_at ASC
    """), {"sid": sid, "cid": chapter_id})).fetchall()

    return [dict(r._mapping) for r in rows]


async def _mark_processing(db, cache_id: str):
    from sqlalchemy import text
    await db.execute(
        text("UPDATE teaching_qa_cache SET pregen_status='processing' WHERE id=CAST(:id AS uuid)"),
        {"id": cache_id}
    )
    await db.commit()

async def _mark_failed(db, cache_id: str, error: str):
    from sqlalchemy import text
    await db.execute(
        text("UPDATE teaching_qa_cache SET pregen_status='failed' WHERE id=CAST(:id AS uuid)"),
        {"id": cache_id}
    )
    await db.commit()
    log.error(f"  FAILED id={cache_id[:8]}: {error}")


async def _save_result(db, row, slides, audio_url_list, total_duration, manim_video_urls, image_urls_map, tier: str):
    from sqlalchemy import text
    from core.embeddings import embed_async, vec_to_pg_str
    from core.local_storage import write_slide_cache

    cache_id = str(row["id"])
    question = row.get("question_text") or ""
    language = row.get("language") or "hi-IN"
    subject  = str(row.get("subject_id") or "")
    q_hash   = row.get("question_hash") or ""

    await db.execute(text("""
        UPDATE teaching_qa_cache
        SET presentation_slides    = CAST(:slides AS jsonb),
            slide_audio_urls       = CAST(:audio  AS jsonb),
            total_duration_seconds = :dur,
            manim_video_urls       = CAST(:manim  AS jsonb),
            image_urls             = CAST(:imgs   AS jsonb),
            access_tier            = :tier,
            pregen_status          = 'done',
            pregen_completed_at    = NOW()
        WHERE id = CAST(:id AS uuid)
    """), {
        "slides": json.dumps(slides),
        "audio":  json.dumps({"language": language, "urls": audio_url_list}),
        "dur":    total_duration,
        "manim":  json.dumps(manim_video_urls),
        "imgs":   json.dumps(image_urls_map),
        "tier":   tier,
        "id":     cache_id,
    })
    await db.commit()
    log.info(f"  Saved | tier={tier} | {len(slides)} slides | {round(total_duration)}s | {len(manim_video_urls)} manim")

    # Embedding
    try:
        vec = await embed_async(question)
        vec_str = vec_to_pg_str(vec)
        await db.execute(text(
            "UPDATE teaching_qa_cache SET question_embedding = CAST(:vec AS vector) WHERE id = CAST(:id AS uuid)"
        ), {"vec": vec_str, "id": cache_id})
        await db.commit()
    except Exception as e:
        log.warning(f"  Embedding failed (non-fatal): {e}")

    # L2 local cache
    if q_hash and subject:
        try:
            await write_slide_cache(subject, q_hash, {
                "cache_id":             cache_id,
                "presentationSlides":   slides,
                "slideAudioUrls":       {"language": language, "urls": audio_url_list},
                "totalDurationSeconds": total_duration,
                "manimVideoUrls":       manim_video_urls,
                "imageUrls":            image_urls_map,
                "access_tier":          tier,
                "cached":               True,
                "cache_layer":          "L2_local_disk",
            })
        except Exception as e:
            log.warning(f"  L2 cache write failed (non-fatal): {e}")

    # Mark question as pregen done
    try:
        await db.execute(text("""
            UPDATE questions SET is_pregen_done = true, cache_id = CAST(:cid AS uuid)
            WHERE question_text = :txt AND chapter_id IS NOT NULL
        """), {"cid": cache_id, "txt": question})
        await db.commit()
    except Exception:
        pass


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace):
    signal.signal(signal.SIGINT, _handle_sigint)

    from db.models import AsyncSessionLocal
    from core.pregen import _pregen_text_only, _pregen_media_only, _pregen_manim_only, _enhance_image_prompt_llm
    from core.ollama_lifecycle import (
        prepare_for_text_generation, prepare_for_media_generation, prepare_for_manim_generation
    )

    tier     = args.tier
    language = args.language
    manim_provider = args.manim_provider

    log.info("=" * 65)
    log.info(f"  run_chapter_pregen.py — Chapter {args.chapter} Generation")
    log.info(f"  Subject: {args.subject} | Tier: {tier} | Lang: {language}")
    log.info(f"  Manim: {manim_provider} | Log: {_LOG_FILE.name}")
    log.info("=" * 65)

    # Resolve subject + chapter
    async with AsyncSessionLocal() as db:
        target = await _resolve_target(db, args.subject, args.chapter)

    log.info(f"\n  Target: {target['subject_name']} — Ch{args.chapter}: {target['chapter_title']}")

    # Ensure cache rows exist + fetch pending
    async with AsyncSessionLocal() as db:
        all_rows = await _ensure_cache_rows(db, target, tier, language)

    total = len(all_rows)
    log.info(f"  Questions to process: {total}\n")

    if total == 0:
        log.info("All questions already fully generated! Nothing to do.")
        return

    if args.dry_run:
        log.info("[DRY RUN] Would process:")
        for i, row in enumerate(all_rows, 1):
            has_slides = bool(row.get("presentation_slides"))
            log.info(f"  [{i}/{total}] status={row.get('pregen_status')} slides={'yes' if has_slides else 'no'} | {(row.get('question_text') or '')[:60]}")
        log.info("[DRY RUN] Done.")
        return

    # ═══════════════════════════════════════════════════════════════
    # PHASE A — TEXT (Ollama)
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 65)
    log.info("  PHASE A — Text Generation (Ollama + Semantic RAG)")
    log.info("=" * 65)

    try:
        ready = await prepare_for_text_generation()
        log.info(f"  Ollama {'✓ ready' if ready else '⚠ not confirmed — will try per-question'}")
    except Exception as e:
        log.warning(f"  Ollama warmup failed (will try per-row): {e}")

    phase_a_done: List[Dict] = []
    phase_a_fail: List[str] = []
    t_a = time.time()

    for i, row in enumerate(all_rows, 1):
        if _stop: log.warning("Stop requested."); break
        cache_id     = str(row["id"])
        question     = (row.get("question_text") or "")[:70]
        slides_exist = bool(row.get("presentation_slides"))

        log.info(f"\n[A {i}/{total}] {question}")
        log.info(f"  cache_id={cache_id[:8]} status={row.get('pregen_status')}")

        if slides_exist:
            log.info(f"  → SKIP: slides already in DB ({len(row.get('presentation_slides') or [])} slides)")
            phase_a_done.append(row); continue

        async with AsyncSessionLocal() as db:
            await _mark_processing(db, cache_id)

        t0 = time.time()
        try:
            async with AsyncSessionLocal() as db:
                row = await _pregen_text_only(row, db, subject_name=target["subject_name"])
            
            # Save generated slides to DB immediately so progress is not lost on crash
            async with AsyncSessionLocal() as db:
                await db.execute(text(
                    "UPDATE teaching_qa_cache SET presentation_slides = CAST(:slides AS jsonb) WHERE id = CAST(:id AS uuid)"
                ), {"slides": json.dumps(row.get("presentation_slides")), "id": cache_id})
                await db.commit()

            phase_a_done.append(row)
            n = len(row.get("presentation_slides") or [])
            for j, s in enumerate(row.get("presentation_slides") or []):
                vt = f" [{s.get('visual_type','img')}]"
                log.info(f"  Slide {j+1}: {s.get('title','?')}{vt}")
            log.info(f"  ✓ {n} slides in {round(time.time()-t0)}s")
        except Exception as e:
            log.error(f"  ✗ FAILED: {e}")
            async with AsyncSessionLocal() as db: await _mark_failed(db, cache_id, str(e))
            phase_a_fail.append(cache_id)

    log.info(f"\n[Phase A] ✓ {len(phase_a_done)} ok | ✗ {len(phase_a_fail)} failed | {round(time.time()-t_a)}s")

    # ═══════════════════════════════════════════════════════════════
    # PHASE A.5 — Image Prompt Enhancement (Ollama still in VRAM)
    #   After all slides are generated, call LLM once per slide to write
    #   a rich, rule-following image prompt.
    #   Must run BEFORE prepare_for_media_generation() evicts Ollama.
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 65)
    log.info("  PHASE A.5 — LLM Image Prompt Enhancement (Ollama in VRAM)")
    log.info("=" * 65)

    for row in phase_a_done:
        if _stop: log.warning("Stop requested."); break
        cache_id = str(row["id"])
        slides   = row.get("presentation_slides") or []
        if not slides:
            continue

        enhanced_any = False
        for i, slide in enumerate(slides):
            try:
                enhanced = await _enhance_image_prompt_llm(slide)
                slides[i]["enhanced_image_prompt"] = enhanced
                enhanced_any = True
                log.info(f"  {cache_id[:8]} slide {i+1}/{len(slides)}: ✓ ({len(enhanced)} chars)")
            except Exception as e:
                log.warning(f"  {cache_id[:8]} slide {i+1}: prompt enhance failed (non-fatal): {e}")

        if enhanced_any:
            try:
                async with AsyncSessionLocal() as db:
                    await db.execute(text(
                        "UPDATE teaching_qa_cache SET presentation_slides = CAST(:s AS jsonb) WHERE id = CAST(:id AS uuid)"
                    ), {"s": json.dumps(slides), "id": cache_id})
                    await db.commit()
                row["presentation_slides"] = slides
                log.info(f"  {cache_id[:8]}: enhanced prompts saved to DB ✓")
            except Exception as e:
                log.warning(f"  {cache_id[:8]}: DB save of enhanced prompts failed (non-fatal): {e}")

    log.info("[Phase A.5] Complete")

    # ═══════════════════════════════════════════════════════════════
    # PHASE B — MEDIA (Wan2GP + VoxCPM)
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 65)
    log.info("  PHASE B — Media Generation (Wan2GP images + VoxCPM audio)")
    log.info("=" * 65)

    try:
        n = await prepare_for_media_generation()
        log.info(f"  ✓ {n} model(s) evicted — GPU ready for Wan2GP + VoxCPM")
    except Exception as e:
        log.warning(f"  Eviction failed (non-fatal): {e}")

    phase_b_done: List[Dict] = []
    phase_b_fail: List[str] = []
    t_b = time.time()

    for i, row in enumerate(phase_a_done, 1):
        if _stop: log.warning("Stop requested."); break
        cache_id = str(row["id"])
        slides   = row.get("presentation_slides") or []
        question = (row.get("question_text") or "")[:70]

        log.info(f"\n[B {i}/{len(phase_a_done)}] {question}")
        log.info(f"  {len(slides)} slides → Wan2GP + VoxCPM...")

        t0 = time.time()
        try:
            async with AsyncSessionLocal() as db:
                enriched, audio_list, total_dur, audio_durs, img_map = \
                    await _pregen_media_only(row, slides, db=db)
            row["_enriched"]    = enriched
            row["_audio_list"]  = audio_list
            row["_total_dur"]   = total_dur
            row["_audio_durs"]  = audio_durs
            row["_img_map"]     = img_map
            phase_b_done.append(row)

            for j, s in enumerate(enriched):
                img = '✓' if (s.get('infographicUrl') or '').startswith('http') else '✗'
                aud = '✓' if (s.get('audioUrl') or '').startswith('http') else '✗'
                log.info(f"  Slide {j+1}: img={img} audio={aud} ({round(s.get('duration',0),1)}s)")
            log.info(f"  ✓ imgs={len(img_map)} audio={len(audio_list)} in {round(time.time()-t0)}s")
        except Exception as e:
            log.error(f"  ✗ FAILED: {e}")
            async with AsyncSessionLocal() as db: await _mark_failed(db, cache_id, str(e))
            phase_b_fail.append(cache_id)

    log.info(f"\n[Phase B] ✓ {len(phase_b_done)} ok | ✗ {len(phase_b_fail)} failed | {round(time.time()-t_b)}s")

    # ═══════════════════════════════════════════════════════════════
    # PHASE C + D — MANIM + SAVE
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 65)
    log.info("  PHASE C/D — Manim (formula slides) + Save to DB")
    log.info("=" * 65)

    total_manim = sum(
        sum(1 for s in (r.get("_enriched") or []) if s.get("visual_type") == "manim")
        for r in phase_b_done
    )
    manim_ready = False
    if total_manim > 0:
        log.info(f"  {total_manim} formula slides need Manim")
        if manim_provider == "openrouter":
            manim_ready = True
        else:
            try:
                manim_ready = await prepare_for_manim_generation()
                log.info(f"  Ollama {'✓ ready for Manim' if manim_ready else '⚠ timeout — will use static image'}")
            except Exception as e:
                log.warning(f"  Ollama Manim load failed: {e}")
    else:
        log.info("  No formula slides — Manim skipped")

    save_ok = save_fail = 0
    for i, row in enumerate(phase_b_done, 1):
        if _stop: break
        cache_id    = str(row["id"])
        enriched    = row["_enriched"]
        audio_list  = row["_audio_list"]
        total_dur   = row["_total_dur"]
        audio_durs  = row["_audio_durs"]
        img_map     = row["_img_map"]
        question    = (row.get("question_text") or "")[:60]

        log.info(f"\n[D {i}/{len(phase_b_done)}] {question}")

        manim_urls: dict = {}
        manim_slides = [s for s in enriched if s.get("visual_type") == "manim"]
        if manim_slides and manim_ready:
            log.info(f"  Generating Manim for {len(manim_slides)} formula slide(s)...")
            try:
                manim_urls = await _pregen_manim_only(row, enriched, audio_durs, provider=manim_provider)
                log.info(f"  ✓ {len(manim_urls)} Manim video(s)")
            except Exception as e:
                log.warning(f"  Manim failed (keeping static): {e}")

        try:
            async with AsyncSessionLocal() as db:
                await _save_result(db, row, enriched, audio_list, total_dur, manim_urls, img_map, tier)
            save_ok += 1
            log.info(f"  ✓ Saved → tier={tier}")
        except Exception as e:
            log.error(f"  ✗ Save failed: {e}")
            save_fail += 1

    # ── Final summary ──────────────────────────────────────────────
    elapsed = round(time.time() - t_a)
    log.info(f"\n{'='*65}")
    log.info(f"  COMPLETE — {target['subject_name']} Ch{args.chapter}: {target['chapter_title']}")
    log.info(f"  Total time : {elapsed}s ({round(elapsed/60,1)} min)")
    log.info(f"  Phase A    : {len(phase_a_done)} ok  {len(phase_a_fail)} failed")
    log.info(f"  Phase B    : {len(phase_b_done)} ok  {len(phase_b_fail)} failed")
    log.info(f"  Saved      : {save_ok} ok  {save_fail} failed")
    log.info(f"  Log file   : {_LOG_FILE}")
    log.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate content for any chapter")
    parser.add_argument("--subject",  required=True, help="science | social | math")
    parser.add_argument("--chapter",  type=int, required=True, help="Chapter number")
    parser.add_argument("--tier",     default="free", choices=["free", "pro"],
                        help="Access tier for generated content (default: free)")
    parser.add_argument("--language", default="hi-IN", help="Audio language (default: hi-IN)")
    parser.add_argument("--dry-run",  action="store_true", help="Preview only, no generation")
    parser.add_argument("--manim-provider", default="openrouter", choices=["openrouter", "local"],
                        help="Manim code provider (default: openrouter)")
    parser.add_argument("--retry-manim", action="store_true",
                        help="Only retry Manim for done rows missing animations")
    asyncio.run(main(parser.parse_args()))
