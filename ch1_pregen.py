"""
ch1_pregen.py — Standalone Chapter 1 Bulk Generation Script
=============================================================

Runs DIRECTLY on the GPU server (not inside Docker).
Generates all remaining Chapter 1 content for 3 subjects:
  - Social Science: "The Advent Of Europeans To India"
  - Science:        "Chemical Reactions and Equations"
  - Mathematics:    "Real Numbers"

Pipeline per question (same as normal pregen):
  Phase A: Text   — Ollama -> generate presentation_slides JSON
  Phase B: Media  — Wan2GP image + VoxCPM audio in parallel (3 slides at a time)
  Phase C: Manim  — generate animations for formula slides (visual_type='manim')

All Chapter 1 questions are saved with access_tier = 'free'.

Usage:
  cd /path/to/ai-teaching-api
  python3 ch1_pregen.py                 # full run (all 3 subjects, all phases)
  python3 ch1_pregen.py --dry-run       # show what would be run, no generation
  python3 ch1_pregen.py --subject math  # only run one subject (social/science/math)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Bootstrap: load .env and add repo root to path ─────────────────────────────
_REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass  # python-dotenv optional — env vars may already be set

# ── Logging setup ───────────────────────────────────────────────────────────────
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FILE = _REPO_ROOT / f"ch1_pregen_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger("ch1_pregen")

# ── Fixed: all Chapter 1 questions are FREE ─────────────────────────────────────
ACCESS_TIER = "free"

# ── Subject alias mapping ───────────────────────────────────────────────────────
SUBJECT_ALIASES = {
    "social":       "Social Science",
    "science":      "Science",
    "math":         "Maths",
    "maths":        "Maths",
    "mathematics":  "Maths",
    "math10":       "Mathematics - Class 10",
}

# ── Graceful stop flag ──────────────────────────────────────────────────────────
_stop_requested = False


def _handle_sigint(sig, frame):
    global _stop_requested
    log.warning("\n[ch1_pregen] Ctrl+C — finishing current question then stopping cleanly...")
    _stop_requested = True


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _get_subjects_and_chapters(db) -> List[Dict[str, Any]]:
    """Resolve subject_ids and Chapter 1 chapter_ids from DB."""
    from sqlalchemy import text

    rows = (await db.execute(text(
        "SELECT subject_id, name FROM subjects ORDER BY name"
    ))).fetchall()
    subject_map = {r.name: r.subject_id for r in rows}

    targets = []
    # Try all known subject name variations in the DB
    for name in ["Social Science", "Science", "Maths", "Mathematics - Class 10"]:
        sid = subject_map.get(name)
        if not sid:
            log.warning(f"Subject '{name}' not found in DB — skipping")
            continue

        ch = (await db.execute(text(
            "SELECT id, title FROM chapters WHERE subject_id = :sid AND chapter_number = 1"
        ), {"sid": sid})).fetchone()
        if not ch:
            log.warning(f"Chapter 1 not found for '{name}' — skipping")
            continue

        targets.append({
            "subject_id":    sid,
            "subject_name":  name,
            "chapter_id":    str(ch.id),
            "chapter_title": ch.title,
        })
    return targets


async def _fetch_pending_rows(db, subject_id: str, chapter_id: str) -> List[Dict[str, Any]]:
    """Fetch all rows that still need generation (pending/failed/stuck/partial media)."""
    from sqlalchemy import text

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
    """), {"sid": subject_id, "cid": chapter_id})).fetchall()

    return [dict(r._mapping) for r in rows]


async def _fetch_manim_retry_rows(db, subject_id: str, chapter_id: str) -> List[Dict[str, Any]]:
    """
    Fetch 'done' rows that have formula slides (visual_type='manim') but
    are missing Manim video URLs. Used by --retry-manim mode.
    """
    from sqlalchemy import text

    rows = (await db.execute(text("""
        SELECT
            id, question_text, question_hash, subject_id, chapter_id,
            document_id, language, pregen_status, access_tier,
            presentation_slides, image_urls, slide_audio_urls, manim_video_urls
        FROM teaching_qa_cache
        WHERE subject_id = :sid
          AND chapter_id = :cid
          AND pregen_status = 'done'
          AND presentation_slides IS NOT NULL
          AND jsonb_array_length(presentation_slides) > 0
          AND (
              manim_video_urls IS NULL
              OR manim_video_urls = '{}'::jsonb
              OR manim_video_urls = 'null'::jsonb
          )
          AND EXISTS (
              SELECT 1 FROM jsonb_array_elements(presentation_slides) AS slide
              WHERE slide->>'visual_type' = 'manim'
          )
        ORDER BY created_at ASC
    """), {"sid": subject_id, "cid": chapter_id})).fetchall()

    return [dict(r._mapping) for r in rows]


async def _mark_processing(db, cache_id: str):
    from sqlalchemy import text
    await db.execute(
        text("UPDATE teaching_qa_cache SET pregen_status='processing' WHERE id=CAST(:id AS uuid)"),
        {"id": cache_id},
    )
    await db.commit()


async def _mark_failed(db, cache_id: str, error: str):
    from sqlalchemy import text
    await db.execute(
        text("UPDATE teaching_qa_cache SET pregen_status='failed' WHERE id=CAST(:id AS uuid)"),
        {"id": cache_id},
    )
    await db.commit()
    log.error(f"FAILED id={cache_id[:8]}: {error}")


# ─────────────────────────────────────────────────────────────────────────────
# Save helper — forces access_tier = 'free'
# ─────────────────────────────────────────────────────────────────────────────

async def _save_result(
    db,
    row: Dict[str, Any],
    slides: list,
    audio_url_list: list,
    total_duration: float,
    manim_video_urls: dict,
    image_urls_map: dict,
) -> None:
    """Save results to DB, mark done, store embedding and L2 cache. Forces access_tier='free'."""
    from sqlalchemy import text
    from core.embeddings import embed_async, vec_to_pg_str
    from core.local_storage import write_slide_cache

    cache_id = str(row["id"])
    question = row.get("question_text") or ""
    language = row.get("language") or "hi-IN"
    subject  = str(row.get("subject_id") or "")
    q_hash   = row.get("question_hash") or ""

    # Step 1: Core save (always free tier)
    await db.execute(text("""
        UPDATE teaching_qa_cache
        SET presentation_slides    = CAST(:slides AS jsonb),
            slide_audio_urls       = CAST(:audio  AS jsonb),
            total_duration_seconds = :dur,
            manim_video_urls       = CAST(:manim  AS jsonb),
            image_urls             = CAST(:imgs   AS jsonb),
            access_tier            = 'free',
            pregen_status          = 'done',
            pregen_completed_at    = NOW()
        WHERE id = CAST(:id AS uuid)
    """), {
        "slides": json.dumps(slides),
        "audio":  json.dumps({"language": language, "urls": audio_url_list}),
        "dur":    total_duration,
        "manim":  json.dumps(manim_video_urls),
        "imgs":   json.dumps(image_urls_map),
        "id":     cache_id,
    })
    await db.commit()
    log.info(f"Saved {cache_id[:8]} | free | {len(slides)} slides | {round(total_duration)}s | {len(manim_video_urls)} manim")

    # Step 2: Embedding (best-effort)
    try:
        vec = await embed_async(question)
        vec_str = vec_to_pg_str(vec)
        await db.execute(text(
            "UPDATE teaching_qa_cache SET question_embedding = CAST(:vec AS vector) WHERE id = CAST(:id AS uuid)"
        ), {"vec": vec_str, "id": cache_id})
        await db.commit()
    except Exception as e:
        log.warning(f"Embedding failed (non-fatal): {e}")

    # Step 3: L2 local slide cache
    if q_hash and subject:
        try:
            await write_slide_cache(subject, q_hash, {
                "cache_id":             cache_id,
                "presentationSlides":   slides,
                "slideAudioUrls":       {"language": language, "urls": audio_url_list},
                "totalDurationSeconds": total_duration,
                "manimVideoUrls":       manim_video_urls,
                "imageUrls":            image_urls_map,
                "access_tier":          "free",
                "cached":               True,
                "cache_layer":          "L2_local_disk",
            })
        except Exception as e:
            log.warning(f"L2 cache write failed (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────

async def _print_final_report(db, targets: List[Dict[str, Any]]):
    from sqlalchemy import text

    log.info("\n" + "=" * 60)
    log.info("  Chapter 1 Final Report")
    log.info("=" * 60)

    for t in targets:
        r = (await db.execute(text("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN presentation_slides->0->>'infographicUrl' LIKE 'http%'
                          AND presentation_slides->0->>'audioUrl' LIKE 'http%' THEN 1 ELSE 0 END) as complete,
                SUM(CASE WHEN pregen_status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM teaching_qa_cache
            WHERE subject_id = :sid AND chapter_id = :cid
        """), {"sid": t["subject_id"], "cid": t["chapter_id"]})).fetchone()

        total    = r.total or 0
        complete = r.complete or 0
        failed   = r.failed or 0

        status = "OK" if complete == total else ("FAILED" if failed > 0 else "PARTIAL")
        log.info(f"[{status}] {t['subject_name']} - {t['chapter_title']}")
        log.info(f"       Fully Done: {complete}/{total} | Failed: {failed} | Remaining: {total - complete - failed}")

    log.info("=" * 60)
    log.info(f"Log file: {_LOG_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    from db.models import AsyncSessionLocal
    from core.pregen import _pregen_text_only, _pregen_media_only, _pregen_manim_only
    from core.ollama_lifecycle import (
        prepare_for_text_generation,
        prepare_for_media_generation,
        prepare_for_manim_generation,
    )

    db_factory = AsyncSessionLocal

    manim_provider = args.manim_provider
    log.info("=" * 60)
    log.info("  ch1_pregen.py -- Chapter 1 Bulk Generation")
    log.info(f"  access_tier=free | manim_provider={manim_provider} | Log: {_LOG_FILE.name}")
    log.info("="  * 60)

    # Resolve subjects + Chapter 1 IDs
    async with db_factory() as db:
        all_targets = await _get_subjects_and_chapters(db)

    if args.subject:
        filter_name = SUBJECT_ALIASES.get(args.subject.lower())
        if not filter_name:
            log.error(f"Unknown --subject '{args.subject}'. Valid: {list(SUBJECT_ALIASES)}")
            sys.exit(1)
        all_targets = [t for t in all_targets if t["subject_name"] == filter_name]

    if not all_targets:
        log.error("No subjects found. Check DB connection.")
        sys.exit(1)

    # Collect all pending rows across subjects
    all_rows: List[Dict[str, Any]] = []
    subject_for_row: Dict[str, str] = {}

    async with db_factory() as db:
        for t in all_targets:
            rows = await _fetch_pending_rows(db, t["subject_id"], t["chapter_id"])
            log.info(f"  {t['subject_name']} Ch1 '{t['chapter_title']}': {len(rows)} rows pending")
            for row in rows:
                all_rows.append(row)
                subject_for_row[str(row["id"])] = t["subject_name"]

    total = len(all_rows)

    # ═══════════════════════════════════════════════════════════
    # --retry-manim: Phase C ONLY for formula slides missing Manim
    # ═══════════════════════════════════════════════════════════
    if args.retry_manim:
        log.info("\n" + "=" * 60)
        log.info("  --retry-manim MODE: Phase C Only (Manim generation)")
        log.info("  Fetching 'done' rows with formula slides but no Manim video")
        log.info("=" * 60)

        manim_rows: List[Dict[str, Any]] = []
        async with db_factory() as db:
            for t in all_targets:
                rows = await _fetch_manim_retry_rows(db, t["subject_id"], t["chapter_id"])
                log.info(f"  {t['subject_name']} Ch1: {len(rows)} rows need Manim")
                for row in rows:
                    manim_rows.append(row)
                    subject_for_row[str(row["id"])] = t["subject_name"]

        if not manim_rows:
            log.info("✅ No missing Manim videos found — all formula slides already have Manim!")
            async with db_factory() as db:
                await _print_final_report(db, all_targets)
            return

        log.info(f"\nTotal rows needing Manim: {len(manim_rows)}\n")

        if args.dry_run:
            log.info("[DRY RUN --retry-manim] Would generate Manim for:")
            for i, row in enumerate(manim_rows, 1):
                subj = subject_for_row.get(str(row["id"]), "?")
                slides = row.get("presentation_slides") or []
                n_manim = sum(1 for s in slides if s.get("visual_type") == "manim")
                log.info(f"  [{i}/{len(manim_rows)}] {subj} | {n_manim} formula slides | {(row.get('question_text') or '')[:60]}")
            log.info("[DRY RUN] Done.")
            return

        # Load Manim provider
        if manim_provider == "openrouter":
            log.info(f"[Manim Retry] Using OpenRouter (cloud) — no local Ollama needed")
            manim_model_ready = True
        else:
            log.info(f"[Manim Retry] Loading local Ollama for Manim code generation...")
            try:
                manim_model_ready = await prepare_for_manim_generation()
                log.info(f"[Manim Retry] Ollama {'✓ ready' if manim_model_ready else '⚠ timed out'}")
            except Exception as e:
                log.warning(f"[Manim Retry] Ollama load failed: {e}")
                manim_model_ready = False

        manim_ok = manim_fail = 0
        run_start = time.time()

        for i, row in enumerate(manim_rows, 1):
            if _stop_requested:
                log.warning("[Manim Retry] Stop requested")
                break

            cache_id     = str(row["id"])
            slides       = row.get("presentation_slides") or []
            language     = row.get("language") or "hi-IN"
            subject_name = subject_for_row.get(cache_id, "?")
            question     = (row.get("question_text") or "")[:60]
            n_manim      = sum(1 for s in slides if s.get("visual_type") == "manim")
            audio_durations: dict = {}

            # Reconstruct audio_durations from slide_audio_urls
            audio_data = (row.get("slide_audio_urls") or {}).get("urls", [])
            for entry in audio_data:
                idx = entry.get("slideIndex")
                dur = entry.get("duration", 0)
                if idx is not None and dur:
                    audio_durations[idx] = dur

            log.info(f"")
            log.info(f"[Manim Retry] ── [{i}/{len(manim_rows)}] {subject_name}")
            log.info(f"[Manim Retry]   Q: {question}")
            log.info(f"[Manim Retry]   → Generating Manim via {manim_provider} for {n_manim} formula slide(s)...")

            try:
                manim_video_urls = await _pregen_manim_only(
                    row, slides, audio_durations, provider=manim_provider
                )
                if manim_video_urls:
                    # Save only the manim_video_urls update to DB
                    from sqlalchemy import text as _text
                    import json as _json
                    async with db_factory() as db:
                        await db.execute(_text("""
                            UPDATE teaching_qa_cache
                            SET manim_video_urls = CAST(:manim AS jsonb)
                            WHERE id = CAST(:id AS uuid)
                        """), {"manim": _json.dumps(manim_video_urls), "id": cache_id})
                        await db.commit()
                    log.info(f"[Manim Retry]   ✓ {len(manim_video_urls)} Manim video(s) saved")
                    manim_ok += 1
                else:
                    log.warning(f"[Manim Retry]   ⚠ No Manim videos generated (formula render may have failed)")
                    manim_fail += 1
            except Exception as e:
                log.error(f"[Manim Retry]   ✗ FAILED: {e}")
                manim_fail += 1

        elapsed = round(time.time() - run_start)
        log.info(f"")
        log.info(f"[Manim Retry] ══ Complete in {elapsed}s ══")
        log.info(f"[Manim Retry]   ✓ Success: {manim_ok} | ✗ Failed: {manim_fail}")
        async with db_factory() as db:
            await _print_final_report(db, all_targets)
        return

    # Normal mode: process pending rows
    if total == 0:
        log.info("All Chapter 1 questions already fully generated!")
        async with db_factory() as db:
            await _print_final_report(db, all_targets)
        return

    log.info(f"\nTotal: {total} questions to process\n")

    # Dry run
    if args.dry_run:
        log.info("[DRY RUN] Would process:")
        for i, row in enumerate(all_rows, 1):
            subj = subject_for_row.get(str(row["id"]), "?")
            has_slides = bool(row.get("presentation_slides"))
            log.info(f"  [{i}/{total}] {subj} | status={row.get('pregen_status')} | slides={'yes' if has_slides else 'no'} | {(row.get('question_text') or '')[:60]}")
        log.info("[DRY RUN] Done. No generation performed.")
        return

    # ═══════════════════════════════════════════════════════════
    # PHASE A — TEXT (Ollama)
    # ═══════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("  PHASE A -- Text Generation (Ollama)")
    log.info("  Generating slides for each question via Ollama LLM")
    log.info("  Skips questions that already have slides in DB")
    log.info("=" * 60)

    log.info("[Phase A] Step 1/3: Loading Ollama text model into GPU VRAM...")
    try:
        ready = await prepare_for_text_generation()
        log.info(f"[Phase A] Ollama model {'✓ loaded and ready' if ready else '⚠ not confirmed — will try per-question'}")
    except Exception as e:
        log.warning(f"[Phase A] Ollama warmup failed (will try per-row): {e}")

    phase_a_done: List[Dict[str, Any]] = []
    phase_a_failed: List[str] = []
    phase_a_start = time.time()

    for i, row in enumerate(all_rows, 1):
        if _stop_requested:
            log.warning("[Phase A] Stop requested — halting after this point")
            break

        cache_id     = str(row["id"])
        subject_name = subject_for_row.get(cache_id, "General")
        question     = (row.get("question_text") or "")[:70]
        slides       = row.get("presentation_slides") or []

        log.info(f"")
        log.info(f"[Phase A] ── Question {i}/{total} ── {subject_name}")
        log.info(f"[Phase A]   Q: {question}")
        log.info(f"[Phase A]   Cache ID: {cache_id[:8]}")

        if slides:
            log.info(f"[Phase A]   → SKIP: slides already exist in DB ({len(slides)} slides) — going straight to media")
            phase_a_done.append(row)
            continue

        log.info(f"[Phase A]   → Calling Ollama to generate 7 slides...")
        async with db_factory() as db:
            await _mark_processing(db, cache_id)

        q_start = time.time()
        try:
            async with db_factory() as db:
                row = await _pregen_text_only(row, db, subject_name=subject_name)
            phase_a_done.append(row)
            n_slides = len(row.get("presentation_slides") or [])
            elapsed = round(time.time() - q_start, 1)
            # Show slide titles so user can verify correctness
            for j, s in enumerate(row.get("presentation_slides") or []):
                vtype = f" [{s.get('visual_type','standard')}]" if s.get('visual_type') else ""
                log.info(f"[Phase A]     Slide {j+1}: {s.get('title','?')}{vtype}")
            log.info(f"[Phase A]   ✓ DONE: {n_slides} slides generated in {elapsed}s")
        except Exception as e:
            log.error(f"[Phase A]   ✗ FAILED: {e}")
            async with db_factory() as db:
                await _mark_failed(db, cache_id, str(e))
            phase_a_failed.append(cache_id)

    elapsed_a = round(time.time() - phase_a_start)
    log.info(f"")
    log.info(f"[Phase A] ═══ Phase A Complete in {elapsed_a}s ═══")
    log.info(f"[Phase A]   ✓ Success: {len(phase_a_done)} | ✗ Failed: {len(phase_a_failed)}")

    # ═══════════════════════════════════════════════════════════
    # PHASE B — MEDIA (Wan2GP + VoxCPM)
    # ═══════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("  PHASE B -- Media Generation (Wan2GP images + VoxCPM audio)")
    log.info("  Each slide gets: 1 AI image (Wan2GP) + 1 audio narration (VoxCPM)")
    log.info("  3 slides processed in parallel per question. Retry: 3 attempts per slide.")
    log.info("=" * 60)

    log.info("[Phase B] Step 2/3: Evicting Ollama from VRAM to free GPU memory for image/audio...")
    try:
        n = await prepare_for_media_generation()
        log.info(f"[Phase B] ✓ {n} model(s) evicted from VRAM — GPU ready for Wan2GP + VoxCPM")
    except Exception as e:
        log.warning(f"[Phase B] Eviction failed (non-fatal): {e}")

    phase_b_done: List[Dict[str, Any]] = []
    phase_b_failed: List[str] = []
    phase_b_start = time.time()

    for i, row in enumerate(phase_a_done, 1):
        if _stop_requested:
            log.warning("[Phase B] Stop requested — halting after this point")
            break

        cache_id = str(row["id"])
        slides   = row.get("presentation_slides") or []
        question = (row.get("question_text") or "")[:70]
        subject_name = subject_for_row.get(cache_id, "?")

        log.info(f"")
        log.info(f"[Phase B] ── Question {i}/{len(phase_a_done)} ── {subject_name}")
        log.info(f"[Phase B]   Q: {question}")
        log.info(f"[Phase B]   Slides to process: {len(slides)}")
        log.info(f"[Phase B]   → Submitting {len(slides)} image jobs to Wan2GP + {len(slides)} audio jobs to VoxCPM...")
        log.info(f"[Phase B]   (Each slide: image gen ~60-120s, audio gen ~30-60s — running in parallel)")

        q_start = time.time()
        try:
            async with db_factory() as db:
                enriched_slides, audio_url_list, total_duration, audio_durations, image_urls_map = \
                    await _pregen_media_only(row, slides, db=db)
            row["_enriched_slides"]  = enriched_slides
            row["_audio_url_list"]   = audio_url_list
            row["_total_duration"]   = total_duration
            row["_audio_durations"]  = audio_durations
            row["_image_urls_map"]   = image_urls_map
            phase_b_done.append(row)

            elapsed_q = round(time.time() - q_start)
            # Show per-slide results
            for j, s in enumerate(enriched_slides):
                has_img = '✓' if (s.get('infographicUrl') or '').startswith('http') else '✗'
                has_aud = '✓' if (s.get('audioUrl') or '').startswith('http') else '✗'
                dur = round(s.get('duration', 0), 1)
                log.info(f"[Phase B]     Slide {j+1}/{len(enriched_slides)}: img={has_img} audio={has_aud} ({dur}s) — {s.get('title','?')[:40]}")

            imgs_ok  = len(image_urls_map)
            audio_ok = len(audio_url_list)
            log.info(f"[Phase B]   ✓ DONE in {elapsed_q}s | Images: {imgs_ok}/{len(slides)} | Audio: {audio_ok}/{len(slides)} | Total audio: {round(total_duration,1)}s")
        except Exception as e:
            log.error(f"[Phase B]   ✗ FAILED: {e}")
            async with db_factory() as db:
                await _mark_failed(db, cache_id, str(e))
            phase_b_failed.append(cache_id)

    elapsed_b = round(time.time() - phase_b_start)
    log.info(f"")
    log.info(f"[Phase B] ═══ Phase B Complete in {elapsed_b}s ═══")
    log.info(f"[Phase B]   ✓ Success: {len(phase_b_done)} | ✗ Failed: {len(phase_b_failed)}")

    # ═══════════════════════════════════════════════════════════
    # PHASE C — MANIM + SAVE
    # ═══════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("  PHASE C -- Manim Animation + Save to DB")
    log.info("  Manim only runs for slides where LLM set visual_type='manim' (formulas)")
    log.info("  All results saved to DB with access_tier='free'")
    log.info("=" * 60)

    log.info("[Phase C] Step 3/3: Checking for formula slides that need Manim animation...")
    manim_rows_needed = any(
        any(s.get("visual_type") == "manim" for s in (r.get("_enriched_slides") or []))
        for r in phase_b_done
    )

    total_manim_slides = sum(
        sum(1 for s in (r.get("_enriched_slides") or []) if s.get("visual_type") == "manim")
        for r in phase_b_done
    )

    manim_ready = False
    if manim_rows_needed:
        log.info(f"[Phase C] Found {total_manim_slides} formula slide(s) across all questions")
        if manim_provider == "openrouter":
            log.info(f"[Phase C] Using OpenRouter (cloud) for Manim code generation — no local Ollama needed")
            manim_ready = True   # OpenRouter is always available; no GPU warmup required
        else:
            log.info(f"[Phase C] Using local Ollama — reloading model for Manim code generation...")
            try:
                manim_ready = await prepare_for_manim_generation()
                log.info(f"[Phase C] Ollama {'✓ ready — will generate Manim animations' if manim_ready else '⚠ timed out — Manim skipped, keeping static images'}")
            except Exception as e:
                log.warning(f"[Phase C] Ollama load failed (non-fatal): {e}")
    else:
        log.info("[Phase C] No formula slides found — Manim phase skipped entirely")

    save_ok = 0
    save_failed = 0
    save_start = time.time()

    for i, row in enumerate(phase_b_done, 1):
        cache_id        = str(row["id"])
        enriched_slides = row["_enriched_slides"]
        audio_url_list  = row["_audio_url_list"]
        total_duration  = row["_total_duration"]
        audio_durations = row["_audio_durations"]
        image_urls_map  = row["_image_urls_map"]
        question        = (row.get("question_text") or "")[:60]
        subject_name    = subject_for_row.get(cache_id, "?")

        log.info(f"")
        log.info(f"[Save] ── Question {i}/{len(phase_b_done)} ── {subject_name}")
        log.info(f"[Save]   Q: {question}")

        # Manim for this row
        manim_video_urls: dict = {}
        manim_slides_here = [s for s in enriched_slides if s.get("visual_type") == "manim"]
        if manim_slides_here and manim_ready:
            log.info(f"[Phase C]   → Generating Manim via {manim_provider} for {len(manim_slides_here)} formula slide(s)...")
            try:
                manim_video_urls = await _pregen_manim_only(
                    row, enriched_slides, audio_durations, provider=manim_provider
                )
                log.info(f"[Phase C]   ✓ {len(manim_video_urls)} Manim video(s) rendered")
            except Exception as e:
                log.warning(f"[Phase C]   ⚠ Manim failed (keeping static image): {e}")
        elif manim_slides_here:
            log.info(f"[Phase C]   {len(manim_slides_here)} formula slide(s) — Manim skipped (Ollama not ready)")

        log.info(f"[Save]   → Writing to DB: access_tier=free | {len(enriched_slides)} slides | {round(total_duration,1)}s audio | {len(manim_video_urls)} manim")
        try:
            async with db_factory() as db:
                await _save_result(
                    db, row, enriched_slides, audio_url_list,
                    total_duration, manim_video_urls, image_urls_map,
                )
            save_ok += 1
            log.info(f"[Save]   ✓ Saved and marked done")
        except Exception as e:
            log.error(f"[Save]   ✗ FAILED: {e}")
            save_failed += 1

    elapsed_total = round(time.time() - phase_a_start)
    # Final summary
    log.info(f"")
    log.info(f"[ch1_pregen] ══════════════ RUN SUMMARY ══════════════")
    log.info(f"[ch1_pregen]   Total elapsed time: {elapsed_total}s ({round(elapsed_total/60, 1)} min)")
    log.info(f"[ch1_pregen]   Phase A (text):  {len(phase_a_done):>3} ok  {len(phase_a_failed):>3} failed")
    log.info(f"[ch1_pregen]   Phase B (media): {len(phase_b_done):>3} ok  {len(phase_b_failed):>3} failed")
    log.info(f"[ch1_pregen]   Save to DB:      {save_ok:>3} ok  {save_failed:>3} failed")

    async with db_factory() as db:
        await _print_final_report(db, all_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 1 bulk generation script for GPU server")
    parser.add_argument("--subject", type=str, default=None,
                        help="Only run one subject: social, science, math (default: all 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without running")
    parser.add_argument(
        "--manim-provider",
        type=str,
        default="openrouter",
        choices=["openrouter", "local"],
        help="LLM provider for Manim code generation: 'openrouter' (cloud, better quality, default) or 'local' (Ollama on GPU server)"
    )
    parser.add_argument(
        "--retry-manim",
        action="store_true",
        help="Only re-run Phase C (Manim) for done rows that have formula slides but missing Manim videos"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
