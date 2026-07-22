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
    "math":         "Mathematics",
    "maths":        "Mathematics",
    "mathematics":  "Mathematics",
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
    for name in ["Social Science", "Science", "Mathematics"]:
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

    log.info("=" * 60)
    log.info("  ch1_pregen.py -- Chapter 1 Bulk Generation")
    log.info(f"  access_tier=free | Log: {_LOG_FILE.name}")
    log.info("=" * 60)

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
    log.info("=" * 60)

    try:
        ready = await prepare_for_text_generation()
        log.info(f"[Phase A] Ollama {'ready' if ready else 'not confirmed ready — proceeding anyway'}")
    except Exception as e:
        log.warning(f"[Phase A] Ollama warmup failed (will try per-row): {e}")

    phase_a_done: List[Dict[str, Any]] = []
    phase_a_failed: List[str] = []

    for i, row in enumerate(all_rows, 1):
        if _stop_requested:
            log.warning("[Phase A] Stop requested")
            break

        cache_id     = str(row["id"])
        subject_name = subject_for_row.get(cache_id, "General")
        question     = (row.get("question_text") or "")[:70]
        slides       = row.get("presentation_slides") or []

        if slides:
            log.info(f"[Phase A] [{i}/{total}] SKIP (slides exist) | {question}")
            phase_a_done.append(row)
            continue

        log.info(f"[Phase A] [{i}/{total}] Generating | {question}")
        async with db_factory() as db:
            await _mark_processing(db, cache_id)

        try:
            async with db_factory() as db:
                row = await _pregen_text_only(row, db, subject_name=subject_name)
            phase_a_done.append(row)
            n_slides = len(row.get("presentation_slides") or [])
            log.info(f"[Phase A] OK {cache_id[:8]} | {n_slides} slides")
        except Exception as e:
            log.error(f"[Phase A] FAIL {cache_id[:8]}: {e}")
            async with db_factory() as db:
                await _mark_failed(db, cache_id, str(e))
            phase_a_failed.append(cache_id)

    log.info(f"[Phase A] Done: {len(phase_a_done)} ok, {len(phase_a_failed)} failed")

    # ═══════════════════════════════════════════════════════════
    # PHASE B — MEDIA (Wan2GP + VoxCPM)
    # ═══════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("  PHASE B -- Media Generation (images + audio)")
    log.info("=" * 60)

    try:
        n = await prepare_for_media_generation()
        log.info(f"[Phase B] Evicted {n} model(s) from VRAM")
    except Exception as e:
        log.warning(f"[Phase B] Eviction failed (non-fatal): {e}")

    phase_b_done: List[Dict[str, Any]] = []
    phase_b_failed: List[str] = []

    for i, row in enumerate(phase_a_done, 1):
        if _stop_requested:
            log.warning("[Phase B] Stop requested")
            break

        cache_id = str(row["id"])
        slides   = row.get("presentation_slides") or []
        question = (row.get("question_text") or "")[:70]

        log.info(f"[Phase B] [{i}/{len(phase_a_done)}] {question}")

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
            log.info(f"[Phase B] OK {cache_id[:8]} | {round(total_duration,1)}s | {len(image_urls_map)} imgs")
        except Exception as e:
            log.error(f"[Phase B] FAIL {cache_id[:8]}: {e}")
            async with db_factory() as db:
                await _mark_failed(db, cache_id, str(e))
            phase_b_failed.append(cache_id)

    log.info(f"[Phase B] Done: {len(phase_b_done)} ok, {len(phase_b_failed)} failed")

    # ═══════════════════════════════════════════════════════════
    # PHASE C — MANIM + SAVE
    # ═══════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("  PHASE C -- Manim + Save Results")
    log.info("=" * 60)

    # Check if any row has formula slides
    manim_rows_needed = any(
        any(s.get("visual_type") == "manim" for s in (r.get("_enriched_slides") or []))
        for r in phase_b_done
    )

    manim_ready = False
    if manim_rows_needed:
        log.info("[Phase C] Formula slides found — loading Ollama for Manim code generation...")
        try:
            manim_ready = await prepare_for_manim_generation()
            log.info(f"[Phase C] Ollama {'ready' if manim_ready else 'timed out — Manim skipped'}")
        except Exception as e:
            log.warning(f"[Phase C] Ollama load failed (non-fatal): {e}")
    else:
        log.info("[Phase C] No formula slides in any row — skipping Manim entirely")

    save_ok = 0
    save_failed = 0

    for i, row in enumerate(phase_b_done, 1):
        cache_id        = str(row["id"])
        enriched_slides = row["_enriched_slides"]
        audio_url_list  = row["_audio_url_list"]
        total_duration  = row["_total_duration"]
        audio_durations = row["_audio_durations"]
        image_urls_map  = row["_image_urls_map"]
        question        = (row.get("question_text") or "")[:60]

        log.info(f"[Save] [{i}/{len(phase_b_done)}] {question}")

        # Manim for this row (only if LLM flagged formula slides)
        manim_video_urls: dict = {}
        manim_slides_here = [s for s in enriched_slides if s.get("visual_type") == "manim"]
        if manim_slides_here and manim_ready:
            log.info(f"[Phase C] Generating Manim for {len(manim_slides_here)} formula slide(s)...")
            try:
                manim_video_urls = await _pregen_manim_only(
                    row, enriched_slides, audio_durations, provider="local"
                )
                log.info(f"[Phase C] {len(manim_video_urls)} manim video(s) generated")
            except Exception as e:
                log.warning(f"[Phase C] Manim failed (non-fatal, keeping static image): {e}")

        try:
            async with db_factory() as db:
                await _save_result(
                    db, row, enriched_slides, audio_url_list,
                    total_duration, manim_video_urls, image_urls_map,
                )
            save_ok += 1
        except Exception as e:
            log.error(f"[Save] FAIL {cache_id[:8]}: {e}")
            save_failed += 1

    # Final report
    log.info(f"\n[ch1_pregen] Summary:")
    log.info(f"  Phase A (text):  {len(phase_a_done)} ok / {len(phase_a_failed)} failed")
    log.info(f"  Phase B (media): {len(phase_b_done)} ok / {len(phase_b_failed)} failed")
    log.info(f"  Save:            {save_ok} ok / {save_failed} failed")

    async with db_factory() as db:
        await _print_final_report(db, all_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 1 bulk generation script for GPU server")
    parser.add_argument("--subject", type=str, default=None,
                        help="Only run one subject: social, science, math (default: all 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without running")
    args = parser.parse_args()
    asyncio.run(main(args))
