#!/usr/bin/env python3
"""
fix_stale_l2_cache.py
─────────────────────
ONE-TIME FIX: Overwrites all stale L2 local-disk JSON cache files with the
correct, fully-completed data (Backblaze B2 audio + image URLs embedded inside
each slide object).

Root cause:
  The pregen pipeline wrote a Phase-A-only cache file (no audio/images) to disk
  early in the process. After Phase B/C completed and saved full media URLs to
  teaching_qa_cache in Postgres, the disk file was never updated.
  Result: every student request hit L2 (stale file) and got slides with no audio
  or images -- even though ~600 questions were fully completed in the DB.

What this script does:
  1. Fetches all teaching_qa_cache rows where pregen_status = 'done'
  2. For each row, builds the correct payload (same shape as /ai-teaching-assistant response)
  3. Overwrites the L2 disk JSON file at:
       {LOCAL_STORAGE_BASE}/subjects/{subject_id}/cache/slides/{question_hash}.json
  4. Deletes the stale L1 Redis key so the next request reads from the fresh L2 file

Run inside the container:
  docker exec ai-teaching-api python3 /app/fix_stale_l2_cache.py

Safe to re-run: write_slide_cache is an atomic overwrite, delete_from_cache is a no-op if key doesn't exist.
"""

import asyncio
import sys
import os

sys.path.insert(0, "/app")

from sqlalchemy import text
from db.models import AsyncSessionLocal
from core.local_storage import write_slide_cache
from core.cache import delete_from_cache, hash_question


async def main():
    print("=" * 60)
    print("Fix: Overwrite stale L2 disk cache from completed DB rows")
    print("=" * 60)

    async with AsyncSessionLocal() as db:
        res = await db.execute(text("""
            SELECT
                id::text          AS cache_id,
                question_hash,
                question_text,
                subject_id,
                access_tier,
                presentation_slides,
                slide_audio_urls,
                image_urls,
                manim_video_urls,
                latex_formulas,
                total_duration_seconds
            FROM teaching_qa_cache
            WHERE pregen_status = 'done'
        """))
        rows = res.fetchall()

    print(f"Found {len(rows)} completed rows in teaching_qa_cache\n")

    refreshed  = 0
    skipped    = 0
    failed     = 0
    no_audio   = 0
    no_image   = 0

    for r in rows:
        slides = r.presentation_slides or []
        if not slides:
            skipped += 1
            continue

        # Check if any slide has audio or image embedded
        has_audio = any(
            s.get("audioUrl") or s.get("audioLocalPath")
            for s in slides
        )
        has_image = any(
            s.get("infographicUrl") or s.get("infographicLocalPath")
            for s in slides
        )

        if not has_audio:
            no_audio += 1
        if not has_image:
            no_image += 1

        # Build slide_audio_urls list from slides (for legacy slideAudioUrls field)
        audio_url_list = []
        for i, s in enumerate(slides):
            aud = s.get("audioUrl")
            if aud and aud.startswith("http"):
                audio_url_list.append({
                    "slideIndex": i,
                    "audioUrl":   aud,
                    "duration":   s.get("duration", 0),
                })

        # Also check the separate slide_audio_urls column as fallback
        if not audio_url_list and r.slide_audio_urls:
            aud_col = r.slide_audio_urls
            audio_url_list = aud_col.get("urls", []) if isinstance(aud_col, dict) else []

        completed_data = {
            "cached":               True,
            "cache_layer":          "L2_local_disk",
            "cache_id":             r.cache_id,
            "presentationSlides":   slides,
            "latexFormulas":        r.latex_formulas,
            "slideAudioUrls":       audio_url_list,
            "totalDurationSeconds": r.total_duration_seconds,
            "manimVideoUrls":       r.manim_video_urls or {},
            "imageUrls":            r.image_urls or {},
            "access_tier":          r.access_tier or "pro",
        }

        try:
            # Overwrite L2 disk cache
            ok = await write_slide_cache(r.subject_id, r.question_hash, completed_data)
            if not ok:
                print(f"  WARN write_slide_cache blocked: {r.cache_id[:8]}")
                failed += 1
                continue

            # Invalidate L1 Redis
            await delete_from_cache(r.question_hash, r.subject_id)

            refreshed += 1
            audio_count = len(audio_url_list)
            img_count = sum(1 for s in slides if (s.get("infographicUrl") or "").startswith("http"))
            print(f"  OK {r.cache_id[:8]} | {r.access_tier:<4} | slides={len(slides)} audio={audio_count} images={img_count} | {r.question_text[:55]}")

        except Exception as e:
            print(f"  FAIL {r.cache_id[:8]}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Done!")
    print(f"  Refreshed (L2 overwritten + L1 invalidated): {refreshed}")
    print(f"  Skipped (no slides in DB):                   {skipped}")
    print(f"  Failed:                                      {failed}")
    print(f"  Rows with no audio URLs found:               {no_audio}")
    print(f"  Rows with no image URLs found:               {no_image}")
    print("=" * 60)
    print()
    print("All student requests will now get correct audio + images from the L2 disk cache.")


if __name__ == "__main__":
    asyncio.run(main())
