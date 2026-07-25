"""
Full deep-check for Social Science Chapter 1.
Checks: PostgreSQL DB, Local disk (slides cache + job files), Redis L1 cache.
Uses the REAL hash_question() function (SHA-256, not md5).
"""
import asyncio, os, json, glob
from db.models import AsyncSessionLocal
from sqlalchemy import text
from core.cache import hash_question, get_redis, cache_key

SUBJECT_NAME = "Social Science"
CHAPTER_NUM  = 1
BASE_PATH    = os.getenv("LOCAL_STORAGE_BASE", "/sdb-disk/ai-teaching")
SUBJECTS_PATH = f"{BASE_PATH}/subjects"

async def main():
    async with AsyncSessionLocal() as db:
        # ── Resolve IDs ────────────────────────────────────────────────────────
        subj = (await db.execute(text("SELECT subject_id FROM subjects WHERE name = :n"), {"n": SUBJECT_NAME})).first()
        if not subj:
            print("Subject not found"); return
        sid = str(subj.subject_id)
        print(f"Subject ID: {sid}\n")

        ch = (await db.execute(text(
            "SELECT id, title FROM chapters WHERE subject_id = :sid AND chapter_number = :n"
        ), {"sid": sid, "n": CHAPTER_NUM})).first()
        if not ch:
            print("Chapter not found"); return
        cid = str(ch.id)
        print(f"=== {SUBJECT_NAME} — Ch{CHAPTER_NUM}: {ch.title} ===\n")

        # ── Get all questions ──────────────────────────────────────────────────
        questions = (await db.execute(text("""
            SELECT id, question_text FROM questions
            WHERE chapter_id = :cid ORDER BY created_at ASC
        """), {"cid": cid})).fetchall()

        print(f"Questions in chapter: {len(questions)}\n")

        # ── Get all cache rows for this chapter (by chapter_id) ────────────────
        cache_rows = (await db.execute(text("""
            SELECT id, question_hash, question_text, pregen_status,
                   presentation_slides IS NOT NULL AS has_slides,
                   jsonb_array_length(presentation_slides) AS slide_count,
                   image_urls IS NOT NULL AS has_image_urls,
                   slide_audio_urls IS NOT NULL AS has_audio_urls,
                   total_duration_seconds,
                   pregen_completed_at
            FROM teaching_qa_cache
            WHERE subject_id = :sid AND chapter_id = :cid
            ORDER BY created_at ASC
        """), {"sid": sid, "cid": cid})).fetchall()

        # Also look up by question hash (in case chapter_id wasn't set)
        q_hashes = [hash_question(q.question_text or "") for q in questions]
        cache_by_hash_rows = (await db.execute(text("""
            SELECT id, question_hash, question_text, pregen_status,
                   presentation_slides IS NOT NULL AS has_slides,
                   jsonb_array_length(presentation_slides) AS slide_count,
                   image_urls IS NOT NULL AS has_image_urls,
                   slide_audio_urls IS NOT NULL AS has_audio_urls,
                   total_duration_seconds,
                   pregen_completed_at
            FROM teaching_qa_cache
            WHERE subject_id = :sid
              AND question_hash = ANY(:hashes)
        """), {"sid": sid, "hashes": q_hashes})).fetchall()

        # Merge: use chapter_id rows + hash-matched rows (deduplicated by id)
        all_cache = {str(r.id): r for r in cache_rows}
        for r in cache_by_hash_rows:
            all_cache[str(r.id)] = r
        all_cache = list(all_cache.values())

        print(f"─── PostgreSQL (teaching_qa_cache) ───")
        print(f"  Found by chapter_id   : {len(cache_rows)}")
        print(f"  Found by hash lookup  : {len(cache_by_hash_rows)}")
        print(f"  Total unique cache rows: {len(all_cache)}")

        status_counts = {}
        for r in all_cache:
            s = r.pregen_status or "none"
            status_counts[s] = status_counts.get(s, 0) + 1
        for k, v in sorted(status_counts.items()):
            print(f"    {k}: {v}")

        has_slides = sum(1 for r in all_cache if r.has_slides)
        has_imgs   = sum(1 for r in all_cache if r.has_image_urls)
        has_aud    = sum(1 for r in all_cache if r.has_audio_urls)
        done_count = sum(1 for r in all_cache if r.pregen_status == 'done')
        print(f"  Has slide text  : {has_slides}/{len(all_cache)}")
        print(f"  Has image_urls  : {has_imgs}/{len(all_cache)}")
        print(f"  Has audio_urls  : {has_aud}/{len(all_cache)}")
        print(f"  Fully done      : {done_count}/{len(all_cache)}")

        # ── Local Disk: slides cache ──────────────────────────────────────────
        slides_dir = f"{SUBJECTS_PATH}/{sid}/cache/slides"
        print(f"\n─── Local Disk: {slides_dir} ───")
        disk_slide_files = []
        if os.path.isdir(slides_dir):
            disk_slide_files = glob.glob(f"{slides_dir}/*.json")
            print(f"  Total slide JSON files on disk: {len(disk_slide_files)}")
            # Check which belong to this chapter's questions
            ch_hashes = set(q_hashes)
            ch_disk = [f for f in disk_slide_files if os.path.basename(f).replace(".json","") in ch_hashes]
            print(f"  Matching Ch{CHAPTER_NUM} questions on disk: {len(ch_disk)}")
            for f in ch_disk:
                size = os.path.getsize(f)
                print(f"    {os.path.basename(f)} ({size:,} bytes)")
        else:
            print(f"  Directory does not exist: {slides_dir}")

        # ── Local Disk: jobs (images + audio) ─────────────────────────────────
        jobs_dir = f"{SUBJECTS_PATH}/{sid}/cache/jobs"
        print(f"\n─── Local Disk: {jobs_dir} ───")
        if os.path.isdir(jobs_dir):
            job_ids = os.listdir(jobs_dir)
            print(f"  Total job folders: {len(job_ids)}")

            # Match to cache row IDs from this chapter
            ch_cache_ids = {str(r.id) for r in all_cache}
            ch_jobs = [j for j in job_ids if j in ch_cache_ids]
            print(f"  Jobs matching Ch{CHAPTER_NUM}: {len(ch_jobs)}")
            for jid in ch_jobs:
                jpath = f"{jobs_dir}/{jid}"
                imgs  = glob.glob(f"{jpath}/images/*.png")
                wavs  = glob.glob(f"{jpath}/audio/**/*.wav", recursive=True)
                print(f"    {jid[:8]}: {len(imgs)} images, {len(wavs)} audio files")
        else:
            print(f"  Directory does not exist: {jobs_dir}")

        # ── Redis L1 ──────────────────────────────────────────────────────────
        print(f"\n─── Redis L1 Cache ───")
        r = get_redis()
        if not r:
            print("  Redis not available")
        else:
            redis_hits = 0
            for qhash in q_hashes:
                key = cache_key(qhash, sid)
                val = await r.get(key)
                if val:
                    redis_hits += 1
            print(f"  Ch{CHAPTER_NUM} questions in Redis: {redis_hits}/{len(questions)}")

        # ── Per-question detail table ─────────────────────────────────────────
        hash_to_cache = {r.question_hash: r for r in all_cache}
        print(f"\n{'─'*90}")
        print(f"{'#':<4} {'DB Status':<12} {'Slides':<7} {'Imgs':<5} {'Aud':<5} {'Disk':<5} Question")
        print(f"{'─'*90}")
        for i, q in enumerate(questions, 1):
            qh   = hash_question(q.question_text or "")
            row  = hash_to_cache.get(qh)
            status = row.pregen_status if row else "NO ROW"
            slides = str(row.slide_count) if row and row.slide_count else "-"
            imgs   = "✓" if row and row.has_image_urls else "✗"
            aud    = "✓" if row and row.has_audio_urls else "✗"
            disk_f = f"{slides_dir}/{qh}.json"
            disk   = "✓" if os.path.exists(disk_f) else "✗"
            qtxt   = (q.question_text or "")[:52]
            print(f"{i:<4} {status:<12} {slides:<7} {imgs:<5} {aud:<5} {disk:<5} {qtxt}")

asyncio.run(main())
