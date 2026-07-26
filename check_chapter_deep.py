"""
Full deep-check for any subject + chapter.
Usage:
  python3 check_chapter_deep.py --subject math --chapter 1
  python3 check_chapter_deep.py --subject social --chapter 3
  python3 check_chapter_deep.py --subject science --chapter 2
"""
import asyncio, os, json, glob, sys, argparse
from db.models import AsyncSessionLocal
from sqlalchemy import text
from core.cache import hash_question, get_redis, cache_key

SUBJECT_ALIASES = {
    "social":      "Social Science",
    "science":     "Science",
    "math":        "Maths",
    "maths":       "Maths",
    "mathematics": "Maths",
}
BASE_PATH    = os.getenv("LOCAL_STORAGE_BASE", "/sdb-disk/ai-teaching")
SUBJECTS_PATH = f"{BASE_PATH}/subjects"

async def main():
    async with AsyncSessionLocal() as db:
        subj = (await db.execute(text("SELECT subject_id FROM subjects WHERE name = :n"), {"n": SUBJECT_NAME})).first()
        if not subj:
            print(f"Subject '{SUBJECT_NAME}' not found"); return
        sid = str(subj.subject_id)
        print(f"Subject ID: {sid}\n")

        ch = (await db.execute(text(
            "SELECT id, title FROM chapters WHERE subject_id = :sid AND chapter_number = :n"
        ), {"sid": sid, "n": CHAPTER_NUM})).first()
        if not ch:
            print(f"Chapter {CHAPTER_NUM} not found"); return
        cid = str(ch.id)
        print(f"=== {SUBJECT_NAME} — Ch{CHAPTER_NUM}: {ch.title} ===\n")

        questions = (await db.execute(text("""
            SELECT id, question_text FROM questions
            WHERE chapter_id = :cid ORDER BY created_at ASC
        """), {"cid": cid})).fetchall()
        print(f"Questions in chapter: {len(questions)}\n")

        q_hashes = [hash_question(q.question_text or "") for q in questions]

        # DB lookup by chapter_id + by hash
        cache_by_cid = (await db.execute(text("""
            SELECT id, question_hash, pregen_status,
                   presentation_slides IS NOT NULL AS has_slides,
                   jsonb_array_length(presentation_slides) AS slide_count,
                   image_urls IS NOT NULL AS has_image_urls,
                   slide_audio_urls IS NOT NULL AS has_audio_urls
            FROM teaching_qa_cache
            WHERE subject_id = :sid AND chapter_id = :cid
        """), {"sid": sid, "cid": cid})).fetchall()

        cache_by_hash = (await db.execute(text("""
            SELECT id, question_hash, pregen_status,
                   presentation_slides IS NOT NULL AS has_slides,
                   jsonb_array_length(presentation_slides) AS slide_count,
                   image_urls IS NOT NULL AS has_image_urls,
                   slide_audio_urls IS NOT NULL AS has_audio_urls
            FROM teaching_qa_cache
            WHERE subject_id = :sid AND question_hash = ANY(:hashes)
        """), {"sid": sid, "hashes": q_hashes})).fetchall()

        all_cache = {str(r.id): r for r in cache_by_cid}
        for r in cache_by_hash:
            all_cache[str(r.id)] = r
        all_cache = list(all_cache.values())

        print(f"─── PostgreSQL (teaching_qa_cache) ───")
        print(f"  Found by chapter_id  : {len(cache_by_cid)}")
        print(f"  Found by hash lookup : {len(cache_by_hash)}")
        print(f"  Total unique rows    : {len(all_cache)}")

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
        print(f"  Has slide text   : {has_slides}/{len(all_cache)}")
        print(f"  Has image_urls   : {has_imgs}/{len(all_cache)}")
        print(f"  Has audio_urls   : {has_aud}/{len(all_cache)}")
        print(f"  Fully done       : {done_count}/{len(all_cache)}")

        # Local disk — slides
        slides_dir = f"{SUBJECTS_PATH}/{sid}/cache/slides"
        print(f"\n─── Local Disk: slides JSON ───")
        disk_slide_files = glob.glob(f"{slides_dir}/*.json") if os.path.isdir(slides_dir) else []
        ch_hashes = set(q_hashes)
        ch_disk = [f for f in disk_slide_files if os.path.basename(f).replace(".json","") in ch_hashes]
        print(f"  Total slide JSON on disk : {len(disk_slide_files)}")
        print(f"  Matching Ch{CHAPTER_NUM} on disk  : {len(ch_disk)}")

        # Local disk — jobs
        jobs_dir = f"{SUBJECTS_PATH}/{sid}/cache/jobs"
        print(f"\n─── Local Disk: job folders ───")
        if os.path.isdir(jobs_dir):
            job_ids = os.listdir(jobs_dir)
            ch_cache_ids = {str(r.id) for r in all_cache}
            ch_jobs = [j for j in job_ids if j in ch_cache_ids]
            print(f"  Total job folders        : {len(job_ids)}")
            print(f"  Jobs matching Ch{CHAPTER_NUM}    : {len(ch_jobs)}")
            total_imgs = total_wavs = 0
            for jid in ch_jobs:
                jpath = f"{jobs_dir}/{jid}"
                imgs = glob.glob(f"{jpath}/images/*.png")
                wavs = glob.glob(f"{jpath}/audio/**/*.wav", recursive=True)
                total_imgs += len(imgs)
                total_wavs += len(wavs)
            if ch_jobs:
                print(f"  Total images in jobs     : {total_imgs}")
                print(f"  Total audio in jobs      : {total_wavs}")
        else:
            print(f"  jobs dir not found: {jobs_dir}")

        # Redis
        print(f"\n─── Redis L1 Cache ───")
        r = get_redis()
        if not r:
            print("  Redis not available")
        else:
            redis_hits = 0
            for qh in q_hashes:
                val = await r.get(cache_key(qh, sid))
                if val:
                    redis_hits += 1
            print(f"  Ch{CHAPTER_NUM} questions in Redis: {redis_hits}/{len(questions)}")

        # Per-question table
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep check for a chapter across all cache layers")
    parser.add_argument("--subject", default="math", help="science | social | math (default: math)")
    parser.add_argument("--chapter", type=int, default=1, help="Chapter number (default: 1)")
    args = parser.parse_args()
    SUBJECT_NAME = SUBJECT_ALIASES.get(args.subject.lower(), args.subject)
    CHAPTER_NUM  = args.chapter
    asyncio.run(main())
