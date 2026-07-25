import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text

SUBJECT_NAME = "Social Science"
CHAPTER_NUM  = 1

async def main():
    async with AsyncSessionLocal() as db:
        # Get IDs
        subj = (await db.execute(text("SELECT subject_id FROM subjects WHERE name = :n"), {"n": SUBJECT_NAME})).first()
        if not subj:
            print("Subject not found"); return
        sid = str(subj.subject_id)

        ch = (await db.execute(text(
            "SELECT id, title FROM chapters WHERE subject_id = :sid AND chapter_number = :n"
        ), {"sid": sid, "n": CHAPTER_NUM})).first()
        if not ch:
            print("Chapter not found"); return
        cid = str(ch.id)

        print(f"=== {SUBJECT_NAME} — Ch{CHAPTER_NUM}: {ch.title} ===\n")

        # All questions
        questions = (await db.execute(text("""
            SELECT q.id, q.question_text, q.is_pregen_done,
                   c.id AS cache_id,
                   c.pregen_status,
                   c.presentation_slides IS NOT NULL AS has_slides,
                   jsonb_array_length(c.presentation_slides) AS slide_count,
                   c.image_urls IS NOT NULL AS has_image_urls,
                   c.slide_audio_urls IS NOT NULL AS has_audio_urls,
                   c.pregen_completed_at
            FROM questions q
            LEFT JOIN teaching_qa_cache c ON c.question_hash = md5(lower(trim(q.question_text)))
                AND c.subject_id = :sid
            WHERE q.chapter_id = :cid
            ORDER BY q.created_at ASC
        """), {"sid": sid, "cid": cid})).fetchall()

        done = sum(1 for r in questions if r.pregen_status == 'done')
        pending = sum(1 for r in questions if r.pregen_status == 'pending')
        processing = sum(1 for r in questions if r.pregen_status == 'processing')
        failed = sum(1 for r in questions if r.pregen_status == 'failed')
        no_cache = sum(1 for r in questions if r.cache_id is None)
        has_slides = sum(1 for r in questions if r.has_slides)
        has_imgs = sum(1 for r in questions if r.has_image_urls)
        has_aud = sum(1 for r in questions if r.has_audio_urls)

        print(f"Total questions : {len(questions)}")
        print(f"  ✅ done        : {done}")
        print(f"  ⏳ pending     : {pending}")
        print(f"  🔄 processing  : {processing}")
        print(f"  ❌ failed      : {failed}")
        print(f"  🚫 no cache row: {no_cache}")
        print(f"\n  Has slide text : {has_slides}")
        print(f"  Has image URLs : {has_imgs}")
        print(f"  Has audio URLs : {has_aud}")

        print(f"\n{'─'*80}")
        print(f"{'#':<4} {'Status':<12} {'Slides':<7} {'Imgs':<5} {'Aud':<5} Question")
        print(f"{'─'*80}")
        for i, r in enumerate(questions, 1):
            status = r.pregen_status or "NO ROW"
            slides = str(r.slide_count) if r.slide_count else "-"
            imgs   = "✓" if r.has_image_urls else "✗"
            aud    = "✓" if r.has_audio_urls else "✗"
            q      = (r.question_text or "")[:55]
            print(f"{i:<4} {status:<12} {slides:<7} {imgs:<5} {aud:<5} {q}")

asyncio.run(main())
