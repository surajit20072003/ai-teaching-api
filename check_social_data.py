import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        # Get Social Science subject_id
        subj = (await db.execute(text("SELECT subject_id, name FROM subjects WHERE name ILIKE '%social%'"))).fetchall()
        print("=== Subjects ===")
        for s in subj:
            print(f"  {s.name} | {s.subject_id}")

        if not subj:
            print("No social science subject found")
            return

        sid = str(subj[0].subject_id)

        # Chapters
        chapters = (await db.execute(text("""
            SELECT chapter_number, title,
                   (SELECT COUNT(*) FROM questions q WHERE q.chapter_id = c.id) AS q_count
            FROM chapters c
            WHERE c.subject_id = :sid
            ORDER BY chapter_number
        """), {"sid": sid})).fetchall()

        print(f"\n=== Chapters ({len(chapters)} total) ===")
        total_q = 0
        for ch in chapters:
            print(f"  Ch{ch.chapter_number}: {ch.title[:55]} | {ch.q_count} questions")
            total_q += ch.q_count
        print(f"\n  Total questions: {total_q}")

        # Cache status
        cache = (await db.execute(text("""
            SELECT pregen_status, COUNT(*) AS cnt
            FROM teaching_qa_cache
            WHERE subject_id = :sid
            GROUP BY pregen_status
            ORDER BY cnt DESC
        """), {"sid": sid})).fetchall()

        print(f"\n=== Cache Status (teaching_qa_cache) ===")
        total_cache = 0
        for row in cache:
            print(f"  {row.pregen_status}: {row.cnt}")
            total_cache += row.cnt
        print(f"  Total cached rows: {total_cache}")

        # Media completion
        media = (await db.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE pregen_status = 'done') AS done,
                COUNT(*) FILTER (WHERE presentation_slides IS NOT NULL AND presentation_slides->>0 != '') AS has_slides,
                COUNT(*) FILTER (WHERE presentation_slides->>0::text LIKE '%infographicUrl%') AS has_images,
                COUNT(*) FILTER (WHERE presentation_slides->>0::text LIKE '%audioUrl%') AS has_audio
            FROM teaching_qa_cache
            WHERE subject_id = :sid
        """), {"sid": sid})).first()

        print(f"\n=== Media Completion ===")
        print(f"  Done (pregen_status=done): {media.done}")
        print(f"  Has slides JSON:           {media.has_slides}")
        print(f"  Has images (infographicUrl): {media.has_images}")
        print(f"  Has audio (audioUrl):        {media.has_audio}")

asyncio.run(main())
