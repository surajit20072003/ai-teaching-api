"""
check_questions_remaining.py
Show question counts per subject + chapter, excluding chapter 1.
"""
import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT
                sub.name            AS subject_name,
                c.chapter_number    AS chapter_num,
                c.title             AS chapter_title,
                COUNT(q.id)         AS question_count
            FROM questions q
            JOIN chapters c  ON q.chapter_id  = c.id
            JOIN subjects sub ON c.subject_id = sub.subject_id
            WHERE c.chapter_number != 1
            GROUP BY sub.name, c.chapter_number, c.title
            ORDER BY sub.name, c.chapter_number
        """))).fetchall()

        if not rows:
            print("No questions found (or subject names differ).")
        else:
            current_subject = None
            subject_total = 0
            grand_total = 0
            for r in rows:
                if r.subject_name != current_subject:
                    if current_subject:
                        print(f"  {'─'*50}")
                        print(f"  TOTAL: {subject_total} questions\n")
                    current_subject = r.subject_name
                    subject_total = 0
                    print(f"\n{'='*60}")
                    print(f"  {r.subject_name.upper()}")
                    print(f"{'='*60}")
                print(f"  Ch{r.chapter_num:>2}: {r.chapter_title[:45]:<45} | {r.question_count:>3} Qs")
                subject_total += r.question_count
                grand_total   += r.question_count
            print(f"  {'─'*50}")
            print(f"  TOTAL: {subject_total} questions\n")
            print(f"{'='*60}")
            print(f"  GRAND TOTAL (all subjects, excl Ch1): {grand_total} questions")
            print(f"{'='*60}")

asyncio.run(main())
