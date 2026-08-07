import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT 
                sub.name AS subject, 
                c.chapter_number, 
                COUNT(q.id) AS total_qs,
                SUM(CASE WHEN q.is_pregen_done THEN 1 ELSE 0 END) AS pregen_done,
                COUNT(cache.id) AS cache_rows
            FROM questions q
            JOIN chapters c ON q.chapter_id = c.id
            JOIN subjects sub ON c.subject_id = sub.subject_id
            LEFT JOIN teaching_qa_cache cache ON q.id::text = cache.id::text
            WHERE LOWER(sub.name) IN ('maths', 'science', 'social science')
            GROUP BY sub.name, c.chapter_number
            ORDER BY sub.name, c.chapter_number
        """))).fetchall()
        
        current_sub = None
        for r in rows:
            if current_sub != r.subject:
                print(f"\n=== {r.subject.upper()} ===")
                current_sub = r.subject
            print(f"Ch {r.chapter_number:>2} | Total Qs: {r.total_qs:>3} | Pregen Done: {r.pregen_done:>3} | Cache Rows: {r.cache_rows:>3}")

asyncio.run(main())
