import asyncio
from sqlalchemy import text
from db.models import AsyncSessionLocal

async def main():
    async with AsyncSessionLocal() as db:
        res1 = await db.execute(text("SELECT is_pregen_done, count(*) FROM questions GROUP BY is_pregen_done"))
        print("Questions Table:")
        for row in res1.fetchall():
            print(f"  is_pregen_done={row.is_pregen_done}: {row.count}")
            
        res2 = await db.execute(text("SELECT pregen_status, count(*) FROM teaching_qa_cache GROUP BY pregen_status"))
        print("\nTeaching QA Cache Table:")
        for row in res2.fetchall():
            print(f"  pregen_status={row.pregen_status}: {row.count}")

asyncio.run(main())
