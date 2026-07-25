import asyncio
from sqlalchemy import text
from db.models import AsyncSessionLocal

async def main():
    async with AsyncSessionLocal() as db:
        res = await db.execute(text("SELECT COUNT(*) FROM text_answer_cache"))
        count = res.scalar()
        print(f"Total records in text_answer_cache: {count}")

asyncio.run(main())
