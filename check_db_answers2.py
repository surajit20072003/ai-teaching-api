import asyncio
from sqlalchemy import text
from db.models import AsyncSessionLocal

async def main():
    async with AsyncSessionLocal() as db:
        res = await db.execute(text("SELECT question_text, answer_text, sources FROM text_answer_cache"))
        rows = res.fetchall()
        for row in rows:
            print(f"Q: {row.question_text}")
            print(f"A: {row.answer_text}")
            print("-" * 50)

asyncio.run(main())
