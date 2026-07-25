import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text
async def f():
    async with AsyncSessionLocal() as db:
        res = await db.execute(text("SELECT answer_text, key_points, example, quick_tip FROM text_answer_cache WHERE question_text='what is rancidity and explain'"))
        row = res.mappings().first()
        print(row)
asyncio.run(f())
