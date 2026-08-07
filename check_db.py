import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("SELECT id, manim_video_urls FROM teaching_qa_cache WHERE id::text LIKE 'e3ec409d%'"))).fetchall()
        for r in rows:
            print(f"ID: {r.id}")
            print(f"Value: {r.manim_video_urls}")
            print(f"Type: {type(r.manim_video_urls)}")

asyncio.run(main())
