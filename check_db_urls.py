import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text
import json

async def main():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT cache.id, cache.presentation_slides
            FROM teaching_qa_cache cache
            JOIN chapters c ON cache.chapter_id::text = c.id::text
            WHERE c.chapter_number = 5 AND cache.pregen_status = 'processing'
            LIMIT 2
        """))).fetchall()
        
        for r in rows:
            print(f"Row ID: {r.id}")
            slides = r.presentation_slides if isinstance(r.presentation_slides, list) else json.loads(r.presentation_slides)
            for i, s in enumerate(slides):
                img = s.get("infographicUrl", "NONE")
                aud = s.get("audioUrl", "NONE")
                print(f"  Slide {i}: img={img[:30]}..., aud={aud[:30]}...")

asyncio.run(main())
