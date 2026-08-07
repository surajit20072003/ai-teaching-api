import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        row = (await db.execute(text("""
            SELECT id, question_text, pregen_status, presentation_slides
            FROM teaching_qa_cache
            WHERE question_text LIKE '%India can be divided into%'
            LIMIT 1
        """))).fetchone()
        
        if row:
            print(f"ID: {row.id}")
            print(f"Question: {row.question_text[:80]}...")
            print(f"Status: {row.pregen_status}")
            
            slides = row.presentation_slides
            if slides is None:
                print("Slides: NULL")
            elif isinstance(slides, list) and len(slides) == 0:
                print("Slides: [] (Empty List)")
            else:
                print(f"Slides: {len(slides)} slides generated!")
        else:
            print("Question not found.")

asyncio.run(main())
