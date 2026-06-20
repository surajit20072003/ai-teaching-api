import asyncio
from sqlalchemy import text
from db.models import AsyncSessionLocal
import json

async def main():
    async with AsyncSessionLocal() as db:
        res = await db.execute(text("""
            SELECT id, question_text, 
                   array_length(question_embedding::numeric[], 1) as embed_dim,
                   presentation_slides, slide_audio_urls, total_duration_seconds
            FROM teaching_qa_cache 
            WHERE question_embedding IS NOT NULL
            LIMIT 1
        """))
        row = res.fetchone()
        if not row:
            print("No rows with embeddings found.")
            return

        print(f"ID: {row.id}")
        print(f"Question: {row.question_text}")
        print(f"Embedding Dimension: {row.embed_dim}")
        
        slides = row.presentation_slides
        if slides:
            print(f"Number of slides: {len(slides)}")
            # Show a snippet of the first slide
            first_slide = json.dumps(slides[0], indent=2)
            print(f"First slide snippet: {first_slide[:300]}...")
        else:
            print("Presentation slides: NULL")

        audios = row.slide_audio_urls
        if audios:
            print(f"Slide audio urls type: {type(audios)}")
            print(f"Audio urls data: {json.dumps(audios, indent=2)}")
        else:
            print("Audio urls: NULL")
            
        print(f"Total Duration: {row.total_duration_seconds}s")

asyncio.run(main())
