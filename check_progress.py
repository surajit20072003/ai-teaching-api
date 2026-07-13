import asyncio
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://teaching_user:teaching_pass@116.202.230.124:5433/teaching_db"

async def main():
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        result = await session.execute(text("SELECT id, presentation_slides, image_urls, slide_audio_urls FROM teaching_qa_cache WHERE pregen_status = 'processing'"))
        rows = result.fetchall()

    print(f"Total processing rows: {len(rows)}")

    text_ready = 0
    media_ready = 0
    manim_ready = 0 # Not requested, but we can check media

    for row in rows:
        cache_id, slides_json, image_urls, slide_audio_urls = row
        
        # Parse JSON
        slides = slides_json if isinstance(slides_json, list) else json.loads(slides_json) if slides_json else []
        img_urls = image_urls if isinstance(image_urls, dict) else json.loads(image_urls) if image_urls else {}
        aud_urls_list = slide_audio_urls.get("urls", []) if isinstance(slide_audio_urls, dict) else (json.loads(slide_audio_urls).get("urls", []) if slide_audio_urls else [])
        
        if not slides:
            continue
            
        text_ready += 1
        
        # check media
        db_audio_by_idx = {entry["slideIndex"]: entry for entry in aud_urls_list if "slideIndex" in entry}
        
        missing = False
        for i, slide in enumerate(slides):
            # check image
            has_img = (slide.get("infographicUrl") or "").startswith("http")
            if not has_img:
                db_img = img_urls.get(str(i), {}).get("url", "")
                if db_img.startswith("http"):
                    has_img = True
            
            # check audio
            has_aud = (slide.get("audioUrl") or "").startswith("http")
            if not has_aud:
                db_aud = db_audio_by_idx.get(i, {})
                if (db_aud.get("audioUrl") or "").startswith("http"):
                    has_aud = True
                    
            if not has_img or not has_aud:
                missing = True
                break
                
        if not missing:
            media_ready += 1

    print(f"Rows with full text (Phase A complete): {text_ready}")
    print(f"Rows with full image+audio (Phase B complete): {media_ready}")

asyncio.run(main())
