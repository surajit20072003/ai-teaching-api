import asyncio
import os
from pathlib import Path
from db.models import AsyncSessionLocal
from sqlalchemy import text
import json

STORAGE_DIR = Path(os.getenv("LOCAL_STORAGE_BASE", "/app/storage"))

async def main():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT 
                cache.id,
                c.chapter_number, 
                cache.pregen_status,
                cache.presentation_slides,
                cache.subject_id
            FROM teaching_qa_cache cache
            JOIN subjects sub ON cache.subject_id::text = sub.subject_id::text
            LEFT JOIN chapters c ON cache.chapter_id::text = c.id::text
            WHERE LOWER(sub.name) = 'social science'
        """))).fetchall()

    total_qs = len(rows)
    
    full_text_only = 0
    all_done = 0
    partially_done = 0
    no_text_yet = 0
    enhancer_done_count = 0

    for r in rows:
        # Check text slides
        has_text = False
        num_slides = 0
        enhancer_done = True
        
        if r.presentation_slides:
            slides = r.presentation_slides if isinstance(r.presentation_slides, list) else json.loads(r.presentation_slides)
            if len(slides) > 0:
                has_text = True
                num_slides = len(slides)
                
                # Check if enhancer is done for all slides
                for s in slides:
                    if not s.get("enhanced_image_prompt"):
                        enhancer_done = False
                        break
            else:
                enhancer_done = False
        else:
            enhancer_done = False
            
        if has_text and enhancer_done:
            enhancer_done_count += 1
                
        # Check disk
        subj_id = str(r.subject_id)
        cache_id = str(r.id)
        job_dir = STORAGE_DIR / "subjects" / subj_id / "cache" / "jobs" / cache_id
        img_dir = job_dir / "images"
        audio_base_dir = job_dir / "audio"
        
        img_count = 0
        aud_count = 0
        
        if r.pregen_status == 'done':
            img_count = num_slides
            aud_count = num_slides
        else:
            if img_dir.exists():
                img_count = len(list(img_dir.glob("*.png")))
            if audio_base_dir.exists():
                for lang_dir in audio_base_dir.iterdir():
                    if lang_dir.is_dir():
                        aud_count += len(list(lang_dir.glob("*.wav")))
                
        if not has_text:
            no_text_yet += 1
        else:
            if img_count >= num_slides and aud_count >= num_slides:
                all_done += 1
            elif img_count == 0 and aud_count == 0:
                full_text_only += 1
            else:
                partially_done += 1

    print("\\n=== SOCIAL SCIENCE GLOBAL PROGRESS SUMMARY ===")
    print(f"Total Questions: {total_qs}")
    print(f"1. Text Done (Phase A Complete): {full_text_only + partially_done + all_done}")
    print(f"   -> Out of those, Enhancer Prompts generated: {enhancer_done_count}")
    print(f"2. Partially Generated Media (Phase B in progress): {partially_done}")
    print(f"3. Text, Image, and Audio ALL Done (Phase B Complete): {all_done}")
    print(f"4. Not Started Yet (Pending Phase A): {no_text_yet}")

asyncio.run(main())
