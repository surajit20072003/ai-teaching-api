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

    ch_stats = {}
    
    for r in rows:
        ch = r.chapter_number if r.chapter_number else '?'
        if ch not in ch_stats:
            ch_stats[ch] = {'total': 0, 'pending': 0, 'process': 0, 'done': 0, 'text_done': 0, 'img_done': 0, 'aud_done': 0}
            
        st = ch_stats[ch]
        st['total'] += 1
        
        if r.pregen_status == 'pending':
            st['pending'] += 1
        elif r.pregen_status == 'processing':
            st['process'] += 1
        elif r.pregen_status == 'done':
            st['done'] += 1
            
        # Check text slides
        has_text = False
        num_slides = 0
        if r.presentation_slides:
            slides = r.presentation_slides if isinstance(r.presentation_slides, list) else json.loads(r.presentation_slides)
            if len(slides) > 0:
                has_text = True
                num_slides = len(slides)
                st['text_done'] += 1
                
        # Check local disk for partial progress
        subj_id = str(r.subject_id)
        cache_id = str(r.id)
        
        job_dir = STORAGE_DIR / "subjects" / subj_id / "cache" / "jobs" / cache_id
        img_dir = job_dir / "images"
        audio_base_dir = job_dir / "audio"
        
        img_count = 0
        aud_count = 0
        
        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.png")))
        if audio_base_dir.exists():
            for lang_dir in audio_base_dir.iterdir():
                if lang_dir.is_dir():
                    aud_count += len(list(lang_dir.glob("*.wav")))
                    
        if img_count > 0 or (r.pregen_status == 'done'):
            st['img_done'] += 1
        if aud_count > 0 or (r.pregen_status == 'done'):
            st['aud_done'] += 1

    print("\n=== SOCIAL SCIENCE DISK-AWARE PROGRESS ===")
    print(f"{'Ch':>3} | {'Total':>5} | {'Pending':>7} | {'Process':>7} | {'Done':>5} | {'Text Done':>10} | {'Img Done':>8} | {'Audio Done':>10}")
    print("-" * 85)
    
    tot = {'total': 0, 'pending': 0, 'process': 0, 'done': 0, 'text_done': 0, 'img_done': 0, 'aud_done': 0}
    
    for ch in sorted(ch_stats.keys(), key=lambda x: int(x) if isinstance(x, int) or str(x).isdigit() else 999):
        st = ch_stats[ch]
        for k in tot:
            tot[k] += st[k]
        print(f"{ch:>3} | {st['total']:>5} | {st['pending']:>7} | {st['process']:>7} | {st['done']:>5} | {st['text_done']:>10} | {st['img_done']:>8} | {st['aud_done']:>10}")
        
    print("-" * 85)
    print(f"{'TOT':>3} | {tot['total']:>5} | {tot['pending']:>7} | {tot['process']:>7} | {tot['done']:>5} | {tot['text_done']:>10} | {tot['img_done']:>8} | {tot['aud_done']:>10}")

asyncio.run(main())
