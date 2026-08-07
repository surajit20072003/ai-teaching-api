import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT 
                sub.name AS subject_name,
                COUNT(cache.id) AS total_qs,
                SUM(CASE WHEN cache.pregen_status = 'pending' THEN 1 ELSE 0 END) AS pending_cnt,
                SUM(CASE WHEN cache.pregen_status = 'processing' THEN 1 ELSE 0 END) AS processing_cnt,
                SUM(CASE WHEN cache.pregen_status = 'done' THEN 1 ELSE 0 END) AS done_cnt,
                SUM(CASE WHEN cache.presentation_slides IS NOT NULL AND jsonb_array_length(cache.presentation_slides) > 0 THEN 1 ELSE 0 END) AS text_done,
                SUM(CASE WHEN cache.image_urls IS NOT NULL AND cache.image_urls::text != '{}' AND cache.image_urls::text != '[]' THEN 1 ELSE 0 END) AS img_done,
                SUM(CASE WHEN cache.slide_audio_urls IS NOT NULL AND cache.slide_audio_urls->>'urls' IS NOT NULL AND jsonb_array_length(cache.slide_audio_urls->'urls') > 0 THEN 1 ELSE 0 END) AS audio_done
            FROM teaching_qa_cache cache
            LEFT JOIN subjects sub ON cache.subject_id::text = sub.subject_id::text
            GROUP BY sub.name
            ORDER BY sub.name
        """))).fetchall()
        
        print("\n=== ALL SUBJECTS TRUE PROGRESS (INCL. IMAGES & AUDIO) ===")
        print(f"{'Subject':>16} | {'Total':>5} | {'Pending':>7} | {'Process':>7} | {'Done':>5} | {'Text Done':>10} | {'Img Done':>8} | {'Audio Done':>10}")
        print("-" * 90)
        
        tot_q = tot_pend = tot_proc = tot_done = tot_slides = tot_img = tot_aud = 0
        for r in rows:
            tot_q += r.total_qs
            tot_pend += r.pending_cnt
            tot_proc += r.processing_cnt
            tot_done += r.done_cnt
            tot_slides += r.text_done
            tot_img += r.img_done
            tot_aud += r.audio_done
            
            sub_name = str(r.subject_name)[:16]
            print(f"{sub_name:>16} | {r.total_qs:>5} | {r.pending_cnt:>7} | {r.processing_cnt:>7} | {r.done_cnt:>5} | {r.text_done:>10} | {r.img_done:>8} | {r.audio_done:>10}")
        
        print("-" * 90)
        print(f"{'TOT':>16} | {tot_q:>5} | {tot_pend:>7} | {tot_proc:>7} | {tot_done:>5} | {tot_slides:>10} | {tot_img:>8} | {tot_aud:>10}")

asyncio.run(main())
