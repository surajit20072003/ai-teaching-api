import asyncio
from sqlalchemy import text
from db.models import AsyncSessionLocal

async def report():
    async with AsyncSessionLocal() as db:
        print("=" * 65)
        print("  OVERALL STATS (all subjects, all chapters)")
        print("=" * 65)
        r = (await db.execute(text("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pregen_status = 'done' THEN 1 ELSE 0 END) as done,
                SUM(CASE WHEN pregen_status = 'done'
                          AND presentation_slides->0->>'infographicUrl' LIKE 'http%'
                          AND presentation_slides->0->>'audioUrl' LIKE 'http%'
                         THEN 1 ELSE 0 END) as full_media,
                SUM(CASE WHEN manim_video_urls IS NOT NULL
                          AND manim_video_urls != '{}'::jsonb
                         THEN 1 ELSE 0 END) as has_manim,
                SUM(CASE WHEN pregen_status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN pregen_status = 'failed'  THEN 1 ELSE 0 END) as failed
            FROM teaching_qa_cache
        """))).fetchone()
        print(f"  Total rows             : {r.total}")
        print(f"  Done (pregen=done)     : {r.done}")
        print(f"  FULLY DONE (img+audio) : {r.full_media}  <-- 'complete'")
        print(f"  Has Manim video        : {r.has_manim}")
        print(f"  Still pending          : {r.pending}")
        print(f"  Failed                 : {r.failed}")

        subjects = (await db.execute(text(
            "SELECT subject_id, name FROM subjects ORDER BY name"
        ))).fetchall()

        print()
        print("=" * 65)
        print("  PER SUBJECT")
        print("=" * 65)
        for s in subjects:
            r2 = (await db.execute(text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pregen_status = 'done'
                              AND presentation_slides->0->>'infographicUrl' LIKE 'http%'
                              AND presentation_slides->0->>'audioUrl' LIKE 'http%'
                             THEN 1 ELSE 0 END) as full_done,
                    SUM(CASE WHEN pregen_status IN ('pending','failed')
                             THEN 1 ELSE 0 END) as remaining,
                    COUNT(DISTINCT chapter_id) as chapters
                FROM teaching_qa_cache
                WHERE subject_id = :sid
            """), {"sid": s.subject_id})).fetchone()
            total = int(r2.total or 0)
            full  = int(r2.full_done or 0)
            rem   = int(r2.remaining or 0)
            chs   = int(r2.chapters or 0)
            pct   = round(100 * full / total) if total else 0
            print(f"  {s.name[:32]:<32} | {full:>4}/{total:<4} ({pct:>3}%) | {chs} chapters | {rem} pending")

        print()
        print("=" * 65)
        print("  PER CHAPTER")
        print("=" * 65)
        for s in subjects:
            chapters = (await db.execute(text(
                "SELECT id, title, chapter_number FROM chapters WHERE subject_id = :sid ORDER BY chapter_number"
            ), {"sid": s.subject_id})).fetchall()
            if not chapters:
                continue
            header_printed = False
            for ch in chapters:
                r3 = (await db.execute(text("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN pregen_status = 'done'
                                  AND presentation_slides->0->>'infographicUrl' LIKE 'http%'
                                  AND presentation_slides->0->>'audioUrl' LIKE 'http%'
                                 THEN 1 ELSE 0 END) as full_done
                    FROM teaching_qa_cache
                    WHERE subject_id = :sid AND chapter_id = :cid
                """), {"sid": s.subject_id, "cid": ch.id})).fetchone()
                total = int(r3.total or 0)
                full  = int(r3.full_done or 0)
                if total == 0:
                    continue
                if not header_printed:
                    print(f"\n  [{s.name}]")
                    header_printed = True
                pct = round(100 * full / total)
                if pct == 100: sym = "✅"
                elif pct > 0:  sym = "⏳"
                else:          sym = "❌"
                bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                print(f"    {sym} Ch{ch.chapter_number:>2} [{bar}] {full:>3}/{total:<3}  {ch.title[:38]}")

asyncio.run(report())
