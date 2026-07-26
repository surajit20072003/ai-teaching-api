"""
Dump all slides for Maths Ch1 — show title, visual_type, formula for each.
"""
import asyncio, json
from db.models import AsyncSessionLocal
from sqlalchemy import text

SUBJECT_NAME = "Maths"
CHAPTER_NUM  = 1

async def main():
    async with AsyncSessionLocal() as db:
        subj = (await db.execute(text("SELECT subject_id FROM subjects WHERE name = :n"), {"n": SUBJECT_NAME})).first()
        sid = str(subj.subject_id)
        ch = (await db.execute(text(
            "SELECT id FROM chapters WHERE subject_id = :sid AND chapter_number = :n"
        ), {"sid": sid, "n": CHAPTER_NUM})).first()
        cid = str(ch.id)

        rows = (await db.execute(text("""
            SELECT id, question_text, presentation_slides
            FROM teaching_qa_cache
            WHERE subject_id = :sid AND chapter_id = :cid
            ORDER BY created_at ASC
        """), {"sid": sid, "cid": cid})).fetchall()

        print(f"{'─'*100}")
        print(f"{'Q#':<4} {'S#':<4} {'VType':<8} {'HasFormula':<12} {'Title':<40} {'Formula/Content'}")
        print(f"{'─'*100}")

        for qi, row in enumerate(rows, 1):
            slides = row.presentation_slides or []
            qtxt = (row.question_text or "")[:45]
            for si, slide in enumerate(slides, 1):
                vt = slide.get("visual_type", "?")
                formula = slide.get("formula", "")
                title = slide.get("title", "")[:38]
                has_f = "YES" if formula.strip() else "no"
                # Show formula if exists, otherwise show infographic snippet
                content = (formula or slide.get("infographic", ""))[:55]
                marker = " ◄ FORMULA" if formula.strip() else ""
                print(f"{qi:<4} {si:<4} {vt:<8} {has_f:<12} {title:<40} {content}{marker}")
            print()

asyncio.run(main())
