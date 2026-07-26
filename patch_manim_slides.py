"""
Patch existing slides in teaching_qa_cache:
- Find all slides with non-empty `formula` field but visual_type = "image"
- Update them to visual_type = "manim"
- Report what was changed
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
        """), {"sid": sid, "cid": cid})).fetchall()

        patched_rows = 0
        patched_slides = 0

        for row in rows:
            slides = list(row.presentation_slides or [])
            changed = False

            for i, slide in enumerate(slides):
                formula = slide.get("formula", "").strip()
                vt = slide.get("visual_type", "")
                is_story = slide.get("isStory", False)
                is_tips = slide.get("isTips", False)

                # Only patch: has formula, not story/tips, currently set to image
                if formula and not is_story and not is_tips and vt != "manim":
                    print(f"  PATCH: {(row.question_text or '')[:50]}")
                    print(f"    slide {i+1}: visual_type '{vt}' → 'manim' | formula: {formula[:60]}")
                    slides[i]["visual_type"] = "manim"
                    changed = True
                    patched_slides += 1

            if changed:
                await db.execute(text("""
                    UPDATE teaching_qa_cache
                    SET presentation_slides = CAST(:s AS jsonb)
                    WHERE id = CAST(:id AS uuid)
                """), {"s": json.dumps(slides), "id": str(row.id)})
                patched_rows += 1

        await db.commit()
        print(f"\n{'='*60}")
        print(f"Patched {patched_slides} slides across {patched_rows} questions")
        print(f"These slides now have visual_type='manim' and are ready for Manim generation")

asyncio.run(main())
