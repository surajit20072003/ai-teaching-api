"""Check visual_type and formula fields for Maths Ch1 slides in DB"""
import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text
from core.cache import hash_question

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
            SELECT question_text, presentation_slides
            FROM teaching_qa_cache
            WHERE subject_id = :sid AND chapter_id = :cid
            ORDER BY created_at ASC
        """), {"sid": sid, "cid": cid})).fetchall()

        manim_count = 0
        image_count = 0
        formula_present = 0

        for row in rows:
            slides = row.presentation_slides or []
            qtxt = (row.question_text or "")[:50]
            for i, s in enumerate(slides):
                vt = s.get("visual_type", "MISSING")
                formula = s.get("formula", "")
                if vt == "manim":
                    manim_count += 1
                    print(f"  MANIM: Q={qtxt} slide={i+1} formula={formula[:60]}")
                else:
                    image_count += 1
                if formula:
                    formula_present += 1

        print(f"\n{'='*60}")
        print(f"Total slides : {manim_count + image_count}")
        print(f"  manim type : {manim_count}")
        print(f"  image type : {image_count}")
        print(f"  has formula: {formula_present}")
        print()

        # Show sample slide to see formula field
        print("=== Sample slide from Q1 ===")
        import json
        if rows:
            slides = rows[0].presentation_slides or []
            for i, s in enumerate(slides[:2]):
                print(f"Slide {i+1}:")
                print(f"  title       : {s.get('title','')}")
                print(f"  visual_type : {s.get('visual_type','MISSING')}")
                print(f"  formula     : {repr(s.get('formula',''))}")
                print(f"  keys        : {list(s.keys())}")

asyncio.run(main())
