"""
Retroactive Manim Patcher
=========================
For already-generated slides that are missing the formula field:
- Calls Ollama to look at slide title + infographic description
- If the slide clearly contains a math formula/equation, extract the LaTeX
- Patch visual_type=manim and formula=<latex> in the DB
- Then you can run --retry-manim to generate the Manim videos

Usage:
  docker exec -it ai-teaching-api python3 -u patch_manim_retroactive.py --subject math --chapter 1
"""
import asyncio, json, argparse, re
import httpx
from db.models import AsyncSessionLocal
from sqlalchemy import text
from core.slide_generator import OLLAMA_URL, OLLAMA_MODEL

SUBJECT_ALIASES = {
    "social": "Social Science", "science": "Science",
    "math": "Maths", "maths": "Maths",
}

FORMULA_EXTRACT_PROMPT = """\
You are a math teacher assistant. Analyze this slide and decide if it contains a mathematical formula, equation, or expression that should be animated.

Slide title: {title}
Slide description: {infographic}
Slide content: {content}

If the slide clearly contains a math formula/equation/expression (like sqrt, HCF formula, proof steps, algebraic identity, etc.), respond with ONLY the LaTeX formula string, e.g.:
  \\sqrt{{7}}
  \\text{{HCF}}(a,b) \\times \\text{{LCM}}(a,b) = a \\times b
  2 + \\sqrt{{3}} = \\frac{{p}}{{q}}

If the slide does NOT have a specific math formula (it's just a diagram, Venn diagram, flowchart, story, tips, number line without equation, etc.), respond with exactly:
  NONE

Respond with ONLY the LaTeX formula or NONE. No explanation."""


async def extract_formula_from_slide(slide: dict) -> str | None:
    """Call Ollama to extract a LaTeX formula from slide content. Returns None if no formula found."""
    title = slide.get("title", "")
    infographic = slide.get("infographic", "")
    content = slide.get("content", "")

    # Skip story/tips slides immediately
    if slide.get("isStory") or slide.get("isTips"):
        return None

    prompt = FORMULA_EXTRACT_PROMPT.format(
        title=title[:200],
        infographic=infographic[:300],
        content=content[:200],
    )

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.1, "num_predict": 100},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            resp.raise_for_status()
        result = (resp.json().get("message") or {}).get("content", "").strip()

        if not result or result.upper() == "NONE" or "NONE" in result.upper():
            return None
        # Clean up the response
        result = result.strip("`").strip()
        # Must be reasonable LaTeX (contains backslash or math chars)
        if not (len(result) > 1 and (result.startswith("\\") or any(c in result for c in ["sqrt", "frac", "^", "_", "=", "+", "times"]))):
            return None
        # Reject trivially simple formulas — not worth a Manim render
        # e.g. \frac{1}{4}, \frac{-4}{1} — simple fractions with plain integers
        import re as _re
        # Skip if formula is just a simple fraction with no irrational/algebraic content
        trivial_fraction = _re.fullmatch(r"\\frac\{-?\d+\}\{-?\d+\}", result.strip())
        if trivial_fraction:
            return None
        # Skip if formula is just a single number or very short plain expression
        if len(result.strip()) < 4:
            return None
        return result
    except Exception as e:
        print(f"    Ollama error: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="math")
    parser.add_argument("--chapter", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't write to DB")
    args = parser.parse_args()

    subject_name = SUBJECT_ALIASES.get(args.subject.lower(), args.subject)

    async with AsyncSessionLocal() as db:
        subj = (await db.execute(text("SELECT subject_id FROM subjects WHERE name = :n"), {"n": subject_name})).first()
        if not subj:
            print(f"Subject not found: {subject_name}"); return
        sid = str(subj.subject_id)

        ch = (await db.execute(text(
            "SELECT id, title FROM chapters WHERE subject_id = :sid AND chapter_number = :n"
        ), {"sid": sid, "n": args.chapter})).first()
        if not ch:
            print(f"Chapter {args.chapter} not found"); return
        cid = str(ch.id)

        rows = (await db.execute(text("""
            SELECT id, question_text, presentation_slides
            FROM teaching_qa_cache
            WHERE subject_id = :sid AND chapter_id = :cid
            ORDER BY created_at ASC
        """), {"sid": sid, "cid": cid})).fetchall()

    print(f"=== Retroactive Manim Patcher ===")
    print(f"Subject: {subject_name} | Ch{args.chapter}: {ch.title}")
    print(f"Questions: {len(rows)} | Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)

    total_patched = 0
    total_skipped = 0

    for qi, row in enumerate(rows, 1):
        slides = list(row.presentation_slides or [])
        qtxt = (row.question_text or "")[:55]
        changed = False

        print(f"\n[Q{qi}] {qtxt}")

        for si, slide in enumerate(slides):
            vt = slide.get("visual_type", "image")
            existing_formula = slide.get("formula", "").strip()
            title = slide.get("title", "")

            # Already manim — skip
            if vt == "manim":
                print(f"  Slide {si+1}: already manim ✓ — skip")
                total_skipped += 1
                continue

            # Already has formula — patch to manim directly
            if existing_formula and not slide.get("isStory") and not slide.get("isTips"):
                print(f"  Slide {si+1}: has formula '{existing_formula[:40]}' → force manim")
                if not args.dry_run:
                    slides[si]["visual_type"] = "manim"
                    changed = True
                total_patched += 1
                continue

            # Ask Ollama to extract formula
            print(f"  Slide {si+1}: '{title[:40]}' → asking Ollama...", end="", flush=True)
            formula = await extract_formula_from_slide(slide)

            if formula:
                print(f" GOT: {formula[:60]}")
                if not args.dry_run:
                    slides[si]["formula"] = formula
                    slides[si]["visual_type"] = "manim"
                    changed = True
                total_patched += 1
            else:
                print(f" none")

        # Save to DB if changed
        if changed and not args.dry_run:
            async with AsyncSessionLocal() as db:
                await db.execute(text("""
                    UPDATE teaching_qa_cache
                    SET presentation_slides = CAST(:s AS jsonb)
                    WHERE id = CAST(:id AS uuid)
                """), {"s": json.dumps(slides), "id": str(row.id)})
                await db.commit()
            print(f"  → DB updated ✓")

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Slides patched to manim : {total_patched}")
    print(f"  Slides skipped (ok)     : {total_skipped}")
    if args.dry_run:
        print(f"\n  DRY RUN — nothing written. Remove --dry-run to apply.")
    else:
        print(f"\n  Now run retry-manim on GPU server:")
        print(f"  docker exec -it ai-teaching-api python3 -u run_chapter_pregen.py \\")
        print(f"    --subject {args.subject} --chapter {args.chapter} --tier free --retry-manim")

asyncio.run(main())
