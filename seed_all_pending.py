"""
seed_all_pending.py
Insert ALL questions (Ch2+) for Maths, Science, Social Science
into teaching_qa_cache with pregen_status='pending'.

After running this, use run_chapter_pregen.py per chapter to process them,
or the background pregen worker will pick them up automatically.

Usage:
    docker exec ai-teaching-api python3 -u seed_all_pending.py
    docker exec ai-teaching-api python3 -u seed_all_pending.py --dry-run
"""
import asyncio
import argparse
import uuid
from sqlalchemy import text

SUBJECT_ALIASES = {
    "math":    "Maths",
    "science": "Science",
    "social":  "Social Science",
}

async def main(dry_run: bool):
    from db.models import AsyncSessionLocal
    from core.cache import hash_question

    async with AsyncSessionLocal() as db:
        # Get subject_ids
        subjects = (await db.execute(text(
            "SELECT subject_id, name FROM subjects ORDER BY name"
        ))).fetchall()
        subject_map = {r.name.lower(): r.subject_id for r in subjects}

        total_inserted = 0
        total_skipped  = 0

        for alias, name in SUBJECT_ALIASES.items():
            sid = subject_map.get(name.lower())
            if not sid:
                print(f"  [WARN] Subject '{name}' not found in DB")
                continue

            # Get all questions for this subject EXCLUDING Ch1
            questions = (await db.execute(text("""
                SELECT q.id, q.question_text, q.chapter_id, q.topic_id,
                       c.chapter_number
                FROM questions q
                JOIN chapters c ON q.chapter_id = c.id
                WHERE c.subject_id = :sid
                  AND c.chapter_number != 1
                  AND q.is_pregen_done = FALSE
                ORDER BY c.chapter_number, q.created_at
            """), {"sid": sid})).fetchall()

            print(f"\n  {name.upper()} — {len(questions)} questions (Ch2+)")

            if dry_run:
                print(f"    [DRY RUN] Would insert {len(questions)} pending rows")
                total_inserted += len(questions)
                continue

            inserted = skipped = 0
            for q in questions:
                q_hash = hash_question(q.question_text or "")
                ch_id  = str(q.chapter_id) if q.chapter_id else None
                t_id   = str(q.topic_id)   if q.topic_id   else None
                new_id = str(uuid.uuid4())

                res = await db.execute(text("""
                    INSERT INTO teaching_qa_cache
                        (id, subject_id, chapter_id, topic_id,
                         question_hash, question_text, variation_number,
                         pregen_status, created_at)
                    VALUES
                        (CAST(:id AS uuid), :sid, CAST(:cid AS uuid), :tid,
                         :qhash, :qtext, 1, 'pending', NOW())
                    ON CONFLICT (question_hash, subject_id, variation_number)
                    DO NOTHING
                """), {
                    "id":    new_id,
                    "sid":   str(sid),
                    "cid":   ch_id,
                    "tid":   t_id,
                    "qhash": q_hash,
                    "qtext": q.question_text,
                })
                if res.rowcount:
                    inserted += 1
                else:
                    skipped += 1

            await db.commit()
            print(f"    ✓ inserted={inserted}  skipped(already exist)={skipped}")
            total_inserted += inserted
            total_skipped  += skipped

    print(f"\n{'='*55}")
    if dry_run:
        print(f"  [DRY RUN] Would insert {total_inserted} pending rows")
    else:
        print(f"  DONE — inserted={total_inserted}  skipped={total_skipped}")
    print(f"{'='*55}")
    print("  Next step: run pregen per chapter, e.g.:")
    print("    python3 run_chapter_pregen.py --subject math --chapter 2 --tier free")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
