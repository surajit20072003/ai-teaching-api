"""
delete_chapter.py — Delete all cached data for a specific chapter
=================================================================

Clears all 4 cache layers for a chapter:
  L1: Redis    — teaching:{q_hash}:{subject_id} keys
  L2: Disk     — {storage}/subjects/{subject_id}/cache/slides/{q_hash}.json
  L3: Postgres — DELETE FROM teaching_qa_cache WHERE questions match
  L4: Disk     — {storage}/subjects/{subject_id}/cache/jobs/{cache_id}/

Also resets is_pregen_done = false, cache_id = NULL on questions table.

Usage:
  python3 delete_chapter.py --subject science --chapter 1
  python3 delete_chapter.py --subject social --chapter 3 --dry-run
  python3 delete_chapter.py --subject science --all-chapters
  python3 delete_chapter.py --subject all --all-chapters   # nuclear: clear everything
"""

from __future__ import annotations
import argparse, asyncio, os, shutil, sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

SUBJECT_ALIASES = {
    "social":      "Social Science",
    "science":     "Science",
    "math":        "Maths",
    "maths":       "Maths",
    "mathematics": "Maths",
}

def _log(msg): print(msg, flush=True)


async def _resolve_targets(db, subject_filter: Optional[str], chapter_number: Optional[int]) -> List[Dict]:
    from sqlalchemy import text
    subjects = (await db.execute(text("SELECT subject_id, name FROM subjects ORDER BY name"))).fetchall()
    subject_map = {r.name: r.subject_id for r in subjects}

    if subject_filter and subject_filter.lower() != "all":
        name = SUBJECT_ALIASES.get(subject_filter.lower())
        if not name:
            _log(f"[Error] Unknown subject '{subject_filter}'. Valid: {list(SUBJECT_ALIASES)}")
            sys.exit(1)
        subject_map = {k: v for k, v in subject_map.items() if k == name}

    targets = []
    for sname, sid in subject_map.items():
        if chapter_number:
            ch = (await db.execute(text(
                "SELECT id, title FROM chapters WHERE subject_id = :sid AND chapter_number = :n"
            ), {"sid": sid, "n": chapter_number})).fetchone()
            if ch:
                targets.append({"subject_id": sid, "subject_name": sname,
                                 "chapter_id": str(ch.id), "chapter_title": ch.title,
                                 "chapter_number": chapter_number})
            else:
                _log(f"  [Skip] {sname}: Chapter {chapter_number} not found")
        else:
            chs = (await db.execute(text(
                "SELECT id, title, chapter_number FROM chapters WHERE subject_id = :sid ORDER BY chapter_number"
            ), {"sid": sid})).fetchall()
            for ch in chs:
                targets.append({"subject_id": sid, "subject_name": sname,
                                 "chapter_id": str(ch.id), "chapter_title": ch.title,
                                 "chapter_number": ch.chapter_number})
    return targets


async def _delete_chapter(db, target: Dict, dry_run: bool) -> Dict[str, int]:
    from sqlalchemy import text
    from core.cache import hash_question, cache_key, get_redis
    from core.local_storage import SUBJECTS_PATH, get_slide_cache_path, get_job_dir

    sid        = target["subject_id"]
    chapter_id = target["chapter_id"]
    stats      = {"redis": 0, "l2_json": 0, "l4_media": 0, "db_rows": 0, "q_reset": 0}

    # Get all questions for this chapter
    questions = (await db.execute(text("""
        SELECT id, question_text FROM questions
        WHERE chapter_id = CAST(:cid AS uuid)
    """), {"cid": chapter_id})).fetchall()

    if not questions:
        _log(f"    No questions found"); return stats

    _log(f"    {len(questions)} questions found in chapter")
    q_hashes = [hash_question(q.question_text or "") for q in questions]

    # Get cache rows
    cache_rows = (await db.execute(text("""
        SELECT id, subject_id FROM teaching_qa_cache
        WHERE question_hash = ANY(:hashes) AND subject_id = :sid
    """), {"hashes": q_hashes, "sid": sid})).fetchall()
    _log(f"    {len(cache_rows)} cache rows found")

    # L1: Redis
    redis = get_redis()
    if redis:
        for h in q_hashes:
            key = cache_key(h, sid)
            if not dry_run:
                try: await redis.delete(key); stats["redis"] += 1
                except: pass
            else: stats["redis"] += 1
        _log(f"    [L1] {'Would clear' if dry_run else 'Cleared'} {stats['redis']} Redis keys")
    else:
        _log(f"    [L1] Redis not configured — skip")

    # L2: Local slide JSON
    for h in q_hashes:
        p = get_slide_cache_path(sid, h)
        if os.path.exists(p):
            if not dry_run: os.remove(p)
            stats["l2_json"] += 1
    _log(f"    [L2] {'Would delete' if dry_run else 'Deleted'} {stats['l2_json']} slide JSON files")

    # L4: Local media directories
    for row in cache_rows:
        job_dir = get_job_dir(str(row.subject_id), str(row.id))
        if os.path.exists(job_dir):
            if not dry_run: shutil.rmtree(job_dir)
            stats["l4_media"] += 1
        for subdir in ["images", "audio"]:
            legacy = f"{SUBJECTS_PATH}/{row.subject_id}/cache/{subdir}/{str(row.id)}"
            if os.path.exists(legacy):
                if not dry_run: shutil.rmtree(legacy)
                stats["l4_media"] += 1
    _log(f"    [L4] {'Would delete' if dry_run else 'Deleted'} {stats['l4_media']} media dirs")

    # L3: Postgres rows
    if not dry_run:
        r = await db.execute(text("""
            DELETE FROM teaching_qa_cache
            WHERE question_hash = ANY(:hashes) AND subject_id = :sid
        """), {"hashes": q_hashes, "sid": sid})
        stats["db_rows"] = r.rowcount
        await db.commit()
    else:
        stats["db_rows"] = len(cache_rows)
    _log(f"    [L3] {'Would delete' if dry_run else 'Deleted'} {stats['db_rows']} DB rows")

    # Reset questions table
    if not dry_run:
        r2 = await db.execute(text("""
            UPDATE questions SET is_pregen_done = false, cache_id = NULL
            WHERE chapter_id = CAST(:cid AS uuid)
        """), {"cid": chapter_id})
        stats["q_reset"] = r2.rowcount
        await db.commit()
        _log(f"    [Questions] Reset {stats['q_reset']} rows → is_pregen_done=false")

    return stats


async def main(args: argparse.Namespace):
    from db.models import AsyncSessionLocal

    dry_run = args.dry_run
    chapter_number = None if args.all_chapters else args.chapter

    _log("=" * 65)
    _log(f"  delete_chapter.py {'[DRY RUN] ' if dry_run else ''}— Chapter Cache Cleanup")
    _log(f"  Subject: {args.subject} | Chapter: {'ALL' if args.all_chapters else args.chapter}")
    _log("=" * 65)

    if not dry_run:
        _log("\n  WARNING: This will permanently delete cached data!")
        answer = input("  Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            _log("Aborted."); sys.exit(0)

    async with AsyncSessionLocal() as db:
        targets = await _resolve_targets(db, args.subject, chapter_number)

    if not targets:
        _log("\nNo targets found. Check subject/chapter args."); sys.exit(1)

    _log(f"\n{len(targets)} chapter(s) to process\n")
    total = {"redis": 0, "l2_json": 0, "l4_media": 0, "db_rows": 0, "q_reset": 0}

    for i, t in enumerate(targets, 1):
        _log(f"\n[{i}/{len(targets)}] {t['subject_name']} — Ch{t['chapter_number']}: {t['chapter_title']}")
        async with AsyncSessionLocal() as db:
            stats = await _delete_chapter(db, t, dry_run)
        for k in total: total[k] += stats[k]
        _log(f"    ✓ redis={stats['redis']} l2={stats['l2_json']} l4={stats['l4_media']} db={stats['db_rows']}")

    _log("\n" + "=" * 65)
    _log(f"  {'[DRY RUN] ' if dry_run else ''}COMPLETE")
    _log(f"  Redis cleared : {total['redis']}")
    _log(f"  L2 JSON files : {total['l2_json']}")
    _log(f"  L4 media dirs : {total['l4_media']}")
    _log(f"  DB rows       : {total['db_rows']}")
    _log("=" * 65)
    if dry_run:
        _log("\n  DRY RUN — nothing was deleted. Remove --dry-run to apply.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete chapter cache data across all layers")
    parser.add_argument("--subject", required=True, help="science | social | math | all")
    parser.add_argument("--chapter", type=int, default=1, help="Chapter number (default: 1)")
    parser.add_argument("--all-chapters", action="store_true", help="Delete ALL chapters for subject")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, delete nothing")
    asyncio.run(main(parser.parse_args()))
