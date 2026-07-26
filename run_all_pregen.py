"""
run_all_pregen.py
Master batch script that runs pre-generation for ALL subjects and chapters
in sequence, resuming from where it left off if interrupted.

Usage:
    docker exec -it ai-teaching-api python3 -u run_all_pregen.py
    docker exec -it ai-teaching-api python3 -u run_all_pregen.py --start-subject science --start-chapter 3
    docker exec -it ai-teaching-api python3 -u run_all_pregen.py --dry-run
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Chapter plan — edit order here ──────────────────────────────────────────
CHAPTER_PLAN = [
    {"subject": "math",    "chapters": list(range(2, 15))},   # Ch2-14  (13 ch, 343 Qs)
    {"subject": "science", "chapters": list(range(2, 14))},   # Ch2-13  (12 ch, 449 Qs)
    {"subject": "social",  "chapters": list(range(2, 34))},   # Ch2-33  (32 ch, 1855 Qs)
]

STATE_FILE = Path("/app/pregen_state.json")
LOG_FILE   = Path(f"/app/all_pregen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

PREGEN_CMD = [
    "python3", "-u", "run_chapter_pregen.py",
    "--tier", "free",
    "--manim-provider", "openrouter",
]


def _log(msg: str, also_print: bool = True):
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}"
    if also_print:
        print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"done": [], "failed": []}


def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _chapter_key(subject: str, chapter: int) -> str:
    return f"{subject}:{chapter}"


def _run_chapter(subject: str, chapter: int, dry_run: bool) -> bool:
    """Run pregen for one chapter. Returns True on success."""
    cmd = PREGEN_CMD + ["--subject", subject, "--chapter", str(chapter)]
    if dry_run:
        _log(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True

    _log(f"\n  Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)  # streams live output
    elapsed = int(time.time() - t0)
    success = result.returncode == 0
    _log(f"  Exit code: {result.returncode} | Time: {elapsed}s")
    return success


def _count_total(plan: list) -> int:
    return sum(len(s["chapters"]) for s in plan)


def main():
    parser = argparse.ArgumentParser(description="Run pregen for all subjects/chapters")
    parser.add_argument("--start-subject", default=None,
                        help="Skip subjects before this one (e.g. 'science')")
    parser.add_argument("--start-chapter", type=int, default=None,
                        help="Skip chapters before this number within --start-subject")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--skip-done", action="store_true", default=True,
                        help="Skip chapters already marked done in state file (default: on)")
    parser.add_argument("--reset-state", action="store_true",
                        help="Clear the state file and start fresh")
    args = parser.parse_args()

    if args.reset_state:
        STATE_FILE.unlink(missing_ok=True)
        print("State file cleared.")

    state = _load_state()
    total_chapters = _count_total(CHAPTER_PLAN)

    _log("=" * 65)
    _log("  FULL PREGEN BATCH RUNNER")
    _log(f"  Total chapters : {total_chapters}")
    _log(f"  Already done   : {len(state['done'])}")
    _log(f"  Previously fail: {len(state['failed'])}")
    _log(f"  Log file       : {LOG_FILE}")
    _log(f"  State file     : {STATE_FILE}")
    _log("=" * 65)

    # Determine where to start
    skip_subject = True if args.start_subject else False
    done_count = fail_count = 0
    chapter_num = 0

    for subject_entry in CHAPTER_PLAN:
        subject = subject_entry["subject"]

        # If --start-subject given, skip until we reach it
        if skip_subject:
            if subject == args.start_subject:
                skip_subject = False
            else:
                _log(f"\n[SKIP] Subject '{subject}' (before --start-subject)")
                continue

        for chapter in subject_entry["chapters"]:
            chapter_num += 1
            key = _chapter_key(subject, chapter)

            # If --start-chapter given (only applies to start-subject)
            if args.start_chapter and subject == args.start_subject and chapter < args.start_chapter:
                _log(f"[SKIP] {subject} Ch{chapter} (before --start-chapter {args.start_chapter})")
                continue

            # Skip if already done
            if args.skip_done and key in state["done"]:
                _log(f"[DONE] {subject} Ch{chapter} — already completed, skipping")
                done_count += 1
                continue

            _log(f"\n{'='*65}")
            _log(f"  [{chapter_num}/{total_chapters}] {subject.upper()} — Chapter {chapter}")
            _log(f"{'='*65}")

            success = _run_chapter(subject, chapter, args.dry_run)

            if success:
                state["done"].append(key)
                # Remove from failed if it was there before
                state["failed"] = [f for f in state["failed"] if f != key]
                _save_state(state)
                done_count += 1
                _log(f"  ✓ {subject} Ch{chapter} DONE")
            else:
                if key not in state["failed"]:
                    state["failed"].append(key)
                _save_state(state)
                fail_count += 1
                _log(f"  ✗ {subject} Ch{chapter} FAILED — continuing to next chapter")

    _log("\n" + "=" * 65)
    _log("  BATCH COMPLETE")
    _log(f"  Chapters done   : {done_count}")
    _log(f"  Chapters failed : {fail_count}")
    if state["failed"]:
        _log(f"  Failed chapters : {', '.join(state['failed'])}")
        _log("  Re-run to retry failed chapters (done ones will be skipped)")
    _log("=" * 65)


if __name__ == "__main__":
    main()
