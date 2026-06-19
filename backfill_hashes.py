#!/usr/bin/env python3
"""
backfill_hashes.py
──────────────────
One-time script: recalculates question_hash for all rows in teaching_qa_cache
using the correct SHA256 algorithm (hash_question from core/cache.py).

Run inside the container:
  docker exec ai-teaching-api python /app/backfill_hashes.py
"""

import sys, os
sys.path.insert(0, "/app")

import hashlib, unicodedata, re, asyncio
import asyncpg


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://teaching_user:teaching_pass@postgres:5432/teaching_db"
)


def hash_question(text: str) -> str:
    """Exact replica of core/cache.py hash_question()"""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    stopwords = {"what", "is", "the", "of", "and", "a", "an", "how", "why",
                 "to", "in", "for", "on", "with"}
    words = text.split()
    filtered = [w for w in words if w not in stopwords]
    sorted_words = sorted(filtered)
    normalized = " ".join(sorted_words)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


async def main():
    # Convert SQLAlchemy URL to asyncpg format
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(url)
    rows = await conn.fetch(
        "SELECT id, question_text, question_hash FROM teaching_qa_cache"
    )

    updated = 0
    skipped = 0
    for row in rows:
        correct = hash_question(row["question_text"])
        if row["question_hash"] == correct:
            skipped += 1
            continue
        await conn.execute(
            "UPDATE teaching_qa_cache SET question_hash = $1 WHERE id = $2",
            correct, row["id"]
        )
        print(f"  Updated: '{row['question_text'][:60]}' | {row['question_hash'][:8]}… → {correct[:8]}…")
        updated += 1

    await conn.close()
    print(f"\n✓ Done: {updated} updated, {skipped} already correct (total {len(rows)} rows)")


if __name__ == "__main__":
    asyncio.run(main())
