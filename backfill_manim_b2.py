"""
backfill_manim_b2.py — Upload existing Manim MP4s to B2 and patch the DB.
Run: docker compose exec -T api python3 /app/backfill_manim_b2.py
"""
import asyncio, boto3, json, os
import asyncpg

B2_KEY_ID  = os.getenv("B2_KEY_ID",  "0058561b2ef9a47000000000a")
B2_APP_KEY = os.getenv("B2_APP_KEY", "K005YmGa87nOvkHpegviJs1X19JHrzM")
B2_BUCKET  = os.getenv("B2_BUCKET",  "Simplelectureaivideo")
B2_ENDPOINT= os.getenv("B2_ENDPOINT","https://s3.us-east-005.backblazeb2.com")

CACHE_ID   = "e5cb396a-96e9-47ae-9bc6-0fc2bd5a964b"
SUBJECT_ID = "d19213b8-9092-4c79-9d9a-a477c504acf8"

DB_RAW = os.getenv("DATABASE_URL", "postgresql://teaching_user:teaching_pass@postgres:5432/teaching_db")
DB_URL = DB_RAW.replace("postgresql+asyncpg://", "postgresql://")

s3 = boto3.client("s3", endpoint_url=B2_ENDPOINT,
    aws_access_key_id=B2_KEY_ID, aws_secret_access_key=B2_APP_KEY)

def parse_json_field(raw):
    if raw is None:
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    return json.loads(raw)

async def main():
    conn = await asyncpg.connect(DB_URL)

    row = await conn.fetchrow(
        "SELECT id, manim_video_urls FROM teaching_qa_cache WHERE subject_id = $1 LIMIT 1",
        SUBJECT_ID
    )
    if not row:
        print("No row found!")
        return

    row_id = str(row["id"])
    manim  = parse_json_field(row["manim_video_urls"])
    print(f"Row ID: {row_id}")
    print(f"Current manim_video_urls:\n{json.dumps(manim, indent=2)}")

    # Upload each MP4 to B2
    for idx in range(4):
        local = f"/sdb-disk/ai-teaching/subjects/{SUBJECT_ID}/cache/jobs/{CACHE_ID}/manim/slide_{idx}.mp4"
        if not os.path.exists(local):
            print(f"  slide {idx}: NOT FOUND at {local}")
            continue

        key = f"ai-teaching/{CACHE_ID}/manim_{idx}.mp4"
        with open(local, "rb") as f:
            s3.put_object(Bucket=B2_BUCKET, Key=key, Body=f.read(), ContentType="video/mp4")
        url = f"{B2_ENDPOINT}/{B2_BUCKET}/{key}"
        print(f"  slide {idx} → {url}")

        existing = manim.get(str(idx), {})
        duration = existing.get("duration_seconds", 0.0) if isinstance(existing, dict) else 0.0
        manim[str(idx)] = {"url": url, "local_mp4": local, "duration_seconds": duration}

    # Persist back to DB
    await conn.execute(
        "UPDATE teaching_qa_cache SET manim_video_urls = $1::jsonb WHERE id = $2::uuid",
        json.dumps(manim), row_id
    )
    print(f"\n✓ DB updated. Final manim_video_urls:\n{json.dumps(manim, indent=2)}")
    await conn.close()

asyncio.run(main())
