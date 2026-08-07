import os
import shutil
import asyncio
from pathlib import Path
from db.models import AsyncSessionLocal
from sqlalchemy import text

STORAGE_DIR = "/sdb-disk/ai-teaching"

async def main():
    if not os.path.exists(STORAGE_DIR):
        print(f"Storage dir not found: {STORAGE_DIR}")
        return

    # 1. Get all active cache IDs from DB
    async with AsyncSessionLocal() as db:
        active_ids = {str(r[0]) for r in (await db.execute(text("SELECT id FROM teaching_qa_cache"))).fetchall()}
    
    print(f"Found {len(active_ids)} active cache records in DB.")

    # 2. Scan disk for orphaned jobs
    orphaned_size = 0
    orphaned_count = 0
    
    subjects_dir = Path(STORAGE_DIR) / "subjects"
    if not subjects_dir.exists():
        return
        
    for subj_path in subjects_dir.iterdir():
        jobs_dir = subj_path / "cache" / "jobs"
        if not jobs_dir.exists():
            continue
            
        for job_path in jobs_dir.iterdir():
            job_id = job_path.name
            if job_id not in active_ids:
                # It's orphaned!
                size = sum(f.stat().st_size for f in job_path.rglob('*') if f.is_file())
                orphaned_size += size
                orphaned_count += 1
                
                # Delete it
                shutil.rmtree(job_path)
                print(f"Deleted orphaned job: {job_id} ({(size / 1024 / 1024):.2f} MB)")

    print(f"Cleanup complete! Removed {orphaned_count} orphaned folders.")
    print(f"Freed up {(orphaned_size / 1024 / 1024):.2f} MB of disk space.")

if __name__ == "__main__":
    asyncio.run(main())
