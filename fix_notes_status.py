"""One-shot migration script: run inside the API container to fix status constraint and update doc statuses."""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import os

DB_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://teaching_user:teaching_pass@postgres:5432/teaching_db")

async def main():
    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        # 1. Expand CHECK constraint
        await conn.execute(text("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_status_check"))
        await conn.execute(text("""
            ALTER TABLE documents ADD CONSTRAINT documents_status_check
            CHECK (status = ANY (ARRAY[
                'processing','ready','failed',
                'notes_pending','notes_generating','notes_done','notes_failed'
            ]))
        """))
        print("✓ Constraint updated")

        # 2. Mark json_import docs as notes_pending
        r = await conn.execute(text("""
            UPDATE documents SET status='notes_pending', updated_at=NOW()
            WHERE import_source='json_import'
              AND status IN ('ready','notes_generating','notes_done','notes_failed')
        """))
        print(f"✓ Marked {r.rowcount} documents as notes_pending")

        # 3. Ensure batch_jobs seed row
        await conn.execute(text("""
            INSERT INTO batch_jobs (id, running, stop_flag, current_doc, current_title,
                                    done, failed, total, subject_id, delay_ms,
                                    last_update, started_at, updated_at)
            VALUES ('notes_batch', false, false, '', '', 0, 0, 0, '', 2000, 'Idle', NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET running=false, stop_flag=false, updated_at=NOW()
        """))
        print("✓ batch_jobs seed row ensured")

        # 4. Show final counts
        r = await conn.execute(text("""
            SELECT status, COALESCE(import_source,'none') as src, COUNT(*)
            FROM documents GROUP BY status, import_source ORDER BY status
        """))
        print("\nDocument status breakdown:")
        for row in r.fetchall():
            print(f"  {row[0]:22s} | {row[1]:12s} | {row[2]} docs")

    await engine.dispose()
    print("\nAll done!")

asyncio.run(main())
