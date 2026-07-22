import asyncio
import asyncpg
import os

MIGRATIONS_DIR = '/app/migrations'

async def run():
    print("Connecting to DB for migration...")
    db_url = os.getenv('DATABASE_URL').replace('+asyncpg', '')
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        sql_files = sorted(f for f in os.listdir(MIGRATIONS_DIR) if f.endswith('.sql'))
        applied_rows = await conn.fetch("SELECT filename FROM schema_migrations")
        applied_set = {r['filename'] for r in applied_rows}

        for filename in sql_files:
            if filename in applied_set:
                print(f"  ➜ {filename} already applied, skipping")
                continue
            path = os.path.join(MIGRATIONS_DIR, filename)
            print(f"Running migration: {filename}")
            with open(path, 'r') as f:
                sql = f.read()
            await conn.execute(sql)
            await conn.execute("INSERT INTO schema_migrations (filename) VALUES ($1) ON CONFLICT DO NOTHING", filename)
            print(f"  ✓ {filename} complete")
        print("All migrations checked successfully.")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(run())
