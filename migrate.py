import asyncio
import asyncpg
import os

MIGRATIONS_DIR = '/app/migrations'

async def run():
    print("Connecting to DB for migration...")
    db_url = os.getenv('DATABASE_URL').replace('+asyncpg', '')
    conn = await asyncpg.connect(db_url)
    try:
        # Run all SQL files in sorted order (001, 002, ...)
        sql_files = sorted(f for f in os.listdir(MIGRATIONS_DIR) if f.endswith('.sql'))
        for filename in sql_files:
            path = os.path.join(MIGRATIONS_DIR, filename)
            print(f"Running migration: {filename}")
            with open(path, 'r') as f:
                sql = f.read()
            await conn.execute(sql)
            print(f"  ✓ {filename} complete")
        print("All migrations completed successfully.")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(run())
