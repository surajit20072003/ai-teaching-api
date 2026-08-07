import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        row = (await db.execute(text("""
            SELECT sub.name AS subject_name, c.chapter_number, c.title, q.question_text
            FROM teaching_qa_cache q
            LEFT JOIN chapters c ON q.chapter_id::text = c.id::text
            LEFT JOIN subjects sub ON q.subject_id::text = sub.subject_id::text
            WHERE q.question_text ILIKE '%blank white page with minor scanning artifacts%'
            LIMIT 1
        """))).fetchone()
        
        if row:
            print(f"\nFound it!")
            print(f"Subject: {row.subject_name}")
            print(f"Chapter Number: {row.chapter_number}")
            print(f"Chapter Title: {row.title}")
            print(f"Full Question: {row.question_text}")
        else:
            print("\nQuestion not found in the database. Let me check the main questions table too.")
            row = (await db.execute(text("""
                SELECT sub.name AS subject_name, c.chapter_number, c.title, q.question_text
                FROM questions q
                LEFT JOIN chapters c ON q.chapter_id = c.id
                LEFT JOIN subjects sub ON q.subject_id = sub.subject_id
                WHERE q.question_text ILIKE '%blank white page with minor scanning artifacts%'
                LIMIT 1
            """))).fetchone()
            if row:
                print(f"\nFound it in main questions table!")
                print(f"Subject: {row.subject_name}")
                print(f"Chapter Number: {row.chapter_number}")
                print(f"Chapter Title: {row.title}")
                print(f"Full Question: {row.question_text}")
            else:
                print("\nQuestion not found in any table.")

asyncio.run(main())
