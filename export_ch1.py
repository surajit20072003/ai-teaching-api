import asyncio
from sqlalchemy import text
from db.models import AsyncSessionLocal

async def main():
    async with AsyncSessionLocal() as db:
        result = await db.execute(text("""
            WITH chapter1_questions AS (
                SELECT q.question_text, s.name as subject_name
                FROM questions q
                JOIN chapters ch ON ch.id::text = q.chapter_id::text
                JOIN subjects s ON s.subject_id = q.subject_id
                WHERE ch.chapter_number = 1
            )
            SELECT s.name as "Subject", c.access_tier as "Tier", REPLACE(c.question_text, E'\n', ' ') as "Question"
            FROM teaching_qa_cache c
            JOIN subjects s ON s.subject_id = c.subject_id
            WHERE c.pregen_status = 'done'
            AND c.question_text IN (SELECT question_text FROM chapter1_questions WHERE subject_name = s.name)
            ORDER BY "Subject", "Question";
        """))
        
        rows = result.fetchall()
        
        with open('/app/chapter1_generated.md', 'w', encoding='utf-8') as f:
            f.write("# Generated Chapter 1 Questions (Free Tier)\n\n")
            f.write("| Subject | Tier | Question |\n")
            f.write("|---------|------|----------|\n")
            for row in rows:
                subject = str(row[0])
                tier = str(row[1])
                question = str(row[2]).strip()
                f.write(f"| {subject} | {tier} | {question} |\n")
        print("Export completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
