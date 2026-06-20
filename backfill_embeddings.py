import asyncio
from sqlalchemy import text
from db.models import AsyncSessionLocal
from core.embeddings import embed_async, vec_to_pg_str

async def main():
    async with AsyncSessionLocal() as db:
        res = await db.execute(text("SELECT id, question_text FROM teaching_qa_cache WHERE question_embedding IS NULL"))
        rows = res.fetchall()
        for r in rows:
            print(f"Generating embedding for {r.id}")
            vec = await embed_async(r.question_text)
            v_str = vec_to_pg_str(vec)
            await db.execute(
                text("UPDATE teaching_qa_cache SET question_embedding = CAST(:vec AS vector) WHERE id = :id"),
                {"vec": v_str, "id": r.id}
            )
        await db.commit()
        print("Done.")

asyncio.run(main())
