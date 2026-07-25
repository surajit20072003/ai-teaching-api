import asyncio
from db.models import AsyncSessionLocal
from sqlalchemy import text
from core.embeddings import embed_async, vec_to_pg_str

SUBJECT_ID = "ceaf73fb-528a-4d4a-947c-4a7be304db2b"

async def check_rag(question: str):
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")
    embedding = await embed_async(question)
    vec_str = vec_to_pg_str(embedding)
    async with AsyncSessionLocal() as db:
        rows = await db.execute(text("""
            SELECT
                dc.chunk_text,
                dc.section_title,
                d.title AS doc_title,
                1 - (dc.chunk_embedding <=> CAST(:vec AS vector)) AS sim
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.subject_id = :subject_id
              AND dc.chunk_embedding IS NOT NULL
            ORDER BY dc.chunk_embedding <=> CAST(:vec AS vector)
            LIMIT 5
        """), {"vec": vec_str, "subject_id": SUBJECT_ID})
        results = rows.mappings().all()
        for r in results:
            print(f"  sim={round(float(r['sim']),4)} | {r['doc_title']} | {r['section_title']}")
            print(f"  chunk: {r['chunk_text'][:120]}...")
            print()

async def main():
    await check_rag("rancidity")
    await check_rag("what is rancidity?")
    await check_rag("what is rancidity and explain")

asyncio.run(main())
