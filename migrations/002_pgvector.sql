-- ─────────────────────────────────────────────────────────
-- Migration 002: pgvector + pg_trgm for semantic search
-- ─────────────────────────────────────────────────────────

-- 1. Vector similarity extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Trigram extension for autocomplete fallback (fast ILIKE)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 3. Add 384-dim embedding column to existing table
--    all-MiniLM-L6-v2 produces 384-dim normalized vectors
ALTER TABLE teaching_qa_cache
    ADD COLUMN IF NOT EXISTS question_embedding vector(384);

-- 4. GIN index for fast trigram LIKE search (autocomplete as-you-type)
CREATE INDEX IF NOT EXISTS idx_qa_trgm
    ON teaching_qa_cache
    USING GIN (question_text gin_trgm_ops);

-- 5. HNSW index for fast approximate nearest-neighbor vector search
--    Much faster than exact scan for 10k+ rows
--    m=16, ef_construction=64 — good balance of speed vs recall
CREATE INDEX IF NOT EXISTS idx_qa_embedding
    ON teaching_qa_cache
    USING hnsw (question_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
