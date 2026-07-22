-- Migration 008: Text Answer Cache
-- Creates a dedicated table for the /ai-text-answer endpoint.
-- Also ensures teaching_qa_cache has an HNSW index for the slide preview lookup.

-- 1. New table for text-only answers
CREATE TABLE IF NOT EXISTS text_answer_cache (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_hash       VARCHAR(32) NOT NULL,
    question_text       TEXT NOT NULL,
    subject_id          VARCHAR,
    language            VARCHAR(10) DEFAULT 'en',
    answer_text         TEXT NOT NULL,
    key_points          JSONB DEFAULT '[]',
    sources             JSONB DEFAULT '[]',      -- [{doc_title, section_title}]
    is_doc_grounded     BOOLEAN DEFAULT FALSE,
    question_embedding  vector(384),             -- for L3 semantic cache
    usage_count         INTEGER DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT now()
); 

-- Unique constraint: one answer per question per subject
CREATE UNIQUE INDEX IF NOT EXISTS uq_text_cache_hash_subj
    ON text_answer_cache (question_hash, subject_id);

-- HNSW index for fast similarity search on text_answer_cache
CREATE INDEX IF NOT EXISTS idx_text_cache_embedding
    ON text_answer_cache USING hnsw (question_embedding vector_cosine_ops);

-- 2. Ensure teaching_qa_cache has question_embedding column and index
--    (may already exist from an earlier migration — IF NOT EXISTS handles that safely)
ALTER TABLE teaching_qa_cache
    ADD COLUMN IF NOT EXISTS question_embedding vector(384);

CREATE INDEX IF NOT EXISTS idx_qa_cache_embedding
    ON teaching_qa_cache USING hnsw (question_embedding vector_cosine_ops);
