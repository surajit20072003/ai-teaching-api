-- Migration 003: Document-Grounded RAG System
-- Creates: documents, document_chunks, document_questions
-- Alters:  teaching_qa_cache (adds document_id, is_doc_grounded)

-- ── 1. documents ─────────────────────────────────────────────────────────────
-- One row per uploaded lecture document (PDF/DOCX/TXT)
CREATE TABLE IF NOT EXISTS documents (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id           TEXT NOT NULL,                         -- MANDATORY, always subject-scoped
    chapter_id           TEXT,                                  -- optional chapter scope
    topic_id             TEXT,                                  -- optional topic scope
    title                TEXT NOT NULL,
    filename             TEXT NOT NULL,
    local_raw_path       TEXT NOT NULL,                         -- /sdb-disk/ai-teaching/subjects/{subject_id}/documents/{id}/raw/{filename}
    local_processed_path TEXT,                                  -- /sdb-disk/ai-teaching/subjects/{subject_id}/documents/{id}/processed/extracted.txt
    b2_url               TEXT,                                  -- Backblaze B2 backup URL
    total_chunks         INTEGER DEFAULT 0,
    pregen_total         INTEGER DEFAULT 0,                     -- total questions to pre-generate
    pregen_done          INTEGER DEFAULT 0,                     -- questions fully pre-generated so far
    status               TEXT DEFAULT 'processing'             -- processing | ready | failed
                         CHECK (status IN ('processing', 'ready', 'failed')),
    language             TEXT DEFAULT 'hi-IN',
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_subject_id ON documents (subject_id);
CREATE INDEX IF NOT EXISTS idx_documents_status     ON documents (status);

-- ── 2. document_chunks ───────────────────────────────────────────────────────
-- One row per text chunk extracted from a document (used for RAG search)
CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    subject_id      TEXT NOT NULL,                             -- denormalized for fast subject-scoped vector search
    chunk_index     INTEGER NOT NULL,                          -- order within document
    section_title   TEXT,                                      -- detected heading (e.g. "Chapter 3: Forces")
    chunk_text      TEXT NOT NULL,                             -- raw passage (~400 words)
    chunk_embedding vector(384),                               -- sentence-transformers all-MiniLM-L6-v2
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HNSW index for fast approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding
    ON document_chunks USING hnsw (chunk_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Regular index for subject-scoped filtering (always used in WHERE clause)
CREATE INDEX IF NOT EXISTS idx_doc_chunks_subject  ON document_chunks (subject_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_document ON document_chunks (document_id);

-- ── 3. document_questions ────────────────────────────────────────────────────
-- Admin-provided questions per document (pre-generated with highest priority)
CREATE TABLE IF NOT EXISTS document_questions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    subject_id      TEXT NOT NULL,
    question_text   TEXT NOT NULL,
    is_pregen_done  BOOLEAN DEFAULT FALSE,
    cache_id        UUID,                                      -- links to teaching_qa_cache once pre-gen is done
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_questions_document    ON document_questions (document_id);
CREATE INDEX IF NOT EXISTS idx_doc_questions_pregen_done ON document_questions (is_pregen_done);

-- ── 4. Alter teaching_qa_cache ───────────────────────────────────────────────
-- Add document traceability columns
ALTER TABLE teaching_qa_cache
    ADD COLUMN IF NOT EXISTS document_id     UUID REFERENCES documents(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS is_doc_grounded BOOLEAN DEFAULT FALSE;

-- Index so we can quickly find all cache entries from a specific document
CREATE INDEX IF NOT EXISTS idx_cache_document_id ON teaching_qa_cache (document_id)
    WHERE document_id IS NOT NULL;

-- ── 5. Auto-update updated_at on documents ───────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
