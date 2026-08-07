-- Migration 011: Textbook Notes + Question Answers
-- Creates: topic_notes (notes + answers per topic, linked to document)
-- Alters: documents (adds content_markdown, parsed_images for textbook generation)

-- ── 1. Alter documents ────────────────────────────────────────────────────────
-- Store the parsed document content so we can generate notes from it later
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_markdown TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parsed_images JSONB DEFAULT '{}';

-- ── 2. topic_notes ────────────────────────────────────────────────────────────
-- One row per topic, containing BOTH textbook notes + question answers
-- Linked to a document so we know the source
CREATE TABLE IF NOT EXISTS topic_notes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id         UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    subject_id          TEXT NOT NULL,
    chapter_id          UUID,
    topic_id            UUID,

    -- Textbook notes (structured sections from document content)
    note_sections       JSONB DEFAULT '[]',
    note_image_urls     JSONB DEFAULT '[]',
    note_latex_formulas JSONB DEFAULT '[]',

    -- Question answers for ALL questions in this topic
    question_answers    JSONB DEFAULT '[]',
    answer_image_urls   JSONB DEFAULT '[]',

    -- Status tracking
    notes_status        TEXT DEFAULT 'pending'
                         CHECK (notes_status IN ('pending', 'generating', 'done', 'failed')),
    error_message       TEXT,
    generated_at        TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(document_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_topic_notes_document ON topic_notes (document_id);
CREATE INDEX IF NOT EXISTS idx_topic_notes_topic ON topic_notes (topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_notes_status ON topic_notes (notes_status);

-- ── 3. Auto-update updated_at ──────────────────────────────────────────────────
DROP TRIGGER IF EXISTS trg_topic_notes_updated_at ON topic_notes;
CREATE TRIGGER trg_topic_notes_updated_at
    BEFORE UPDATE ON topic_notes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
