-- Migration 015: Create note_documents table (notes-only document registry)
-- ========================================================================
-- Problem:
--   The `documents` table is shared between slide generation and notes generation.
--   documents.chapter_id is stored as free text (set during upload), while
--   topic_notes.chapter_id is cast to UUID. This causes a UUID mismatch: the
--   value in topic_notes.chapter_id is not the same UUID as chapters.id, so
--   /notes/chapter/{chapter_id} 404s even though notes exist.
--
-- Fix:
--   Create a dedicated `note_documents` table for the notes pipeline with a
--   proper FK to chapters.id. topic_notes.document_id now points to
--   note_documents.id instead of documents.id.
--
-- Slide data is completely untouched. documents, document_chunks, and
-- document_questions are not modified.

-- 1. Create note_documents
CREATE TABLE IF NOT EXISTS note_documents (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id           TEXT    NOT NULL,
    chapter_id           UUID    REFERENCES chapters(id) ON DELETE SET NULL,
    topic_id             UUID    REFERENCES topics(id)   ON DELETE SET NULL,
    title                TEXT    NOT NULL,
    filename             TEXT    NOT NULL,
    local_raw_path       TEXT    NOT NULL,
    local_processed_path TEXT,
    b2_url               TEXT,
    content_markdown     TEXT,
    parsed_images        JSONB   DEFAULT '{}'::jsonb,
    language             TEXT    DEFAULT 'hi-IN',
    access_tier          TEXT    NOT NULL DEFAULT 'pro',
    notes_status         TEXT    DEFAULT 'pending',  -- pending | generating | done | failed
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE note_documents IS
  'Dedicated document registry for the notes pipeline. Separate from documents (slide pipeline).';

-- 2. Migrate existing data: copy every documents row that has topic_notes pointing to it
INSERT INTO note_documents
    (id, subject_id, chapter_id, topic_id, title, filename,
     local_raw_path, local_processed_path, b2_url,
     content_markdown, parsed_images, language, access_tier, notes_status,
     created_at, updated_at)
SELECT
    d.id,
    d.subject_id,
    tn.chapter_id                       AS chapter_id,   -- already a UUID from topic_notes
    tn.topic_id                         AS topic_id,
    d.title,
    d.filename,
    d.local_raw_path,
    d.local_processed_path,
    d.b2_url,
    d.content_markdown,
    d.parsed_images,
    d.language,
    d.access_tier,
    COALESCE(tn.notes_status, 'pending') AS notes_status,
    d.created_at,
    d.updated_at
FROM documents d
JOIN topic_notes tn ON tn.document_id = d.id
ON CONFLICT (id) DO NOTHING;

-- 3. Re-point topic_notes.document_id → note_documents.id (same UUID, new table)
--    The INSERT above uses d.id, so the UUID values are identical — just FK target changes.
--    Since note_documents.id was created FROM documents.id, no data change is needed;
--    the FK constraint on topic_notes will enforce the new target on future writes.
--    We add the FK now so any orphaned topic_notes rows (document deleted) surface.
ALTER TABLE topic_notes
    DROP CONSTRAINT IF EXISTS topic_notes_document_id_fkey;

ALTER TABLE topic_notes
    ADD CONSTRAINT topic_notes_document_id_fkey
        FOREIGN KEY (document_id) REFERENCES note_documents(id) ON DELETE CASCADE;

-- 4. Index for common lookups
CREATE INDEX IF NOT EXISTS idx_note_documents_subject ON note_documents(subject_id);
CREATE INDEX IF NOT EXISTS idx_note_documents_chapter ON note_documents(chapter_id);
CREATE INDEX IF NOT EXISTS idx_note_documents_topic   ON note_documents(topic_id);

-- 5. updated_at trigger for note_documents
CREATE OR REPLACE FUNCTION update_note_documents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_note_documents_updated_at ON note_documents;
CREATE TRIGGER trg_note_documents_updated_at
    BEFORE UPDATE ON note_documents
    FOR EACH ROW EXECUTE FUNCTION update_note_documents_updated_at();
