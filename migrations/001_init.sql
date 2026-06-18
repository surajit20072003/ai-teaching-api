CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS teaching_qa_cache (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_hash          TEXT NOT NULL,
    question_text          TEXT NOT NULL,
    subject_id             TEXT,
    topic_id               TEXT,
    chapter_id             TEXT,
    language               TEXT DEFAULT 'hi-IN',
    variation_number       INT CHECK (variation_number BETWEEN 1 AND 3),
    presentation_slides    JSONB DEFAULT '[]',
    latex_formulas         JSONB DEFAULT '[]',
    slide_audio_urls       JSONB DEFAULT '{}',
    total_duration_seconds FLOAT DEFAULT 0,
    usage_count            INT DEFAULT 0,
    created_at             TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(question_hash, subject_id, variation_number)
);

CREATE INDEX IF NOT EXISTS idx_qa_hash  ON teaching_qa_cache(question_hash);
CREATE INDEX IF NOT EXISTS idx_qa_topic ON teaching_qa_cache(topic_id, usage_count DESC);
CREATE INDEX IF NOT EXISTS idx_qa_subj  ON teaching_qa_cache(subject_id, usage_count DESC);
