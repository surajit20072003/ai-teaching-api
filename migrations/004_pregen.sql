-- Migration 004: Add pregen tracking columns to teaching_qa_cache
-- Pre-generation status per row: NULL | pending | processing | done | failed

ALTER TABLE teaching_qa_cache
    ADD COLUMN IF NOT EXISTS pregen_status       VARCHAR(20)   DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS pregen_completed_at TIMESTAMPTZ   DEFAULT NULL;

-- Index for fast "find all pending for subject_id" queries used by run_pregen_batch
CREATE INDEX IF NOT EXISTS idx_qa_cache_pregen
    ON teaching_qa_cache (subject_id, pregen_status)
    WHERE pregen_status IS NULL OR pregen_status IN ('pending', 'failed');

-- Index on image jobs: quickly look up wan2gp image status
-- (no schema change needed; handled in application layer)
