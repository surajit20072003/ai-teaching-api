-- Migration 012: Expand documents.status CHECK constraint + add import_source column
-- Safe: only adds new allowed values, does not remove or modify existing data

-- 1. Expand the status check constraint to include notes pipeline statuses
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_status_check;
ALTER TABLE documents ADD CONSTRAINT documents_status_check
  CHECK (status = ANY (ARRAY[
    'processing'::text,
    'ready'::text,
    'failed'::text,
    'notes_generating'::text,
    'notes_done'::text,
    'notes_failed'::text
  ]));

-- 2. Add import_source column if it doesn't already exist
ALTER TABLE documents ADD COLUMN IF NOT EXISTS import_source VARCHAR(20) DEFAULT 'file_upload';

-- 3. Backfill: any existing docs without import_source → file_upload
UPDATE documents SET import_source = 'file_upload' WHERE import_source IS NULL;
