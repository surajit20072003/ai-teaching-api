-- Migration: add 'notes_pending' to documents.status CHECK constraint
ALTER TABLE documents DROP CONSTRAINT documents_status_check;
ALTER TABLE documents ADD CONSTRAINT documents_status_check
  CHECK (status = ANY (ARRAY[
    'processing',
    'ready',
    'failed',
    'notes_pending',
    'notes_generating',
    'notes_done',
    'notes_failed'
  ]));
