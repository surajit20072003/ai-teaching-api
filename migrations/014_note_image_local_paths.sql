-- Migration 014: Add note_image_local_paths to topic_notes
-- Stores local filesystem paths parallel to note_image_urls (B2 URLs)
-- Frontend can use either the B2 URL (cloud) or /local-images/... (local server)

ALTER TABLE topic_notes ADD COLUMN IF NOT EXISTS note_image_local_paths JSONB DEFAULT '[]';

COMMENT ON COLUMN topic_notes.note_image_local_paths IS
  'Parallel array to note_image_urls. Each entry is the local filesystem path
   (e.g. /sdb-disk/ai-teaching/subjects/{sid}/documents/{did}/notes/images/note_001.png)
   for the corresponding image in note_image_urls.';
