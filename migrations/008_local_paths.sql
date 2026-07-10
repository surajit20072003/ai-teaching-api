-- ================================================================
-- Migration 008: Local File Paths
-- ================================================================
-- Stores the physical local file paths on the CPU server alongside
-- the B2 cloud URLs. This allows L2 disk serving without B2 round-trips
-- and confirms which pregen files have been downloaded to CPU disk.
--
-- local_image_paths:  {"0": "/app/storage/subjects/{id}/cache/jobs/{cid}/images/slide_0.png"}
-- local_audio_paths:  {"0": "/app/storage/subjects/{id}/cache/jobs/{cid}/audio/hi-IN/slide_0.wav"}
-- local_manim_paths:  {"0": "/app/storage/subjects/{id}/cache/jobs/{cid}/manim/slide_0.mp4"}
-- ================================================================

ALTER TABLE teaching_qa_cache
  ADD COLUMN IF NOT EXISTS local_image_paths JSONB DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS local_audio_paths JSONB DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS local_manim_paths JSONB DEFAULT NULL;

-- Index: helps find rows that have been pregen'd but not yet
-- downloaded to CPU local disk (local_image_paths IS NULL but image_urls not null)
CREATE INDEX IF NOT EXISTS idx_qa_missing_local_images
  ON teaching_qa_cache (pregen_status)
  WHERE image_urls IS NOT NULL
    AND local_image_paths IS NULL;
