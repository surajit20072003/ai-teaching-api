-- 006_manim.sql
-- Add Manim video support to teaching_qa_cache
-- Run: python migrate.py (or psql -f migrations/006_manim.sql)

-- Add manim_video_urls column: JSONB map of slide_index -> {url, local_path, duration_seconds}
-- Example: {"0": {"url": "https://f005.backblazeb2.com/.../slide_0.mp4", "local_path": "/sdb-disk/.../manim/slide_0.mp4", "duration_seconds": 14.3}}
ALTER TABLE teaching_qa_cache
    ADD COLUMN IF NOT EXISTS manim_video_urls JSONB DEFAULT '{}'::jsonb;

-- Index for finding rows that have manim videos (for debugging/admin)
CREATE INDEX IF NOT EXISTS idx_qa_cache_has_manim
    ON teaching_qa_cache ((manim_video_urls != '{}'::jsonb))
    WHERE manim_video_urls != '{}'::jsonb;

-- Add image_urls column: JSONB map of slide_index -> {url, local_path}
-- (mirrors existing slide_audio_urls pattern)
ALTER TABLE teaching_qa_cache
    ADD COLUMN IF NOT EXISTS image_urls JSONB DEFAULT '{}'::jsonb;

COMMENT ON COLUMN teaching_qa_cache.manim_video_urls IS
    'Per-slide Manim rendered video URLs. Key = slide index (string). '
    'Value = {url: B2 public URL, local_path: local disk path, duration_seconds: float}';

COMMENT ON COLUMN teaching_qa_cache.image_urls IS
    'Per-slide static image URLs. Key = slide index (string). '
    'Value = {url: B2 public URL, local_path: local disk path}';
