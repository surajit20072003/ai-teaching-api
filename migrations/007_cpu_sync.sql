-- ================================================================
-- Migration 007: CPU Sync Columns
-- ================================================================
-- realtime_generated: TRUE when this row was created by the CPU
--                     server's L5 live generation (OpenRouter +
--                     Sarvam), NOT by the GPU pregen pipeline.
--
-- realtime_tier:      Quality tier of the generator:
--                       1 = GPU pregen (Ollama + Wan2GP + VoxCPM)
--                       2 = CPU realtime (OpenRouter + Sarvam)
--
-- synced_to_cpu:      Set TRUE by the GPU server after confirming
--                     a pregen row has reached the CPU replica.
--                     Used by the periodic safety-net sync script.
-- ================================================================

ALTER TABLE teaching_qa_cache
  ADD COLUMN IF NOT EXISTS realtime_generated BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS realtime_tier      INTEGER DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS synced_to_cpu      BOOLEAN DEFAULT FALSE;

-- Index: helps GPU enrichment queue find CPU-generated rows that
-- need upgrading to higher-quality Wan2GP images + VoxCPM audio.
-- Filtered index — very small, only covers realtime rows.
CREATE INDEX IF NOT EXISTS idx_qa_realtime_enrich
  ON teaching_qa_cache (usage_count DESC)
  WHERE realtime_generated = TRUE
    AND pregen_status = 'done';
