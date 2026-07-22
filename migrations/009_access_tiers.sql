-- migrations/009_access_tiers.sql
-- Add access_tier to documents and document_chunks for tiered RAG filtering

-- 1. Documents table
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS access_tier VARCHAR(20) NOT NULL DEFAULT 'pro';

-- 2. Document chunks (denormalized for fast RAG query filtering without JOIN)
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS access_tier VARCHAR(20) NOT NULL DEFAULT 'pro';

-- 3. Index for fast tier-filtered RAG queries
CREATE INDEX IF NOT EXISTS idx_chunks_subject_tier
    ON document_chunks (subject_id, access_tier);

-- ──────────────────────────────────────────────────────────────────────────────
-- 4. CHAPTER 1 FREE TIER — Mark Chapter 1 of all 3 subjects as 'free'
--    chapter_ids confirmed from live DB.  All other chapters remain 'pro'.
-- ──────────────────────────────────────────────────────────────────────────────

-- Science: Chapter 1 — CHEMICAL REACTION AND EQUATION (3 docs, 16 chunks)
UPDATE documents    SET access_tier = 'free'
    WHERE chapter_id = 'fb445fdf-67d3-470d-8911-09d1c810fab0';
UPDATE document_chunks SET access_tier = 'free'
    WHERE document_id IN (
        SELECT id FROM documents WHERE chapter_id = 'fb445fdf-67d3-470d-8911-09d1c810fab0');

-- Maths: Chapter 1 — REAL NUMBERS (3 docs, 14 chunks)
UPDATE documents    SET access_tier = 'free'
    WHERE chapter_id = 'c9523153-d111-4c58-a813-fa94398ad61a';
UPDATE document_chunks SET access_tier = 'free'
    WHERE document_id IN (
        SELECT id FROM documents WHERE chapter_id = 'c9523153-d111-4c58-a813-fa94398ad61a');

-- Social Science: Chapter 1 — THE ADVENT OF EUROPEANS TO INDIA (5 docs, 42 chunks)
UPDATE documents    SET access_tier = 'free'
    WHERE chapter_id = '222db5d2-f036-4480-adb3-6e714edb7cc3';
UPDATE document_chunks SET access_tier = 'free'
    WHERE document_id IN (
        SELECT id FROM documents WHERE chapter_id = '222db5d2-f036-4480-adb3-6e714edb7cc3');

-- Verify:
SELECT s.name, d.access_tier, COUNT(DISTINCT d.id) AS docs, COUNT(dc.id) AS chunks
FROM documents d
JOIN subjects s ON s.subject_id = d.subject_id
LEFT JOIN document_chunks dc ON dc.document_id = d.id
GROUP BY s.name, d.access_tier
ORDER BY s.name, d.access_tier;
