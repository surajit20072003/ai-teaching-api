-- Migration 005: Hierarchical Curriculum + Rich Question Bank
-- Creates: chapters, topics, questions
-- Alters:  subjects (add slug), teaching_qa_cache (add chapter_id, topic_id)

-- ── 1. subjects table (idempotent — may already exist) ────────────────────────
CREATE TABLE IF NOT EXISTS subjects (
    subject_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE subjects ADD COLUMN IF NOT EXISTS slug TEXT;

-- ── 2. chapters ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chapters (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id     TEXT NOT NULL,          -- matches subjects.subject_id
    chapter_number INTEGER NOT NULL,
    title          TEXT NOT NULL,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(subject_id, chapter_number)
);
CREATE INDEX IF NOT EXISTS idx_chapters_subject ON chapters(subject_id);

-- ── 3. topics ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS topics (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chapter_id   UUID NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    subject_id   TEXT NOT NULL,            -- denormalized for fast subject-scoped queries
    topic_number TEXT NOT NULL,            -- e.g. "1.2"
    title        TEXT NOT NULL,
    created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(chapter_id, topic_number)
);
CREATE INDEX IF NOT EXISTS idx_topics_chapter ON topics(chapter_id);
CREATE INDEX IF NOT EXISTS idx_topics_subject ON topics(subject_id);

-- ── 4. questions ─────────────────────────────────────────────────────────────
-- Rich question bank: MCQ and subjective, sourced from external system
CREATE TABLE IF NOT EXISTS questions (
    id                      UUID PRIMARY KEY,   -- preserve external system's UUID
    subject_id              TEXT NOT NULL,
    chapter_id              UUID REFERENCES chapters(id) ON DELETE SET NULL,
    topic_id                UUID REFERENCES topics(id) ON DELETE SET NULL,
    source_document_id      UUID,               -- external doc reference (stored only)
    source_document_purpose TEXT DEFAULT 'general',

    question_text           TEXT NOT NULL,
    question_type           TEXT DEFAULT 'subjective',   -- subjective | objective
    question_format         TEXT DEFAULT 'subjective',   -- subjective | mcq (stored, not used for slides)
    options                 JSONB DEFAULT '{}',           -- {"A":{"text":"..."},...}
    option_images           JSONB DEFAULT '{}',
    question_image_url      TEXT,
    correct_answer          TEXT,
    explanation             TEXT,
    difficulty              TEXT DEFAULT 'Medium',        -- Easy | Medium | Hard
    marks                   INTEGER DEFAULT 4,

    is_verified             BOOLEAN DEFAULT FALSE,
    is_ai_generated         BOOLEAN DEFAULT TRUE,

    -- Pre-generation tracking
    is_pregen_done          BOOLEAN DEFAULT FALSE,
    cache_id                UUID,              -- → teaching_qa_cache.id once pregen done

    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_questions_topic   ON questions(topic_id);
CREATE INDEX IF NOT EXISTS idx_questions_subject ON questions(subject_id);
CREATE INDEX IF NOT EXISTS idx_questions_chapter ON questions(chapter_id);
CREATE INDEX IF NOT EXISTS idx_questions_pregen  ON questions(is_pregen_done)
    WHERE is_pregen_done = FALSE;

-- ── 5. Extend teaching_qa_cache with chapter/topic ────────────────────────────
ALTER TABLE teaching_qa_cache
    ADD COLUMN IF NOT EXISTS chapter_id TEXT,
    ADD COLUMN IF NOT EXISTS topic_id   TEXT;

CREATE INDEX IF NOT EXISTS idx_cache_topic   ON teaching_qa_cache(topic_id)   WHERE topic_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cache_chapter ON teaching_qa_cache(chapter_id) WHERE chapter_id IS NOT NULL;
