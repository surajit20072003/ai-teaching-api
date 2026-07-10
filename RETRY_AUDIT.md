# Retry System Audit — Code Analysis & Fix Plan

## Verdict Summary

| Retry Layer | Works? | Issue |
|---|---|---|
| Phase A — Re-generate missing text | ✅ Works correctly | None |
| Phase B — Retry missing images | ⚠️ Partially broken | Re-runs ALL slides, not just missing ones |
| Phase B — Retry missing audio | ⚠️ Partially broken | Re-runs ALL slides, not just missing ones |
| Phase C — Retry missing Manim | ❌ Broken | SQL query does not detect "done but Manim missing" rows |

---

## Bug 1: Phase B — Image & Audio Re-runs Everything (Wastes Time & Money)

### The Code (line 1522–1530)
```python
needs_img = any(
    not (s.get("infographicUrl") or "").startswith("http")
    for s in slides
)
needs_aud = any(
    not (s.get("audioUrl") or "").startswith("http")
    for s in slides
)
```

This checks if **ANY slide at all** is missing a URL. If yes, it calls `_pregen_media_only(row_dict, slides)` which re-generates **ALL slides in parallel** (see line 596).

### The Real Problem
Imagine a question has 7 slides. Slides 1–6 are perfect. Slide 7's audio failed. The retry will:
1. Re-request 7 images (6 already exist → wasted API calls)
2. Re-generate 7 audio files (6 already exist → wasted TTS calls)

This wastes money and slows down the retry pass significantly.

### The Fix Needed
`_pregen_media_only` should accept a set of "skip" slide indices, or better yet, only be called for the missing slides. The check should be **per-slide**, not **per-row**.

---

## Bug 2: Phase C — Manim Retry Doesn't Detect Rows That Need It

### The SQL Query (lines 1421–1447)
```sql
WHERE subject_id = :subj
  AND (
    pregen_status IN ('failed', 'pending')
    OR (
      pregen_status = 'done'
      AND (
        jsonb_array_length(COALESCE(presentation_slides,'[]'::jsonb)) = 0
        OR jsonb_array_length(COALESCE(slide_audio_urls->'urls','[]'::jsonb)) = 0
        OR EXISTS(SELECT 1 ... WHERE infographicUrl IS NULL)
        OR EXISTS(SELECT 1 ... WHERE audioUrl IS NULL)
      )
    )
  )
```

**There is NO check for missing Manim videos.** The SQL only detects rows missing images or audio. A row where every image and audio succeeded (status=`done`) but Manim failed is INVISIBLE to the retry scanner.

### The Proof
From your own logs:
```
[2026-07-07T08:47:17Z] [Retry] Found 0 incomplete rows — nothing to do.
```
Both questions were status `done` with images and audio present, so the query returned 0 rows — even though all 8 Manim slots had no video URLs!

### The Fix Needed
Add a 4th OR condition to the SQL to also detect rows where `visual_type=manim` slides exist but have no corresponding entry in `manim_video_urls`.

---

## Fix Plan

### Fix 1: Add Manim Detection to SQL Query

**File:** `core/pregen.py`, lines ~1426–1443

Replace the existing WHERE clause with this:
```sql
WHERE subject_id = :subj
  AND (
    pregen_status IN ('failed', 'pending')
    OR (
      pregen_status = 'done'
      AND (
        jsonb_array_length(COALESCE(presentation_slides,'[]'::jsonb)) = 0
        OR jsonb_array_length(COALESCE(slide_audio_urls->'urls','[]'::jsonb)) = 0
        OR EXISTS(
          SELECT 1 FROM jsonb_array_elements(
            COALESCE(presentation_slides,'[]'::jsonb)) s
          WHERE (s->>'infographicUrl') IS NULL OR (s->>'infographicUrl') = ''
        )
        OR EXISTS(
          SELECT 1 FROM jsonb_array_elements(
            COALESCE(presentation_slides,'[]'::jsonb)) s
          WHERE (s->>'audioUrl') IS NULL OR (s->>'audioUrl') = ''
        )
        -- NEW: detect manim slides that are missing their video URL
        OR EXISTS(
          SELECT 1 FROM jsonb_array_elements(
            COALESCE(presentation_slides,'[]'::jsonb)) WITH ORDINALITY AS t(s, pos)
          WHERE (s->>'visual_type') = 'manim'
            AND (
              manim_video_urls IS NULL
              OR NOT manim_video_urls ? (pos - 1)::text
              OR (manim_video_urls->>((pos - 1)::text))::jsonb->>'url' = ''
            )
        )
      )
    )
  )
```

### Fix 2: Per-Slide Granularity for Image & Audio Retry

**File:** `core/pregen.py` — the Phase B retry loop (lines ~1516–1572)

Instead of calling `_pregen_media_only` for the whole row, build a list of which slides need what and only process those:

```python
# Identify WHICH slides specifically are missing
slides_missing_img = {i for i, s in enumerate(slides) if not (s.get("infographicUrl") or "").startswith("http")}
slides_missing_aud = {i for i, s in enumerate(slides) if not (s.get("audioUrl") or "").startswith("http")}

if not slides_missing_img and not slides_missing_aud:
    # skip to Phase C (already works)
    ...
else:
    # Only process the specific missing slides
    for idx in slides_missing_img | slides_missing_aud:
        await _process_slide(idx, slides[idx], cache_id, language, subject)
```

This would be done by modifying `_pregen_media_only` to accept an `only_indices: set[int] | None = None` parameter so it skips already-completed slides.

---

## Priority

| Priority | Fix | Impact |
|---|---|---|
| 🔴 Critical | Fix 1 — SQL Manim detection | Manim will NEVER be retried without this |
| 🟡 Medium | Fix 2 — Per-slide media retry | Saves API cost and time on large batches |

Shall I implement both fixes now?
