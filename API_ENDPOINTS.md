# AI Teaching API — Complete Endpoint Reference

> **Architecture:** 2-Server Hybrid (CPU Master + GPU Worker)
>
> | Server | Role | IP | Port | URL |
> |--------|------|----|------|-----|
> | 🖥️ **CPU Master** | Student-facing API, Database, Redis | `116.202.230.124` | `8000` | `http://116.202.230.124:8000` |
> | ⚡ **GPU Worker** | Pre-generation factory (Ollama + Wan2GP + VoxCPM) | `69.197.145.4` | `8000` | `http://69.197.145.4:8000` |

---

## Server Assignment Legend

| Symbol | Meaning |
|--------|---------|
| 🖥️ CPU | Endpoint lives on CPU Master `116.202.230.124:8000` |
| ⚡ GPU | Endpoint lives on GPU Worker `69.197.145.4:8000` |
| 🔄 BOTH | Endpoint is available on both servers (same code) |
| 🚫 CPU BLOCKED | Endpoint exists on both but is **disabled** on CPU — returns HTTP 503 |

---

## 1. Core / System Endpoints

| Method | Path | Server | Description |
|--------|------|--------|-------------|
| `GET` | `/` | 🔄 BOTH | Serves the test frontend HTML dashboard |
| `GET` | `/health` | 🔄 BOTH | Health check — returns server role (`cpu_student_server` or `gpu_pregen_factory`) |
| `GET` | `/pregen-dashboard` | 🔄 BOTH | Pre-generation monitoring & control HTML dashboard |

### Health Check Example
```
GET http://116.202.230.124:8000/health
GET http://69.197.145.4:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "version": "2.0.0",
  "role": "cpu_student_server"   // or "gpu_pregen_factory"
}
```

---

## 2. Student / Teaching Endpoints

> These endpoints are the student-facing API. They should **only be called against the CPU server**.
> The CPU server has 16 workers and the persistent database/Redis needed to serve cache hits in milliseconds.

| Method | Path | Server | Description |
|--------|------|--------|-------------|
| `POST` | `/ai-teaching-assistant` | 🖥️ CPU | **Main endpoint.** Takes a question, runs 4-layer cache lookup (Redis → Local Disk → Postgres → Semantic), returns slides + audio + video. Falls back to live generation if all cache layers miss. |
| `GET` | `/search-questions` | 🖥️ CPU | Dual-source semantic search across the Q&A cache and uploaded document chunks using `pgvector`. |
| `GET` | `/subjects` | 🔄 BOTH | List all subjects with Q&A count and document count. |
| `POST` | `/subjects` | 🔄 BOTH | Create a new named subject. Deduplicates by name (case-insensitive). |

### POST `/ai-teaching-assistant` — Full Request Body
```json
{
  "question": "What is Newton's second law?",
  "subjectId": "uuid-of-subject",
  "subjectName": "Physics",
  "language": "hi-IN",
  "mode": "full"
}
```
**Modes:**
- `full` — Full 4-layer cache + fallback live generation (default)
- `doubt` — Explains why an answer is right/wrong for a student
- `detect_topic` — Detects which topic/subject the question belongs to

### GET `/search-questions` — Query Params
```
GET /search-questions?q=photosynthesis&subject_id=<uuid>&limit=5
```

---

## 3. Media / TTS / Image Endpoints

> Used by the CPU server's live-generation path (when all cache layers miss) and directly by frontend testing.

| Method | Path | Server | Description |
|--------|------|--------|-------------|
| `POST` | `/sarvam-tts` | 🖥️ CPU | Convert narration text to speech using Sarvam AI TTS. Returns base64-encoded WAV audio. |
| `POST` | `/ai-generate-image` | 🖥️ CPU | Generate a single educational infographic image via Wan2GP. Returns a B2 CDN URL. |
| `POST` | `/save-presentation-audio` | 🖥️ CPU | Accept base64 WAV chunks, upload to Backblaze B2, and update the `teaching_qa_cache` row with audio URLs. |

### POST `/sarvam-tts`
```json
{
  "text": "Newton's second law states that Force equals mass times acceleration.",
  "languageCode": "hi-IN",
  "gender": "male"
}
```

### POST `/ai-generate-image`
```json
{
  "cacheId": "optional-uuid",
  "slideIndex": 0,
  "prompt": "Diagram showing force mass and acceleration relationship"
}
```

### POST `/save-presentation-audio`
```json
{
  "cache_id": "uuid-of-cache-row",
  "language": "hi-IN",
  "slides": [
    { "slideIndex": 0, "base64Chunks": ["...base64..."] },
    { "slideIndex": 1, "base64Chunks": ["...base64..."] }
  ]
}
```

---

## 4. Pre-Generation Endpoints (`/pregen/*`)

> **Critical Rule:** `/pregen/start` and `/pregen/retry-media` are **GPU-only**.
> The CPU server will return HTTP 503 if you call them on it.
> All `/pregen/status` and count endpoints are read-only and work on both servers.

| Method | Path | Server | Description |
|--------|------|--------|-------------|
| `POST` | `/pregen/start` | ⚡ GPU ONLY | Start batch pre-generation for a subject. Runs in background — returns immediately. **Blocked on CPU (503).** |
| `POST` | `/pregen/stop` | ⚡ GPU | Request a graceful stop of the running batch. Current row finishes before stopping. |
| `GET` | `/pregen/status` | 🔄 BOTH | Live in-memory progress of the running (or last completed) pregen job. |
| `GET` | `/pregen/pending-count` | 🔄 BOTH | Count rows in `pending` or `processing` status. Can be scoped by `?subject_id=`. |
| `GET` | `/pregen/failed-count` | 🔄 BOTH | Count rows in `failed` status. Can be scoped by `?subject_id=`. |
| `POST` | `/pregen/retry-failed` | 🔄 BOTH | Reset all `failed` rows back to `pending` so next `/pregen/start` will re-process them. |
| `POST` | `/pregen/add-question` | 🔄 BOTH | Manually add a single question to the pregen queue for a given subject. |
| `POST` | `/pregen/retry-media` | ⚡ GPU ONLY | Smart media retry — re-generates **only** missing images/audio for `done` rows without re-running Ollama. **Blocked on CPU (503).** |
| `GET` | `/pregen/retry-status` | 🔄 BOTH | Live progress of the running (or last completed) media retry job. |

### POST `/pregen/start` — Full Request Body (call on GPU: `69.197.145.4:8000`)
```json
{
  "subjectId": "uuid-of-subject",
  "limit": 500,
  "topicId": "optional-uuid",
  "chapterId": "optional-uuid",
  "manim_provider": "local"
}
```
- `limit`: max questions to process in this batch (default: 500)
- `manim_provider`: `"local"` (Ollama devstral:24b) or `"openrouter"` (DeepSeek API)

### POST `/pregen/retry-media` (call on GPU: `69.197.145.4:8000`)
```json
{
  "subjectId": "uuid-of-subject"
}
```

### POST `/pregen/retry-failed`
```json
{
  "subjectId": "uuid-of-subject"
}
```

### POST `/pregen/add-question`
```json
{
  "subjectId": "uuid-of-subject",
  "question": "Explain the process of osmosis."
}
```

### GET `/pregen/pending-count` — Query Params
```
GET /pregen/pending-count                        # all subjects
GET /pregen/pending-count?subject_id=<uuid>      # scoped to one subject
```

---

## 5. Document Endpoints (`/documents/*`)

> Document uploads **must go to the CPU server**. The CPU server handles chunking, embedding generation, and B2 upload. The GPU server has no `/app/storage/` volume configured for documents.

| Method | Path | Server | Description |
|--------|------|--------|-------------|
| `POST` | `/documents/upload` | 🖥️ CPU | Upload a PDF/DOCX/TXT. Chunks it, generates `pgvector` embeddings, stores to local disk + B2. Accepts optional list of questions to auto-queue for pregen. |
| `GET` | `/documents` | 🖥️ CPU | List all documents for a subject. Filter by `?status=ready\|processing\|failed`. |
| `GET` | `/documents/{doc_id}/status` | 🖥️ CPU | Pre-generation status for a specific document — shows chunks, % complete. |
| `DELETE` | `/documents/{doc_id}` | 🖥️ CPU | Delete a document and all its chunks, cache entries, and local files. Also attempts B2 cleanup. |

### POST `/documents/upload` — Multipart Form Data
```
POST http://116.202.230.124:8000/documents/upload
Content-Type: multipart/form-data

subject_id:  uuid-of-subject
title:       "Physics Chapter 1 - Motion"
language:    "hi-IN"
chapter_id:  optional-uuid
topic_id:    optional-uuid
questions:   "What is velocity?\nWhat is acceleration?\nExplain Newton's first law."
file:        <binary PDF/DOCX/TXT — max 50MB>
```

### GET `/documents`
```
GET http://116.202.230.124:8000/documents?subject_id=<uuid>&status=ready
```

---

## 6. Question Bank Endpoints (`/questions/*`)

> Used by admin systems to import questions from external platforms (e.g., a school's exam management system). Also used to monitor pregen progress per chapter/topic.

| Method | Path | Server | Description |
|--------|------|--------|-------------|
| `POST` | `/questions/import` | 🖥️ CPU | Bulk import questions with full subject/chapter/topic hierarchy and optional document RAG content. Idempotent — safe to call multiple times. |
| `GET` | `/questions` | 🔄 BOTH | List questions with optional filters. |
| `GET` | `/questions/status` | 🔄 BOTH | All-subjects overview of question bank + pregen progress. |
| `GET` | `/questions/status/{subject_id}` | 🔄 BOTH | Detailed pregen progress broken down by chapter → topic tree for one subject. |
| `GET` | `/questions/{question_id}` | 🔄 BOTH | Single question detail with full pipeline state (cache status, slides count, audio, images). |

### POST `/questions/import` — Full Request Body
```json
{
  "subject": {
    "id": "external-subject-uuid",
    "name": "Mathematics",
    "slug": "mathematics"
  },
  "chapter": {
    "id": "external-chapter-uuid",
    "chapter_number": 1,
    "title": "Real Numbers"
  },
  "topic": {
    "id": "external-topic-uuid",
    "topic_number": "1.1",
    "title": "Euclid's Division Lemma"
  },
  "document": {
    "id": "external-doc-uuid",
    "display_name": "NCERT Chapter 1",
    "parsed_json": {
      "content_markdown": "## Real Numbers\nReal numbers are..."
    }
  },
  "questions": [
    {
      "id": "external-question-uuid",
      "question_text": "State Euclid's Division Lemma.",
      "question_format": "subjective",
      "correct_answer": "For any positive integers a and b...",
      "difficulty": "Medium",
      "marks": 4
    }
  ]
}
```

### GET `/questions` — Query Params
```
GET /questions?subject_id=<uuid>&chapter_id=<uuid>&topic_id=<uuid>&is_pregen_done=false&limit=50&offset=0
```

---

## 7. Infrastructure / Utility

| Service | Server | Port | URL | Purpose |
|---------|--------|------|-----|---------|
| **FastAPI Docs (Swagger)** | 🖥️ CPU | `8000` | `http://116.202.230.124:8000/docs` | Interactive API explorer |
| **FastAPI Docs (Swagger)** | ⚡ GPU | `8000` | `http://69.197.145.4:8000/docs` | Interactive API explorer |
| **Adminer (DB GUI)** | 🖥️ CPU | `8085` | `http://116.202.230.124:8085` | PostgreSQL admin panel |
| **PostgreSQL** | 🖥️ CPU | `5433` | `116.202.230.124:5433` | Database (GPU connects here remotely) |
| **Redis** | 🖥️ CPU | `6381` | `116.202.230.124:6381` | Cache/state (GPU connects here remotely) |

---

## 8. Typical Operational Workflows

### Workflow A: Import Questions & Run Pregen
```
1. POST http://116.202.230.124:8000/questions/import    (CPU — import question bank)
2. POST http://116.202.230.124:8000/documents/upload    (CPU — upload textbook)
3. POST http://69.197.145.4:8000/pregen/start           (GPU — start batch)
4. GET  http://69.197.145.4:8000/pregen/status          (GPU — poll progress)
5. POST http://69.197.145.4:8000/pregen/retry-media     (GPU — fix any missing media)
```

### Workflow B: Student Asks a Question
```
1. POST http://116.202.230.124:8000/ai-teaching-assistant   (CPU — get presentation)
   → L1: Redis (~0.5ms hit)
   → L2: Local disk (~1ms hit)
   → L3: Postgres exact hash (~3ms hit)
   → L4: pgvector semantic search + LLM judge (~200ms hit)
   → L5: Live generation (~3 min, CPU uses OpenRouter + Sarvam)
```

### Workflow C: Fix Failed Pregen Rows
```
1. GET  http://69.197.145.4:8000/pregen/failed-count?subject_id=<uuid>
2. POST http://69.197.145.4:8000/pregen/retry-failed   { "subjectId": "<uuid>" }
3. POST http://69.197.145.4:8000/pregen/start          { "subjectId": "<uuid>", "limit": 500 }
```

---

## 9. GPU-Only Endpoint Behaviour on CPU

If you accidentally call a GPU-only endpoint against the CPU server, it will return:

```json
HTTP 503 Service Unavailable
{
  "detail": "Pre-generation runs on GPU server only. This is the CPU server."
}
```

This is controlled by the `IS_CPU_SERVER=true` environment variable set in the CPU server's `.env`.

---

## 10. Quick Reference — Base URLs

```
# Student-facing & Admin (CPU — always use this for reads and document uploads)
BASE_CPU = http://116.202.230.124:8000

# Pre-generation control (GPU — only for pregen batch operations)
BASE_GPU = http://69.197.145.4:8000

# DB Admin
ADMINER  = http://116.202.230.124:8085
API_DOCS_CPU = http://116.202.230.124:8000/docs
API_DOCS_GPU = http://69.197.145.4:8000/docs
```
