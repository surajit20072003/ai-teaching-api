# AI Teaching API — Project Understanding

> Full project documentation for future reference.
> Last updated: 2026-07-28

---

## 1. What This Project Does

The AI Teaching API is an automated educational content platform that:
1. Accepts document uploads (PDF, DOCX, TXT) from admins
2. Extracts and chunks the text, creates vector embeddings
3. Predicts questions students might ask from the content
4. Pre-generates answers, images, audio, and video explanations
5. Serves a mobile-facing AI Teaching Assistant that answers student questions in real-time

**Target users:** School students (ages 14–18) studying NCERT/state-board subjects in Hindi + English.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        INTERNET / MOBILE APP                     │
└──────────────────────────────┬───────────────────────────────────┘
                               │ HTTPS
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  CPU SERVER (main.py — FastAPI)                                 │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ documents  │ │ questions  │ │ pregen     │ │   notes    │   │
│  │   router   │ │   router   │ │   router   │ │   router   │   │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘   │
│        │              │              │              │           │
│  ┌─────┴──────────────┴──────────────┴──────────────┴──────┐   │
│  │              CORE MODULES (CPU-side)                      │   │
│  │  document_processor → chunking + embedding               │   │
│  │  text_answer_generator →  free LLM answers               │   │
│  │  note_service → orchestrate notes + answers               │   │
│  │  slide_generator → Manim video (HTTP to GPU)              │   │
│  │  image_generator → Flux images (HTTP to GPU)              │   │
│  │  tts_client → ElevenLabs / Gemini TTS                     │   │
│  │  b2_client → Backblaze B2 upload                          │   │
│  │  local_storage → disk I/O + write_image / write_audio     │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP (internal Docker network)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  GPU SERVER (main_gpu.py — FastAPI)                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Wan2GP (port 9090) — Manim + Flux image generation       │   │
│  │  - Receives job requests from CPU server                  │   │
│  │  - Runs Manim Python scripts → MP4 videos                 │   │
│  │  - Runs Flux diffusion → PNG images                       │   │
│  │  - Returns B2 upload URLs to CPU                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  EXTERNAL SERVICES                                               │
│  -  API →  free LLMs (Nemotron, Gemma, etc.)          │
│  - Backblaze B2 → image/audio/video storage                     │
│  - ElevenLabs / Gemini TTS → voice synthesis                    │
│  - PostgreSQL + pgvector → DB + vector search                   │
│  - Redis (optional) → caching                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Two-Server Model: CPU vs GPU

| Aspect | CPU Server | GPU Server |
|--------|-----------|------------|
| **Env flag** | `IS_CPU_SERVER=true` | Not set (or `false`) |
| **Runs** | FastAPI app (main.py) | FastAPI app (main_gpu.py) |
| **LLM calls** | Yes —  free models or Ollama | No |
| **Image/Video gen** | No (forwards to GPU via HTTP) | Yes — Wan2GP |
| **Port** | 8000 | 8001 |
| **Heavy work** | Orchestration, DB, embeddings, text | GPU rendering (Manim, Flux) |

**Communication:** CPU → GPU via HTTP (`http://host.docker.internal:9090` inside Docker).

**Pattern used everywhere:**
```python
_IS_CPU = os.getenv("IS_CPU_SERVER", "false").lower() == "true"
# In GPU-only endpoints:
if _IS_CPU:
    raise HTTPException(503, "Runs on GPU server only")
```

---

## 4. Directory Structure

```
ai-teaching-api/
├── main.py                    # CPU server entrypoint (FastAPI)
├── main_gpu.py                # GPU server entrypoint (if exists)
│
├── db/
│   ├── models.py              # SQLAlchemy ORM models
│   └── migrations/
│       ├── 001_initial.sql
│       ├── ...
│       └── 011_topic_notes.sql  # NEW: topic_notes table
│
├── routers/
│   ├── documents.py           # Document upload/list/delete
│   ├── questions.py           # Student Q&A API
│   ├── text_answer.py         # Text answer endpoint
│   ├── pregen.py              # Pre-generation control (batch jobs)
│   ├── notes.py               # NEW: Textbook notes CRUD
│   ├── notes_worker.py        # NEW: GPU worker re-export
│   ├── admin_tiers.py         # Admin tier management
│   └── content.py             # Subject/chapter/topic CRUD
│
├── core/
│   ├── document_processor.py  # PDF/DOCX → chunks + embeddings
│   ├── text_answer_generator.py  #  free LLM text answers
│   ├── slide_generator.py     # Manim video via HTTP to GPU
│   ├── image_generator.py     # Flux images via HTTP to GPU
│   ├── tts_client.py          # Text-to-speech (ElevenLabs/Gemini)
│   ├── pregen.py              # Batch pre-generation engine
│   ├── b2_client.py           # Backblaze B2 upload
│   ├── local_storage.py       # Disk I/O, write_image, write_audio
│   ├── cache.py               # Question caching + Redis
│   ├── semantic_check.py      # Vector similarity checks
│   ├── subject_gate.py        # Subject validation
│   │
│   ├── prompts/               # LLM prompt text files
│   │   ├── manim_system_prompt.txt
│   │   ├── manim_repair_prompt.txt
│   │   ├── image_system_prompt.txt
│   │   ├── note_generation_prompt.txt   # NEW
│   │   └── answer_generation_prompt.txt # NEW
│   │
│   └── notes/                 # NEW: Notes subsystem
│       ├── __init__.py
│       ├── note_service.py    # CPU orchestrator
│       └── note_generator.py  # GPU worker (Wan2GP images)
│
├── .env                       # Environment variables
├── .claude/
│   ├── settings.json
│   └── workflows/
└── PROJECT_UNDERSTANDING.md   # This file
```

---

## 5. Database Schema (Key Tables)

### `documents`
| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID | Primary key |
| `subject_id` | TEXT | Subject (e.g. "physics") |
| `chapter_id` | UUID | Chapter reference |
| `topic_id` | UUID | Topic reference |
| `title` | TEXT | Document title |
| `filename` | TEXT | Original filename |
| `local_raw_path` | TEXT | Path to original file on disk |
| `local_processed_path` | TEXT | Path to extracted.txt on disk |
| `b2_url` | TEXT | Backblaze B2 URL of raw file |
| `total_chunks` | INT | Number of text chunks |
| `status` | TEXT | `processing` → `ready` → `notes_generating` → `done`/`notes_failed` |
| `language` | TEXT | e.g. `hi-IN`, `en-US` |
| `pregen_total` | INT | Total questions to pre-generate |
| `pregen_done` | INT | Completed pregen rows |
| `content_markdown` | TEXT | **NEW**: Full extracted markdown for notes generation |

### `topic_notes` (NEW — migration 011)
| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID | Primary key |
| `document_id` | UUID | FK → documents |
| `subject_id` | TEXT | Subject |
| `chapter_id` | UUID | Optional chapter |
| `topic_id` | UUID | Optional topic (unique with document_id) |
| `note_sections` | JSONB | Array of structured note sections |
| `note_image_urls` | JSONB | B2 URLs for note diagrams |
| `note_latex_formulas` | JSONB | Extracted formulas with LaTeX |
| `question_answers` | JSONB | Full Q&A pairs for this topic |
| `answer_image_urls` | JSONB | B2 URLs for answer diagrams |
| `notes_status` | TEXT | `pending` / `generating` / `done` / `failed` |
| `error_message` | TEXT | Failure reason if failed |
| `generated_at` | TIMESTAMP | Completion time |

### `document_chunks`
| Column | Type | Purpose |
|--------|------|---------|
| `document_id` | UUID | FK → documents |
| `chunk_index` | INT | Order in document |
| `section_title` | TEXT | Detected heading |
| `chunk_text` | TEXT | The text content |
| `chunk_embedding` | vector(384) | Sentence-transformer embedding |

### `teaching_qa_cache`
| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID | Primary key |
| `subject_id` | TEXT | Subject |
| `question_hash` | TEXT | SHA256 of question text (dedup key) |
| `question_text` | TEXT | The question |
| `variation_number` | INT | Which variation of this question |
| `pregen_status` | TEXT | `pending` / `processing` / `done` / `failed` |
| `answer_text` | TEXT | Pre-generated answer |
| `answer_images` | JSONB | B2 URLs for answer images |
| `audio_b2_url` | TEXT | B2 URL of TTS audio |
| `slide_b2_url` | TEXT | B2 URL of Manim video |
| `used_count` | INT | How many times served (for cache stats) |

### `questions`
| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID | Primary key |
| `subject_id` | TEXT | Subject |
| `chapter_id` | UUID | FK → chapters |
| `topic_id` | UUID | FK → topics |
| `question_text` | TEXT | The question |
| `question_format` | TEXT | `subjective` / `mcq` / `fill_blank` |
| `options` | JSONB | MCQ options if applicable |
| `correct_answer` | TEXT | Correct answer |

---

## 6. Key Pipelines

### 6.1 Document Upload Pipeline

```
Admin uploads PDF
    │
    ▼
POST /documents/upload
    │
    ├─► process_document() → extract text, chunk, embed
    │       │
    │       ├─ Save extracted.txt → local_processed_path
    │       ├─ Save chunks → document_chunks table (with embeddings)
    │       └─ Return: local_raw_path, local_processed_path, chunks[]
    │
    ├─► Upload raw file → Backblaze B2
    │
    ├─► Insert Document row (status="processing")
    │
    ├─► Insert DocumentQuestion rows for admin-provided questions
    │
    ├─► Insert teaching_qa_cache rows (pregen_status="pending")
    │
    ├─► background_tasks.add_task(_launch_pregen)
    │       │
    │       ├─► Mark Document.status = "ready"
    │       ├─► Read extracted.txt → save to content_markdown
    │       └─► background_tasks.add_task(_auto_generate_notes)
    │               │
    │               └─► generate_notes_for_document()
    │                       │
    │                       ├─ Split markdown by topic
    │                       ├─ For each topic (parallel, max 3):
    │                       │   ├─ Call FREE_MODELS → structured notes
    │                       │   ├─ Generate images → Wan2GP → B2 URLs
    │                       │   └─ Answer all questions for topic
    │                       ├─ Save to topic_notes (JSONB)
    │                       └─ Mark document status done/partial
    │
    └─► Return success to client
```

### 6.2 Student Question Flow (Real-Time)

```
Student asks question in app
    │
    ▼
POST /questions/ask
    │
    ├─► subject_gate.gate_subject() → verify allowed subject
    │
    ├─► Cache lookup (hash_question → Redis/DB)
    │   └─ If cached + fresh → return immediately
    │
    ├─► Vector search (pgvector) → find similar questions + chunks
    │
    ├─► text_answer_generator →  free model answer
    │   ├─ Try nvidia/nemotron-3-ultra-550b:free
    │   ├─ Fallback: google/gemma-4-31b-it:free
    │   └─ ... (6 models in priority order)
    │
    ├─► LLM judge → pick best answer from candidates
    │
    ├─► image_generator → generate images (GPU)
    │
    ├─► tts_client → generate audio (ElevenLabs or Gemini)
    │
    ├─► slide_generator → Manim video (GPU)
    │
    ├─► Save to teaching_qa_cache
    │
    └─► Return answer + images + audio + video URLs
```

### 6.3 Pre-Generation Pipeline (Batch)

```
Admin calls POST /pregen/start { subjectId, limit }
    │
    ▼
run_pregen_batch()
    │
    ├─► Query teaching_qa_cache WHERE pregen_status = 'pending'
    │
    ├─► For each question (batched, parallel):
    │   ├─► Vector search → find relevant chunks
    │   ├─► text_answer_generator → LLM answer
    │   ├─► llm_judge → pick best
    │   ├─► image_generator → images (GPU)
    │   ├─► tts_client → audio
    │   ├─► slide_generator → Manim video (GPU)
    │   └─► Update teaching_qa_cache row (pregen_status='done')
    │
    └─► Update document.pregen_done counter
```

---

## 7. Free LLM Strategy

**Why free models?** The app is for government school students — zero API cost is critical.

**Current provider:**  free tier

**Model fallback chain (same in `text_answer_generator.py` and `core/notes/note_service.py`):**
```
1. nvidia/nemotron-3-ultra-550b-a55b:free   ← try first
2. google/gemma-4-31b-it:free               ← strong instruction following
3. google/gemma-4-26b-a4b-it:free           ← fast alternative
4. nvidia/nemotron-3-super-120b-a12b:free
5. nvidia/nemotron-3-nano-30b-a3b:free      ← smallest, last resort
6. /auto                           ←  picks
```

**Fallback to local Ollama:**
```bash
NOTES_USE_LOCAL_OLLAMA=true  # or TEXT_USE_LOCAL_OLLAMA
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=:32b
```

**Pattern (from `subject_gate.py`):**
```python
async def _call_llm():
    if NOTES_USE_LOCAL_OLLAMA:
        return await _call_ollama()
    return await _call_openrouter_free()
```

**Rate limiting:** All callers handle HTTP 429 by trying the next model in the list.

---

## 8. Storage Layout (Local Disk)

```
/sdb-disk/
├── raw/                  # Original uploaded files
│   └── {subject_id}/{doc_id}/{filename}
├── processed/            # Extracted text
│   └── {subject_id}/{doc_id}/extracted.txt
├── audio/                # TTS audio files
│   └── {subject_id}/{doc_id}/{question_hash}.mp3
├── slides/               # Manim-generated videos
│   └── {subject_id}/{doc_id}/{question_hash}.mp4
├── images/               # Generated images
│   └── {subject_id}/{doc_id}/{hash}.png
└── logs/                 # Operation logs
    └── uploads/
    └── errors/
```

**Helper functions** (in `core/local_storage.py`):
- `write_doc_meta()` — save metadata JSON
- `write_image()` — save PNG + return local path
- `write_audio()` — save MP3 + return local path
- `read_slide_cache()` — check if video already exists
- `write_slide_cache()` — save Manim output

---

## 9. Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `IS_CPU_SERVER` | Gate CPU-only endpoints | `false` |
| `DATABASE_URL` | PostgreSQL connection string | — |
| `OPENROUTER_API_KEY` | Free LLM access | — |
| `OPENROUTER_URL` |  base URL | ` |
| `OLLAMA_URL` | Local Ollama server | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | Default Ollama model | `:32b` |
| `NOTES_USE_LOCAL_OLLAMA` | Toggle notes to use Ollama | `false` |
| `WAN2GP_URL` | GPU worker base URL | `http://host.docker.internal:9090` |
| `WAN2GP_API_KEY` | GPU worker auth | `""` |
| `B2_BUCKET` | Backblaze B2 bucket name | — |
| `B2_APP_KEY_ID` | B2 auth | — |
| `B2_APP_KEY` | B2 auth | — |
| `ELEVENLABS_API_KEY` | TTS provider | — |
| `REDIS_URL` | Cache backend | — |

---

## 10. New Feature: Textbook Notes (Implemented 2026-07-28)

### What it does
Automatically generates structured textbook-style notes from uploaded documents, including:
- **Structured sections** with explanations, key points, and image descriptions
- **LaTeX formulas** extracted from the content
- **Generated images** (diagrams) for visual learning
- **Question answers** for every question in the document's topics

### Flow
1. Admin uploads a document → `POST /documents/upload`
2. After processing, `_launch_pregen` saves `content_markdown` to the document
3. `_auto_generate_notes` fires as a background task
4. `note_service.generate_notes_for_document()`:
   - Splits markdown by topic
   - Calls FREE_MODELS to structure notes
   - Calls Wan2GP (GPU) to generate diagrams
   - Generates answers for all questions
   - Saves everything to `topic_notes` table

### API Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/notes/generate/{document_id}` | Queue notes generation |
| `GET` | `/notes/status/{document_id}` | Check progress |
| `GET` | `/notes/topic/{topic_id}` | Get notes for one topic |
| `GET` | `/notes/document/{document_id}` | Get all notes for document |
| `POST` | `/notes/retry/{document_id}` | Retry failed generations |

### Files Added
| File | Purpose |
|------|---------|
| `core/notes/__init__.py` | Package marker |
| `core/notes/note_service.py` | CPU orchestrator (LLM calls + DB writes) |
| `core/notes/note_generator.py` | GPU worker (Wan2GP image gen) |
| `routers/notes.py` | CPU API router |
| `routers/notes_worker.py` | GPU worker router re-export |
| `core/prompts/note_generation_prompt.txt` | LLM prompt for note structuring |
| `core/prompts/answer_generation_prompt.txt` | LLM prompt for question answers |
| `migrations/011_topic_notes.sql` | DB migration for topic_notes table |
| `db/models.py` (edited) | Added TopicNote SQLAlchemy model |

### Cost
**$0.00** — uses the same free  models as the rest of the system.

---

## 11. Important Patterns to Follow

### A. CPU/GPU Split
```python
_IS_CPU = os.getenv("IS_CPU_SERVER", "false").lower() == "true"

@router.post("/something")
async def do_something():
    if _IS_CPU:
        raise HTTPException(503, "Runs on GPU server only")
    # ... GPU work ...
```

### B. Free Model Fallback Chain
```python
FREE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "/auto",
]

for model in FREE_MODELS:
    try:
        resp = await call_openrouter(model, ...)
        if resp.ok: return resp.json()
    except: continue
```

### C. JSON Extraction from LLM
```python
def _extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0).strip() if match else text.strip()
```

### D. Background Tasks in FastAPI
```python
@router.post("/something")
async def endpoint(background_tasks: BackgroundTasks, db = Depends(get_db)):
    # Save to DB first
    await db.commit()
    # Then queue background work
    background_tasks.add_task(_background_work, arg1, arg2)
    return {"status": "queued"}

async def _background_work(arg1, arg2):
    async with AsyncSessionLocal() as db:
        # Do work with fresh DB session
        pass
```

### E. Image Storage Pattern
```python
# 1. Generate on GPU → get bytes
img_bytes = await wan2gp_generate(prompt)

# 2. Save locally
local_path = write_image(subject_id, doc_id, filename, img_bytes)

# 3. Upload to B2
b2_url = upload_to_b2(local_path, f"notes/{subject_id}/{doc_id}/")

# 4. Store b2_url in JSONB column
```

---

## 12. Running the Servers

```bash
# CPU server
cd /home2/ai-teaching-api
IS_CPU_SERVER=true uvicorn main:app --host 0.0.0.0 --port 8000

# GPU server
cd /home2/ai-teaching-api
uvicorn main_gpu:app --host 0.0.0.0 --port 8001
```

**With Docker Compose:**
```yaml
services:
  cpu:
    build: .
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    environment:
      - IS_CPU_SERVER=true
    ports: ["8000:8000"]

  gpu:
    build: .
    command: uvicorn main_gpu:app --host 0.0.0.0 --port 8001
    ports: ["8001:8001"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 13. Common Issues & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| "rate limited (429)" on all models |  free tier throttled | Wait 60s, retry; or switch to Ollama |
| Manim videos not generating | Wan2GP not reachable | Check `WAN2GP_URL` env var, GPU server running |
| Cache always miss | Hash mismatch | Use `hash_question()` from `core/cache.py` (SHA256) |
| B2 upload fails | Wrong credentials | Check `.env` B2_* variables |
| Notes generation slow | Many topics + LLM latency | Normal — each topic is ~30-60s |
| Document status stuck on "processing" | `_launch_pregen` crashed | Check logs, retry via `/pregen/start` |

---

## 14. File Map: "Where Does X Happen?"

| Feature | Primary File | Secondary Files |
|---------|-------------|-----------------|
| Document upload | `routers/documents.py` | `core/document_processor.py` |
| Text chunking | `core/document_processor.py` | — |
| Vector embeddings | `core/document_processor.py` | `core/embeddings.py` |
| LLM text answer | `core/text_answer_generator.py` | — |
| Question caching | `core/cache.py` | `routers/questions.py` |
| Subject validation | `core/subject_gate.py` | — |
| Image generation | `core/image_generator.py` | GPU → `Wan2GP` |
| Manim video | `core/slide_generator.py` | GPU → `Wan2GP` |
| TTS audio | `core/tts_client.py` | — |
| Pre-generation batch | `core/pregen.py` | `routers/pregen.py` |
| Textbook notes | `core/notes/note_service.py` | `routers/notes.py` |
| Notes GPU images | `core/notes/note_generator.py` | GPU → `Wan2GP` |
| B2 uploads | `core/b2_client.py` | — |
| Disk I/O | `core/local_storage.py` | — |
| DB models | `db/models.py` | migrations/*.sql |

---

## 15. Git Workflow

```bash
# Check status
git status

# See what changed
git diff

# Commit (NEVER use --no-verify or --no-gpg-sign)
git add <specific files>
git commit -m "Description"

# Push (ask user first!)
git push
```

**Rules:**
- Never skip hooks
- Never force push to main
- Always create new commits (don't amend unless asked)
- Review `git status` before any destructive command

---

*End of PROJECT_UNDERSTANDING.md*
