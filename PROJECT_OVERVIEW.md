# AI Teaching API - Project Overview

The **AI Teaching API** is a comprehensive educational backend system designed to automatically generate rich, interactive slide-based video presentations for academic questions. It leverages a modern asynchronous Python stack (FastAPI), local and cloud-based Large Language Models (LLMs), and an array of external generation services for audio, images, and high-quality mathematical animations.

## Core Features & Workflow

The system takes raw questions (e.g., math, science, history) and a curriculum syllabus (uploaded documents), and automatically transforms them into a series of highly engaging "slides".

### 1. Document RAG & Curriculum Ingestion
- **Documents & Hierarchies:** Subjects have Chapters, which have Topics. Teachers upload textbook PDFs or `.txt` files.
- **RAG (Retrieval-Augmented Generation):** Uploaded documents are chunked into paragraphs and embedded using `sentence-transformers` (`all-MiniLM-L6-v2`). The embeddings are stored in PostgreSQL using `pgvector`.
- When an AI generates an explanation for a question, it searches these document chunks to ensure the explanation stays grounded in the specific textbook/curriculum.

### 2. The Pre-Generation Pipeline (Pregen)
To avoid making students wait for slow AI generation, the system proactively generates and caches content using a **3-Phase Background Orchestrator** (`core/pregen.py`).

* **Phase A: Text Generation (Ollama)**
  * It queries the local **Ollama** model (usually a high-parameter Llama or Qwen model) to break the question down into a logical sequence of slides.
  * The LLM creates JSON defining each slide's content, speaker script, formulas, and visual type (e.g., `static`, `equation`, `manim`).
  
* **Phase B: Media Generation (Evict & Fetch)**
  * Generating images and audio requires VRAM (GPU memory). The orchestrator specifically **evicts Ollama from VRAM** to free up memory before this phase.
  * **Images:** Calls out to the **Wan2GP** image generation service to create visual representations for slides that need them.
  * **Audio:** Calls out to the **VoxCPM** or **Sarvam TTS** services to synthesize high-quality human voiceovers based on the speaker script.
  
* **Phase C: Manim Animations (Local/Cloud)**
  * For complex math/physics concepts, the slide's visual type is marked as `manim`.
  * The system calls an advanced coding LLM (either via **OpenRouter** `deepseek-chat` or a local Ollama coder model) to write a Python script using the **Manim** library (the math animation engine built by 3Blue1Brown).
  * The script is executed inside a safe, isolated subprocess to render an `.mp4` video animation.

### 3. Caching & Storage Strategy
* **Local Storage Cache:** All raw generated assets (wav files, pngs, mp4s) are saved directly to the high-speed local NVMe drive (`/sdb-disk/ai-teaching/subjects/{subject_id}/cache/jobs/...`).
* **Cloud Backup (Backblaze B2):** As assets are finalized, they are synced asynchronously to a Backblaze B2 bucket (`Simplelectureaivideo`).
* **Database (teaching_qa_cache):** The final slide JSON, along with URLs to the images/audio/videos, are saved in the PostgreSQL database. When a student requests a question, the API responds instantly in milliseconds.

### 4. Smart Retry Mechanism
* Since network requests (TTS, Image Gen, LLM APIs) can occasionally fail, the pipeline has a "Layer 2" smart retry system (`/pregen/retry-media`).
* It periodically scans the database for rows that successfully finished Phase A but are missing specific images or audio files due to timeouts. It attempts to regenerate *only* the missing pieces without re-running the expensive text generation phase.

## Tech Stack

* **Web Framework:** FastAPI (Asynchronous Python)
* **Database:** PostgreSQL + `pgvector` extension (via `asyncpg` and SQLAlchemy)
* **Message Broker / Cache:** Redis (used for tracking pre-generation job limits and locks)
* **LLM Engine:** Local Ollama + Cloud OpenRouter
* **Media:** Manim (Video), VoxCPM/Sarvam (Audio), Wan2GP (Images)
* **Infrastructure:** Docker & Docker Compose

## Directory Structure

* **`core/`**: Contains the main business logic.
  * `pregen.py`: The background job orchestrator.
  * `slide_generator.py`: LLM prompting and slide logic.
  * `manim_generator.py`: Manim Python script generation and subprocess execution.
  * `image_generator.py`, `tts_client.py`: External API wrappers for media.
  * `embeddings.py`: Sentence-transformer RAG logic.
* **`routers/`**: FastAPI endpoints.
  * `pregen.py`: Triggers, status, and retries for background jobs.
  * `documents.py`: File uploads and chunking.
  * `questions.py`: CRUD for the core question bank.
* **`migrations/`**: Raw SQL schema setup files (e.g., `005_new_schema.sql`).
