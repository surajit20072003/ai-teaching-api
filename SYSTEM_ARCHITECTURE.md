# AI Teaching API — Complete System Architecture

This document provides a comprehensive overview of how the AI Teaching API is structured across its dual-server architecture, how the database handles vector search and caching, and how the content generation pipelines work.

---

## 1. Dual-Server Architecture

To ensure high availability for students while running extremely heavy AI background tasks, the system is split across two separate physical servers:

### A. The CPU Server (Student-Facing)
- **Role:** Handles all live HTTP traffic from students and admin dashboards.
- **Components:**
  - **FastAPI Application:** Serves the `/ai-teaching-assistant` endpoint and the new `/ai-text-answer` endpoint.
  - **PostgreSQL Database (`ai-teaching-postgres`):** The primary, central database. It runs *only* on this server.
  - **Redis Cache:** Handles fast exact-match lookups and distributed locking.
- **Capabilities:** Can generate real-time answers by delegating to fast cloud APIs (OpenRouter, local FreeLLMAPI) but relies heavily on retrieving pre-generated content from the database.

### B. The GPU Server (Background Factory)
- **Role:** A stateless, heavy-lifting factory that continuously runs in the background to pre-generate hundreds of thousands of slides, videos, and audio clips.
- **Components:**
  - **Ollama:** Runs local, high-parameter LLMs (like Qwen or Llama 3) for text generation.
  - **Wan2GP / Flux:** Generates images.
  - **VoxCPM / Sarvam:** Generates high-quality text-to-speech audio.
  - **Manim Engine:** Renders complex mathematical animations.
- **Workflow:** It periodically connects to the CPU Server's database over an SSH tunnel (Port 81), fetches pending questions, runs the heavy generation models, saves the media files to Backblaze B2, and writes the final JSON slides back into the CPU Server's `teaching_qa_cache` table.

---

## 2. Database & pgvector Architecture

The system uses a single source of truth: the PostgreSQL database running on the CPU server. 

### Core Tables
1. **`document_chunks` (The Brain):**
   - When a teacher uploads a textbook, it is split into small paragraphs.
   - Each paragraph is passed through an embedding model (`all-MiniLM-L6-v2`) which turns the text into an array of 384 numbers (a vector).
   - `pgvector` stores this vector using a highly optimized HNSW index.
   - **Usage:** Whenever a student asks a question, the API converts their question into a vector and asks PostgreSQL: *"Find the 3 chunks whose vectors are closest to this question's vector."* This is called **RAG (Retrieval-Augmented Generation)**. It ensures the AI never hallucinates and only teaches what is in the syllabus.

2. **`teaching_qa_cache` (The Speed Layer):**
   - Stores the massive, finalized JSON payloads containing slides, scripts, and media URLs.
   - **Usage:** If a student asks a question that the GPU server has already pre-generated, the API pulls it directly from this table and returns it in ~3ms. No AI generation is needed.

3. **`text_answer_cache` (New):**
   - Similar to the QA cache, but specifically stores shorter, text-only answers.
   - Also uses `pgvector` to store a vector of the question. This allows the system to reuse an old answer if a student asks a *semantically similar* question (e.g., "What is Newton's 2nd law?" vs "Explain the second law of motion").

---

## 3. The 5-Layer Caching System

To minimize API costs and maximize speed, the real-time endpoints (`/ai-teaching-assistant` and `/ai-text-answer`) use a 5-layer cascading cache:

- **[L1] Redis Exact Match (~0.5ms):** Checks Redis for the exact question hash.
- **[L2] Local Disk JSON (~1ms):** Checks the local NVMe drive for a finalized JSON file.
- **[L3] Postgres Exact Match (~3ms):** Checks the database for the exact hash.
- **[L4] pgvector Semantic Search (~30ms):** Uses `pgvector` to find a *similar* question that was previously answered. If found, an LLM acts as a "Judge" to verify if the cached answer is good enough for the new question.
- **[L5] RAG Generation (Real-Time):** If all caches miss, the system fetches textbook context from `document_chunks`, calls an LLM (FreeLLMAPI or OpenRouter) to generate a fresh answer, saves it to all cache layers, and returns it to the user.

---

## 4. Media Storage

Media files (audio, images, videos) are massive and would crash the database if stored inside it.
1. **Local NVMe (`/app/storage` or `/sdb-disk`):** All generated media is first written to the local disk.
2. **Backblaze B2:** A background process uploads these local files to a cloud bucket (`Simplelectureaivideo`).
3. **Database Links:** The database only stores the lightweight public URL strings pointing to Backblaze. This keeps the database lean and lightning fast.
