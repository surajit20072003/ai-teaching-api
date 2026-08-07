"""
core/notes/note_service.py — CPU Orchestrator
=============================================

Orchestrates textbook note generation for a document's topics.

Pipeline per document:
  1. Load document + content_markdown
  2. Split markdown by topic (or treat whole doc as one section if no topics)
  3. For each topic (in parallel):
     a. Call OpusMax (Anthropic-compatible) or Ollama to generate structured notes
     b. Extract formulas
     c. Generate image descriptions from the notes
     d. Send image descriptions to GPU worker (Wan2GP) → get B2 URLs
     e. Load all questions for this topic
     f. For each question: generate a full answer via OpusMax/Ollama
  4. Save everything to topic_notes row (JSONB)
  5. Update documents.pregen_total / pregen_done pattern (or notes_status)

Design:
  - Default: OpusMax API (https://api.opusmax.pro) using claude-haiku-4-5-20251001
    — cheapest Claude model, fast, reliable JSON output, 200k context
  - Toggle: NOTES_USE_LOCAL_OLLAMA=true → use local Ollama instead
  - CPU calls GPU for image generation only (via HTTP to Wan2GP)
  - Single model call — no fallback loop needed (Claude is reliable)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ── Model selection ────────────────────────────────────────────────────────────
NOTES_USE_LOCAL_OLLAMA = os.getenv("NOTES_USE_LOCAL_OLLAMA", "false").lower() == "true"
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")

# OpusMax — Anthropic-compatible proxy (direct call from container)
# claude-haiku-4-5-20251001 = cheapest Claude model, reliable JSON, 200k context
OPUSMAX_URL   = "https://api.opusmax.pro"
OPUSMAX_KEY   = os.getenv("OPUSMAX_API_KEY", "")
OPUSMAX_MODEL = os.getenv("OPUSMAX_MODEL", "claude-haiku-4-5-20251001")

# ── Prompt loader ──────────────────────────────────────────────────────────────
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"

def _load_prompt(filename: str) -> str:
    try:
        return (_PROMPT_DIR / filename).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[Notes] Could not load {filename}: {e}")
        return ""

NOTE_PROMPT  = _load_prompt("note_generation_prompt.txt")
ANSWER_PROMPT = _load_prompt("answer_generation_prompt.txt")

# ── JSON helpers ───────────────────────────────────────────────────────────────
def _extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0).strip() if match else text.strip()


# ── LLM callers ────────────────────────────────────────────────────────────────
async def _call_llm_with_retry(system_prompt: str, user_prompt: str,
                                temperature: float = 0.3, max_tokens: int = 32000,
                                retries: int = 2, delay_s: float = 5.0) -> dict:
    """Call the configured LLM (OpusMax or Ollama) with retry on transient failure."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return await _call_llm(system_prompt, user_prompt, temperature, max_tokens)
        except RuntimeError as e:
            last_err = e
            if attempt < retries:
                logger.warning(f"[Notes] LLM attempt {attempt}/{retries} failed: {e}. Retrying in {delay_s}s...")
                await asyncio.sleep(delay_s)
    raise RuntimeError(f"[Notes] LLM failed after {retries} attempts: {last_err}")


async def _call_opusmax(system_prompt: str, user_prompt: str,
                        temperature: float = 0.3, max_tokens: int = 32000) -> dict:
    """Call OpusMax via Anthropic Messages API. Raises RuntimeError on failure."""
    if not OPUSMAX_KEY:
        raise RuntimeError("OPUSMAX_API_KEY not set — cannot use OpusMax")

    headers = {
        "x-api-key": OPUSMAX_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OPUSMAX_URL}/v1/messages",
            headers=headers,
            json={
                "model": OPUSMAX_MODEL,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "thinking": {"type": "disabled"},
            },
        )

    if resp.status_code != 200:
        raise RuntimeError(f"OpusMax HTTP {resp.status_code}: {resp.text[:200]}")

    # Find the text content block (skip thinking/other types)
    content_blocks = resp.json().get("content", [])
    raw = next((b["text"] for b in content_blocks if b.get("type") == "text"), None)
    if not raw:
        raise RuntimeError(f"OpusMax returned no text content. Blocks: {content_blocks}")
    cleaned = _extract_json(raw)
    result = json.loads(cleaned)
    logger.info(f"[Notes] ✓ OpusMax ({OPUSMAX_MODEL}) responded")
    return result


async def _call_ollama(system_prompt: str, user_prompt: str,
                       temperature: float = 0.3, max_tokens: int = 32000) -> dict:
    """Call local Ollama. Raises on failure."""
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()

    raw = resp.json().get("message", {}).get("content", "")
    cleaned = _extract_json(raw)
    return json.loads(cleaned)


async def _call_llm(system_prompt: str, user_prompt: str,
                    temperature: float = 0.3, max_tokens: int = 32000) -> dict:
    """Dispatch to OpusMax or local Ollama based on NOTES_USE_LOCAL_OLLAMA env var."""
    if NOTES_USE_LOCAL_OLLAMA:
        logger.info("[Notes] Using local Ollama")
        return await _call_ollama(system_prompt, user_prompt, temperature, max_tokens)
    logger.info(f"[Notes] Using OpusMax ({OPUSMAX_MODEL})")
    return await _call_opusmax(system_prompt, user_prompt, temperature, max_tokens)


# ── GPU image generation (Wan2GP via HTTP) ─────────────────────────────────────
async def _generate_single_image(desc: str, ctx: Optional[Dict],
                                  subject_id: str, document_id: str,
                                  image_index: int,
                                  headers: dict, wan2gp_url: str) -> tuple[str, str]:
    """
    Build prompt from description via the keyword-based template (fast, no extra LLM call),
    submit to Wan2GP, poll until done, download, save locally + upload to B2.

    Returns (local_path, b2_url) on success, or ("", "") on any failure.
    Local save path: subjects/{subject_id}/documents/{document_id}/notes/images/note_{index:03d}.png
    B2 path:         notes/{subject_id}/{document_id}/note_{index:03d}.png
    """
    from core.b2_client import upload_to_b2
    from core.local_storage import write_note_image
    from core.image_prompt import enhance_image_prompt

    try:
        # Use Claude-enhanced prompt for higher quality infographics and diagrams
        prompt = await enhance_image_prompt(desc, ctx)

        # Step 1: Submit job
        submit_timeout = httpx.Timeout(connect=10, read=60, write=30, pool=5)
        async with httpx.AsyncClient(timeout=submit_timeout) as client:
            submit = await client.post(
                f"{wan2gp_url}/generate-image",
                headers=headers,
                json={
                    "prompt":         prompt,
                    "model":          "flux_dev",
                    "resolution":     "1024x1024",
                    "steps":          50,
                    "guidance_scale": 7.5,
                    "seed":           -1,
                },
            )
            submit.raise_for_status()
            job_id = submit.json().get("job_id")
            if not job_id:
                logger.warning(f"[Notes] Wan2GP: no job_id for: {desc[:60]}")
                return ("", "")

        logger.info(f"[Notes] Wan2GP job {job_id} submitted for note #{image_index:03d}: {desc[:50]}")

        # Step 2: Poll — 360 × 5s = 30 min max
        poll_timeout = httpx.Timeout(connect=10, read=30, write=10, pool=5)
        completed = False
        async with httpx.AsyncClient(timeout=poll_timeout) as poll_client:
            for attempt in range(360):
                await asyncio.sleep(5)
                try:
                    stat_resp = await poll_client.get(
                        f"{wan2gp_url}/status/{job_id}", headers=headers
                    )
                    if stat_resp.is_success:
                        st = stat_resp.json().get("status")
                        if st == "completed":
                            completed = True
                            break
                        if st == "failed":
                            logger.warning(f"[Notes] Wan2GP job failed: {stat_resp.json().get('error')}")
                            return ("", "")
                except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as conn_err:
                    logger.warning(f"[Notes] Wan2GP poll disconnect (attempt {attempt+1}): {conn_err}")
                    await asyncio.sleep(10)
                    continue
                except Exception:
                    pass  # other transient errors — keep polling

        if not completed:
            logger.warning(f"[Notes] Wan2GP timeout (30 min) for: {desc[:60]}")
            return ("", "")

        # Step 3: Download
        dl_timeout = httpx.Timeout(connect=10, read=60, write=10, pool=5)
        async with httpx.AsyncClient(timeout=dl_timeout) as client:
            dl = await client.get(f"{wan2gp_url}/download-image/{job_id}", headers=headers)
            dl.raise_for_status()
            img_bytes = dl.content

        # Step 4: Save locally (structured path) + upload to B2
        local_path = await write_note_image(subject_id, document_id, image_index, img_bytes)
        b2_key = f"notes/{subject_id}/{document_id}/note_{image_index:03d}.png"
        b2_url = await upload_to_b2(Path(local_path).read_bytes(), b2_key, "image/png")
        logger.info(f"[Notes] ✓ [PROGRESS] Image #{image_index:03d} successfully generated & uploaded: {b2_url}")
        return (local_path, b2_url)

    except Exception as e:
        logger.warning(f"[Notes] Image #{image_index:03d} generation failed: {e}")
        return ("", "")


async def _generate_images_on_gpu(image_descriptions: List[str],
                                   subject_id: str, document_id: str,
                                   section_contexts: Optional[List[Dict]] = None) -> List[tuple[str, str]]:
    """
    Send image descriptions to GPU worker (Wan2GP) and return (local_path, b2_url) tuples.
    Processes 3 images concurrently per batch to maximise GPU throughput.
    Image indices are sequential (1-based) so local paths are deterministic.
    Returns list of (local_path, b2_url) tuples (both '' on failure), preserving order.
    """
    if not image_descriptions:
        return []

    WAN2GP_URL = os.getenv("WAN2GP_URL", "http://host.docker.internal:9090")
    WAN2GP_KEY = os.getenv("WAN2GP_API_KEY", "")

    headers = {"Content-Type": "application/json"}
    if WAN2GP_KEY:
        headers["X-API-Key"] = WAN2GP_KEY

    BATCH_SIZE = 9
    results: List[tuple[str, str]] = []
    image_index = 1  # sequential 1-based index across all batches

    for batch_start in range(0, len(image_descriptions), BATCH_SIZE):
        batch_descs = image_descriptions[batch_start:batch_start + BATCH_SIZE]
        batch_ctxs  = (section_contexts[batch_start:batch_start + BATCH_SIZE]
                       if section_contexts else [None] * len(batch_descs))
        # Pad contexts if shorter than descriptions
        while len(batch_ctxs) < len(batch_descs):
            batch_ctxs.append(None)

        logger.info(f"[Notes] Submitting image batch {batch_start // BATCH_SIZE + 1}: "
                    f"{len(batch_descs)} images in parallel")

        # Run all images in this batch concurrently
        batch_results = await asyncio.gather(*[
            _generate_single_image(desc, ctx, subject_id, document_id,
                                   image_index + i, headers, WAN2GP_URL)
            for i, (desc, ctx) in enumerate(zip(batch_descs, batch_ctxs))
        ])
        results.extend(batch_results)
        image_index += len(batch_descs)

    return results


# ── Markdown splitter ──────────────────────────────────────────────────────────
def _split_markdown_by_topic(markdown: str, topic_titles: List[str]) -> List[Dict[str, str]]:
    """
    Split raw markdown into chunks per topic using topic titles as delimiters.
    Returns list of {"title": ..., "content": ...} dicts.
    If no topic titles match, returns [{"title": "Full Document", "content": markdown}].
    """
    if not markdown or not markdown.strip():
        return [{"title": "Full Document", "content": "No content available."}]

    if not topic_titles:
        return [{"title": "Full Document", "content": markdown}]

    # Build a regex that matches any topic title as a heading
    # Topic titles may be "1.2 Newton's Law" etc. — escape regex chars
    escaped = [re.escape(t) for t in topic_titles]
    pattern = re.compile(r'^(#{1,6}\s*(' + '|'.join(escaped) + r'))', re.MULTILINE | re.IGNORECASE)

    splits: List[Dict[str, str]] = []
    last_end = 0
    last_title = "Introduction"

    for match in pattern.finditer(markdown):
        start = match.start()
        if start > last_end:
            splits.append({"title": last_title, "content": markdown[last_end:start].strip()})
        last_title = match.group(2)
        last_end = match.end()

    # Last chunk
    remaining = markdown[last_end:].strip()
    if remaining:
        splits.append({"title": last_title, "content": remaining})

    return splits if splits else [{"title": "Full Document", "content": markdown}]


def _build_topic_chunks(content_md: str, topic_rows: List[Dict]) -> List[Dict]:
    """
    Pair each topic to its actual section in the markdown.

    Strategy:
      1. Try to split markdown by topic headings. If headings match topic titles,
         pair each split to its topic by order of appearance.
      2. If no headings match (or only one big split is returned), split the full
         content evenly across all topics so each topic gets distinct content.
         This avoids duplicating the whole document N times.

    Returns list of {"title", "content", "topic_id"} dicts — one per topic.
    """
    if not topic_rows:
        return [{"title": "Full Document", "content": content_md, "topic_id": None}]

    topic_titles = [t["title"] for t in topic_rows]
    splits = _split_markdown_by_topic(content_md, topic_titles)

    # Case A: headings matched — we have multiple meaningful splits
    if len(splits) > 1:
        chunks = []
        for i, sp in enumerate(splits):
            topic = topic_rows[i] if i < len(topic_rows) else topic_rows[-1]
            chunks.append({
                "title": sp["title"],
                "content": sp["content"],
                "topic_id": topic["id"],
            })
        return chunks

    # Case B: single split (no headings matched, or whole doc is one block)
    # Split content evenly across all topics so each gets distinct material
    n = len(topic_rows)
    content_len = len(content_md)
    if n == 1:
        return [{
            "title": topic_rows[0]["title"],
            "content": content_md,
            "topic_id": topic_rows[0]["id"],
        }]

    chunk_size = content_len // n
    chunks = []
    for i, topic in enumerate(topic_rows):
        start = i * chunk_size
        # Last chunk gets the remainder
        end = (i + 1) * chunk_size if i < n - 1 else content_len
        segment = content_md[start:end].strip()
        # Fallback: if a segment is empty (e.g. short doc), give it the whole doc
        if not segment:
            segment = content_md
        chunks.append({
            "title": topic["title"],
            "content": segment,
            "topic_id": topic["id"],
        })

    return chunks


# ── DB helpers ─────────────────────────────────────────────────────────────────
async def _load_questions_for_topic(db: AsyncSession, subject_id: str,
                                    chapter_id: Optional[uuid.UUID],
                                    topic_id: Optional[uuid.UUID]) -> List[Dict]:
    """Load all questions for a specific topic."""
    if topic_id:
        result = await db.execute(
            text("SELECT id, question_text, question_format, options, correct_answer "
                 "FROM questions WHERE topic_id = CAST(:tid AS uuid)"),
            {"tid": str(topic_id)}
        )
    elif chapter_id:
        result = await db.execute(
            text("SELECT id, question_text, question_format, options, correct_answer "
                 "FROM questions WHERE chapter_id = CAST(:cid AS uuid)"),
            {"cid": str(chapter_id)}
        )
    else:
        result = await db.execute(
            text("SELECT id, question_text, question_format, options, correct_answer "
                 "FROM questions WHERE subject_id = :sid AND topic_id IS NULL AND chapter_id IS NULL"),
            {"sid": subject_id}
        )

    rows = result.fetchall()
    return [
        {
            "id": str(r[0]),
            "question_text": r[1],
            "format": r[2],
            "options": r[3] or {},
            "correct_answer": r[4],
        }
        for r in rows
    ]


async def _upsert_topic_note(db: AsyncSession, doc_id: str, subject_id: str,
                              chapter_id: Optional[str], topic_id: Optional[str],
                              sections: list, formulas: list, note_img_urls: list,
                              note_image_local_paths: list,
                              q_answers: list, answer_img_urls: list,
                              status: str, error: str = "") -> str:
    """Upsert a topic_notes row. Returns the row UUID."""
    # Reuse existing row id if one already exists for (document_id, topic_id),
    # including the NULL topic_id case where topic_id IS NULL means document-global notes.
    existing = await db.execute(text("""
        SELECT id FROM topic_notes
        WHERE document_id = CAST(:d AS uuid)
          AND ((CAST(:t AS uuid) IS NULL AND topic_id IS NULL)
               OR topic_id = CAST(:t AS uuid))
        LIMIT 1
    """), {"d": doc_id, "t": topic_id})
    row = existing.fetchone()
    note_id = str(row[0]) if row else str(uuid.uuid4())

    await db.execute(text("""
        INSERT INTO topic_notes
            (id, document_id, subject_id, chapter_id, topic_id,
             note_sections, note_latex_formulas, note_image_urls,
             note_image_local_paths,
             question_answers, answer_image_urls,
             notes_status, error_message, generated_at)
        VALUES
            (:id, :doc_id, :sid, :cid, :tid,
             :sections, :formulas, :note_imgs,
             :note_local,
             :q_answers, :ans_imgs,
             :status, :err, NOW())
        ON CONFLICT (document_id, topic_id) DO UPDATE SET
            note_sections           = EXCLUDED.note_sections,
            note_latex_formulas     = EXCLUDED.note_latex_formulas,
            note_image_urls         = EXCLUDED.note_image_urls,
            note_image_local_paths  = EXCLUDED.note_image_local_paths,
            question_answers        = EXCLUDED.question_answers,
            answer_image_urls       = EXCLUDED.answer_image_urls,
            notes_status            = EXCLUDED.notes_status,
            error_message           = EXCLUDED.error_message,
            generated_at            = EXCLUDED.generated_at,
            updated_at              = NOW()
    """), {
        "id": note_id,
        "doc_id": doc_id,
        "sid": subject_id,
        "cid": chapter_id,
        "tid": topic_id,
        "sections": json.dumps(sections),
        "formulas": json.dumps(formulas),
        "note_imgs": json.dumps(note_img_urls),
        "note_local": json.dumps(note_image_local_paths),
        "q_answers": json.dumps(q_answers),
        "ans_imgs": json.dumps(answer_img_urls),
        "status": status,
        "err": error,
    })

    return note_id


# ── Main pipeline ──────────────────────────────────────────────────────────────
async def generate_notes_for_document(document_id: str, db: AsyncSession,
                                      background_tasks=None) -> dict:
    """
    Main entry point: generate textbook notes + question answers for a document.

    Args:
        document_id: UUID of the uploaded document
        db:          Active DB session
        background_tasks: FastAPI BackgroundTasks (optional — runs final DB commit in it if provided)

    Returns:
        {"status": "queued|done|failed", "topics_processed": N, "error": "..."}
    """
    log_prefix = f"[Notes/doc={document_id[:8]}]"

    # 1. Load document
    result = await db.execute(
        text("SELECT id, subject_id, chapter_id, topic_id, title, content_markdown, notes_status "
             "FROM note_documents WHERE id = CAST(:did AS uuid)"),
        {"did": document_id}
    )
    row = result.fetchone()
    if not row:
        raise RuntimeError(f"Document {document_id} not found")

    doc_id_raw, subject_id, doc_chapter_id_raw, doc_topic_id_raw, title, content_md, doc_status = row
    doc_id = str(doc_id_raw)
    doc_chapter_id = str(doc_chapter_id_raw) if doc_chapter_id_raw else None
    doc_topic_id = str(doc_topic_id_raw) if doc_topic_id_raw else None
    # Accept 'ready', 'notes_pending' (json_import), 'notes_generating' (batch), or 'generating' (api)
    allowed_statuses = ("ready", "notes_pending", "notes_generating", "generating")
    if doc_status not in allowed_statuses:
        raise RuntimeError(
            f"Document status is '{doc_status}' — only {allowed_statuses} allowed. "
            "Use /notes/reset/{doc_id} to unblock."
        )

    if not content_md or not content_md.strip():
        raise RuntimeError("Document has no content_markdown — cannot generate notes")

    logger.info(f"{log_prefix} Starting notes generation for '{title}' (status={doc_status})")

    # 2. Load topic titles for splitting
    topic_rows = []
    if doc_topic_id:
        result = await db.execute(
            text("SELECT id, title FROM topics WHERE id = CAST(:tid AS uuid)"),
            {"tid": str(doc_topic_id)}
        )
        topic_rows = [dict(id=str(r[0]), title=r[1]) for r in result.fetchall()]
    elif doc_chapter_id:
        result = await db.execute(
            text("SELECT id, title FROM topics WHERE chapter_id = CAST(:cid AS uuid) ORDER BY topic_number"),
            {"cid": str(doc_chapter_id)}
        )
        topic_rows = [dict(id=str(r[0]), title=r[1]) for r in result.fetchall()]

    topic_titles = [t["title"] for t in topic_rows]

    # 3. Build per-topic chunks (pairs each topic to its section, or splits evenly)
    chunks = _build_topic_chunks(content_md, topic_rows)
    logger.info(f"{log_prefix} Built {len(chunks)} per-topic chunks across {len(topic_rows)} topics")

    async def _process_chunk(chunk_idx: int, chunk: dict) -> dict:
        """Process one content chunk → generate notes + answers."""
        chunk_title = chunk["title"]
        chunk_content = chunk["content"]
        # topic_id is set by _build_topic_chunks (paired to heading, or evenly split)
        matched_topic_id = chunk.get("topic_id")

        result = {
            "chunk_idx": chunk_idx,
            "title": chunk_title,
            "topic_id": matched_topic_id,
            "status": "failed",
            "error": "",
        }

        try:
            # ── Step A: Generate structured notes ──────────────────────────────
            note_user_prompt = (
                f"Convert the following lecture content into structured textbook notes.\n\n"
                f"Content:\n{chunk_content[:15000]}"  # safety limit
            )
            note_result = await _call_llm_with_retry(
                system_prompt=NOTE_PROMPT,
                user_prompt=note_user_prompt,
                temperature=0.3,
                max_tokens=32000,
            )

            sections   = note_result.get("sections", [])
            formulas   = note_result.get("formulas", [])

            # ── Step B: Generate images from descriptions ──────────────────────
            # Build aligned contexts so each image knows its parent section's
            # key points + heading → richer enhanced prompt.
            all_img_descs: List[str] = []
            section_contexts: List[Dict] = []
            for sec in sections:
                for img_desc in sec.get("image_descriptions", []):  # LLM decides count
                    all_img_descs.append(img_desc)
                    section_contexts.append({
                        "key_points": sec.get("key_points", []),
                        "formula":    sec.get("formula", ""),
                        "is_story":   sec.get("section_type") == "story",
                        "is_tips":    sec.get("section_type") == "tips",
                    })

            note_img_urls: List[str] = []
            note_local_paths: List[str] = []
            if all_img_descs:
                logger.info(f"{log_prefix} [IMAGE COUNTS] Sections requested {len(all_img_descs)} images (across {len(sections)} sections)")
                raw = await _generate_images_on_gpu(
                    all_img_descs,  # no limit — LLM decides
                    subject_id, document_id,
                    section_contexts=section_contexts,
                )
                note_local_paths = [loc for (loc, b2) in raw if loc]
                note_img_urls    = [b2   for (loc, b2) in raw if b2]

            # ── Step C: Generate question answers ──────────────────────────────
            questions = await _load_questions_for_topic(
                db, subject_id,
                uuid.UUID(doc_chapter_id) if doc_chapter_id else None,
                uuid.UUID(matched_topic_id) if matched_topic_id else None,
            )
            logger.info(f"{log_prefix} Chunk '{chunk_title}': {len(questions)} questions to answer")

            q_answers = []
            for q in questions:
                try:
                    answer_prompt = (
                        f"Question: {q['question_text']}\n\n"
                        f"Source material:\n{chunk_content[:8000]}"
                    )
                    answer_result = await _call_llm_with_retry(
                        system_prompt=ANSWER_PROMPT,
                        user_prompt=answer_prompt,
                        temperature=0.3,
                        max_tokens=8000,
                    )
                    q_answers.append({
                        "question_id": q["id"],
                        "question_text": q["question_text"],
                        "format": q["format"],
                        **answer_result,
                    })
                except Exception as e:
                    logger.warning(f"{log_prefix} Answer failed for q={q['id']}: {e}")
                    q_answers.append({
                        "question_id": q["id"],
                        "question_text": q["question_text"],
                        "format": q.get("format", "subjective"),
                        "answer": f"Error generating answer: {e}",
                        "key_points": [],
                        "error": str(e),
                    })

            # ── Step D: Generate images for answers ────────────────────────────
            # Gather image_descriptions from answers (max 1 per answer to keep it fast)
            ans_img_descs: List[str] = []
            ans_img_answer_indices: List[int] = []
            for ai, qa in enumerate(q_answers):
                descs = qa.get("image_descriptions", [])
                if descs:
                    ans_img_descs.append(descs[0])  # 1 image per answer
                    ans_img_answer_indices.append(ai)

            if ans_img_descs:
                logger.info(f"{log_prefix} [IMAGE COUNTS] Q&A requested {len(ans_img_descs)} images (across {len(q_answers)} answers)")
                ans_img_raw = await _generate_images_on_gpu(
                    ans_img_descs, subject_id, document_id,
                )
                for list_idx, answer_idx in enumerate(ans_img_answer_indices):
                    loc, b2 = ans_img_raw[list_idx] if list_idx < len(ans_img_raw) else ("", "")
                    pub_url = ""
                    if loc:
                        rel = loc.replace("/app/storage/subjects/", "")
                        pub_url = f"/local-images/{rel}"
                    q_answers[answer_idx]["answer_images"] = [
                        {"url": b2, "local_url": pub_url}
                    ] if (loc or b2) else []
            else:
                for qa in q_answers:
                    qa.setdefault("answer_images", [])

            # ── Step E: Save to DB ─────────────────────────────────────────────
            await _upsert_topic_note(
                db, document_id, subject_id,
                doc_chapter_id, matched_topic_id,
                sections=sections,
                formulas=formulas,
                note_img_urls=note_img_urls,
                note_image_local_paths=note_local_paths,
                q_answers=q_answers,
                answer_img_urls=[],  # answer images are now embedded in q_answers
                status="done",
            )

            result["status"] = "done"
            result["sections_count"] = len(sections)
            result["answers_count"] = len(q_answers)
            logger.info(f"{log_prefix} ✓ Chunk '{chunk_title}' done: {len(sections)} sections, {len(q_answers)} answers")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"{log_prefix} ✗ Chunk '{chunk_title}' failed: {e}")
            # Save failed status
            await _upsert_topic_note(
                db, document_id, subject_id,
                doc_chapter_id, matched_topic_id,
                sections=[], formulas=[], note_img_urls=[],
                note_image_local_paths=[],
                q_answers=[], answer_img_urls=[],
                status="failed", error=str(e),
            )

        return result

    # 5. Process all chunks sequentially to avoid overwhelming free OpenRouter tier
    results = []
    for i, chunk in enumerate(chunks):
        result = await _process_chunk(i, chunk)
        results.append(result)
        # Commit each chunk's topic_notes immediately so they're not lost
        # if a later chunk or the final status update fails
        await db.commit()

    done_count   = sum(1 for r in results if r["status"] == "done")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    # 6. Update note_document status
    if done_count == 0 and failed_count > 0:
        final_status = "failed"
    else:
        final_status = "done"  # at least some topics succeeded

    await db.execute(
        text("UPDATE note_documents SET notes_status = :s, updated_at = NOW() WHERE id = CAST(:did AS uuid)"),
        {"s": final_status, "did": document_id}
    )
    await db.commit()

    logger.info(f"{log_prefix} Complete: {done_count} done, {failed_count} failed out of {len(chunks)} — doc status={final_status}")
    return {
        "status": final_status,
        "topics_processed": len(chunks),
        "topics_done": done_count,
        "topics_failed": failed_count,
        "results": results,
    }
