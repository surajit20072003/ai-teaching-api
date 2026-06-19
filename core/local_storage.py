"""
core/local_storage.py
─────────────────────
All local disk I/O for the AI Teaching system.

Folder structure (per subject, document-centric):
/sdb-disk/ai-teaching/subjects/
    {subject_id}/
        documents/
            {doc_id}/
                raw/            <- original uploaded PDF/DOCX/TXT
                processed/      <- extracted plain text
                meta.json       <- document metadata snapshot
        cache/
            slides/             <- {question_hash}.json (slide data)
            images/             <- {cache_id}/slide_N.png
            audio/              <- {cache_id}/{language}/slide_N.wav
logs/
    uploads.log
    pregen.log
    errors.log
"""

import os
import json
import logging
import aiofiles
import aiofiles.os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

BASE_PATH = "/sdb-disk/ai-teaching"
SUBJECTS_PATH = f"{BASE_PATH}/subjects"
LOGS_PATH = f"{BASE_PATH}/logs"

# ── Path Helpers ──────────────────────────────────────────────────────────────

def get_doc_raw_dir(subject_id: str, doc_id: str) -> str:
    return f"{SUBJECTS_PATH}/{subject_id}/documents/{doc_id}/raw"

def get_doc_raw_path(subject_id: str, doc_id: str, filename: str) -> str:
    return f"{get_doc_raw_dir(subject_id, doc_id)}/{filename}"

def get_doc_processed_path(subject_id: str, doc_id: str) -> str:
    return f"{SUBJECTS_PATH}/{subject_id}/documents/{doc_id}/processed/extracted.txt"

def get_doc_meta_path(subject_id: str, doc_id: str) -> str:
    return f"{SUBJECTS_PATH}/{subject_id}/documents/{doc_id}/meta.json"

def get_slide_cache_path(subject_id: str, question_hash: str) -> str:
    return f"{SUBJECTS_PATH}/{subject_id}/cache/slides/{question_hash}.json"

def get_image_dir(subject_id: str, cache_id: str) -> str:
    return f"{SUBJECTS_PATH}/{subject_id}/cache/images/{cache_id}"

def get_image_path(subject_id: str, cache_id: str, slide_index: int) -> str:
    return f"{get_image_dir(subject_id, cache_id)}/slide_{slide_index}.png"

def get_audio_dir(subject_id: str, cache_id: str, language: str) -> str:
    return f"{SUBJECTS_PATH}/{subject_id}/cache/audio/{cache_id}/{language}"

def get_audio_path(subject_id: str, cache_id: str, language: str, slide_index: int) -> str:
    return f"{get_audio_dir(subject_id, cache_id, language)}/slide_{slide_index}.wav"

def get_log_path(log_name: str) -> str:
    """log_name: 'uploads', 'pregen', 'errors'"""
    return f"{LOGS_PATH}/{log_name}.log"

# ── Directory Setup ───────────────────────────────────────────────────────────

def ensure_subject_dirs(subject_id: str) -> None:
    """
    Create all required subdirectories for a subject synchronously.
    Called on first document upload for a subject.
    """
    dirs = [
        f"{SUBJECTS_PATH}/{subject_id}/documents",
        f"{SUBJECTS_PATH}/{subject_id}/cache/slides",
        f"{SUBJECTS_PATH}/{subject_id}/cache/images",
        f"{SUBJECTS_PATH}/{subject_id}/cache/audio",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info(f"[local_storage] Subject dirs ready: {subject_id}")

def ensure_doc_dirs(subject_id: str, doc_id: str) -> None:
    """Create raw/ and processed/ folders for a specific document."""
    dirs = [
        f"{SUBJECTS_PATH}/{subject_id}/documents/{doc_id}/raw",
        f"{SUBJECTS_PATH}/{subject_id}/documents/{doc_id}/processed",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def ensure_base_dirs() -> None:
    """Create top-level folders on startup if not present."""
    os.makedirs(SUBJECTS_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)

# ── Slide Cache (JSON) ────────────────────────────────────────────────────────

async def read_slide_cache(subject_id: str, question_hash: str) -> Optional[dict]:
    """
    L2 cache: read slide JSON from local disk.
    Returns None if file doesn't exist (cache miss).
    ~1ms read time.
    """
    path = get_slide_cache_path(subject_id, question_hash)
    try:
        if not await aiofiles.os.path.exists(path):
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"[local_storage] read_slide_cache failed for {question_hash}: {e}")
        return None

async def write_slide_cache(subject_id: str, question_hash: str, data: dict) -> bool:
    """
    Write slide JSON to local disk. Creates parent dirs if needed.
    Returns True on success.
    """
    path = get_slide_cache_path(subject_id, question_hash)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False))
        return True
    except Exception as e:
        logger.error(f"[local_storage] write_slide_cache failed for {question_hash}: {e}")
        return False

async def delete_slide_cache(subject_id: str, question_hash: str) -> None:
    """Delete a slide cache file (used when deleting a document)."""
    path = get_slide_cache_path(subject_id, question_hash)
    try:
        if await aiofiles.os.path.exists(path):
            await aiofiles.os.remove(path)
    except Exception as e:
        logger.warning(f"[local_storage] delete_slide_cache failed: {e}")

# ── Document Files ────────────────────────────────────────────────────────────

async def write_raw_file(subject_id: str, doc_id: str, filename: str, file_bytes: bytes) -> str:
    """
    Save original uploaded file to local disk.
    Returns the local path.
    """
    ensure_doc_dirs(subject_id, doc_id)
    path = get_doc_raw_path(subject_id, doc_id, filename)
    async with aiofiles.open(path, "wb") as f:
        await f.write(file_bytes)
    logger.info(f"[local_storage] Raw file saved: {path} ({len(file_bytes):,} bytes)")
    return path

async def write_processed_text(subject_id: str, doc_id: str, text: str) -> str:
    """
    Save extracted plain text to local disk.
    Returns the local path.
    """
    path = get_doc_processed_path(subject_id, doc_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(text)
    logger.info(f"[local_storage] Processed text saved: {path} ({len(text):,} chars)")
    return path

async def read_processed_text(subject_id: str, doc_id: str) -> Optional[str]:
    """Read extracted text from disk. Returns None if not found."""
    path = get_doc_processed_path(subject_id, doc_id)
    try:
        if not await aiofiles.os.path.exists(path):
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()
    except Exception as e:
        logger.warning(f"[local_storage] read_processed_text failed: {e}")
        return None

# ── Document Meta ─────────────────────────────────────────────────────────────

async def write_doc_meta(subject_id: str, doc_id: str, meta: dict) -> None:
    """
    Save metadata snapshot as meta.json inside the document folder.
    Useful for inspecting documents without hitting the database.
    """
    path = get_doc_meta_path(subject_id, doc_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta["_written_at"] = datetime.now(timezone.utc).isoformat()
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(meta, indent=2, ensure_ascii=False))

async def read_doc_meta(subject_id: str, doc_id: str) -> Optional[dict]:
    """Read document meta.json. Returns None if not found."""
    path = get_doc_meta_path(subject_id, doc_id)
    try:
        if not await aiofiles.os.path.exists(path):
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return json.loads(await f.read())
    except Exception as e:
        logger.warning(f"[local_storage] read_doc_meta failed: {e}")
        return None

# ── Image / Audio Files ───────────────────────────────────────────────────────

async def write_image(subject_id: str, cache_id: str, slide_index: int, image_bytes: bytes) -> str:
    """Save generated image to local disk. Returns local path."""
    path = get_image_path(subject_id, cache_id, slide_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiofiles.open(path, "wb") as f:
        await f.write(image_bytes)
    return path

async def write_audio(subject_id: str, cache_id: str, language: str, slide_index: int, audio_bytes: bytes) -> str:
    """Save generated audio to local disk. Returns local path."""
    path = get_audio_path(subject_id, cache_id, language, slide_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiofiles.open(path, "wb") as f:
        await f.write(audio_bytes)
    return path

# ── Cleanup ───────────────────────────────────────────────────────────────────

def delete_document_files(subject_id: str, doc_id: str) -> None:
    """
    Synchronously delete all local files for a document.
    Called during DELETE /documents/{id}.
    """
    import shutil
    doc_dir = f"{SUBJECTS_PATH}/{subject_id}/documents/{doc_id}"
    if os.path.exists(doc_dir):
        shutil.rmtree(doc_dir)
        logger.info(f"[local_storage] Deleted document dir: {doc_dir}")

def delete_cache_files(subject_id: str, cache_id: str) -> None:
    """Delete all image and audio files for a cache entry."""
    import shutil
    for subdir in ["images", "audio"]:
        path = f"{SUBJECTS_PATH}/{subject_id}/cache/{subdir}/{cache_id}"
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info(f"[local_storage] Deleted cache dir: {path}")

# ── Log Helpers ───────────────────────────────────────────────────────────────

async def append_log(log_name: str, message: str) -> None:
    """Append a timestamped line to a log file."""
    path = get_log_path(log_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    async with aiofiles.open(path, "a", encoding="utf-8") as f:
        await f.write(f"[{ts}] {message}\n")

# ── Storage Health Check ──────────────────────────────────────────────────────

def check_storage_available() -> dict:
    """
    Check if /sdb-disk is mounted and writable.
    Returns status dict. Called by GET /health.
    """
    try:
        stat = os.statvfs(BASE_PATH)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
        writable = os.access(BASE_PATH, os.W_OK)
        return {
            "status": "ok" if writable else "read_only",
            "path": BASE_PATH,
            "free_gb": round(free_gb, 1),
            "total_gb": round(total_gb, 1),
            "used_pct": round((1 - stat.f_bavail / stat.f_blocks) * 100, 1),
        }
    except Exception as e:
        return {"status": "error", "path": BASE_PATH, "error": str(e)}
