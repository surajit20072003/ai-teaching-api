"""
core/local_sync.py
──────────────────
Lazy download: when CPU serves a pregen cache hit, media files
(images, audio, manim) exist only in B2. This module downloads
them on first access and saves to CPU /app/storage/ using the same
folder structure as the GPU /sdb-disk/.

After saving, updates Postgres local_*_paths columns so next access
is served directly from local disk (L2 cache) without B2 round-trip.
"""

import os, json, logging, asyncio
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from core.local_storage import (
    get_image_path, get_audio_path, get_manim_video_path,
    get_image_dir, get_audio_dir, get_manim_dir,
)

logger = logging.getLogger(__name__)


async def _download_file(url: str, local_path: str) -> bool:
    """Download a file from B2 URL and save to local_path. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        logger.info(f"[local_sync] Downloaded: {local_path} ({len(r.content):,} bytes)")
        return True
    except Exception as e:
        logger.warning(f"[local_sync] Download failed {url}: {e}")
        return False


async def ensure_images_local(
    subject_id: str,
    cache_id: str,
    image_urls: dict,          # {"0": {"url": "https://B2/..."}, "1": {...}}
    db: AsyncSession,
) -> dict:
    """
    For each slide image: check if local copy exists on CPU disk.
    If not, download from B2 and save to /app/storage/...
    Updates local_image_paths in Postgres.
    Returns dict of {slide_index: local_path}.
    """
    local_paths = {}
    any_downloaded = False

    for idx_str, val in image_urls.items():
        url = val.get("url", "") if isinstance(val, dict) else val
        if not url:
            continue
        idx = int(idx_str)
        local_path = get_image_path(subject_id, cache_id, idx)

        if not os.path.exists(local_path):
            ok = await _download_file(url, local_path)
            if ok:
                any_downloaded = True
        
        if os.path.exists(local_path):
            local_paths[idx_str] = local_path

    # Update Postgres if any files were downloaded
    if any_downloaded and local_paths:
        try:
            await db.execute(
                text("UPDATE teaching_qa_cache SET local_image_paths = :p WHERE id = :id"),
                {"p": json.dumps(local_paths), "id": cache_id}
            )
            await db.commit()
        except Exception as e:
            logger.warning(f"[local_sync] Failed to update local_image_paths: {e}")

    return local_paths


async def ensure_audio_local(
    subject_id: str,
    cache_id: str,
    audio_urls: list,          # [{"slideIndex": 0, "audioUrl": "https://B2/..."}]
    language: str,
    db: AsyncSession,
) -> dict:
    """
    For each slide audio: check if local copy exists on CPU disk.
    If not, download from B2 and save to /app/storage/...
    Updates local_audio_paths in Postgres.
    Returns dict of {slide_index: local_path}.
    """
    local_paths = {}
    any_downloaded = False

    for item in audio_urls:
        url = item.get("audioUrl", "")
        idx = item.get("slideIndex", 0)
        if not url:
            continue
        local_path = get_audio_path(subject_id, cache_id, language, idx)

        if not os.path.exists(local_path):
            ok = await _download_file(url, local_path)
            if ok:
                any_downloaded = True

        if os.path.exists(local_path):
            local_paths[str(idx)] = local_path

    if any_downloaded and local_paths:
        try:
            await db.execute(
                text("UPDATE teaching_qa_cache SET local_audio_paths = :p WHERE id = :id"),
                {"p": json.dumps(local_paths), "id": cache_id}
            )
            await db.commit()
        except Exception as e:
            logger.warning(f"[local_sync] Failed to update local_audio_paths: {e}")

    return local_paths


async def ensure_manim_local(
    subject_id: str,
    cache_id: str,
    manim_urls: dict,          # {"0": {"url": "https://B2/..."}}
    db: AsyncSession,
) -> dict:
    """
    For each manim video: check if local copy exists on CPU disk.
    If not, download from B2 and save to /app/storage/...
    Updates local_manim_paths in Postgres.
    Returns dict of {slide_index: local_path}.
    """
    local_paths = {}
    any_downloaded = False

    for idx_str, val in manim_urls.items():
        url = val.get("url", "") if isinstance(val, dict) else val
        if not url:
            continue
        idx = int(idx_str)
        local_path = get_manim_video_path(subject_id, cache_id, idx)

        if not os.path.exists(local_path):
            ok = await _download_file(url, local_path)
            if ok:
                any_downloaded = True

        if os.path.exists(local_path):
            local_paths[idx_str] = local_path

    if any_downloaded and local_paths:
        try:
            await db.execute(
                text("UPDATE teaching_qa_cache SET local_manim_paths = :p WHERE id = :id"),
                {"p": json.dumps(local_paths), "id": cache_id}
            )
            await db.commit()
        except Exception as e:
            logger.warning(f"[local_sync] Failed to update local_manim_paths: {e}")

    return local_paths


async def sync_cache_row_to_local(
    subject_id: str,
    cache_id: str,
    row_data: dict,            # the data dict returned from Postgres row
    language: str,
    db: AsyncSession,
) -> None:
    """
    Convenience: run all three sync operations in parallel for one cache row.
    Called non-blocking (fire-and-forget) on cache hits so response is not delayed.
    """
    tasks = []
    image_urls = row_data.get("imageUrls") or {}
    audio_urls = row_data.get("slideAudioUrls") or []
    manim_urls = row_data.get("manimVideoUrls") or {}

    if image_urls:
        tasks.append(ensure_images_local(subject_id, cache_id, image_urls, db))
    if audio_urls:
        tasks.append(ensure_audio_local(subject_id, cache_id, audio_urls, language, db))
    if manim_urls:
        tasks.append(ensure_manim_local(subject_id, cache_id, manim_urls, db))

    if tasks:
        # Run in background — don't block student response
        asyncio.create_task(asyncio.gather(*tasks, return_exceptions=True))
