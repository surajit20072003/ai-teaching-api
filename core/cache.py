import hashlib, unicodedata, re, json, os, asyncio
import redis.asyncio as aioredis

_redis = None
_REDIS_URL = os.getenv("REDIS_URL", "")   # empty = Redis not configured (GPU server)

def get_redis():
    global _redis
    if not _REDIS_URL:
        return None   # GPU server: no Redis — all cache calls will be no-ops
    if _redis is None:
        _redis = aioredis.from_url(_REDIS_URL, decode_responses=True)
    return _redis

def hash_question(text: str) -> str:
    """Normalize and SHA-256 hash a question string cleanly without destroying word order."""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    
    # Split and remove empty strings, but keep order
    words = [w for w in text.split() if w]
    
    normalized_text = " ".join(words)
    return hashlib.sha256(normalized_text.encode()).hexdigest()[:32]

def cache_key(q_hash: str, subject_id: str = "") -> str:
    return f"teaching:{q_hash}:{subject_id}"

def lock_key(q_hash: str, subject_id: str = "") -> str:
    return f"lock:{q_hash}:{subject_id}"

async def get_from_cache(q_hash: str, subject_id: str = ""):
    r = get_redis()
    if not r:
        return None   # Redis unavailable — fall through to Postgres
    try:
        val = await r.get(cache_key(q_hash, subject_id))
        return json.loads(val) if val else None
    except Exception:
        return None

async def set_to_cache(q_hash: str, subject_id: str, value: dict, ttl: int = 604800):
    r = get_redis()
    if not r:
        return
    try:
        await r.set(cache_key(q_hash, subject_id), json.dumps(value), ex=ttl)
    except Exception:
        pass

async def delete_from_cache(q_hash: str, subject_id: str = ""):
    r = get_redis()
    if not r:
        return
    try:
        await r.delete(cache_key(q_hash, subject_id))
    except Exception:
        pass

async def increment_usage(q_hash: str, subject_id: str = ""):
    r = get_redis()
    if not r:
        return
    try:
        await r.incr(f"usage:{q_hash}:{subject_id}")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────
# DISTRIBUTED LOCK — prevents cache stampede
# ─────────────────────────────────────────────────────────

async def acquire_lock(q_hash: str, subject_id: str = "", ttl: int = 60) -> bool:
    """
    Try to acquire a Redis lock for this question.
    Returns True if lock acquired (this worker should call AI).
    Returns False if another worker already has the lock (should wait).
    TTL=60s: lock auto-expires if AI call crashes.
    If Redis is unavailable, always returns True (no lock = no coordination).
    """
    r = get_redis()
    if not r:
        return True   # no Redis — assume lock acquired, proceed
    try:
        key = lock_key(q_hash, subject_id)
        result = await r.set(key, "1", nx=True, ex=ttl)
        return result is True
    except Exception:
        return True

async def release_lock(q_hash: str, subject_id: str = ""):
    """Release the lock after AI generation is complete."""
    r = get_redis()
    if not r:
        return
    try:
        await r.delete(lock_key(q_hash, subject_id))
    except Exception:
        pass

async def wait_for_cache(q_hash: str, subject_id: str = "",
                         max_wait: int = 55, poll_interval: float = 0.5):
    """
    Poll Redis every 0.5s waiting for another worker to populate the cache.
    Returns cached data if found, None if timeout.
    If Redis unavailable, returns None immediately (caller handles it).
    """
    r = get_redis()
    if not r:
        return None   # no Redis — caller will generate directly
    waited = 0.0
    while waited < max_wait:
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        try:
            data = await get_from_cache(q_hash, subject_id)
            if data:
                return data
        except Exception:
            return None
    return None  # timeout


# ─────────────────────────────────────────────────────────
# TEXT ANSWER CACHE — separate namespace from slides cache
# ─────────────────────────────────────────────────────────

TEXT_CACHE_PREFIX = "text_answer"

def text_cache_key(q_hash: str, subject_id: str = "") -> str:
    return f"{TEXT_CACHE_PREFIX}:{subject_id}:{q_hash}"

async def get_text_from_cache(q_hash: str, subject_id: str = ""):
    """Return cached text answer dict or None."""
    r = get_redis()
    if not r:
        return None
    try:
        val = await r.get(text_cache_key(q_hash, subject_id))
        return json.loads(val) if val else None
    except Exception:
        return None

async def set_text_to_cache(q_hash: str, subject_id: str, value: dict, ttl: int = 604800):
    """Cache a text answer for 7 days."""
    r = get_redis()
    if not r:
        return
    try:
        await r.set(text_cache_key(q_hash, subject_id), json.dumps(value), ex=ttl)
    except Exception:
        pass

async def acquire_text_lock(q_hash: str, subject_id: str = "", ttl: int = 60) -> bool:
    """Distributed lock for text answer generation (prevents stampede)."""
    r = get_redis()
    if not r:
        return True
    try:
        key = f"textlock:{q_hash}:{subject_id}"
        result = await r.set(key, "1", nx=True, ex=ttl)
        return result is True
    except Exception:
        return True

async def release_text_lock(q_hash: str, subject_id: str = ""):
    r = get_redis()
    if not r:
        return
    try:
        await r.delete(f"textlock:{q_hash}:{subject_id}")
    except Exception:
        pass

async def wait_for_text_cache(q_hash: str, subject_id: str = "",
                               max_wait: int = 55, poll_interval: float = 0.5):
    """Poll Redis for a text answer being generated by another worker."""
    r = get_redis()
    if not r:
        return None
    waited = 0.0
    while waited < max_wait:
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        try:
            data = await get_text_from_cache(q_hash, subject_id)
            if data:
                return data
        except Exception:
            return None
    return None
