import hashlib, unicodedata, re, json, os, asyncio
import redis.asyncio as aioredis

_redis = None

def get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"), decode_responses=True)
    return _redis

def hash_question(text: str) -> str:
    """Normalize and SHA-256 hash a question string by sorting keywords."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Simple stopword list
    stopwords = {"what", "is", "the", "of", "and", "a", "an", "how", "why", "to", "in", "for", "on", "with"}
    
    # Split, filter, sort
    words = text.split()
    filtered_words = [w for w in words if w not in stopwords]
    sorted_words = sorted(filtered_words)
    
    normalized_text = " ".join(sorted_words)
    return hashlib.sha256(normalized_text.encode()).hexdigest()[:32]

def cache_key(q_hash: str, subject_id: str = "") -> str:
    return f"teaching:{q_hash}:{subject_id}"

def lock_key(q_hash: str, subject_id: str = "") -> str:
    return f"lock:{q_hash}:{subject_id}"

async def get_from_cache(q_hash: str, subject_id: str = ""):
    r = get_redis()
    val = await r.get(cache_key(q_hash, subject_id))
    return json.loads(val) if val else None

async def set_to_cache(q_hash: str, subject_id: str, value: dict, ttl: int = 604800):
    r = get_redis()
    await r.set(cache_key(q_hash, subject_id), json.dumps(value), ex=ttl)

async def delete_from_cache(q_hash: str, subject_id: str = ""):
    r = get_redis()
    await r.delete(cache_key(q_hash, subject_id))

async def increment_usage(q_hash: str, subject_id: str = ""):
    r = get_redis()
    await r.incr(f"usage:{q_hash}:{subject_id}")

# ─────────────────────────────────────────────────────────
# DISTRIBUTED LOCK — prevents cache stampede
# ─────────────────────────────────────────────────────────

async def acquire_lock(q_hash: str, subject_id: str = "", ttl: int = 60) -> bool:
    """
    Try to acquire a Redis lock for this question.
    Returns True if lock acquired (this worker should call AI).
    Returns False if another worker already has the lock (should wait).
    TTL=60s: lock auto-expires if AI call crashes.
    """
    r = get_redis()
    key = lock_key(q_hash, subject_id)
    # NX = only set if Not eXists, EX = expire in ttl seconds
    result = await r.set(key, "1", nx=True, ex=ttl)
    return result is True

async def release_lock(q_hash: str, subject_id: str = ""):
    """Release the lock after AI generation is complete."""
    r = get_redis()
    await r.delete(lock_key(q_hash, subject_id))

async def wait_for_cache(q_hash: str, subject_id: str = "",
                         max_wait: int = 55, poll_interval: float = 0.5):
    """
    Poll Redis every 0.5s waiting for another worker to populate the cache.
    Returns cached data if found, None if timeout.
    max_wait=55s (slightly less than lock TTL of 60s).
    """
    waited = 0.0
    while waited < max_wait:
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        data = await get_from_cache(q_hash, subject_id)
        if data:
            return data
    return None  # timeout — caller should try AI itself as fallback
