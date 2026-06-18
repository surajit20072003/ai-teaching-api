"""
core/embeddings.py
──────────────────
Local sentence-transformer embedding module.
Model: all-MiniLM-L6-v2 (22MB, 384 dimensions)
Cost: $0 — runs entirely on CPU using your server hardware.

The model loads ONCE on first call and stays in memory.
embed_async() runs in a thread pool so it never blocks the async event loop.
"""
import asyncio

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    """Lazy-load embedding model (only on first embed call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embeddings] Loading model '{EMBED_MODEL_NAME}' (first time only)...")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        print("[Embeddings] ✓ Model loaded and cached in memory.")
    return _model


def embed_text(text: str) -> list[float]:
    """
    Encode text → 384-dim normalized float list.
    Runs on CPU, typically ~5ms per call.
    normalize_embeddings=True ensures cosine similarity = dot product.
    """
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


async def embed_async(text: str) -> list[float]:
    """
    Async wrapper: runs embed_text() in the default ThreadPoolExecutor
    so it doesn't block the FastAPI event loop during embedding.
    Safe to call from any async endpoint.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_text, text)


def vec_to_pg_str(vec: list[float]) -> str:
    """Convert Python list to PostgreSQL vector literal string '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
