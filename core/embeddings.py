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
    """Lazy-load embedding model (only on first embed call). Forced to CPU."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embeddings] Loading model '{EMBED_MODEL_NAME}' on CPU (first time only)...")
        # device='cpu' prevents the 'meta tensor' crash that occurs in
        # multi-worker Uvicorn when device_map='auto' puts tensors on a
        # meta device that cannot be used for inference.
        _model = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
        print("[Embeddings] ✓ Model loaded and cached in memory.")
    return _model


def embed_text(text: str) -> list[float]:
    """
    Encode text → 384-dim normalized float list.
    Runs on CPU, typically ~5ms per call.
    normalize_embeddings=True ensures cosine similarity = dot product.
    Retries once if encode() fails (meta tensor or other transient error).
    """
    global _model
    try:
        model = get_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        print(f"[Embeddings] encode() failed ({e}), reloading model and retrying...")
        _model = None  # force reload on next call
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
            vec = _model.encode(text, normalize_embeddings=True)
            print("[Embeddings] ✓ Model reloaded successfully after failure.")
            return vec.tolist()
        except Exception as e2:
            raise RuntimeError(f"[Embeddings] Fatal: model reload also failed: {e2}") from e2


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
