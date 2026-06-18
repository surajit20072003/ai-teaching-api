"""
core/semantic_check.py
LLM-powered equivalence check for semantic cache.

Uses the cheapest available model (Gemini Flash 8B) to answer
one yes/no question: "Are these two questions asking about exactly
the same topic?"

Cost: ~$0.000001 per call. Only called in the gray zone (0.70–0.97).
"""
import httpx, os

async def llm_same_topic(q1: str, q2: str) -> bool:
    """
    Returns True if the LLM considers q1 and q2 to be about the
    same specific topic — i.e. the cached content would answer q1.

    Examples:
      "explain Newton 2nd law" vs "what is Newton 2nd law?" → True
      "Newton 2nd law"         vs "Newton 3rd law"          → False
      "explain photosynthesis" vs "what is photosynthesis"   → True
      "chapter 3 summary"      vs "chapter 5 summary"        → False
    """
    prompt = (
        "You are a strict topic-equivalence checker for an education cache system.\n"
        "Decide if two student questions are asking about EXACTLY the same specific topic.\n"
        "Even small differences like '2nd law' vs '3rd law' mean they are NOT the same.\n\n"
        f"Q1: {q1}\n"
        f"Q2: {q2}\n\n"
        "Reply with a single word: YES or NO"
    )

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                json={
                    # Smallest / cheapest model — only needs to say YES/NO
                    "model": "google/gemini-2.5-flash-lite-preview-06-17",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0,
                }
            )
            resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        result = answer.startswith("YES")
        print(f"[LLM Equivalence] '{q1}' vs '{q2}' → {answer} → {'SAME' if result else 'DIFFERENT'}")
        return result
    except Exception as e:
        # On failure → be conservative, don't serve wrong cache
        print(f"[LLM Equivalence] Error (fail-safe: DIFFERENT): {e}")
        return False
