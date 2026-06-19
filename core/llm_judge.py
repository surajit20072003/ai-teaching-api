"""
core/llm_judge.py — single LLM call picks best cached match or triggers NEW.
"""
import os, httpx, logging
logger = logging.getLogger(__name__)

JUDGE_MODEL    = "google/gemini-2.5-flash-lite"
MIN_THRESHOLD  = 0.60
MAX_CANDIDATES = 5


async def llm_pick_best_match(user_question: str, candidates: list[dict]) -> str:
    """Candidates: list of {question, score}. Returns '1'-'5' or 'NEW'."""
    if not candidates:
        return "NEW"
    best = max(c["score"] for c in candidates)
    if best < MIN_THRESHOLD:
        logger.info(f"[llm_judge] All scores<{MIN_THRESHOLD} (best={best:.3f}) → NEW (no LLM)")
        return "NEW"

    lines = "\n".join(
        f"{i+1}. \"{c['question']}\"  [similarity: {c['score']:.2f}]"
        for i, c in enumerate(candidates)
    )
    n = len(candidates)
    prompt = (
        f"You are a teaching assistant cache evaluator.\n\n"
        f"A student asked: \"{user_question}\"\n\n"
        f"Cached answers available:\n{lines}\n\n"
        f"Rules:\n"
        f"- Match must cover the EXACT SAME concept (Newton 2nd ≠ Newton 3rd).\n"
        f"- F=ma = Explain Newton 2nd Law → reply 1\n"
        f"- Chapter 3 ≠ Chapter 5 → reply NEW\n\n"
        f"Reply with ONLY a digit 1-{n} or the word NEW:"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                json={"model": JUDGE_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 5, "temperature": 0},
            )
            r.raise_for_status()
        raw    = r.json()["choices"][0]["message"]["content"].strip().upper()
        valid  = {str(i+1) for i in range(n)} | {"NEW"}
        result = raw if raw in valid else "NEW"
        logger.info(f"[llm_judge] '{user_question[:50]}' best={best:.3f} → {result}")
        return result
    except Exception as e:
        logger.warning(f"[llm_judge] Error (→ NEW): {e}")
        return "NEW"


async def judge_and_pick(user_question: str, candidates: list[dict], subject_id: str) -> dict | None:
    """High-level wrapper: returns winning candidate dict or None."""
    if not candidates:
        return None
    top  = candidates[:MAX_CANDIDATES]
    pick = await llm_pick_best_match(user_question, top)
    if pick == "NEW":
        return None
    winner = top[int(pick) - 1]
    logger.info(f"[llm_judge] ✅ Cache hit: subject={subject_id} score={winner['score']:.3f}")
    return winner
