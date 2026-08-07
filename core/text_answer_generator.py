"""
core/text_answer_generator.py
─────────────────────────────
Generates a rich, educational text answer using OpenRouter (OpenAI-compatible API).

Strategy:
  - Course material is used as the PRIMARY source of facts/concepts
  - LLM enriches with natural language explanations (no "according to the document" citations)
  - Output is beautifully structured markdown with optional example + quick_tip fields

Model: google/gemini-flash-1.5 (fast, cheap, via OpenRouter)
"""
import os, json, re, httpx

OPENROUTER_URL   = "https://openrouter.ai/api/v1"
OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3.5-flash")

SYSTEM_PROMPT = """\
You are Professor AI — a brilliant, friendly teacher who makes complex topics simple and exciting.

You have access to course material for reference. Use it as your PRIMARY source of facts.
Then explain naturally and confidently, like a real teacher in a classroom — NOT like reading from a book.

STRICT RULES:
- NEVER say "according to the document", "the text states", "the material mentions" or any similar citation phrasing
- Write confidently and naturally, as if YOU know this topic deeply
- Use simple language suitable for a 15–16 year old student
- Use **bold** to highlight key terms and concepts
- Use short paragraphs and `-` bullet points for clarity
- Keep answers focused, accurate, and educational
- Only add an example if it genuinely helps understanding
- Only add a quick_tip if it's a genuinely useful exam/memory trick
"""

USER_PROMPT_TEMPLATE = """\
A student asked: "{question}"

Here is relevant course material for reference:
{context}

Write a beautiful, teacher-quality answer. Respond ONLY with valid JSON in exactly this format:
{{
  "answer": "Your clear explanation here (150–250 words). Use **bold** for key terms. Write in paragraphs.",
  "key_points": ["Concise fact 1", "Concise fact 2", "Concise fact 3"],
  "example": "One real-world example or analogy that makes this click (1–2 sentences). OMIT this field entirely if no genuinely helpful example exists.",
  "quick_tip": "One memorable exam tip or memory hook. OMIT this field entirely if not helpful."
}}

Do NOT add markdown fences (```) around the JSON. Return ONLY the JSON object."""


def _extract_json(text: str) -> str:
    """Extract the first JSON object from the model response."""
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    # Find first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0).strip() if match else text.strip()


async def generate_text_answer(question: str, context: str) -> dict:
    """
    Call OpenRouter (OpenAI-compatible) to generate a structured educational answer.
    Returns dict: {answer, key_points, example?, quick_tip?}
    Raises RuntimeError if the call fails.
    """
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")

    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            print(f"[TextGen] Calling OpenRouter ({OPENROUTER_MODEL}){' (Retry ' + str(attempt) + ')' if attempt > 0 else ''}")
            async with httpx.AsyncClient(timeout=90) as client:
                resp = await client.post(
                    f"{OPENROUTER_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENROUTER_MODEL,
                        "max_tokens": 5000,
                        "temperature": 0.5,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                    },
                )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter HTTP {resp.status_code}: {resp.text[:200]}"
                )

            resp_json = resp.json()
            raw_text = resp_json.get("choices", [{}])[0].get("message", {}).get("content", None)
            if not raw_text:
                raise RuntimeError(f"OpenRouter returned no text content. Response: {resp_json}")

            cleaned = _extract_json(raw_text)
            result  = json.loads(cleaned)

            if "answer" not in result:
                raise RuntimeError(f"OpenRouter response missing 'answer' key. Raw: {raw_text[:120]}")

            # Ensure key_points is always a list
            if not isinstance(result.get("key_points"), list):
                result["key_points"] = []

            print(f"[TextGen] ✓ OpenRouter — answer={len(result['answer'])} chars")
            return result

        except json.decoder.JSONDecodeError as e:
            if attempt == max_retries:
                print(f"[TextGen] Failed to parse JSON after {max_retries + 1} attempts.")
                raise e
            print(f"[TextGen] JSON parse error: {e}. Retrying...")
