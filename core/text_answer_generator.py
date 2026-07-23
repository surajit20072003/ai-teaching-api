"""
core/text_answer_generator.py
─────────────────────────────
Generates a rich, educational text answer using OpenRouter FREE models.

Strategy:
  - Course material is used as the PRIMARY source of facts/concepts
  - LLM enriches with natural language explanations (no "according to the document" citations)
  - Output is beautifully structured markdown with optional example + quick_tip fields

Cost: $0.00 — all models below have prompt_price = 0.
"""
import os, json, re, httpx

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "")

# Free models in priority order — all have $0 prompt/completion pricing
FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/auto",
]

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
    Try each free OpenRouter model in order until one returns valid JSON.
    Returns dict: {answer, key_points, example?, quick_tip?}
    Raises RuntimeError if all models fail.
    """
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")

    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
    last_error = None
    raw_text = ""

    for model in FREE_MODELS:
        print(f"[TextGen] Trying {model}")
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                resp = await client.post(
                    f"{OPENROUTER_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://ai-teaching-api.internal",
                        "X-Title": "AI Teaching Assistant",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        "temperature": 0.5,
                        "max_tokens": 900,
                    },
                )

            if resp.status_code == 429:
                last_error = f"{model}: rate limited (429)"
                print(f"[TextGen] {model} → rate limited, trying next")
                continue

            if resp.status_code != 200:
                last_error = f"{model}: HTTP {resp.status_code} — {resp.text[:150]}"
                print(f"[TextGen] {model} → {last_error}")
                continue

            data     = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            cleaned  = _extract_json(raw_text)
            result   = json.loads(cleaned)

            if "answer" not in result:
                last_error = f"{model}: JSON missing 'answer' key"
                continue

            # Ensure key_points is always a list
            if not isinstance(result.get("key_points"), list):
                result["key_points"] = []

            print(f"[TextGen] ✓ {model} — answer={len(result['answer'])} chars")
            return result

        except json.JSONDecodeError as e:
            last_error = f"{model}: bad JSON — {e} | raw: {raw_text[:120]}"
            print(f"[TextGen] {model} → JSON parse failed: {e}")
            continue
        except KeyError as e:
            last_error = f"{model}: unexpected response shape — {e}"
            print(f"[TextGen] {model} → KeyError: {e}")
            continue
        except httpx.RequestError as e:
            last_error = f"{model}: network error — {e}"
            print(f"[TextGen] {model} → network error: {e}")
            continue

    raise RuntimeError(
        f"All free OpenRouter models failed. Last error: {last_error}"
    )
