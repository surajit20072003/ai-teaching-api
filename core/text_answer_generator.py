"""
core/text_answer_generator.py
─────────────────────────────
Generates a concise, RAG-grounded text answer using OpenRouter FREE models.
Cost: $0.00 — all models below have prompt_price = 0.

Priority:
  1. nvidia/nemotron-3-super-120b-a12b:free   (120B, 1M ctx, best quality)
  2. meta-llama/llama-3.3-70b-instruct:free   (70B, reliable JSON output)
  3. qwen/qwen3-coder:free                    (1M ctx, excellent JSON)
  4. nvidia/nemotron-3-nano-30b-a3b:free      (30B, fastest)
  5. openrouter/free                          (last resort auto-pick)

If all fail → raises RuntimeError (caller returns 503).
"""
import os, json, httpx

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "")

# Free models in priority order — all have $0 prompt/completion pricing
FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/free",
]

SYSTEM_PROMPT = (
    "You are Professor AI, an expert teacher. "
    "Answer the student's question STRICTLY based only on the provided document context. "
    "If the context does not contain enough information, say so honestly. "
    "Never hallucinate or invent facts."
)

USER_PROMPT_TEMPLATE = """\
A student asked: "{question}"

Here is relevant content from the official course material:
--- BEGIN DOCUMENT CONTEXT ---
{context}
--- END DOCUMENT CONTEXT ---

Generate a clear, concise answer based ONLY on the above document content.
Reply ONLY with valid JSON. Do NOT copy the placeholder text below, you must write the actual answer!
{{"answer": "Write the actual full explanation here (200-300 words, simple language)", "key_points": ["Write actual key point 1 here", "Write actual key point 2 here"]}}"""


def _strip_fences(text: str) -> str:
    """Extract JSON object from text, ignoring markdown and conversational filler."""
    import re
    text = text.strip()
    
    # Try to find a JSON block between curly braces
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    
    return text.strip()


async def generate_text_answer(question: str, context: str) -> dict:
    """
    Try each free OpenRouter model in order until one returns valid JSON.
    Returns dict: {answer, key_points}
    Raises RuntimeError if all models fail.
    """
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")

    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
    last_error = None

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
                        "temperature": 0.4,
                        "max_tokens": 700,
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

            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            cleaned  = _strip_fences(raw_text)
            result   = json.loads(cleaned)

            # Validate shape
            if "answer" not in result:
                last_error = f"{model}: JSON missing 'answer' key"
                continue

            print(f"[TextGen] ✓ {model} answered successfully")
            return result

        except json.JSONDecodeError as e:
            last_error = f"{model}: bad JSON — {e} | raw: {raw_text[:100]}"
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
