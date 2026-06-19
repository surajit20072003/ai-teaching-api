import httpx, os, json, re, logging

logger = logging.getLogger(__name__)

# ── Env config ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_LLM     = "google/gemini-2.5-flash"          # real-time: fast cloud

OLLAMA_URL         = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # pre-gen: local GPU
OLLAMA_TIMEOUT     = int(os.getenv("OLLAMA_TIMEOUT", "600"))   # 10 min — local models are slow

# ── Prompts ───────────────────────────────────────────────────────────────────
_BASE_FORMAT = """
Reply ONLY with this exact JSON (no markdown fences, no extra text):
{{
  "presentation_slides": [
    {{
      "title": "Introduction to ...",
      "content": "Main explanation text...",
      "narration": "See, what happens here is... [conversational 150-250 words]",
      "keyPoints": ["Point 1", "Point 2", "Point 3"],
      "formula": "F = ma",
      "infographic": "Diagram showing ...",
      "isStory": false,
      "isTips": false
    }}
  ],
  "latex_formulas": [{{"formula": "F=ma", "explanation": "Force equals mass times acceleration"}}],
  "key_points": ["Summary point 1", "Summary point 2"],
  "follow_up_questions": ["Can you explain inertia?", "What is impulse?"]
}}"""

SLIDE_PROMPT = (
    "You are Professor AI, an expert Indian teacher specializing in {subject}.\n"
    "Generate a 6-7 slide mini-lecture for this student question: \"{question}\"\n\n"
    "RULES:\n"
    "- Narration should be 150-250 words (will be converted to audio)\n"
    "- Last slide MUST have isStory: true (story or analogy)\n"
    "- Second-to-last slide MUST have isTips: true (memory tips)\n"
    "- formula field: LaTeX string or empty string \"\"\n"
) + _BASE_FORMAT

RAG_PROMPT = (
    "You are Professor AI, an expert Indian teacher specializing in {subject}.\n"
    "A student asked: \"{question}\"\n\n"
    "Here is relevant content from the official course material:\n"
    "--- BEGIN DOCUMENT CONTEXT ---\n"
    "{context}\n"
    "--- END DOCUMENT CONTEXT ---\n\n"
    "Generate a 6-7 slide mini-lecture BASED ON the document content above.\n"
    "Use the document's specific explanations, examples, and terminology.\n\n"
    "RULES:\n"
    "- Narration should be 150-250 words (will be converted to audio)\n"
    "- Last slide MUST have isStory: true\n"
    "- Second-to-last slide MUST have isTips: true\n"
    "- formula field: LaTeX string or empty string \"\"\n"
    "- Add \"is_doc_grounded\": true to the JSON root\n"
) + _BASE_FORMAT.replace(
    '"presentation_slides"',
    '"is_doc_grounded": true,\n  "presentation_slides"'
)


# ── JSON helpers ──────────────────────────────────────────────────────────────
def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*',     '', raw)
    raw = re.sub(r'\s*```$',     '', raw)
    return raw.strip()


def _parse_slides(raw: str) -> dict:
    raw = _clean_json(raw)
    try:
        import json_repair
        data = json_repair.loads(raw)
        if not isinstance(data, dict):
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json_repair.loads(match.group()) if match else {}
    except Exception:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Cannot parse LLM response as JSON: {raw[:200]}")

    slides = data.get("presentation_slides", [])
    if not any(s.get("isTips") for s in slides):
        slides.append({
            "title": "Tips & Tricks to Remember",
            "content": "Key memory aids for this concept",
            "narration": "Here are tips to remember this concept easily. Use mnemonics and visual associations.",
            "keyPoints": ["Make a diagram", "Use mnemonics", "Practice with examples"],
            "formula": "", "infographic": "Lightbulb with tips", "isStory": False, "isTips": True
        })
    if not any(s.get("isStory") for s in slides):
        slides.append({
            "title": "A Story to Remember",
            "content": "Real world analogy",
            "narration": "Let me tell you a story that will help you remember this concept forever.",
            "keyPoints": ["Real world connection", "Easy to visualize", "Never forget"],
            "formula": "", "infographic": "Story illustration", "isStory": True, "isTips": False
        })
    data["presentation_slides"] = slides
    return data


# ── Ollama client ─────────────────────────────────────────────────────────────
async def _generate_via_ollama(prompt: str) -> str:
    """
    PRE-GENERATION PATH: Call local Ollama — free, local GPU, no API cost.
    NO fallback. If Ollama is down, the batch row is marked failed and retried later.
    Timeout: 600s (10 min) to accommodate slow local models.
    """
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model":   OLLAMA_MODEL,
        "stream":  False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.7, "num_predict": 4096},
    }
    logger.info(f"[SlideGen/Ollama] POST {url} model={OLLAMA_MODEL} timeout={OLLAMA_TIMEOUT}s")
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
    content = resp.json()["message"]["content"]
    logger.info(f"[SlideGen/Ollama] OK — {len(content)} chars")
    return content


# ── OpenRouter client ─────────────────────────────────────────────────────────
async def _generate_via_openrouter(prompt: str) -> str:
    """
    REAL-TIME PATH: Call OpenRouter (Gemini 2.5 Flash) — fast cloud response.
    Used only when a student hits a cache miss at runtime.
    """
    logger.info(f"[SlideGen/OpenRouter] model={OPENROUTER_LLM}")
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            OPENROUTER_BASE,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model":    OPENROUTER_LLM,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if resp.status_code == 402:
            raise RuntimeError(
                "OpenRouter credits exhausted. Top up at https://openrouter.ai/credits"
            )
        if resp.status_code == 429:
            raise RuntimeError(
                "OpenRouter rate limit — retry in a few seconds."
            )
        resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    logger.info(f"[SlideGen/OpenRouter] OK — {len(content)} chars")
    return content


# ── Public API ────────────────────────────────────────────────────────────────
async def generate_slides(
    question:   str,
    subject:    str,
    context:    str  = "",
    use_local:  bool = False,
) -> dict:
    """
    Generate 6-7 structured lecture slides.

    use_local=True  → Ollama (offline pre-gen, FREE, local GPU, timeout=600s)
                      NO fallback — failure = row marked failed, retried later
    use_local=False → OpenRouter (real-time user request, fast cloud, timeout=90s)
    """
    prompt = (
        RAG_PROMPT.format(question=question, subject=subject, context=context)
        if context else
        SLIDE_PROMPT.format(question=question, subject=subject)
    )

    if use_local:
        # ── Pre-generation: Ollama only — NO fallback ─────────────────────
        raw = await _generate_via_ollama(prompt)
    else:
        # ── Real-time: OpenRouter only ────────────────────────────────────
        raw = await _generate_via_openrouter(prompt)

    return _parse_slides(raw)
