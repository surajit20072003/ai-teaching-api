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
    "1. Each slide covers a DISTINCT sub-topic — no repetition across slides.\n"
    "2. Narration: MINIMUM 220 words. Count them. Conversational teacher voice. "
    "   Explain step-by-step. Include a concrete real-life example in EVERY slide. "
    "   Do NOT start with 'Hello', 'Welcome', 'Good morning', or any greeting.\n"
    "3. 'content' field: clear 2-3 sentence explanation of that slide's sub-topic.\n"
    "4. 'infographic' field: describe a SPECIFIC visual (e.g. 'Factor tree of 72 = 2x2x2x3x3', NOT 'diagram of concept').\n"
    "5. 'keyPoints': exactly 3 crisp bullet points for the slide.\n"
    "6. 'formula': LaTeX string if slide has math, else empty string \"\".\n"
    "7. Last slide MUST have isStory: true — a real-world analogy or story that makes the concept unforgettable.\n"
    "8. Second-to-last slide MUST have isTips: true — mnemonic tricks and memory aids.\n"
    "9. You MUST generate AT LEAST 5 slides. Aim for 6-7.\n"
) + _BASE_FORMAT

RAG_PROMPT = (
    "You are Professor AI, an expert Indian teacher specializing in {subject}.\n"
    "A student asked: \"{question}\"\n\n"
    "Here is relevant content from the official course material:\n"
    "--- BEGIN DOCUMENT CONTEXT ---\n"
    "{context}\n"
    "--- END DOCUMENT CONTEXT ---\n\n"
    "Generate a 6-7 slide mini-lecture STRICTLY based ONLY on the document content above.\n\n"
    "CRITICAL RULES — MUST FOLLOW ALL:\n"
    "1. ONLY use facts, examples, definitions, and numbers from the document. DO NOT invent anything.\n"
    "2. Quote the document's specific examples verbatim where possible (e.g. if doc says '4^n = 2^2n', use it).\n"
    "3. If the document has numbered theorems or definitions, reference them by number (e.g. 'Theorem 1.2 states...').\n"
    "4. Narration: MINIMUM 220 words. Count them. Sound like a teacher reading the textbook aloud and "
    "   explaining each line step by step. Do NOT start with 'Hello', 'Welcome', or any greeting.\n"
    "5. 'content' field: 2-3 sentence direct summary pulled from document text.\n"
    "6. 'infographic' field: describe a SPECIFIC diagram tied to the document's example\n"
    "   (e.g. 'Factor tree: 12 = 2 x 2 x 3, shown as branching tree diagram' NOT 'diagram of concept').\n"
    "7. 'keyPoints': exactly 3 bullet points — each must be a fact or formula FROM the document.\n"
    "8. 'formula': exact LaTeX of any formula in the document for this slide, else empty string \"\".\n"
    "9. Each slide must cover a DISTINCT section of the document — spread content across all slides.\n"
    "10. Last slide MUST have isStory: true — a real-world analogy tied directly to the document topic.\n"
    "11. Second-to-last slide MUST have isTips: true — memory tricks for the document's key theorems/formulas.\n"
    "12. Add \"is_doc_grounded\": true to the JSON root.\n"
    "13. You MUST generate AT LEAST 5 slides. Aim for 6-7.\n"
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
            "narration": "Here are tips to remember this concept easily. Use mnemonics and visual associations. "
                         "First, create a mental image of the key idea. Second, link it to something you already know. "
                         "Third, practice writing the formula or definition three times without looking. "
                         "Fourth, explain it out loud to someone as if you are the teacher. "
                         "Fifth, make a short note card with the key formula on one side and an example on the other. "
                         "These five techniques together will lock this concept in your long-term memory.",
            "keyPoints": ["Make a diagram", "Use mnemonics", "Practice with examples"],
            "formula": "", "infographic": "Lightbulb with numbered tips list", "isStory": False, "isTips": True
        })
    if not any(s.get("isStory") for s in slides):
        slides.append({
            "title": "A Story to Remember",
            "content": "Real world analogy",
            "narration": "Let me tell you a story that will help you remember this concept forever. "
                         "Imagine you are explaining this to a friend who has never studied this topic before. "
                         "You would start with something they already know from daily life, then connect it step by step "
                         "to the idea we just learned. This is exactly how great teachers make difficult concepts stick. "
                         "The real world is full of examples of this concept if you look carefully. "
                         "So next time you encounter this in an exam or in life, you will recognize it immediately "
                         "because you have this story anchored in your memory.",
            "keyPoints": ["Real world connection", "Easy to visualize", "Never forget"],
            "formula": "", "infographic": "Story illustration with characters", "isStory": True, "isTips": False
        })
    # Enforce minimum 5 slides
    if len(slides) < 5:
        logger.warning(f"[SlideGen] Only {len(slides)} slides generated — model produced too few. Check prompt/token limit.")
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
        "format":  "json",                                    # Forces Ollama to output valid JSON always
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.7, "num_predict": 8000},  # 8000 tokens → richer 220+ word narrations
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
