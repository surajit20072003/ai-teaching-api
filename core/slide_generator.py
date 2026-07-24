import httpx, os, json, re, logging

logger = logging.getLogger(__name__)

# ── Env config ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_LLM     = "google/gemini-2.5-flash"          # real-time: fast cloud

OLLAMA_URL         = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
OLLAMA_TIMEOUT     = int(os.getenv("OLLAMA_TIMEOUT", "900"))

# ── Base JSON format spec (shared by all prompts) ─────────────────────────────
_BASE_FORMAT = """
Reply ONLY with this exact JSON (no markdown fences, no extra text):
{{
  "presentation_slides": [
    {{
      "title": "Slide title here",
      "content": "Clear 2-3 sentence explanation of this slide's sub-topic.",
      "narration": "Conversational teacher voice — 180-280 words. Explain step by step. Include a real-life example. Do NOT start with Hello/Welcome/Good morning.",
      "keyPoints": ["Point 1", "Point 2", "Point 3"],
      "formula": "LaTeX string if math present, else empty string",
      "infographic": "Specific, concrete diagram description (e.g. 'Factor tree of 72 = 2×2×2×3×3 as branching diagram with labeled nodes', NOT 'concept diagram')",
      "isStory": false,
      "isTips": false,
      "visual_type": "image"
    }}
  ],
  "latex_formulas": [{{"formula": "F=ma", "explanation": "Force equals mass times acceleration"}}],
  "key_points": ["Summary point 1", "Summary point 2"],
  "follow_up_questions": ["Follow-up question 1?", "Follow-up question 2?"]
}}"""

# ── Prompt: Pure LLM (no document context) ────────────────────────────────────
SLIDE_PROMPT = """\
You are Professor AI — a brilliant, engaging Indian teacher specialising in {subject}.
A student asked: "{question}"

━━━ STEP 1 — DECIDE SLIDE COUNT (do this BEFORE writing any slide) ━━━
Analyse the question complexity and choose the right number of slides:
  • 1–2 slides → simple facts, dates, single definitions, yes/no answers
  • 3–4 slides → concepts, processes, comparisons, cause-effect
  • 5–6 slides → multi-part topics, laws/theories with proofs, multi-stage systems
MAXIMUM is 6 slides. NEVER write more than 6.

━━━ STEP 2 — SLIDE RULES ━━━
1. Each slide covers a DISTINCT sub-topic — zero repetition across slides.
2. Narration: 180–280 words. Sound like an enthusiastic teacher in a classroom.
   Use simple language (suitable for a 14–16 year old). Include one real-life analogy per slide.
   Do NOT start with "Hello", "Welcome", "Good morning", or any greeting.
3. "content": clear 2-3 sentence explanation of that slide's sub-topic.
4. "infographic": describe ONE specific visual with exact content
   (e.g. "Diagram of food chain: Sun → Grass → Rabbit → Fox, with arrows and labels"
    NOT "diagram of concept").
5. "keyPoints": exactly 3 crisp bullet points per slide.
6. "formula": exact LaTeX if slide has math, else empty string "".
7. "visual_type": "manim" for math formulas/graphs/geometry. "image" for everything else.
   isStory and isTips slides MUST always be "image".

━━━ STEP 3 — SPECIAL SLIDES (only if total slides ≥ 3) ━━━
• Second-to-last slide → isTips: true — memorable mnemonics and memory tricks.
• Last slide → isStory: true — a vivid real-world story/analogy that makes the concept unforgettable.
  (Skip these for 1–2 slide answers — they don't need padding)

""" + _BASE_FORMAT

# ── Prompt: Document-first + LLM-enriched (with RAG context) ─────────────────
RAG_PROMPT = """\
You are Professor AI — a brilliant, engaging Indian teacher specialising in {subject}.
A student asked: "{question}"

Here is relevant content from the official course material:
--- BEGIN DOCUMENT CONTEXT ---
{context}
--- END DOCUMENT CONTEXT ---

━━━ STEP 0 — SLIDE 1: DIRECTLY ANSWER THE QUESTION FIRST ━━━
Slide 1 is the "answer slide". The student needs the answer IMMEDIATELY — not after background.

  If the question is MCQ (has options like (a) (b) (c) (d)):
    • title:      "Correct Answer: (b) Plants"  ← exact option letter + text
    • content:    First sentence names the correct answer. Next 2 sentences explain WHY it is correct.
    • keyPoints:  ["Correct option: (b) Plants", "Why: reason 1", "Why: reason 2"]
    • infographic: All 4 options shown in a 2x2 grid. Correct option box highlighted bright green.
                  Wrong options greyed out. Question text at the top in bold.

  If the question is Descriptive/Explain/Define:
    • title:    Core answer in 5–7 words (e.g. "Ecosystem = Living + Non-Living System")
    • content:  Sentence 1 = direct 1-line answer. Sentences 2–3 = brief expansion.
    • NEVER use "Introduction", "Overview", "Background" as slide 1 title.
    • NEVER delay the answer to slide 2 or later.

Slides 2 onwards → detailed explanation, deeper concepts, examples, process, steps.

━━━ STEP 1 — DECIDE SLIDE COUNT (do this BEFORE writing any slide) ━━━
Analyse the question complexity and choose the right number of slides:
  • 1–2 slides → simple facts, dates, single definitions, yes/no answers
  • 3–4 slides → concepts, processes, comparisons, cause-effect
  • 5–6 slides → multi-part topics, laws/theories with proofs, multi-stage systems
MAXIMUM is 6 slides. NEVER write more than 6.

━━━ STEP 2 — CONTENT STRATEGY ━━━
Use the document content as your PRIMARY source of facts, numbers, and definitions.
Then ENRICH with your own knowledge to make explanations clear, vivid, and memorable.

RULES:
• NEVER say "according to the document", "the text states", or any citation phrase.
• Write confidently, like a teacher who truly understands the topic — not like someone reading from a book.
• Prioritise accuracy to the document's facts, but explain them in natural, engaging language.
• Add helpful analogies, step-by-step reasoning, or real-world examples the document may not include.
• Add "is_doc_grounded": true to the JSON root.

━━━ STEP 3 — SLIDE RULES ━━━
1. Each slide covers a DISTINCT sub-topic — zero repetition across slides.
2. Narration: 180–280 words. Conversational teacher voice. Simple language (14–16 year old level).
   Include one concrete real-life example per slide.
   Do NOT start with "Hello", "Welcome", "Good morning", or any greeting.
3. "content": clear 2-3 sentence explanation drawn from the document facts.
4. "infographic": describe ONE specific visual with exact content
   (e.g. "Venn diagram comparing photosynthesis vs respiration with 3 differences labelled"
    NOT "diagram of concept").
5. "keyPoints": exactly 3 crisp, fact-based bullet points per slide.
6. "formula": exact LaTeX if slide has math, else empty string "".
7. "visual_type": "manim" for math/graphs/geometry slides. "image" for all others.
   isStory and isTips slides MUST always be "image".

━━━ STEP 4 — SPECIAL SLIDES (only if total slides ≥ 3) ━━━
• Second-to-last slide → isTips: true — memorable mnemonics and memory tricks for exam.
• Last slide → isStory: true — a vivid real-world analogy that makes the concept unforgettable.
  (Skip these for 1–2 slide answers — they don't need padding)

""" + _BASE_FORMAT.replace(
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

    # ── Cap at 6 slides (truncate if LLM went over) ───────────────────────────
    if len(slides) > 6:
        logger.warning(f"[SlideGen] LLM returned {len(slides)} slides — truncating to 6")
        # Try to preserve the last isStory/isTips slides if they exist
        story_slides = [s for s in slides if s.get("isStory")]
        tips_slides  = [s for s in slides if s.get("isTips")]
        body_slides  = [s for s in slides if not s.get("isStory") and not s.get("isTips")]
        slides = body_slides[:6 - len(story_slides) - len(tips_slides)] + tips_slides + story_slides
        slides = slides[:6]

    # ── Only add fallback Tips/Story if total slides >= 3 ─────────────────────
    if len(slides) >= 3:
        if not any(s.get("isTips") for s in slides):
            slides.insert(-1 if len(slides) > 1 else len(slides), {
                "title": "Tips & Tricks to Remember",
                "content": "Key memory aids and mnemonics for this concept.",
                "narration": (
                    "Now let me give you some powerful tricks to lock this concept in your memory forever. "
                    "The best students don't just read — they create mental hooks. "
                    "First, link the new concept to something you already know from daily life. "
                    "Second, write down the key formula or definition three times without looking. "
                    "Third, make a mind map connecting the main idea to its sub-topics. "
                    "Fourth, explain it out loud as if you are teaching a friend — "
                    "this forces your brain to actually understand rather than just recognise. "
                    "Fifth, create a simple acronym or rhyme for any list you need to memorise. "
                    "Use these five techniques together, and this concept will stay with you "
                    "long after the exam is over."
                ),
                "keyPoints": ["Create a mental hook or analogy", "Write it 3 times to lock it in", "Teach it to someone else"],
                "formula": "", "infographic": "Lightbulb with 5 numbered tips, colourful icons for each step",
                "isStory": False, "isTips": True, "visual_type": "image"
            })
        if not any(s.get("isStory") for s in slides):
            slides.append({
                "title": "A Story to Remember",
                "content": "A real-world analogy to make this concept unforgettable.",
                "narration": (
                    "Let me leave you with a story that will make this concept impossible to forget. "
                    "The best way to truly understand any idea is to see it alive in the world around you. "
                    "Think about how this concept shows up in everyday life — in the kitchen, on a road trip, "
                    "in a cricket match, or in something you use every single day. "
                    "When you can connect what you learn in the classroom to something real, "
                    "you have not just memorised it — you have understood it. "
                    "And understanding is what leads to top marks, not just memorisation. "
                    "So the next time you encounter this topic in an exam or in life, "
                    "remember this story, remember the connection, and the answer will come naturally."
                ),
                "keyPoints": ["Connect to real life", "See it, visualise it", "Understanding beats memorisation"],
                "formula": "", "infographic": "Warm classroom scene with teacher and students, light bulb of insight above",
                "isStory": True, "isTips": False, "visual_type": "image"
            })

    if not slides:
        logger.warning("[SlideGen] No slides generated — model produced empty response.")
    elif len(slides) < 3:
        logger.info(f"[SlideGen] {len(slides)} slide(s) generated — short answer mode.")

    # ── Ensure visual_type defaults correctly ────────────────────────────────
    for slide in slides:
        if not slide.get("visual_type"):
            if slide.get("isStory") or slide.get("isTips"):
                slide["visual_type"] = "image"
            elif slide.get("formula", "").strip():
                slide["visual_type"] = "manim"
            else:
                slide["visual_type"] = "image"
        # Safety: story/tips must always be image
        if slide.get("isStory") or slide.get("isTips"):
            slide["visual_type"] = "image"

    data["presentation_slides"] = slides
    return data


# ── Ollama client ─────────────────────────────────────────────────────────────
async def _generate_via_ollama(prompt: str) -> str:
    """
    PRE-GENERATION PATH: Call local Ollama — free, local GPU, no API cost.
    NO fallback. If Ollama is down, the batch row is marked failed and retried later.
    """
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model":      OLLAMA_MODEL,
        "stream":     False,
        "format":     "json",
        "messages":   [{"role": "user", "content": prompt}],
        "keep_alive": -1,
        "options":    {"temperature": 0.65, "num_predict": 8000, "num_ctx": 8192},
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
            raise RuntimeError("OpenRouter credits exhausted. Top up at https://openrouter.ai/credits")
        if resp.status_code == 429:
            raise RuntimeError("OpenRouter rate limit — retry in a few seconds.")
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
    Generate 1–6 structured lecture slides (count decided by LLM based on complexity).

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
        raw = await _generate_via_ollama(prompt)
    else:
        raw = await _generate_via_openrouter(prompt)

    return _parse_slides(raw)
