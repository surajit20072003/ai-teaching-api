import httpx, os, json, re

SLIDE_PROMPT = """You are Professor AI, an expert Indian teacher specializing in {subject}.
Generate a 6-7 slide mini-lecture for this student question: "{question}"

RULES:
- Slides must be in English
- Narration should be 150-250 words (will be converted to audio)
- Last slide MUST have isStory: true (a story or analogy to remember the concept)
- Second-to-last slide MUST have isTips: true (memory tips)
- formula field: LaTeX string or empty string ""

Reply ONLY with this exact JSON (no markdown fences):
{{
  "presentation_slides": [
    {{
      "title": "Introduction to ...",
      "content": "Main explanation text...",
      "narration": "See, what happens here is... [conversational 200 words]",
      "keyPoints": ["Point 1", "Point 2", "Point 3"],
      "formula": "F = ma",
      "infographic": "Diagram showing a car being pushed with arrows labelled Force Mass Acceleration",
      "isStory": false,
      "isTips": false
    }}
  ],
  "latex_formulas": [{{"formula": "F=ma", "explanation": "Force equals mass times acceleration"}}],
  "key_points": ["Summary point 1", "Summary point 2"],
  "follow_up_questions": ["Can you explain inertia?", "What is impulse?"]
}}"""


def _clean_json(raw: str) -> str:
    """Try multiple strategies to extract valid JSON."""
    raw = raw.strip()
    # Remove markdown fences
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


async def generate_slides(question: str, subject: str) -> dict:
    """Call Gemini to generate 6-7 structured lecture slides."""
    prompt = SLIDE_PROMPT.format(question=question, subject=subject)

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
            json={
                "model": "google/gemini-2.5-flash",
                "messages": [{"role": "user", "content": prompt}]
            }
        )

        # Handle billing/rate-limit errors with clear messages
        if resp.status_code == 402:
            raise RuntimeError(
                "AI service credits exhausted. Please top up your OpenRouter balance at "
                "https://openrouter.ai/credits — no content was generated."
            )
        if resp.status_code == 429:
            raise RuntimeError(
                "AI service rate limit reached. The system is under high load — please try again in a few seconds."
            )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    raw = _clean_json(raw)

    try:
        import json_repair
        data = json_repair.loads(raw)
        if not isinstance(data, dict):
            # In case json_repair returns something else, fallback
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json_repair.loads(match.group())
            else:
                raise ValueError("Could not parse Gemini response as JSON object")
    except Exception as e:
        # Final fallback: try to extract JSON object with standard json
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse Gemini response as JSON: {e}")

    slides = data.get("presentation_slides", [])

    # Ensure story and tips slides exist
    has_story = any(s.get("isStory") for s in slides)
    has_tips = any(s.get("isTips") for s in slides)

    if not has_tips:
        slides.append({
            "title": "Tips & Tricks to Remember",
            "content": f"Key memory aids for {question}",
            "narration": f"Here are some tips to remember this concept easily. Use mnemonics and visual associations to lock it in memory.",
            "keyPoints": ["Make a diagram", "Use mnemonics", "Practice with examples"],
            "formula": "", "infographic": "A lightbulb with tips written around it", "isStory": False, "isTips": True
        })

    if not has_story:
        slides.append({
            "title": "A Story to Remember",
            "content": f"Real world analogy for {question}",
            "narration": f"Let me tell you a story that will help you remember this concept forever.",
            "keyPoints": ["Real world connection", "Easy to visualize", "Never forget"],
            "formula": "", "infographic": "A story illustration with characters", "isStory": True, "isTips": False
        })

    data["presentation_slides"] = slides
    return data
