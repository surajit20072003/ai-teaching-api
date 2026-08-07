"""
core/image_prompt.py — Shared educational image prompt builder
==============================================================

Used by:
  - core/pregen.py      (slide-wise pre-generation pipeline)
  - core/notes/note_service.py (textbook notes feature)

Provides:
  - _load_prompt_file()   — load image_system_prompt.txt from core/prompts/
  - _build_image_prompt() — template-based fallback (keyword heuristics)
  - enhance_image_prompt() — LLM-enhanced via Ollama + image_system_prompt.txt
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

# ── Prompt file loader ──────────────────────────────────────────────────────────
_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_prompt_file(filename: str) -> str:
    """Load a prompt from core/prompts/ — no restart needed to update."""
    try:
        return (_PROMPT_DIR / filename).read_text(encoding="utf-8")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[image_prompt] Could not load {filename}: {e}")
        return ""


# ── Template-based fallback ─────────────────────────────────────────────────────
def _build_image_prompt(description: str, context: dict | None = None) -> str:
    """
    Build a rich, context-aware educational image prompt using keyword heuristics.
    Falls back to generic textbook style when no keywords match.
    """
    desc_lower = description.lower()

    # Pick style/color/composition based on content keywords
    if any(w in desc_lower for w in ["cell", "organ", "dna", "molecule", "anatomy", "biology",
                                       "physiology", "tissue", "membrane", "protein", "nerve"]):
        style = (
            "detailed scientific illustration, cross-section biological diagram, "
            "labeled anatomical parts with arrows, textbook medical style, "
            "semi-transparent layers showing internal structure, "
            "teal and rose color palette, white annotation labels"
        )
        composition = "cross-section or diagram centered, labeled parts with leader lines"
        bg = "clean white scientific background"

    elif any(w in desc_lower for w in ["how", "step", "process", "stage", "phase", "cycle",
                                       "pathway", "reaction", "mechanism"]):
        style = (
            "vector art infographic, clean minimalist flat design, UI/UX style data visualization, "
            "numbered step flowchart with connected boxes and thick directional arrows, "
            "vibrant modern color palette, sharp vector lines, high contrast typography, "
            "professional corporate diagram, clear sequential flow, 2D graphic design"
        )
        composition = "centered flowchart, steps arranged logically with clear spacing and thick arrows"
        bg = "solid clean white background"

    elif any(w in desc_lower for w in ["map", "location", "place", "region", "country",
                                       "geography", "continent"]):
        style = (
            "simple outline map infographic, key locations labeled with dots and short text, "
            "earth tone flat colors, textbook map style, clean geographical illustration"
        )
        composition = "map centered, labels pointing to locations"
        bg = "light tan or soft green background"

    elif any(w in desc_lower for w in ["formula", "equation", "math", "calculate",
                                       "algebra", "derivation"]):
        style = (
            "clean academic mathematical illustration, formula large and centered, "
            "variable labels with arrows, professional textbook style, "
            "clear step-by-step math visualization"
        )
        composition = "formula centered, variables labeled on sides"
        bg = "clean white background with subtle grid"

    elif any(w in desc_lower for w in ["history", "historical", "war", "empire",
                                       "civilization", "ancient", "revolution"]):
        style = (
            "historical educational illustration, period-appropriate scene, "
            "labeled key elements, textbook history style, clear narrative visual"
        )
        composition = "central historical scene, labeled elements around edges"
        bg = "warm parchment-like background"

    else:
        style = (
            "modern educational diagram, clean vector illustration, UI/UX graphic design, "
            "crisp labeled elements, vibrant flat colors with soft gradients, "
            "professional academic infographic, high-end textbook style"
        )
        composition = "title at top, main visual centered with clear spacing, key labels at sides"
        bg = "solid clean white background"

    formula_part = ""
    if context and context.get("formula"):
        formula_part = f" Mathematical formula shown: {context['formula']}."
    kp_part = ""
    if context and context.get("key_points"):
        kp_part = f" Key concepts: {', '.join(context['key_points'][:4])}."

    prompt = (
        f"{style}. "
        f"Topic: {description[:120]}. "
        f"{formula_part}{kp_part} "
        f"Composition: {composition}. "
        f"Background: {bg}. "
        f"high quality, sharp focus, 4K resolution, detailed, "
        f"professional educational illustration, suitable for classroom projection, "
        f"no watermarks, no text artifacts, crisp edges, well-balanced composition. "
        f"Avoid: blurry, low quality, pixelated, ugly, distorted, watermark, "
        f"dark muddy colors, overexposed, cluttered layout, overlapping elements."
    )
    return prompt


# ── LLM-enhanced prompt (uses Claude via OpusMax) ─────────────────────────────
async def enhance_image_prompt(description: str, context: dict | None = None) -> str:
    """
    Use Claude (via OpusMax) with image_system_prompt.txt as system prompt to produce a
    rich, rule-following educational image prompt (80+ words).

    Falls back to _build_image_prompt() on any error.

    Args:
        description: Raw image description from LLM (e.g. "diagram of mitochondria").
        context: Optional dict with keys: formula, key_points, slide_type, is_story, is_tips.

    Returns:
        A fully-formed Wan2GP image prompt string.
    """
    system_prompt = _load_prompt_file("image_system_prompt.txt")
    if not system_prompt:
        return _build_image_prompt(description, context)

    # Determine slide_type from context
    slide_type = "concept"
    if context:
        if context.get("is_story"):
            slide_type = "story"
        elif context.get("is_tips"):
            slide_type = "tips"
        elif context.get("formula") or context.get("visual_type") == "manim":
            slide_type = "formula"

    user_content = json.dumps({
        "title":       description[:100],
        "infographic": description,
        "key_points":  (context.get("key_points") or [])[:4] if context else [],
        "slide_type":  slide_type,
        "formula":     (context.get("formula") or "") if context else "",
    }, ensure_ascii=False)

    opusmax_url   = "https://api.opusmax.pro"
    opusmax_key   = os.getenv("OPUSMAX_API_KEY", "")
    opusmax_model = os.getenv("OPUSMAX_MODEL", "claude-haiku-4-5-20251001")

    if not opusmax_key:
        return _build_image_prompt(description, context)

    payload = {
        "model":      opusmax_model,
        "max_tokens": 4000,
        "temperature": 0.4,
        "system":     system_prompt + "\n\nCRITICAL: DO NOT use <thinking> tags. DO NOT output any reasoning or thought process. Output ONLY the final image prompt directly.",
        "messages": [
            {"role": "user", "content": f"Generate the image prompt for this content:\n{user_content}"},
        ],
    }

    import asyncio
    
    # Global semaphore to ensure we only call OpusMax one at a time to prevent rate limits
    if not hasattr(enhance_image_prompt, "_semaphore"):
        enhance_image_prompt._semaphore = asyncio.Semaphore(1)

    try:
        async with enhance_image_prompt._semaphore:
            async with httpx.AsyncClient(timeout=90) as client:
                resp = await client.post(
                    f"{opusmax_url}/v1/messages",
                    headers={
                        "x-api-key":    opusmax_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()

        content_blocks = resp.json().get("content", [])
        content = next((b["text"] for b in content_blocks if b.get("type") == "text"), "").strip()

        # Validate: must be meaningful (>40 chars) and not a JSON dump
        if len(content) > 40 and not content.startswith("{"):
            # Strip hex color codes — Wan2GP renders them as literal text artifacts
            content = re.sub(r'#[0-9A-Fa-f]{3,8}\b', '', content)
            content = ' '.join(content.split())
            import logging
            logging.getLogger(__name__).info(f"[image_prompt] ✓ Enhanced via Claude: {len(content)} chars")
            return content

        import logging
        logging.getLogger(__name__).warning(
            f"[image_prompt] Claude returned short/invalid output ({len(content)} chars) — using template. Raw response: {resp.text[:500]}"
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[image_prompt] Claude enhance failed: {str(e)}")

    return _build_image_prompt(description, context)

