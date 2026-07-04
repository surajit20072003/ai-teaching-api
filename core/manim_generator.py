"""
core/manim_generator.py
────────────────────────
Generate Manim Python code via Ollama (qwen3-coder:latest),
enforce audio-sync timing, render to MP4, and upload to B2.

Pipeline per slide:
  1. generate_manim_code()     — Ollama → complete Python file (using prompt files)
  2. _validate_syntax()        — AST parse before render (no wasted render time)
  3. _enforce_timing()         — scale self.wait() to match audio duration (no LLM)
  4. render_manim_code()       — subprocess: `manim -qm slide_N.py SlideScene`
  5. local save + B2 upload    — via local_storage + b2_storage

Prompt files (edit without Docker restart):
  core/prompts/manim_system_prompt.txt  — system rules (zero-failure mandate)
  core/prompts/manim_user_template.txt  — per-slide user prompt template
  core/prompts/manim_repair_prompt.txt  — repair prompt when render fails

Environment variables:
  OLLAMA_URL            — Ollama base URL (default: http://host.docker.internal:11434)
  OLLAMA_MODEL          — model to use   (default: qwen3-coder:latest)
  OLLAMA_TIMEOUT        — seconds         (default: 900)
  MANIM_MAX_RETRIES     — code-gen retries (default: 3)
  MANIM_RENDER_TIMEOUT  — seconds for manim CLI (default: 300)
  MANIM_QUALITY         — l/m/h/k (default: m = 720p)
"""

import ast
import asyncio
import logging
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import httpx

from core import local_storage

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────────
OLLAMA_URL           = os.getenv("OLLAMA_URL",    "http://host.docker.internal:11434")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL",  "qwen3-coder:latest")
OLLAMA_TIMEOUT       = int(os.getenv("OLLAMA_TIMEOUT",       "900"))
MANIM_MAX_RETRIES    = int(os.getenv("MANIM_MAX_RETRIES",    "3"))
MANIM_RENDER_TIMEOUT = int(os.getenv("MANIM_RENDER_TIMEOUT", "300"))
MANIM_QUALITY        = os.getenv("MANIM_QUALITY", "m")   # l=480p, m=720p, h=1080p

# ── Prompt file loader ──────────────────────────────────────────────────────────
_PROMPT_DIR = Path(__file__).parent / "prompts"

def _load_prompt(filename: str) -> str:
    """Load a prompt from core/prompts/ — falls back to empty string on error."""
    path = _PROMPT_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"[ManimGen] Failed to load prompt file {filename}: {e}")
        return ""


def _split_narration_segments(narration: str, total_duration: float) -> list:
    """
    Split narration text into proportionally-timed segments.
    Each sentence becomes a segment with start/end timestamps.
    This gives Manim the cue to sync animations to actual speech.
    """
    import re as _re
    # Split on sentence boundaries
    sentences = _re.split(r'(?<=[.!?])\s+', narration.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [{"index": 1, "start": 0.0, "end": total_duration, "duration": total_duration,
                 "text": narration[:200]}]

    # Estimate duration proportional to word count
    word_counts = [len(s.split()) for s in sentences]
    total_words  = sum(word_counts) or 1
    segments = []
    t = 0.0
    for i, (sentence, words) in enumerate(zip(sentences, word_counts)):
        dur = round((words / total_words) * total_duration, 2)
        dur = max(dur, 1.0)   # floor of 1s per segment
        end = min(t + dur, total_duration)
        segments.append({
            "index":    i + 1,
            "start":    round(t, 2),
            "end":      round(end, 2),
            "duration": round(end - t, 2),
            "text":     sentence[:200],
        })
        t = end
    return segments


def _format_narration_segments(segments: list) -> str:
    """Format segments for the user prompt."""
    lines = []
    for seg in segments:
        lines.append(
            f"  Segment {seg['index']} ({seg['start']:.1f}s – {seg['end']:.1f}s, "
            f"{seg['duration']:.1f}s): \"{seg['text']}\""
        )
    return "\n".join(lines)


def _build_user_prompt(slide: dict, audio_duration: float) -> str:
    """Fill the manim_user_template.txt with narration-segment timing."""
    template  = _load_prompt("manim_user_template.txt")
    narration = slide.get("narration", "") or slide.get("content", "") or ""
    formula   = slide.get("formula", "") or ""
    key_pts   = slide.get("keyPoints") or []
    content   = slide.get("content", "") or slide.get("infographic", "") or ""

    segments  = _split_narration_segments(narration, audio_duration)
    seg_text  = _format_narration_segments(segments)

    # Provide first two segment durations for the skeleton in the template
    seg1_dur  = segments[0]["duration"] if len(segments) > 0 else audio_duration / 2
    seg2_dur  = segments[1]["duration"] if len(segments) > 1 else audio_duration / 2

    return template.format(
        title              = slide.get("title", ""),
        narration_segments = seg_text,
        formula            = formula,
        key_points         = ", ".join(key_pts) if key_pts else "None",
        content            = content,
        duration           = audio_duration,
        seg1_duration      = seg1_dur,
        seg2_duration      = seg2_dur,
    )


# ── Python AST syntax validator ─────────────────────────────────────────────────
def _validate_syntax(code: str) -> tuple[bool, str]:
    """
    Run ast.parse() on the generated code.
    Returns (True, "") on success or (False, error_message) on SyntaxError.
    Avoids wasting up to 5 minutes on a render that will definitely fail.
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg} — {e.text}"
    except Exception as e:
        return False, str(e)


def _basic_sanity_check(code: str) -> tuple[bool, str]:
    """
    Check for known forbidden patterns that cause runtime crashes.
    Returns (True, "") if clean, (False, reason) if suspicious.
    """
    # MathTex subscript indexing: formula[0], eq[2], etc.
    if re.search(r'\w+\[\d+\]', code):
        bad = re.findall(r'\w+\[\d+\]', code)
        return False, f"Forbidden MathTex/list indexing detected: {bad[:3]}"

    # get_part_by_tex — always returns None for complex LaTeX
    if "get_part_by_tex" in code:
        return False, "Forbidden: get_part_by_tex() detected — use Indicate() instead"

    # Class name must be SlideScene
    if "class SlideScene" not in code:
        return False, "Missing: class SlideScene(Scene) not found in generated code"

    return True, ""


# ── Timing enforcer ──────────────────────────────────────────────────────────────
def _parse_animation_duration(code: str) -> tuple[float, list]:
    """Parse total animation duration from Manim code."""
    wait_pattern = re.compile(r"self\.wait\(\s*([\d.]+)\s*\)")
    wait_locations = []
    total_wait = 0.0
    for match in wait_pattern.finditer(code):
        dur = float(match.group(1))
        wait_locations.append((match.start(), match.end(), dur))
        total_wait += dur

    runtime_pattern = re.compile(r"run_time\s*=\s*([\d.]+)")
    total_runtime = sum(float(m.group(1)) for m in runtime_pattern.finditer(code))

    return total_wait + total_runtime, wait_locations


def _append_final_wait(code: str, duration: float) -> str:
    """Append self.wait(X) before the last line in the construct() method."""
    wait_line = f"\n        self.wait({duration:.3f})\n"
    lines = code.splitlines(keepends=True)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() and not lines[i].strip().startswith("#"):
            lines.insert(i + 1, wait_line)
            break
    else:
        lines.append(wait_line)
    return "".join(lines)


def _enforce_timing(code: str, audio_duration: float) -> str:
    """
    Scale self.wait() calls in Manim code to match real audio duration.
    Tolerance: ±0.1s — don't patch if already close enough.
    """
    if audio_duration <= 0:
        logger.warning("[ManimGen] audio_duration=0 — skipping timing enforcement")
        return code

    total_duration, wait_locations = _parse_animation_duration(code)

    if total_duration <= 0:
        logger.info(f"[ManimGen] No timing found — appending self.wait({audio_duration:.3f})")
        return _append_final_wait(code, audio_duration)

    deficit = audio_duration - total_duration

    if abs(deficit) < 0.1:
        logger.info(f"[ManimGen] Duration {total_duration:.2f}s ≈ target {audio_duration:.2f}s ✓")
        return code

    if deficit > 0:
        if wait_locations:
            last_start, last_end, last_dur = wait_locations[-1]
            new_dur = last_dur + deficit
            patched = code[:last_start] + f"self.wait({new_dur:.3f})" + code[last_end:]
            logger.info(f"[ManimGen] Extended last wait {last_dur:.2f}s → {new_dur:.2f}s")
            return patched
        else:
            return _append_final_wait(code, deficit)
    else:
        total_wait = sum(dur for _, _, dur in wait_locations)
        if total_wait <= 0:
            logger.warning("[ManimGen] No waits to scale — animation exceeds target")
            return code
        runtime_total = total_duration - total_wait
        available_wait = max(0.1, audio_duration - runtime_total)
        scale = available_wait / total_wait
        patched = code
        for start, end, dur in reversed(wait_locations):
            new_dur = max(0.1, dur * scale)
            patched = patched[:start] + f"self.wait({new_dur:.3f})" + patched[end:]
        logger.info(f"[ManimGen] Scaled waits by {scale:.3f}x to fit {audio_duration:.2f}s")
        return patched


# ── Ollama code generation ───────────────────────────────────────────────────────
async def generate_manim_code(slide: dict, slide_index: int, audio_duration: float) -> str:
    """
    Call Ollama (qwen3-coder:latest) to generate complete Manim Python code.
    Uses prompt files from core/prompts/ for zero-restart prompt iteration.
    Returns the Python code string. Raises RuntimeError on failure after MAX_RETRIES.
    """
    system_prompt = _load_prompt("manim_system_prompt.txt")
    user_prompt   = _build_user_prompt(slide, audio_duration)
    last_err      = None
    last_code     = ""

    for attempt in range(1, MANIM_MAX_RETRIES + 1):
        try:
            logger.info(
                f"[ManimGen] Calling Ollama for slide {slide_index} "
                f"(attempt {attempt}/{MANIM_MAX_RETRIES}, duration={audio_duration:.2f}s)"
            )

            # On 2nd+ attempt, prepend repair context if previous code was generated
            prompt = user_prompt
            if attempt > 1 and last_code and last_err:
                repair_template = _load_prompt("manim_repair_prompt.txt")
                prompt = repair_template.format(
                    code=last_code[:3000],
                    error_log=str(last_err)[:1000],
                )

            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model":  OLLAMA_MODEL,
                        "system": system_prompt,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,   # low for code — near-deterministic
                            "num_predict": 4096,
                        },
                    },
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")

            code = _clean_python_code(raw)
            last_code = code

            # ── Validation gate 1: basic structure ──────────────────────────
            if "class SlideScene" not in code or "def construct" not in code:
                last_err = "Generated code missing SlideScene or construct()"
                logger.warning(f"[ManimGen] Attempt {attempt}: {last_err}")
                if attempt < MANIM_MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                continue

            # ── Validation gate 2: AST syntax check ─────────────────────────
            ok, syntax_err = _validate_syntax(code)
            if not ok:
                last_err = syntax_err
                logger.warning(f"[ManimGen] Attempt {attempt}: AST syntax error — {syntax_err}")
                if attempt < MANIM_MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                continue

            # ── Validation gate 3: forbidden pattern check ───────────────────
            ok, pattern_err = _basic_sanity_check(code)
            if not ok:
                last_err = pattern_err
                logger.warning(f"[ManimGen] Attempt {attempt}: Forbidden pattern — {pattern_err}")
                if attempt < MANIM_MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                continue

            logger.info(f"[ManimGen] ✓ Code validated for slide {slide_index} ({len(code)} chars)")
            return code

        except Exception as e:
            last_err = e
            logger.warning(f"[ManimGen] Attempt {attempt} exception for slide {slide_index}: {e}")
            if attempt < MANIM_MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(
        f"[ManimGen] Failed to generate valid Manim code for slide {slide_index} "
        f"after {MANIM_MAX_RETRIES} attempts. Last error: {last_err}"
    )


def _clean_python_code(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace from LLM output."""
    raw = raw.strip()
    raw = re.sub(r'^```python\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^```\s*', '',    raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '',    raw, flags=re.MULTILINE)
    return raw.strip()


# ── Manim CLI renderer ────────────────────────────────────────────────────────────
def render_manim_code(
    py_code: str,
    slide_index: int,
    output_dir: str,
) -> Optional[str]:
    """
    Write py_code to a temp .py file and render with `manim -qm`.
    Returns path to the rendered .mp4 file, or None on failure.
    -qm = 720p (medium quality) — better than -ql 480p for classrooms.
    """
    os.makedirs(output_dir, exist_ok=True)

    py_path = os.path.join(output_dir, f"slide_{slide_index}.py")
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(py_code)

    quality_flag = f"-q{MANIM_QUALITY}"   # l=480p, m=720p, h=1080p
    out_name     = f"slide_{slide_index}.mp4"
    cmd = [
        "manim", "render",
        quality_flag,
        "--fps", "30",              # explicit 30fps — more stable than default 60
        "--disable_caching",
        "-o", out_name,             # deterministic output filename
        "--media_dir", output_dir,
        py_path,
        "SlideScene",
    ]

    logger.info(f"[ManimGen] Rendering slide {slide_index}: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MANIM_RENDER_TIMEOUT,
            cwd=output_dir,
        )
        if result.returncode != 0:
            logger.error(
                f"[ManimGen] manim render failed (slide {slide_index}) "
                f"returncode={result.returncode}\n"
                f"STDERR: {result.stderr[-2000:]}"
            )
            # Write crash log for post-mortem debugging
            try:
                crash_log = Path(output_dir) / f"crash_slide_{slide_index}.log"
                crash_log.write_text(
                    f"EXIT CODE: {result.returncode}\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}\n"
                    f"CODE:\n{py_code if 'py_code' in dir() else '(not available)'}\n",
                    encoding="utf-8",
                )
                logger.info(f"[ManimGen] Crash log → {crash_log}")
            except Exception as _le:
                logger.debug(f"[ManimGen] Could not write crash log: {_le}")
            return None

        mp4_path = _find_rendered_mp4(output_dir, slide_index)
        if mp4_path:
            logger.info(f"[ManimGen] ✓ Rendered: {mp4_path}")
        else:
            logger.error(f"[ManimGen] Render succeeded but mp4 not found in {output_dir}")
        return mp4_path

    except subprocess.TimeoutExpired:
        logger.error(f"[ManimGen] manim render timed out after {MANIM_RENDER_TIMEOUT}s (slide {slide_index})")
        return None
    except Exception as e:
        logger.error(f"[ManimGen] render_manim_code failed for slide {slide_index}: {e}")
        return None


def _find_rendered_mp4(output_dir: str, slide_index: int) -> Optional[str]:
    """
    Deep-walk search for the rendered mp4.
    Priority:
      1. slide_{index}.mp4 anywhere under output_dir (from -o flag)
      2. SlideScene.mp4 anywhere under output_dir (legacy fallback)
      3. Any .mp4 directly in output_dir
    """
    named = f"slide_{slide_index}.mp4"

    # Priority 1: exact output name we requested with -o
    for root, _dirs, files in os.walk(output_dir):
        if named in files:
            return os.path.join(root, named)

    # Priority 2: legacy SlideScene.mp4 (old command format)
    for root, _dirs, files in os.walk(output_dir):
        if "SlideScene.mp4" in files:
            return os.path.join(root, "SlideScene.mp4")

    # Priority 3: any .mp4 directly in output_dir
    for fname in os.listdir(output_dir):
        if fname.endswith(".mp4"):
            return os.path.join(output_dir, fname)

    return None



# ── Full pipeline per slide ───────────────────────────────────────────────────────
async def generate_and_render_slide_manim(
    slide: dict,
    slide_index: int,
    cache_id: str,
    subject_id: str,
    audio_duration: float,
    b2_storage=None,
) -> Optional[dict]:
    """
    Full Manim pipeline for one slide:
      1. Generate Python code via Ollama (using prompt files)
      2. Validate syntax (AST) and forbidden patterns
      3. Enforce audio timing
      4. Render to MP4 with retry (2 render attempts)
      5. Save .py + .mp4 to local disk
      6. Upload .mp4 to B2 (if b2_storage provided)

    Returns dict:
      {local_py, local_mp4, b2_url, duration_seconds}
    or None on failure.
    """
    manim_dir = local_storage.get_manim_dir(subject_id, cache_id)

    # ── Step 1+2: Code gen + validation ──────────────────────────────────────
    try:
        py_code = await generate_manim_code(slide, slide_index, audio_duration)
    except RuntimeError as e:
        logger.error(f"[ManimGen] Code gen failed for slide {slide_index}: {e}")
        return None

    # ── Step 3: Timing enforcement ────────────────────────────────────────────
    py_code = _enforce_timing(py_code, audio_duration)

    # ── Step 4: Save .py to disk ──────────────────────────────────────────────
    local_py = await local_storage.write_manim_code(subject_id, cache_id, slide_index, py_code)

    # ── Step 5: Render with retry ─────────────────────────────────────────────
    local_mp4 = None
    for attempt in range(1, 3):  # up to 2 render attempts
        local_mp4 = await asyncio.get_event_loop().run_in_executor(
            None,
            render_manim_code,
            py_code,
            slide_index,
            manim_dir,
        )
        if local_mp4:
            break
        logger.warning(f"[ManimGen] Render attempt {attempt} failed for slide {slide_index}")
        if attempt < 2:
            await asyncio.sleep(3)

    if not local_mp4:
        logger.error(f"[ManimGen] All render attempts failed for slide {slide_index}")
        return None

    # ── Step 6: Copy to canonical path ────────────────────────────────────────
    canonical_mp4 = local_storage.get_manim_video_path(subject_id, cache_id, slide_index)
    if local_mp4 != canonical_mp4:
        os.makedirs(os.path.dirname(canonical_mp4), exist_ok=True)
        shutil.copy2(local_mp4, canonical_mp4)
        local_mp4 = canonical_mp4

    # ── Step 7: Upload to B2 ──────────────────────────────────────────────────
    b2_url = None
    if b2_storage is not None:
        try:
            b2_key = f"manim/{subject_id}/{cache_id}/slide_{slide_index}.mp4"
            with open(local_mp4, "rb") as f:
                mp4_bytes = f.read()
            b2_url = await b2_storage.upload_bytes(mp4_bytes, b2_key, content_type="video/mp4")
            logger.info(f"[ManimGen] ✓ B2 upload: {b2_url}")
        except Exception as e:
            logger.warning(f"[ManimGen] B2 upload failed for slide {slide_index}: {e}")

    return {
        "local_py":         local_py,
        "local_mp4":        local_mp4,
        "b2_url":           b2_url,
        "duration_seconds": audio_duration,
    }
