"""
core/manim_generator.py
────────────────────────
Generate Manim Python code via Ollama, enforce audio-sync timing per segment,
dry-run validate, render to MP4, and upload to B2.

Pipeline per slide:
  1. generate_manim_code()          — Ollama → complete Python file
  2. _check_completeness()          — truncation detection before render
  3. _enforce_timing_per_segment()  — per-segment hard sync (v2.6)
  4. _scrub_invalid_waits()         — remove self.wait(0) crashes
  5. _validate_runtime_dry()        — manim --dry_run fast check
  6. render_manim_code()            — subprocess: manim -qm slide_N.py SlideScene
  7. local save + B2 upload

Environment variables:
  OLLAMA_URL            — Ollama base URL (default: http://host.docker.internal:11434)
  OLLAMA_MODEL          — model for slides (default: qwen3-coder:latest)
  MANIM_OLLAMA_MODEL    — model for Manim code (default: devstral:24b)
  OLLAMA_TIMEOUT        — seconds (default: 900)
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
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL",  "qwen3-coder:latest")   # slides
MANIM_OLLAMA_MODEL   = os.getenv("MANIM_OLLAMA_MODEL", "devstral:24b")    # Manim code
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
    """Format segments in v2.6 two-line style for clearer LLM alignment."""
    lines = []
    for seg in segments:
        lines.append(
            f"  Segment {seg['index']} ({seg['start']:.1f}s - {seg['end']:.1f}s, "
            f"duration {seg['duration']:.1f}s):"
        )
        lines.append(f"    Narration: \"{seg['text']}\"")
    return "\n".join(lines)


def _build_user_prompt(slide: dict, audio_duration: float) -> str:
    """Fill the manim_user_template.txt with narration-segment timing."""
    template          = _load_prompt("manim_user_template.txt")
    narration         = slide.get("narration", "") or slide.get("content", "") or ""
    formula           = slide.get("formula", "") or ""
    key_pts           = slide.get("keyPoints") or []
    content           = slide.get("content", "") or slide.get("infographic", "") or ""
    visual_description = (
        slide.get("visual_description") or
        slide.get("manim_spec") or
        slide.get("infographic") or
        ""
    )

    segments = _split_narration_segments(narration, audio_duration)
    seg_text = _format_narration_segments(segments)

    seg1_dur = segments[0]["duration"] if len(segments) > 0 else audio_duration / 2
    seg2_dur = segments[1]["duration"] if len(segments) > 1 else audio_duration / 2

    return template.format(
        title              = slide.get("title", ""),
        narration_segments = seg_text,
        formula            = formula,
        key_points         = ", ".join(key_pts) if key_pts else "None",
        content            = content,
        visual_description = visual_description,
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

    # Class name: allow SlideScene or MainScene
    if "class SlideScene" not in code and "class MainScene" not in code:
        return False, "Missing: class SlideScene(Scene) or MainScene(Scene) not found"

    return True, ""


# ── Completeness check (truncation detection) ────────────────────────────────────
def _check_completeness(code: str) -> list[str]:
    """
    Detect truncated code BEFORE timing enforcement.
    Catches LLM outputs that were cut mid-statement.
    """
    errors = []
    lines = code.strip().split("\n")
    if not lines:
        return ["Empty code"]
    last = lines[-1].strip()
    # Bare identifier on last line (truncation artifact)
    if re.match(r'^[a-zA-Z_]\w*$', last):
        errors.append(f"Possible truncation — last line is bare identifier: '{last}'")
    # Line ending with binary operator
    if re.search(r'[\+\-\*\/\=\,\(]$', last):
        errors.append(f"Possible truncation — last line ends with operator: '{last}'")
    # compile() syntax test
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
    return errors


def _scrub_invalid_waits(code: str) -> str:
    """Remove self.wait(0) and self.wait(0.0) which crash Manim 0.19."""
    return re.sub(r"self\.wait\(\s*0(?:\.0+)?\s*\)", "", code)


# ── Per-segment hard sync timing enforcer (v2.6) ─────────────────────────────────
def _enforce_timing_per_segment(code: str, segments: list) -> str:
    """
    For each '# Segment N' block: measure actual animation time, then inject
    self.wait(deficit) + FadeOut to exactly match that segment's audio duration.
    Much more precise than global scaling.
    """
    if not segments or "# Segment" not in code:
        # Fallback: legacy global timing
        return _enforce_timing_global(code, sum(s.get("duration", 5.0) for s in segments))

    lines = code.split("\n")
    new_lines: list = []
    segment_lines: list = []
    current_seg_idx = -1
    indent = "        "  # 8 spaces — inside construct()

    def _sum_block_timing(block_lines: list) -> float:
        block_text = "\n".join(block_lines)
        waits = sum(float(m) for m in re.findall(r"self\.wait\(\s*([\d.]+)\s*\)", block_text))
        runs  = sum(float(m) for m in re.findall(r"run_time\s*=\s*([\d.]+)", block_text))
        return waits + runs

    def process_block(block_lines: list, seg_idx: int) -> list:
        if seg_idx < 0 or seg_idx >= len(segments):
            return block_lines
        target = float(segments[seg_idx].get("duration", 5.0))
        actual = _sum_block_timing(block_lines)
        block_text = "\n".join(block_lines)

        # Inject FadeOut if missing
        fadeout_dur = 0.5
        if "FadeOut(" not in block_text:
            block_lines.append(f"{indent}# V2.6: auto-injected cleanup")
            block_lines.append(f"{indent}self.play(FadeOut(*self.mobjects), run_time={fadeout_dur})")
            actual += fadeout_dur

        # Inject wait for remaining deficit
        deficit = target - actual
        if deficit > 0.05:
            block_lines.append(
                f"{indent}# HardSync: {actual:.2f}s → {target:.2f}s"
            )
            block_lines.append(f"{indent}self.wait({deficit:.3f})")
        elif deficit < -0.5:
            block_lines.append(
                f"{indent}# HardSync WARNING: animation {abs(deficit):.2f}s over target"
            )
        return block_lines

    for line in lines:
        seg_match = re.search(r"#\s*Segment\s*(\d+)", line, re.IGNORECASE)
        if seg_match:
            if current_seg_idx >= 0:
                new_lines.extend(process_block(segment_lines, current_seg_idx))
                segment_lines = []
            try:
                current_seg_idx = int(seg_match.group(1)) - 1  # 1-based → 0-based
            except Exception:
                current_seg_idx = -1

        if current_seg_idx >= 0:
            segment_lines.append(line)
        else:
            new_lines.append(line)

    # Process final segment
    if current_seg_idx >= 0 and segment_lines:
        new_lines.extend(process_block(segment_lines, current_seg_idx))

    return "\n".join(new_lines)


def _enforce_timing_global(code: str, audio_duration: float) -> str:
    """Legacy global timing scaler — fallback when no segment markers found."""
    if audio_duration <= 0:
        return code
    wait_pat = re.compile(r"self\.wait\(\s*([\d.]+)\s*\)")
    run_pat  = re.compile(r"run_time\s*=\s*([\d.]+)")
    waits    = [(m.start(), m.end(), float(m.group(1))) for m in wait_pat.finditer(code)]
    total    = sum(d for _, _, d in waits) + sum(float(m.group(1)) for m in run_pat.finditer(code))
    if total <= 0:
        return code + f"\n        self.wait({audio_duration:.3f})\n"
    deficit = audio_duration - total
    if abs(deficit) < 0.1:
        return code
    if deficit > 0 and waits:
        s, e, d = waits[-1]
        return code[:s] + f"self.wait({d + deficit:.3f})" + code[e:]
    total_wait = sum(d for _, _, d in waits)
    if total_wait <= 0:
        return code
    scale  = max(0.0, (total_wait + deficit)) / total_wait
    patched = code
    for s, e, d in reversed(waits):
        patched = patched[:s] + f"self.wait({max(0.1, d * scale):.3f})" + patched[e:]
    return patched


# ── Dry-run validator (fast, no video frames) ────────────────────────────────────
def _validate_runtime_dry(py_code: str, slide_index: int, tmp_dir: str) -> Optional[str]:
    """
    Run manim --dry_run before full render to catch runtime errors fast.
    Returns error string, or None if clean.
    """
    import tempfile
    tmp_py = os.path.join(tmp_dir, f"dryrun_{slide_index}.py")
    try:
        with open(tmp_py, "w", encoding="utf-8") as f:
            f.write(py_code)
        r = subprocess.run(
            ["manim", "-q", "l", "--dry_run", "--disable_caching", tmp_py, "SlideScene"],
            capture_output=True, text=True, timeout=30,
        )
        if os.path.exists(tmp_py):
            os.unlink(tmp_py)
        if r.returncode != 0:
            return f"DryRun RC={r.returncode}:\n{r.stderr[-800:]}"
        return None
    except subprocess.TimeoutExpired:
        return "DryRun timed out (30s)"
    except Exception as e:
        return f"DryRun error: {e}"


# ── Ollama code generation ───────────────────────────────────────────────────────
async def generate_manim_code(slide: dict, slide_index: int, audio_duration: float) -> str:
    """
    Call Ollama (MANIM_OLLAMA_MODEL, default devstral:24b) to generate Manim code.
    Returns the Python code string. Raises RuntimeError after MAX_RETRIES.
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
                        "model":  MANIM_OLLAMA_MODEL,   # dedicated Manim model
                        "system": system_prompt,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 6000,
                        },
                    },
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")

            code = _clean_python_code(raw)
            last_code = code

            # ── Gate 1: structure ────────────────────────────────────────────
            if "def construct" not in code:
                last_err = "Generated code missing construct()"
                logger.warning(f"[ManimGen] Attempt {attempt}: {last_err}")
                if attempt < MANIM_MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                continue

            # ── Gate 2: completeness (truncation) ───────────────────────────
            completeness_errs = _check_completeness(code)
            if completeness_errs:
                last_err = " | ".join(completeness_errs)
                logger.warning(f"[ManimGen] Attempt {attempt}: Truncated — {last_err}")
                if attempt < MANIM_MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                continue

            # ── Gate 3: forbidden patterns ───────────────────────────────────
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
    narration  = slide.get("narration", "") or slide.get("content", "") or ""
    segments   = _split_narration_segments(narration, audio_duration)

    # ── Step 1: Code gen + validation ────────────────────────────────────────
    try:
        py_code = await generate_manim_code(slide, slide_index, audio_duration)
    except RuntimeError as e:
        logger.error(f"[ManimGen] Code gen failed for slide {slide_index}: {e}")
        return None

    # ── Step 2: Scrub zero-waits (Manim 0.19 crash) ──────────────────────────
    py_code = _scrub_invalid_waits(py_code)

    # ── Step 3: Per-segment hard sync timing ─────────────────────────────────
    py_code = _enforce_timing_per_segment(py_code, segments)
    logger.info(f"[ManimGen] Per-segment sync applied ({len(segments)} segments, {audio_duration:.2f}s)")

    # ── Step 4: Dry-run validation (fast, no frames) ─────────────────────────
    dry_err = _validate_runtime_dry(py_code, slide_index, manim_dir)
    if dry_err:
        logger.warning(f"[ManimGen] DryRun failed slide {slide_index}: {dry_err[:300]}")
        # Write crash context for post-mortem, but don't abort — try render anyway
        try:
            Path(manim_dir).mkdir(parents=True, exist_ok=True)
            (Path(manim_dir) / f"dryrun_fail_{slide_index}.log").write_text(dry_err)
        except Exception:
            pass

    # ── Step 5: Save .py to disk ──────────────────────────────────────────────
    local_py = await local_storage.write_manim_code(subject_id, cache_id, slide_index, py_code)

    # ── Step 6: Render with retry ─────────────────────────────────────────────
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

    # ── Step 7: Copy to canonical path ────────────────────────────────────────
    canonical_mp4 = local_storage.get_manim_video_path(subject_id, cache_id, slide_index)
    if local_mp4 != canonical_mp4:
        os.makedirs(os.path.dirname(canonical_mp4), exist_ok=True)
        shutil.copy2(local_mp4, canonical_mp4)
        local_mp4 = canonical_mp4

    # ── Step 8: Upload to B2 ──────────────────────────────────────────────────
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
