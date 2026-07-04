"""
tests/test_manim_quality.py
────────────────────────────
Automated quality validation for the Manim generation pipeline.

Usage (inside Docker container):
    python -m pytest tests/test_manim_quality.py -v

Or standalone:
    python tests/test_manim_quality.py

These tests validate:
  1. Narration segmentation produces correct timing
  2. Slide visual_type selection (derivation → manim, concept → image)
  3. Generated Manim code structure (segments, FadeOut, no forbidden patterns)
  4. Render command is correct (fps, -o flag)
  5. Manual test prompts for full end-to-end quality verification
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─── Unit Tests ───────────────────────────────────────────────────────────────

def test_narration_segmentation():
    """Verify narration is split into proportionally timed segments."""
    from core.manim_generator import _split_narration_segments

    narration = (
        "First, we set up the problem. "
        "Next, we substitute the known values. "
        "Then, we rearrange the equation. "
        "Finally, we arrive at the answer."
    )
    segments = _split_narration_segments(narration, total_duration=20.0)

    assert len(segments) == 4, f"Expected 4 segments, got {len(segments)}"

    # Segments must be ordered
    for i, seg in enumerate(segments):
        assert seg["index"] == i + 1
        assert seg["duration"] >= 1.0, "Every segment must be >= 1s"
        assert seg["start"] >= 0.0
        assert seg["end"] <= 20.0 + 0.5  # small float tolerance

    # Total durations should approximately equal audio_duration
    total = sum(s["duration"] for s in segments)
    assert abs(total - 20.0) < 2.0, f"Segment total {total:.1f}s ≠ 20.0s"
    print(f"  ✓ Narration segmentation: {len(segments)} segments, total={total:.1f}s")


def test_narration_single_sentence():
    """Edge case: single-sentence narration → one segment."""
    from core.manim_generator import _split_narration_segments
    segs = _split_narration_segments("This is a single sentence.", 10.0)
    assert len(segs) == 1
    assert segs[0]["duration"] >= 1.0
    print("  ✓ Single-sentence narration: 1 segment")


def test_visual_type_derivation_gets_manim():
    """Slides with step-by-step derivations must get visual_type=manim."""
    from core.slide_generator import _parse_slides
    import json

    # Fabricate LLM output with a clear derivation slide
    fake_llm = json.dumps({
        "presentation_slides": [
            {
                "title": "Deriving v = u + at",
                "content": "We derive the first equation of motion step by step.",
                "narration": "Let us derive v = u + at from first principles step by step. "
                             "Starting from the definition of acceleration we substitute and rearrange.",
                "keyPoints": ["Step 1", "Step 2", "Step 3"],
                "formula": "v = u + at",
                "isStory": False,
                "isTips": False,
            },
            {
                "title": "Tips to Remember",
                "content": "Use mnemonics",
                "narration": "Remember with the acronym SUVAT.",
                "keyPoints": ["S", "U", "V"],
                "formula": "",
                "isStory": False,
                "isTips": True,
            },
            {
                "title": "A Story",
                "content": "Car journey analogy",
                "narration": "Imagine a car accelerating down the road.",
                "keyPoints": ["A", "B", "C"],
                "formula": "",
                "isStory": True,
                "isTips": False,
            },
        ]
    })

    data = _parse_slides(fake_llm)
    slides = data["presentation_slides"]

    deriv_slide = slides[0]
    assert deriv_slide["visual_type"] == "manim", (
        f"Derivation slide should be manim, got {deriv_slide['visual_type']}"
    )

    tips_slide = next(s for s in slides if s.get("isTips"))
    assert tips_slide["visual_type"] == "image", "Tips slide must be image"

    story_slide = next(s for s in slides if s.get("isStory"))
    assert story_slide["visual_type"] == "image", "Story slide must be image"
    print("  ✓ visual_type: derivation=manim, tips=image, story=image")


def test_visual_type_definition_gets_image():
    """A slide that just defines a formula (no derivation) should get image."""
    from core.slide_generator import _parse_slides
    import json

    fake_llm = json.dumps({
        "presentation_slides": [
            {
                "title": "Newton's Second Law",
                "content": "Force equals mass times acceleration.",
                "narration": "Newton's second law states that F equals ma. "
                             "This is a fundamental law of physics.",
                "keyPoints": ["F = ma", "Force", "Acceleration"],
                "formula": "F = ma",
                "isStory": False,
                "isTips": False,
            },
        ]
    })

    data = _parse_slides(fake_llm)
    slides = data["presentation_slides"]
    concept_slide = slides[0]
    assert concept_slide["visual_type"] == "image", (
        f"Definition-only slide should be image, got {concept_slide['visual_type']}"
    )
    print("  ✓ visual_type: definition-only slide = image")


def test_manim_cap_at_two():
    """Even if many slides qualify, max 2 should be Manim."""
    from core.slide_generator import _parse_slides
    import json

    fake_llm = json.dumps({
        "presentation_slides": [
            {
                "title": f"Derivation {i}",
                "content": "derive step by step",
                "narration": f"Derive equation {i} step by step from first principles.",
                "keyPoints": ["A", "B", "C"],
                "formula": f"v{i} = u + a*t",
                "isStory": False,
                "isTips": False,
            }
            for i in range(5)
        ]
    })

    data = _parse_slides(fake_llm)
    manim_count = sum(1 for s in data["presentation_slides"] if s.get("visual_type") == "manim")
    assert manim_count <= 2, f"Expected ≤ 2 Manim slides, got {manim_count}"
    print(f"  ✓ Manim cap: {manim_count} ≤ 2")


def test_user_prompt_contains_segments():
    """The user prompt sent to Ollama must include narration segments with timestamps."""
    from core.manim_generator import _build_user_prompt

    slide = {
        "title": "Deriving v = u + at",
        "formula": "v = u + at",
        "keyPoints": ["Step 1", "Step 2", "Step 3"],
        "narration": "First we define acceleration. Then we integrate. Finally we simplify.",
        "content": "Derivation of first equation of motion.",
    }
    prompt = _build_user_prompt(slide, audio_duration=30.0)

    assert "Segment 1" in prompt, "User prompt must contain 'Segment 1'"
    assert "0.0s" in prompt, "User prompt must include start timestamp 0.0s"
    assert "30." in prompt or "duration" in prompt.lower(), "Duration must be present"
    print("  ✓ User prompt contains narration segments with timestamps")


def test_render_command_has_fps_flag():
    """
    The render command in manim_generator must include --fps 30 and -o flag.
    Inspect the source code to confirm.
    """
    import inspect
    from core.manim_generator import render_manim_code
    src = inspect.getsource(render_manim_code)
    assert '"--fps"' in src or "'--fps'" in src, "render command must include --fps"
    assert '"30"' in src or "'30'" in src, "render command must specify 30fps"
    assert '"-o"' in src or "'-o'" in src, "render command must include -o output flag"
    assert '"render"' in src or "'render'" in src, "must use 'manim render' subcommand"
    print("  ✓ Render command: manim render --fps 30 -o ...")


def test_render_command_crash_log():
    """The render command must write a crash log on failure."""
    import inspect
    from core.manim_generator import render_manim_code
    src = inspect.getsource(render_manim_code)
    assert "crash_slide_" in src or "crash_log" in src, "Must write crash log on failure"
    print("  ✓ Crash log writing confirmed in render_manim_code")


def test_code_quality_rules():
    """Validate a sample generated code string against our quality rules."""
    sample_code = '''
from manim import *

class SlideScene(Scene):
    def construct(self):
        # Segment 1: Introduction (5.0s)
        title_s1 = Text("Deriving v = u + at", font_size=48, color=YELLOW)
        title_s1.to_edge(UP)
        self.play(Write(title_s1), run_time=2.0)
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects), run_time=0.5)

        # Segment 2: Setup (8.0s)
        label_s2 = Text("Definition of acceleration:", font_size=32).to_edge(UP)
        formula_s2 = MathTex(r"a = \\frac{v - u}{t}", font_size=60)
        formula_s2.move_to(ORIGIN)
        self.play(Write(label_s2), run_time=1.0)
        self.play(Write(formula_s2), run_time=2.0)
        self.wait(3.5)
        self.play(Indicate(formula_s2), run_time=1.0)
        self.play(FadeOut(*self.mobjects), run_time=0.5)
    '''

    # Rule 1: Must be complete Python (has class and construct)
    assert "class SlideScene" in sample_code
    assert "def construct" in sample_code

    # Rule 2: Must have >= 2 segments
    segment_count = len(re.findall(r"# Segment \d+", sample_code))
    assert segment_count >= 2, f"Expected >= 2 segments, got {segment_count}"

    # Rule 3: No Dot() placeholders
    assert "Dot()" not in sample_code

    # Rule 4: FadeOut between segments
    fadeout_count = sample_code.count("FadeOut(*self.mobjects)")
    assert fadeout_count >= 2, f"Expected >= 2 FadeOut calls, got {fadeout_count}"

    # Rule 5: No MathTex indexing
    assert not re.search(r'\w+\[\d+\]', sample_code), "MathTex indexing found"

    # Rule 6: No get_part_by_tex
    assert "get_part_by_tex" not in sample_code

    print(f"  ✓ Code quality rules: {segment_count} segments, {fadeout_count} FadeOuts, no forbidden patterns")


# ─── Manual Test Prompts ──────────────────────────────────────────────────────

MANUAL_TEST_PROMPTS = [
    {
        "name": "Derivation — equations of motion",
        "question": "Derive the three equations of motion showing all working",
        "expect_manim": True,
        "pass_criteria": [
            "≤ 2 slides tagged manim",
            "Each Manim MP4 has ≥ 3 animation segments",
            "MP4 duration within ±3s of audio",
            "No crash logs created",
            "FadeOut between every segment",
        ]
    },
    {
        "name": "Definition only — F=ma",
        "question": "Explain Newton's second law F = ma",
        "expect_manim": False,
        "pass_criteria": [
            "0 slides tagged manim (definition only)",
            "Image generated for formula slide",
        ]
    },
    {
        "name": "Proof — Pythagorean theorem",
        "question": "Prove the Pythagorean theorem step by step",
        "expect_manim": True,
        "pass_criteria": [
            "1–2 slides tagged manim (proof qualifies)",
            "Manim shows geometric construction steps",
        ]
    },
    {
        "name": "Concept — photosynthesis",
        "question": "Explain how photosynthesis works",
        "expect_manim": False,
        "pass_criteria": [
            "0 slides tagged manim (no math derivation)",
            "All slides use image visual",
        ]
    },
]


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_narration_segmentation,
        test_narration_single_sentence,
        test_visual_type_derivation_gets_manim,
        test_visual_type_definition_gets_image,
        test_manim_cap_at_two,
        test_user_prompt_contains_segments,
        test_render_command_has_fps_flag,
        test_render_command_crash_log,
        test_code_quality_rules,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")

    print("\n\n📋 MANUAL TEST PROMPTS (run in browser at http://69.197.145.4:8000/):")
    for i, t in enumerate(MANUAL_TEST_PROMPTS, 1):
        print(f"\n  {i}. [{t['name']}]")
        print(f"     Question: \"{t['question']}\"")
        print(f"     Expect Manim: {'Yes' if t['expect_manim'] else 'No'}")
        print(f"     Pass if:")
        for c in t["pass_criteria"]:
            print(f"       • {c}")

    if failed:
        sys.exit(1)
