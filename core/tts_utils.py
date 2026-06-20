"""
core/tts_utils.py
─────────────────
Shared text normalization utilities for ALL TTS engines (Sarvam + VoxCPM).

Called before text is sent to any TTS engine:
  - clean_narration()     — strips AI greeting filler ("Hello everyone!", etc.)
  - normalize_for_tts()   — fixes math, dash/hyphen, LaTeX, markdown
"""

import re

# ── Greeting patterns to strip from narration start ───────────────────────────
_GREETING_PATTERNS = [
    r"^Hello[,\s]+(everyone|students|class|friends|learners)[!,.]?\s*",
    r"^Welcome\s+to\s+(our|this|today'?s?)\s+[\w\s]+[!,.]\s*",
    r"^Good\s+(morning|afternoon|evening)[,\s]+[\w\s]+[!,.]\s*",
    r"^Namaste[,!.]\s*",
    r"^Alright[,!.]\s*(so\s+|everyone\s+)?",
    r"^Okay[,!.]\s*(so\s+|everyone\s+)?",
    r"^Let'?s\s+(begin|start|dive\s+in|get\s+started)[,!.]\s*",
    r"^Today[,\s]+we('re| are| will\s+be)\s+(going\s+to\s+)?",
    r"^In\s+this\s+(slide|lecture|lesson|mini.?lecture)[,\s]+we('ll| will)?\s+",
]

# Compiled once at module load for performance
_GREETING_RE = re.compile(
    "|".join(_GREETING_PATTERNS),
    re.IGNORECASE,
)

# ── Greek letter map ───────────────────────────────────────────────────────────
_GREEK = {
    r"\\alpha":   "alpha",
    r"\\beta":    "beta",
    r"\\gamma":   "gamma",
    r"\\delta":   "delta",
    r"\\epsilon": "epsilon",
    r"\\theta":   "theta",
    r"\\lambda":  "lambda",
    r"\\mu":      "mu",
    r"\\nu":      "nu",
    r"\\xi":      "xi",
    r"\\pi":      "pi",
    r"\\rho":     "rho",
    r"\\sigma":   "sigma",
    r"\\tau":     "tau",
    r"\\phi":     "phi",
    r"\\psi":     "psi",
    r"\\omega":   "omega",
}


# ── Public functions ───────────────────────────────────────────────────────────

def clean_narration(text: str) -> str:
    """
    Remove AI-generated greeting filler from the START of narration text.

    Examples:
      "Hello everyone! Welcome to our mini-lecture on..."
      → "Welcome to our mini-lecture on..."  (first pattern stripped)
      → "..."                                (second pattern stripped)

    Only strips from the beginning — preserves all mid-sentence content.
    """
    if not text:
        return text

    # Apply repeatedly until no more matches at start (handles chained greetings)
    for _ in range(5):
        stripped = _GREETING_RE.sub("", text, count=1).lstrip()
        if stripped == text:
            break
        text = stripped

    return text.strip()


def _expand_math(text: str) -> str:
    """
    Expand mathematical notation into spoken English words.

    MUST run BEFORE LaTeX delimiters are stripped — otherwise content inside
    $...$ is deleted and TTS hears nothing (e.g. "$a^2$" → "" → just silence).

    Covers:
      a^2           → "a squared"
      a^3           → "a cubed"
      a^n           → "a to the power n"
      a^{n+1}       → "a to the power n+1"
      (a+b)^2       → "(a plus b) squared"
      \\sqrt{x}     → "square root of x"
      \\frac{a}{b}  → "a over b"
      \\alpha, \\pi → "alpha", "pi"  (Greek letters)
      a+b           → "a plus b"  (letter+letter math context)
    """
    # ── Greek letters ──────────────────────────────────────────────────────────
    for pattern, word in _GREEK.items():
        text = re.sub(pattern, word, text)

    # ── Square root: \sqrt{expr} or √symbol ───────────────────────────────────
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'square root of \1', text)
    text = re.sub(r'√([a-zA-Z0-9]+)', r'square root of \1', text)

    # ── Fraction: \frac{a}{b} → "a over b" ────────────────────────────────────
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 over \2', text)

    # ── Powers: braced form first (highest specificity) ───────────────────────
    text = re.sub(r'\^\{2\}',       ' squared',           text)
    text = re.sub(r'\^\{3\}',       ' cubed',             text)
    text = re.sub(r'\^\{([^}]+)\}', r' to the power \1',  text)

    # ── Powers: bare form (^2, ^3, ^n) — after braced form ────────────────────
    text = re.sub(r'\^2\b', ' squared',                   text)
    text = re.sub(r'\^3\b', ' cubed',                     text)
    text = re.sub(r'\^([a-zA-Z0-9]+)', r' to the power \1', text)

    # ── Plus between letters/numbers (math context) → " plus " ────────────────
    text = re.sub(r'(?<=[a-zA-Z0-9])\+(?=[a-zA-Z0-9])', ' plus ', text)

    return text


def normalize_for_tts(text: str) -> str:
    """
    Normalize text for TTS so it sounds correct when spoken aloud.
    Applied to ALL TTS engines (Sarvam + VoxCPM local voice clone).

    Rules (in priority order):
    0. _expand_math()            — ^2→squared, ^n→power n, sqrt, frac, Greek
    1. Strip LaTeX delimiters    — $...$, $$...$$, \\(...\\)  (safe: content already expanded)
    2. Single-letter minus: a-b  → "a minus b"
    3. Number minus:   2-3       → "2 minus 3"
    4. Compound words: edge-case → "edge case"  (NO "minus")
    5. Strip markdown bold/italic (**)
    6. Strip backticks
    7. Collapse multiple spaces
    """
    if not text:
        return text

    # 0. Expand math FIRST — before any stripping removes content
    text = _expand_math(text)

    # 1. Strip LaTeX delimiters — content already expanded, safe to remove markers
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', '', text, flags=re.DOTALL)
    text = re.sub(r'\$([^$]+?)\$', r'\1', text)  # inline $...$ → keep inner text
    text = re.sub(r'\$', '', text)                # remove any stray $

    # 2. Single-letter math: a-b, x-y, A-B → "a minus b"
    text = re.sub(r'\b([a-zA-Z])\s*-\s*([a-zA-Z])\b', r'\1 minus \2', text)

    # 3. Number minus number: 10-5, 2-3 → "10 minus 5"
    text = re.sub(r'\b(\d+)\s*-\s*(\d+)\b', r'\1 minus \2', text)

    # 4. Compound English hyphenated words (2+ chars on each side) → space
    #    Runs AFTER rules 2 & 3 so single-letter math is already converted
    text = re.sub(r'(?<=[a-zA-Z]{2})-(?=[a-zA-Z]{2})', ' ', text)

    # 5. Strip markdown bold/italic asterisks
    text = re.sub(r'\*+', '', text)

    # 6. Strip backticks
    text = re.sub(r'`+', '', text)

    # 7. Collapse extra whitespace
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def prepare_for_tts(text: str) -> str:
    """
    Full pipeline: clean greetings → normalize (math expand + dash/LaTeX fix).
    Single call for both Sarvam and VoxCPM.
    """
    text = clean_narration(text)
    text = normalize_for_tts(text)
    return text
