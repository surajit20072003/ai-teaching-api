import httpx, os, base64, asyncio, re, subprocess, tempfile

# Sarvam TTS Growth plan: ~60 requests/min
# Each slide narration = ~6 chunks = 6 API calls
# Semaphore(20) = max 20 chunk calls at the same time
# Prevents "429 Too Many Requests" from Sarvam
SARVAM_TTS_SEMAPHORE = asyncio.Semaphore(20)

# Sarvam language code → speaker mapping
LANGUAGE_MAP = {
    "en-IN": {"lang": "hi-IN",  "speaker": "abhilash"},   # English narration → Hindi speaker
    "hi-IN": {"lang": "hi-IN",  "speaker": "abhilash"},
    "ta-IN": {"lang": "ta-IN",  "speaker": "abhilash"},
    "te-IN": {"lang": "te-IN",  "speaker": "abhilash"},
    "kn-IN": {"lang": "kn-IN",  "speaker": "abhilash"},
    "ml-IN": {"lang": "ml-IN",  "speaker": "abhilash"},
    "bn-IN": {"lang": "bn-IN",  "speaker": "abhilash"},
    "mr-IN": {"lang": "mr-IN",  "speaker": "abhilash"},
    "gu-IN": {"lang": "gu-IN",  "speaker": "abhilash"},
    "pa-IN": {"lang": "pa-IN",  "speaker": "abhilash"},
}


def _split_text(text: str, max_chars: int = 250) -> list[str]:
    """Split narration at sentence boundaries for Sarvam 250-char limit."""
    if len(text) <= max_chars:
        return [text]
    chunks, current = [], ""
    for sentence in re.split(r'(?<=[।॥?!.])\s+', text):
        if not sentence.strip():
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text[:250]]


def _merge_wav_sync(parts: list[bytes]) -> bytes:
    """Merge WAV byte chunks using ffmpeg (sync, runs in executor)."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, p in enumerate(parts):
            fp = os.path.join(tmp, f"part_{i}.wav")
            open(fp, "wb").write(p)
            paths.append(fp)
        list_file = os.path.join(tmp, "list.txt")
        open(list_file, "w").write("\n".join(f"file '{p}'" for p in paths))
        out = os.path.join(tmp, "merged.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out],
            check=True, capture_output=True
        )
        return open(out, "rb").read()


async def synthesize(text: str, language_code: str = "hi-IN", gender: str = "male") -> str:
    """
    Convert text to speech using Sarvam AI TTS (bulbul:v2).
    Returns base64-encoded WAV audio.
    Auto-chunks text at 250-char limit.
    """
    cfg = LANGUAGE_MAP.get(language_code, LANGUAGE_MAP["hi-IN"])
    chunks = _split_text(text)
    audio_parts = []

    async with httpx.AsyncClient(timeout=30) as client:
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            # ── Semaphore: max 20 Sarvam calls at a time ──
            async with SARVAM_TTS_SEMAPHORE:
                print(f"[TTS] Chunk {i+1}/{len(chunks)} → calling Sarvam (slot acquired)")
                resp = await client.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={
                        "API-Subscription-Key": os.getenv("SARVAM_API_KEY", ""),
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": [chunk],
                        "target_language_code": cfg["lang"],
                        "speaker": cfg["speaker"],
                        "pitch": 0,
                        "pace": 1.0,
                        "loudness": 1.5,
                        "speech_sample_rate": 24000,
                        "enable_preprocessing": True,
                        "model": "bulbul:v2"
                    }
                )
            # ── Slot released here automatically ──

            if resp.status_code == 200:
                audio_parts.append(base64.b64decode(resp.json()["audios"][0]))
            else:
                print(f"[TTS] Chunk {i} failed: {resp.status_code} {resp.text}")

    if not audio_parts:
        raise ValueError("Sarvam TTS returned no audio")

    if len(audio_parts) == 1:
        return base64.b64encode(audio_parts[0]).decode()

    # Merge multiple WAV chunks via ffmpeg
    loop = asyncio.get_event_loop()
    merged = await loop.run_in_executor(None, _merge_wav_sync, audio_parts)
    return base64.b64encode(merged).decode()
