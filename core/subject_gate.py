import httpx, os, json

async def gate_subject(question: str, subject_name: str) -> dict:
    """Use Gemini to verify question belongs to the given subject."""
    prompt = f"""You are a strict subject gatekeeper for an ed-tech app.
Allowed subject: {subject_name}
Student question: "{question}"

Reply with ONLY valid JSON (no markdown):
{{"allowed": true, "detected_subject": "Physics", "reason": "Question is about Newton's law"}}
or
{{"allowed": false, "detected_subject": "Mathematics", "reason": "Question is about algebra"}}"""

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                json={
                    "model": "google/gemini-2.5-flash",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        # On failure, allow the question (fail open)
        return {"allowed": True, "detected_subject": subject_name, "reason": f"gate_error: {e}"}


async def detect_topic(question: str, subject_name: str) -> dict:
    """Fast topic detection for instant UI feedback."""
    prompt = f"""Detect the topic of this student question about {subject_name}.
Question: "{question}"
Reply ONLY valid JSON:
{{"detected_topic": "Newton's Second Law", "description": "Relationship between force, mass, acceleration", "related_concepts": ["Force", "Mass", "Acceleration"], "confidence": 0.95}}"""

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                json={
                    "model": "google/gemini-2.5-flash",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        return {"detected_topic": question[:50], "description": "", "related_concepts": [], "confidence": 0.5}
