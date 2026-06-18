import os
import json
import base64
import asyncio
import httpx
from .b2_client import upload_to_b2

# Semaphore limits concurrent image requests (set higher since OpenRouter allows more)
IMAGE_SEMAPHORE = asyncio.Semaphore(10)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

async def generate_one_image(slide: dict, cache_id: str, idx: int) -> str:
    """Generate one infographic image for a slide via OpenRouter/Gemini."""
    title = slide.get("title", "")
    info = slide.get("infographic", "educational diagram")
    
    prompt = f"Professional educational infographic about {title}: {info}. Clean vector art style, minimalist, no text, colorful, high quality."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-document-presentation.local",
        "X-Title": "AI Teaching API",
    }
    payload = {
        "model": "google/gemini-3.1-flash-image-preview",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with IMAGE_SEMAPHORE:
            print(f"[Image] Slide {idx} → calling OpenRouter (Gemini)")
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            
            choices = data.get("choices", [])
            if not choices:
                raise Exception(f"No choices returned: {data}")
            
            message = choices[0].get("message", {})
            image_b64 = None
            
            # 1. Check 'images' field
            images = message.get("images") or []
            if images:
                img = images[0]
                if isinstance(img, dict):
                    if img.get("type") == "image_url":
                        image_b64 = img.get("image_url", {}).get("url", "")
                    else:
                        image_b64 = img.get("url") or img.get("data") or img.get("b64_json") or ""
                elif isinstance(img, str):
                    image_b64 = img
            
            # 2. Check 'content' for data URIs
            if not image_b64:
                content = message.get("content")
                if isinstance(content, str) and content.startswith("data:image"):
                    image_b64 = content

            if not image_b64:
                print(f"[Image] Slide {idx} → Unexpected OpenRouter format: {data}")
                raise Exception("Could not find image in OpenRouter response")

            if image_b64.startswith("data:image"):
                image_b64 = image_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(image_b64)
            
        path = f"ai-presentations/{cache_id}/slide_{idx}.png"
        b2_url = await upload_to_b2(img_bytes, path, "image/png")
        print(f"[Image] Slide {idx} → {b2_url}")
        return b2_url

    except Exception as e:
        print(f"[Image] Error for slide {idx}: {e}")
        return ""


async def generate_all_images(slides: list, cache_id: str) -> list[str]:
    """Generate ALL slide images in PARALLEL using asyncio.gather."""
    tasks = [generate_one_image(s, cache_id, i) for i, s in enumerate(slides)]
    return await asyncio.gather(*tasks)
