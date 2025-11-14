"""Gemini API agent wrapper for generating content."""
import httpx
import logging
import os
from typing import Optional
from fastapi import HTTPException
from dotenv import load_dotenv
from time import sleep

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BACKOFF = float(os.getenv("GEMINI_RETRY_BACKOFF", "1.0"))
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "300"))


async def call_gemini(prompt: str) -> str:
    """Call Gemini API to generate content"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        attempt = 0
        while True:
            try:
                logger.info(f"Gemini request model={GEMINI_MODEL} prompt_len={len(prompt)} attempt={attempt}")
                response = await client.post(url, params=params, json=payload, timeout=GEMINI_TIMEOUT)
                logger.info(response)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    logger.warning(f"Gemini rate limited. Retry-After: {retry_after}")
                    raise HTTPException(status_code=429, detail="Rate limited by AI service", headers={"Retry-After": retry_after})

                if response.status_code == 404:
                    # Likely invalid model or wrong base URL version
                    raise HTTPException(status_code=404, detail=f"Gemini model not found or unavailable: {GEMINI_MODEL}")

                # Retry on transient 5xx
                if 500 <= response.status_code < 600 and attempt < GEMINI_MAX_RETRIES:
                    backoff = GEMINI_RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Gemini 5xx {response.status_code}, retrying in {backoff:.1f}s")
                    sleep(backoff)
                    attempt += 1
                    continue

                response.raise_for_status()
                data = response.json()
                # Extract text from candidates
                candidates = data.get("candidates", [])
                if not candidates:
                    raise HTTPException(status_code=500, detail="No candidates returned by Gemini")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
                logger.info(f"Gemini response chars={len(text)}")
                return text
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else 500
                if status == 429:
                    retry_after = e.response.headers.get("Retry-After", "60") if e.response else "60"
                    raise HTTPException(status_code=429, detail="Rate limited by AI service", headers={"Retry-After": retry_after})
                if status == 404:
                    raise HTTPException(status_code=404, detail=f"Gemini model not found or unavailable: {GEMINI_MODEL}")
                if 500 <= status < 600 and attempt < GEMINI_MAX_RETRIES:
                    backoff = GEMINI_RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Gemini HTTP error {status}, retrying in {backoff:.1f}s")
                    sleep(backoff)
                    attempt += 1
                    continue
                body_preview = ""
                try:
                    body_preview = (e.response.text or "")[:500]
                except Exception:
                    body_preview = ""
                logger.error(f"Gemini HTTP error: {e} body={body_preview}")
                detail = f"AI service error: {str(e)}"
                if body_preview:
                    detail += f" | body: {body_preview}"
                raise HTTPException(status_code=status, detail=detail)
            except httpx.RequestError as e:
                if attempt < GEMINI_MAX_RETRIES:
                    backoff = GEMINI_RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Gemini request error {e.__class__.__name__}, retrying in {backoff:.1f}s")
                    sleep(backoff)
                    attempt += 1
                    continue
                logger.error(f"Gemini request error: {repr(e)}")
                raise HTTPException(status_code=503, detail=f"AI service network error: {e.__class__.__name__}: {str(e)}")
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

