# backend/assistant/stt_client.py
"""
STT client for Alina (Deepgram, async, production-safe)

Key features:
- Accepts `lang` argument (ru / en / th) WITHOUT breaking old callers
- Ignores unknown kwargs safely
- Uses Deepgram REST API (stable on Railway)
"""

from __future__ import annotations

import os
import mimetypes
import httpx
from typing import Optional, Any

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set")

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# Map UI language → Deepgram language
LANG_MAP = {
    "ru": "ru",
    "en": "en",
    "th": "th",
}


async def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    mimetype: Optional[str] = None,
    lang: Optional[str] = None,     # ✅ ВАЖНО: теперь принимаем lang
    **kwargs: Any,                  # ✅ глотаем всё лишнее
) -> str:
    """
    Transcribe audio via Deepgram.

    Args:
        audio_bytes: raw audio bytes
        filename: original filename
        mimetype: audio MIME type
        lang: "ru" | "en" | "th" (optional)
        **kwargs: ignored (for backward compatibility)

    Returns:
        transcript text (str)
    """

    if not audio_bytes:
        return ""

    # --- Detect mimetype ---
    if not mimetype:
        mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    # --- Language handling (SAFE) ---
    dg_language = None
    if lang:
        dg_language = LANG_MAP.get(lang)

    # --- Build Deepgram query params ---
    params = {
        "model": "nova-2",
        "punctuate": "true",
        "smart_format": "true",
    }

    if dg_language:
        params["language"] = dg_language

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": mimetype,
    }

    # --- Call Deepgram ---
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            DEEPGRAM_URL,
            params=params,
            headers=headers,
            content=audio_bytes,
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Deepgram error {resp.status_code}: {resp.text}"
        )

    data = resp.json()

    # --- Extract transcript safely ---
    try:
        transcript = (
            data["results"]["channels"][0]["alternatives"][0]["transcript"]
        )
    except Exception:
        raise RuntimeError(
            f"Failed to parse Deepgram response: {data}"
        )

    return transcript.strip()
