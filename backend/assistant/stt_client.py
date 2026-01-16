# backend/assistant/stt_client.py
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


async def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.webm",
    mimetype: Optional[str] = None,
    lang: str = "th",
) -> str:
    """
    STT router:
      - If DEEPGRAM_API_KEY exists -> use Deepgram (best for Thai demo, supports webm/mp3 well)
      - Else fallback to OpenAI Whisper (optional)
    """
    if not audio_bytes:
        return ""

    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if deepgram_key:
        from .deepgram_client import deepgram_client  # local import to avoid import-time crash

        model = os.getenv("DEEPGRAM_MODEL", "nova-3")
        return await deepgram_client.transcribe_bytes(
            audio_bytes=audio_bytes,
            mimetype=mimetype,
            lang=lang or "th",
            model=model,
            smart_format=True,
        )

    # Optional fallback: OpenAI Whisper (if you ever want it)
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise RuntimeError("No STT provider configured: set DEEPGRAM_API_KEY or OPENAI_API_KEY")

    from openai import OpenAI

    client = OpenAI(
        api_key=openai_key,
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        timeout=float(os.getenv("OPENAI_TIMEOUT_SEC", "30")),
    )

    try:
        resp = client.audio.transcriptions.create(
            model=os.getenv("OPENAI_STT_MODEL", "whisper-1"),
            file=(filename, audio_bytes),
            # language hint improves results for Thai (if Whisper is used)
            language=(lang or "th"),
        )
        return (resp.text or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI STT failed: {e}")
