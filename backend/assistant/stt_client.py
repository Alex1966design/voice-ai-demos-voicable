# backend/assistant/stt_client.py
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()

# OpenAI STT client (Whisper)
if OPENAI_API_KEY:
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        timeout=float(os.getenv("OPENAI_TIMEOUT_SEC", "30")),
    )
else:
    openai_client = None

# Deepgram client (optional)
deepgram_client = None
if DEEPGRAM_API_KEY:
    try:
        from assistant.deepgram_client import deepgram_client as _dg  # type: ignore
        deepgram_client = _dg
    except Exception as e:
        # If Deepgram SDK isn't installed/compatible, you will see it in logs later.
        deepgram_client = None


async def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    lang: str = "th",
    content_type: Optional[str] = None,
) -> str:
    """
    STT router:
      - Thai (th) -> Deepgram (best for browser webm)
      - Otherwise -> OpenAI Whisper
    """
    if not audio_bytes:
        return ""

    # 1) TH priority -> Deepgram
    if lang == "th" and deepgram_client is not None:
        try:
            # Deepgram prefers correct mimetype for webm/ogg
            return (await deepgram_client.transcribe_bytes(audio_bytes, mimetype=content_type, lang="th")).strip()
        except Exception as e:
            # fall through to Whisper if configured
            print(f"Deepgram STT failed, fallback to Whisper: {e}")

    # 2) Whisper fallback
    if openai_client is None:
        raise RuntimeError("No STT provider available: set DEEPGRAM_API_KEY (recommended for TH) or OPENAI_API_KEY")

    try:
        resp = openai_client.audio.transcriptions.create(
            model=os.getenv("OPENAI_STT_MODEL", "whisper-1"),
            file=(filename, audio_bytes),
        )
        return (resp.text or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI STT failed: {e}")
