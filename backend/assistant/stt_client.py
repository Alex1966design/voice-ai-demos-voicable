from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# ENV
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")

OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))

# -------------------------
# OpenAI client
# -------------------------
client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=OPENAI_MAX_RETRIES,
    timeout=OPENAI_TIMEOUT_SEC,
)

# -------------------------
# Language mapping
# -------------------------
LANGUAGE_MAP = {
    "ru": "ru",
    "en": "en",
    "th": "th",
}

# -------------------------
# STT
# -------------------------
def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    content_type: str = "application/octet-stream",
    lang: str = "en",
) -> str:
    """
    Speech-to-Text using OpenAI Whisper.

    Supports:
      - webm / opus (browser mic)
      - mp3 / wav / m4a
      - Thai (th), English (en), Russian (ru)

    Args:
        audio_bytes: raw audio bytes
        filename: original filename (used by Whisper)
        content_type: MIME type (e.g. audio/webm;codecs=opus)
        lang: language hint ("th", "en", "ru")

    Returns:
        Transcribed text (string)
    """
    if not audio_bytes:
        return ""

    # Normalize language
    language = LANGUAGE_MAP.get(lang, None)

    try:
        # Whisper accepts (filename, bytes) tuple
        response = client.audio.transcriptions.create(
            model=OPENAI_STT_MODEL,
            file=(filename, audio_bytes),
            language=language,          # 👈 КЛЮЧЕВО ДЛЯ TH
            response_format="json",
        )

        text = getatt
