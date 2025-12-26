# backend/assistant/stt_client.py

from __future__ import annotations
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env or environment")

# Оптимизированный клиент: timeout + retries
client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    timeout=float(os.getenv("OPENAI_TIMEOUT_SEC", "30")),
)


def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    STT (Whisper) — возвращает распознанный текст.
    timeout + retries помогают избежать зависаний.
    """
    if not audio_bytes:
        return ""

    try:
        response = client.audio.transcriptions.create(
            model=os.getenv("OPENAI_STT_MODEL", "whisper-1"),
            file=(filename, audio_bytes),
        )
        # Убедимся, что всегда возвращаем строку
        return (response.text or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI STT failed: {e}")
