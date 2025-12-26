# backend/assistant/stt_client.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")

client = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """
    Batch STT (Whisper-1).
    NOTE: Для снижения latency на следующем этапе будем переходить к VAD/streaming,
          но сейчас фиксируем валидацию и явные ошибки.
    """
    if not audio_bytes:
        return ""

    try:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, audio_bytes),
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI STT failed: {e}")

    return (response.text or "").strip()
