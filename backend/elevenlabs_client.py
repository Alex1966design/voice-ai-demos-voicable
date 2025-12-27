from __future__ import annotations

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE")

# Таймауты: можно переопределить в .env
# ELEVEN_TIMEOUT_CONNECT=5
# ELEVEN_TIMEOUT_READ=45
ELEVEN_TIMEOUT_CONNECT = float(os.getenv("ELEVEN_TIMEOUT_CONNECT", "5"))
ELEVEN_TIMEOUT_READ = float(os.getenv("ELEVEN_TIMEOUT_READ", "45"))

if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY not found in .env")
if not ELEVENLABS_VOICE:
    raise RuntimeError("ELEVENLABS_VOICE not found in .env")


# Session = keep-alive + retries (очень помогает от случайных подвисаний/дропа)
_session = requests.Session()
_retries = Retry(
    total=2,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("POST",),
)
_adapter = HTTPAdapter(max_retries=_retries, pool_connections=10, pool_maxsize=10)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def tts_elevenlabs(text: str) -> bytes:
    """
    Генерирует речь (mp3) из текста через ElevenLabs и возвращает байты.

    Для latency:
    - обязательно timeout
    - Session keep-alive
    - ограничивай длину текста до разумной (это делаем в LLM max_tokens)
    """
    text = (text or "").strip()
    if not text:
        return b""

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
        },
    }

    resp = _session.post(
        url,
        headers=headers,
        json=payload,
        timeout=(ELEVEN_TIMEOUT_CONNECT, ELEVEN_TIMEOUT_READ),
    )
    resp.raise_for_status()
    return resp.content
