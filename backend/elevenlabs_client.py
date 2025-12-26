import os
import requests
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE")

if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY not found in .env")
if not ELEVENLABS_VOICE:
    raise RuntimeError("ELEVENLABS_VOICE not found in .env")


def tts_elevenlabs(text: str) -> bytes:
    """
    Генерирует речь (mp3) из текста через ElevenLabs и возвращает байты.
    """
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

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.content
