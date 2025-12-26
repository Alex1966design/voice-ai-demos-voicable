# backend/assistant/tts_client.py

"""
Модуль TTS-клиента для Алины.

Переиспользуем функцию tts_elevenlabs из elevenlabs_client.py.
"""

import base64
from elevenlabs_client import tts_elevenlabs


def synthesize_voice(text: str) -> bytes:
    """
    Обёртка над tts_elevenlabs.
    Предполагаем, что tts_elevenlabs возвращает байты аудио (mp3).
    """
    return tts_elevenlabs(text)


def text_to_speech_base64(text: str) -> str:
    """
    На вход: текст.
    На выход: base64-строка с аудио (которую фронт кладёт в audio.src).
    """
    audio_bytes = synthesize_voice(text)

    # На случай, если tts_elevenlabs вернёт строку:
    if isinstance(audio_bytes, str):
        audio_bytes = audio_bytes.encode("utf-8")

    return base64.b64encode(audio_bytes).decode("utf-8")
