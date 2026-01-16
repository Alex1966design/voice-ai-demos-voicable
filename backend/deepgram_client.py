# backend/assistant/deepgram_client.py
from __future__ import annotations

import os
from typing import Optional

import httpx


DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set in environment")


class DeepgramClient:
    """
    Async REST client for Deepgram STT.
    Works with raw audio bytes received from browser uploads (mp3/webm/wav/...).
    """

    def __init__(self) -> None:
        self.api_key = DEEPGRAM_API_KEY
        self.base_url = "https://api.deepgram.com/v1/listen"

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        mimetype: Optional[str] = None,
        lang: str = "th",
        model: str = "nova-3",
        smart_format: bool = True,
    ) -> str:
        if not audio_bytes:
            return ""

        headers = {
            "Authorization": f"Token {self.api_key}",
            # If we don't know mimetype - Deepgram still can handle it, but better pass if known.
            "Content-Type": (mimetype or "application/octet-stream"),
        }

        params = {
            "model": model,
            "smart_format": "true" if smart_format else "false",
        }

        # Deepgram language codes: th, en, ru, etc.
        # For Thai demo we force th.
        if lang:
            params["language"] = lang

        timeout = float(os.getenv("DEEPGRAM_TIMEOUT_SEC", "45"))

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(self.base_url, params=params, headers=headers, content=audio_bytes)

        if r.status_code >= 400:
            raise RuntimeError(f"Deepgram STT failed ({r.status_code}): {r.text[:800]}")

        data = r.json()
        try:
            transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        except Exception:
            raise RuntimeError(f"Deepgram response parse error: {str(data)[:800]}")

        return (transcript or "").strip()


deepgram_client = DeepgramClient()
