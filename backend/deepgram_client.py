# backend/assistant/deepgram_client.py
from __future__ import annotations

import os
from typing import Optional

from deepgram import DeepgramClient as SyncDeepgramClient  # deepgram-sdk v3+
from deepgram import PrerecordedOptions


class DeepgramClient:
    """
    Deepgram STT for small prerecorded audio.
    Accepts bytes from browser (often audio/webm).
    """

    def __init__(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        self.client = SyncDeepgramClient(api_key)

        self.model = os.getenv("DEEPGRAM_MODEL", "nova-3").strip() or "nova-3"

    async def transcribe_bytes(self, audio_bytes: bytes, mimetype: Optional[str] = None, lang: str = "th") -> str:
        """
        Uses Deepgram prerecorded endpoint.
        """
        if not audio_bytes:
            return ""

        # Deepgram language codes: "th" is valid.
        options = PrerecordedOptions(
            model=self.model,
            language=lang,
            smart_format=True,
            punctuate=True,
        )

        payload = {"buffer": audio_bytes}
        # mimetype important for webm/ogg
        if mimetype:
            payload["mimetype"] = mimetype

        # SDK provides asyncio interface via .asyncprerecorded
        resp = await self.client.asyncprerecorded.v("1").transcribe_file(payload, options)

        # Robust extraction
        try:
            return (
                resp["results"]["channels"][0]["alternatives"][0].get("transcript", "") or ""
            ).strip()
        except Exception:
            # last resort: stringify
            return str(resp).strip()


deepgram_client = DeepgramClient()
