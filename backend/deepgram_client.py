from __future__ import annotations

from typing import Optional

from deepgram import AsyncDeepgramClient
from .config import settings


class DeepgramClient:
    """
    Async Deepgram client for STT.
    Designed for browser microphone input (webm/opus) and audio files.
    """

    def __init__(self) -> None:
        if not settings.deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        # Explicit API key for reliability
        self.client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        mimetype: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe raw audio bytes via Deepgram.

        Args:
            audio_bytes: raw audio bytes from browser or file
            mimetype: e.g. "audio/webm", "audio/webm;codecs=opus", "audio/mpeg"
            language: "th", "en", etc.

        Returns:
            Transcribed text
        """
        if not audio_bytes:
            return ""

        # Deepgram parameters
        options = {
            "model": settings.deepgram_model or "nova-2",
            "smart_format": True,
        }

        # Explicit language hint (VERY IMPORTANT for Thai)
        if language:
            options["language"] = language

        # Deepgram SDK accepts bytes directly
        response = await self.client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            mimetype=mimetype,     # 👈 КЛЮЧЕВО для webm/opus
            **options,
        )

        try:
            transcript = (
                response.results
                .channels[0]
                .alternatives[0]
                .transcript
            )
        except Exception:
            # Safe fallback if Deepgram response shape changes
            transcript = ""

        return transcript.strip()


# Singleton instance
deepgram_client = DeepgramClient()
