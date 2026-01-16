from typing import Optional
from deepgram import AsyncDeepgramClient
from .config import settings


class DeepgramClient:
    def __init__(self) -> None:
        if not settings.deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")

        self.client = AsyncDeepgramClient(
            api_key=settings.deepgram_api_key
        )

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        mimetype: Optional[str] = "audio/webm"
    ) -> str:
        """
        STT через Deepgram.
        Браузер -> MediaRecorder -> audio/webm (opus)
        """

        response = await self.client.listen.v1.media.transcribe_file(
            request=audio_bytes,

            # 🔥 ВАЖНО
            mimetype=mimetype,          # <-- ВОТ ЗДЕСЬ
            model=settings.deepgram_model or "nova-2",
            language="th",              # тайский
            smart_format=True,
        )

        transcript = (
            response.results.channels[0]
            .alternatives[0]
            .transcript
        )

        return transcript.strip()


deepgram_client = DeepgramClient()
