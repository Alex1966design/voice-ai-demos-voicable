from typing import Optional

from deepgram import AsyncDeepgramClient
from .config import settings


class DeepgramClient:
    """
    Реальный клиент Deepgram для расшифровки небольших аудио-файлов.
    Работает с байтами (audio_bytes), которые мы получаем из браузера.
    """

    def __init__(self) -> None:
        if not settings.deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")

        # Клиент сам возьмёт ключ из DEEPGRAM_API_KEY,
        # но мы передаём явно для надёжности
        self.client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)

    async def transcribe_bytes(self, audio_bytes: bytes, mimetype: Optional[str] = None) -> str:
        """
        Отправляет байты аудио в Deepgram и возвращает текстовую расшифровку.
        """
        # model и smart_format можно при желании настраивать через .env
        response = await self.client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=settings.deepgram_model or "nova-3",
            smart_format=True,
        )

        # Берём первую гипотезу из первого канала
        transcript = (
            response.results.channels[0]
            .alternatives[0]
            .transcript
        )

        return transcript.strip()


deepgram_client = DeepgramClient()
