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
        # но передаём явно для надёжности
        self.client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        """
        Приводит язык к формату, который ожидает Deepgram: "ru" / "en" / "th".
        Принимаем варианты: RU/ru/ru-RU, EN/en/en-US, TH/th/th-TH и т.п.
        """
        if not language:
            return None

        lang = language.strip().lower()

        # Частые форматы из UI/браузера: "ru-RU", "en-US", "th-TH"
        if lang.startswith("ru"):
            return "ru"
        if lang.startswith("en"):
            return "en"
        if lang.startswith("th"):
            return "th"

        # Иногда приходит "gb" (как флаг Великобритании)
        if lang == "gb":
            return "en"

        # Если пришло что-то неожиданное — не ломаемся, просто не задаём язык явно
        return None

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        mimetype: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Отправляет байты аудио в Deepgram и возвращает текстовую расшифровку.

        Важно: для стабильного распознавания RU/TH лучше явно передавать language.
        """
        dg_language = self._normalize_language(language)

        # Собираем аргументы запроса
        kwargs = dict(
            request=audio_bytes,
            model=settings.deepgram_model or "nova-3",
            smart_format=True,
        )

        # Явно задаём язык, если он понятен
        if dg_language:
            kwargs["language"] = dg_language

        # Deepgram SDK сам определяет формат по данным,
        # mimetype обычно не обязателен; оставляем параметр для совместимости.
        # При желании можно будет использовать mimetype для доп.настроек.

        response = await self.client.listen.v1.media.transcribe_file(**kwargs)

        # Берём первую гипотезу из первого канала
        transcript = (
            response.results.channels[0]
            .alternatives[0]
            .transcript
        )

        return (transcript or "").strip()


deepgram_client = DeepgramClient()
