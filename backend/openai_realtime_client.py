from openai import OpenAI

from .config import settings


class OpenAIChatClient:
    """
    Пока используем обычный ChatCompletion.
    Позже сюда можно воткнуть OpenAI Realtime.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def chat(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""


openai_chat_client = OpenAIChatClient()
