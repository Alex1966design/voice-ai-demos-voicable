from typing import List, Dict, Optional
import os

from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения из .env (если используешь)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")

# Инициализируем клиента OpenAI один раз на модуль
client = OpenAI(api_key=OPENAI_API_KEY)


def chat_with_alina(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Универсальная обёртка над OpenAI Chat Completions для Алины.

    messages:
      [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
      ]

    model:
      По умолчанию gpt-4o-mini — быстрый, недорогой, многоязычный.

    temperature:
      0.2–0.4 даёт достаточно стабильные, но не «деревянные» ответы.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        # Здесь можно залогировать e, если добавишь логгер
        raise RuntimeError(f"OpenAI chat completion failed: {e}")

    # Берём контент первой гипотезы
    content = response.choices[0].message.content
    return content or ""
