from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")

# Таймауты: можно переопределить в .env
# OPENAI_TIMEOUT_SEC=30
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))

# Инициализируем клиента OpenAI один раз на модуль
client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SEC)


@dataclass
class CancelToken:
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return bool(self.cancelled)


def chat_with_alina(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: Optional[int] = 220,
) -> str:
    """
    Обычный (не-streaming) запрос.
    Для latency: держим max_tokens умеренным, temperature ниже.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI chat completion failed: {e}")

    content = response.choices[0].message.content
    return (content or "").strip()


def chat_with_alina_stream(
    messages: List[Dict[str, str]],
    cancel_token: Optional[CancelToken] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: Optional[int] = 220,
) -> str:
    """
    Streaming-генерация с поддержкой отмены (barge-in).
    Важно: мы всё равно возвращаем полный текст, но можем остановиться раньше.
    Это полезно, когда пользователь прерывает (cancel).
    """
    if cancel_token and cancel_token.is_cancelled():
        return ""

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI streaming completion failed: {e}")

    chunks: list[str] = []
    last_yield = time.time()

    for event in stream:
        if cancel_token and cancel_token.is_cancelled():
            break

        # OpenAI Python SDK: delta content приходит в choices[0].delta.content
        try:
            delta = event.choices[0].delta.content
        except Exception:
            delta = None

        if delta:
            chunks.append(delta)

        # мягкая “пульсация” (не обязательна), но снижает риск долгих блокировок
        # если поток странно себя ведёт
        if time.time() - last_yield > 15 and (cancel_token and cancel_token.is_cancelled()):
            break

    return "".join(chunks).strip()
