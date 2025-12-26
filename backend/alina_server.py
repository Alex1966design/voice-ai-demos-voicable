# backend/assistant/alina.py

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from assistant.stt_client import transcribe

# LLM: у тебя может быть либо chat_with_alina (non-stream),
# либо streaming-обёртка (если ты её добавлял позже).
try:
    from assistant.llm_client import chat_with_alina  # type: ignore
except Exception as e:
    raise RuntimeError(f"Failed to import chat_with_alina from assistant.llm_client: {e}")

# TTS: у тебя в проекте встречается elevenlabs_client.py (в assistant/)
# Если файл лежит именно в backend/assistant/elevenlabs_client.py — импорт как ниже корректный.
try:
    from assistant.elevenlabs_client import tts_elevenlabs  # type: ignore
except Exception as e:
    raise RuntimeError(f"Failed to import tts_elevenlabs from assistant.elevenlabs_client: {e}")


@dataclass
class AssistantConfig:
    mode: str = "ru"  # ru|en|th
    model: str = "gpt-4o-mini"
    temperature: float = 0.4
    max_history_messages: int = 12  # сколько последних реплик хранить (user+assistant)


def _system_prompt(mode: str) -> str:
    # Короткий, “дешёвый” промпт — меньше токенов = быстрее и дешевле.
    if mode == "en":
        return (
            "You are Alina, a concise, helpful voice assistant. "
            "Respond clearly, with short paragraphs and actionable steps. "
            "If the user asks about Thailand/Phuket, answer with practical local advice."
        )
    if mode == "th":
        # UI/режим тайский: отвечаем по-тайски, если возможно.
        return (
            "คุณคือ Alina ผู้ช่วยเสียงที่ตอบสั้น กระชับ และเป็นประโยชน์. "
            "ตอบเป็นภาษาไทย หากผู้ใช้พูดภาษาไทย. "
            "หากผู้ใช้ถามเรื่องประเทศไทย/ภูเก็ต ให้คำแนะนำที่ใช้ได้จริง."
        )
    # ru
    return (
        "Ты — Алина, голосовой ассистент. Отвечай кратко, по делу, структурировано. "
        "Если вопрос про Таиланд/Пхукет — давай практичные рекомендации."
    )


def _trim_history(history: List[Dict[str, str]], max_messages: int) -> List[Dict[str, str]]:
    """
    Оставляем последние max_messages сообщений истории (без system),
    чтобы не раздувать контекст (и latency).
    """
    if max_messages <= 0:
        return []
    return history[-max_messages:]


class AlinaAssistant:
    """
    Локальное состояние Алины (history) на уровне процесса.
    В проде лучше хранить историю по session_id, но пока оставляем простой вариант.
    """

    def __init__(self, mode: str = "ru", model: str = "gpt-4o-mini") -> None:
        self.cfg = AssistantConfig(mode=mode, model=model)
        self.history: List[Dict[str, str]] = []  # только user/assistant, без system

    def reset(self) -> None:
        self.history = []

    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": _system_prompt(self.cfg.mode)}]
        msgs.extend(_trim_history(self.history, self.cfg.max_history_messages))
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        cancel_token: Optional[Any] = None,
        use_llm_stream: bool = False,
    ) -> str:
        """
        Сейчас у тебя в репо “streaming/cancel” может быть, а может и нет.
        Поэтому:
        - если use_llm_stream=True и в llm_client есть streaming-функция — используем
        - иначе fallback на обычный chat_with_alina
        """
        if use_llm_stream:
            # Пытаемся найти streaming-обёртку в llm_client (если ты её добавлял)
            try:
                from assistant.llm_client import chat_with_alina_stream  # type: ignore

                return chat_with_alina_stream(
                    messages=messages,
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    cancel_token=cancel_token,
                ) or ""
            except Exception:
                # нет streaming реализации — используем обычную
                pass

        return chat_with_alina(
            messages=messages,
            model=self.cfg.model,
            temperature=self.cfg.temperature,
        ) or ""

    def handle_user_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        cancel_token: Optional[Any] = None,
        use_llm_stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Pipeline: STT → LLM → TTS
        Возвращает payload для фронта.
        """
        t0 = time.perf_counter()

        # 1) STT
        t_stt0 = time.perf_counter()
        transcript = transcribe(audio_bytes, filename=filename)
        stt_ms = int((time.perf_counter() - t_stt0) * 1000)

        transcript = (transcript or "").strip()
        if not transcript:
            # Не распознали — вернём пусто, но быстро
            total_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "transcript": "",
                "answer": "",
                "audio_base64": "",
                "audio_mime": "audio/mpeg",
                "history": self.history,
                "timings": {"stt_ms": stt_ms, "llm_ms": 0, "tts_ms": 0, "total_ms": total_ms},
            }

        # 2) LLM
        t_llm0 = time.perf_counter()
        messages = self._build_messages(transcript)
        answer = self._call_llm(messages, cancel_token=cancel_token, use_llm_stream=use_llm_stream).strip()
        llm_ms = int((time.perf_counter() - t_llm0) * 1000)

        # Обновляем историю (только user/assistant)
        self.history.append({"role": "user", "content": transcript})
        self.history.append({"role": "assistant", "content": answer})
        self.history = _trim_history(self.history, self.cfg.max_history_messages)

        # 3) TTS
        t_tts0 = time.perf_counter()
        audio_out: bytes = b""
        audio_mime = "audio/mpeg"

        # Если был cancel во время LLM — можно “срезать” TTS и вернуть пусто
        # (работает только если cancel_token имеет поле/метод, но мы не навязываем интерфейс)
        try:
            if cancel_token is not None:
                # поддержим варианты: cancel_token.cancelled / cancel_token.is_cancelled / cancel_token.value
                cancelled = bool(
                    getattr(cancel_token, "cancelled", False)
                    or getattr(cancel_token, "is_cancelled", False)
                    or getattr(cancel_token, "value", False)
                )
                if cancelled:
                    total_ms = int((time.perf_counter() - t0) * 1000)
                    return {
                        "transcript": transcript,
                        "answer": answer,
                        "audio_base64": "",
                        "audio_mime": audio_mime,
                        "history": self.history,
                        "timings": {"stt_ms": stt_ms, "llm_ms": llm_ms, "tts_ms": 0, "total_ms": total_ms},
                    }
        except Exception:
            pass

        if answer:
            audio_out = tts_elevenlabs(answer)
        tts_ms = int((time.perf_counter() - t_tts0) * 1000)

        audio_base64 = base64.b64encode(audio_out).decode("utf-8") if audio_out else ""
        total_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "transcript": transcript,
            "answer": answer,
            "audio_base64": audio_base64,
            "audio_mime": audio_mime,
            "history": self.history,
            "timings": {
                "stt_ms": stt_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": total_ms,
            },
        }
