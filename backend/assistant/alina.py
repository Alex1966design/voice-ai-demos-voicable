# backend/assistant/alina.py
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# --- Optional imports (под разные версии твоих файлов) ---

# STT
try:
    # ожидаем, что у тебя есть stt_client.py с функцией transcribe(...)
    from .stt_client import transcribe_audio as stt_transcribe  # type: ignore
except Exception:
    try:
        from .stt_client import transcribe as stt_transcribe  # type: ignore
    except Exception:
        stt_transcribe = None  # type: ignore

# LLM
try:
    from .llm_client import chat_with_alina  # твоя базовая функция (non-stream)
except Exception:
    chat_with_alina = None  # type: ignore

# CancelToken (для barge-in / отмены)
try:
    from .llm_client import CancelToken  # type: ignore
except Exception:
    @dataclass
    class CancelToken:
        cancelled: bool = False

        def cancel(self) -> None:
            self.cancelled = True

# Streaming LLM (если реализовано)
# (не обязательно — просто используем, если есть)
try:
    from .llm_client import chat_with_alina_stream  # type: ignore
except Exception:
    chat_with_alina_stream = None  # type: ignore

# TTS
try:
    from .elevenlabs_client import tts_elevenlabs  # type: ignore
except Exception:
    # иногда лежит как backend/elevenlabs_client.py (не в assistant/)
    try:
        from elevenlabs_client import tts_elevenlabs  # type: ignore
    except Exception:
        tts_elevenlabs = None  # type: ignore


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def _lang_norm(mode: str) -> str:
    mode = (mode or "ru").strip().lower()
    if mode in ("ru", "rus", "russian"):
        return "ru"
    if mode in ("en", "eng", "english"):
        return "en"
    if mode in ("th", "thai"):
        return "th"
    return "ru"


def _system_prompt(lang: str) -> str:
    # Можно тонко настроить стиль Алины под твой “нейропродажник”
    if lang == "en":
        return (
            "You are Alina, a concise, helpful voice assistant. "
            "Answer clearly, keep it short, avoid long disclaimers. "
            "If user asks for steps, give structured steps."
        )
    if lang == "th":
        return (
            "คุณคือ Alina ผู้ช่วยเสียงที่ตอบสั้น ชัดเจน และมีโครงสร้าง "
            "ตอบเป็นภาษาไทยเป็นหลัก ถ้าผู้ใช้ขอขั้นตอน ให้ตอบเป็นข้อ ๆ"
        )
    return (
        "Ты Алина — голосовой ассистент. "
        "Отвечай кратко, по делу, структурно. "
        "Если нужен план — дай шаги. Не пиши длинных дисклеймеров."
    )


class AlinaAssistant:
    """
    Алиса-подобный ассистент: STT -> LLM -> TTS
    Хранит историю в памяти (на процесс).
    """

    def __init__(self, mode: str = "ru", max_history_turns: int = 8):
        self.lang = _lang_norm(mode)
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, str]] = []
        self._reset_history()

    def _reset_history(self) -> None:
        self.history = [{"role": "system", "content": _system_prompt(self.lang)}]

    def _trim_history(self) -> None:
        # история вида: system + (user/assistant пары)
        if len(self.history) <= 1:
            return
        # оставляем system + последние N*2 сообщений
        keep = 1 + self.max_history_turns * 2
        if len(self.history) > keep:
            self.history = [self.history[0]] + self.history[-(keep - 1):]

    def handle_user_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        cancel_token: Optional[CancelToken] = None,
        use_llm_stream: bool = True,
    ) -> Dict[str, Any]:
        """
        Возвращает dict для FastAPI:
          {
            "transcript": "...",
            "answer": "...",
            "audio_base64": "...",
            "audio_mime": "audio/mpeg",
            "history": [...],
            "timings": {...}
          }
        """
        timings: Dict[str, float] = {}
        t0 = _now_ms()

        if cancel_token is None:
            cancel_token = CancelToken(False)

        # --- STT ---
        if stt_transcribe is None:
            raise RuntimeError("STT client is not available: stt_client.transcribe_audio/transcribe not found")

        t_stt0 = _now_ms()
        transcript = stt_transcribe(audio_bytes, filename=filename, lang=self.lang)  # type: ignore
        timings["stt_ms"] = _now_ms() - t_stt0

        if cancel_token.cancelled:
            # Пользователь перебил — не продолжаем
            timings["total_ms"] = _now_ms() - t0
            return {
                "transcript": transcript,
                "answer": "",
                "audio_base64": "",
                "audio_mime": "audio/mpeg",
                "history": self.history,
                "timings": timings,
                "cancelled": True,
            }

        # --- LLM ---
        if not transcript or not str(transcript).strip():
            raise RuntimeError("Empty transcript from STT")

        self.history.append({"role": "user", "content": str(transcript).strip()})
        self._trim_history()

        t_llm0 = _now_ms()

        # Если есть streaming-реализация и включено use_llm_stream — используем её
        if use_llm_stream and chat_with_alina_stream is not None:
            answer = self._llm_streaming(self.history, cancel_token=cancel_token)
        else:
            if chat_with_alina is None:
                raise RuntimeError("LLM client is not available: llm_client.chat_with_alina not found")
            answer = chat_with_alina(self.history)  # type: ignore

        timings["llm_ms"] = _now_ms() - t_llm0

        if cancel_token.cancelled:
            timings["total_ms"] = _now_ms() - t0
            return {
                "transcript": transcript,
                "answer": "",
                "audio_base64": "",
                "audio_mime": "audio/mpeg",
                "history": self.history,
                "timings": timings,
                "cancelled": True,
            }

        answer = (answer or "").strip()
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()

        # --- TTS ---
        if tts_elevenlabs is None:
            raise RuntimeError("TTS client is not available: elevenlabs_client.tts_elevenlabs not found")

        t_tts0 = _now_ms()
        audio_mp3: bytes = tts_elevenlabs(answer)  # type: ignore
        timings["tts_ms"] = _now_ms() - t_tts0

        timings["total_ms"] = _now_ms() - t0

        return {
            "transcript": transcript,
            "answer": answer,
            "audio_base64": _b64(audio_mp3),
            "audio_mime": "audio/mpeg",
            "history": self.history,
            "timings": timings,
        }

    def _llm_streaming(
        self,
        messages: List[Dict[str, str]],
        cancel_token: CancelToken,
    ) -> str:
        """
        Стриминговая сборка ответа (если в llm_client есть chat_with_alina_stream).
        Ожидаем, что chat_with_alina_stream(messages, cancel_token=...) -> iterator[str] или list[str]
        """
        if chat_with_alina_stream is None:
            # fallback
            if chat_with_alina is None:
                raise RuntimeError("No LLM functions available")
            return chat_with_alina(messages)  # type: ignore

        chunks: List[str] = []
        try:
            stream = chat_with_alina_stream(messages, cancel_token=cancel_token)  # type: ignore
            for part in stream:
                if cancel_token.cancelled:
                    break
                if part:
                    chunks.append(str(part))
        except TypeError:
            # если сигнатура без cancel_token
            stream = chat_with_alina_stream(messages)  # type: ignore
            for part in stream:
                if cancel_token.cancelled:
                    break
                if part:
                    chunks.append(str(part))

        return "".join(chunks)
