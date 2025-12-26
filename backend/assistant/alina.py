# backend/assistant/alina.py

from __future__ import annotations

from typing import List, Dict, Any, Optional
import time

from . import stt_client, llm_client, tts_client


class AlinaAssistant:
    """
    Alina — premium voice assistant with modes:
    - ru: Russian persona
    - en: English persona
    - th: Thai persona

    Pipeline:
    1) STT -> user text
    2) LLM -> Alina's answer with dialogue memory
    3) TTS -> audio answer (base64) for the frontend player
    """

    def __init__(self, mode: str = "en", max_history_turns: int = 6) -> None:
        if mode not in ("ru", "en", "th"):
            raise ValueError("mode must be 'ru', 'en', or 'th'")
        self.mode = mode

        self.history: List[Dict[str, str]] = []
        self.max_history_turns = max_history_turns

    @property
    def system_prompt(self) -> str:
        if self.mode == "en":
            return (
                "You are Alina, a premium ENGLISH-SPEAKING voice sales assistant.\n\n"
                "LANGUAGE RULES:\n"
                "• You MUST ALWAYS answer in English only.\n"
                "• Even if the user speaks Russian or mixes languages, your reply must be fully in English.\n"
                "• Do NOT switch to Russian. Do NOT mix languages in one answer.\n\n"
                "ROLE:\n"
                "1) Identify the client's needs by asking clear, focused questions.\n"
                "2) Propose solutions, pricing options and next steps so that the client feels care and expertise.\n"
                "3) Explain complex things in a simple, human and friendly way.\n\n"
                "STYLE:\n"
                "• Tone: friendly, confident, professional.\n"
                "• Structure your answers: short summary first, then 3–5 bullet points with details.\n"
                "• Avoid bureaucratic language and clichés.\n"
                "• Be proactive: suggest the next step, ask clarifying questions.\n"
            )

        if self.mode == "th":
            return (
                "คุณคือ Alina ผู้ช่วยเสียงระดับพรีเมียมสำหรับการขายและให้คำปรึกษา\n\n"
                "กติกาภาษา:\n"
                "• ต้องตอบเป็นภาษาไทยเท่านั้น\n"
                "• ห้ามผสมภาษาในคำตอบเดียว (หลีกเลี่ยงอังกฤษ/รัสเซีย)\n\n"
                "บทบาท:\n"
                "1) ถามคำถามสั้น ๆ ชัดเจนเพื่อเข้าใจความต้องการ\n"
                "2) เสนอทางเลือก/ขั้นตอนถัดไปอย่างมืออาชีพ\n"
                "3) อธิบายเรื่องยากให้เข้าใจง่าย เป็นมิตร และเป็นมนุษย์\n\n"
                "สไตล์:\n"
                "• น้ำเสียง: เป็นมิตร มั่นใจ มืออาชีพ\n"
                "• โครงสร้าง: สรุปสั้น ๆ ก่อน แล้วตามด้วย 3–5 bullet points\n"
                "• กระตือรือร้น: เสนอ next step และถามคำถามต่อยอด\n"
            )

        # Russian persona (mode == "ru")
        return (
            "Ты — Алина, премиальный голосовой нейро-продавец и ассистент по продажам.\n\n"
            "ЯЗЫК:\n"
            "• Всегда отвечай по-русски.\n"
            "• Даже если пользователь говорит по-английски или смешивает языки, твой ответ должен быть полностью на русском.\n\n"
            "РОЛЬ:\n"
            "1) Выяснять потребности клиента, задавая уточняющие вопросы.\n"
            "2) Подбирать решения и варианты так, чтобы клиент чувствовал заботу и экспертность.\n"
            "3) Объяснять сложные вещи простым, живым, человеческим языком.\n\n"
            "СТИЛЬ:\n"
            "• Тон — доброжелательный, уверенный, профессиональный.\n"
            "• Структурируй ответы: краткий вывод, затем 3–5 пунктов с подробностями.\n"
            "• Избегай канцелярита и штампованных фраз.\n"
            "• Будь проактивной: предлагай следующий шаг, задавай уточняющие вопросы.\n"
        )

    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

        if self.history:
            trimmed = self.history[-2 * self.max_history_turns :]
            messages.extend(trimmed)

        messages.append({"role": "user", "content": user_text})
        return messages

    def _update_history(self, user_text: str, answer_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer_text})

        if len(self.history) > 2 * self.max_history_turns:
            self.history = self.history[-2 * self.max_history_turns :]

    def handle_user_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        cancel_token: Optional[llm_client.CancelToken] = None,
        use_llm_stream: bool = True,
    ) -> Dict[str, Any]:
        """
        Input: raw audio bytes.
        Output: dict with transcript, answer, audio_base64, audio_mime, history, timings.
        """

        t0 = time.perf_counter()

        # 1) STT
        user_text: str = stt_client.transcribe_audio(audio_bytes, filename)
        user_text = (user_text or "").strip()
        t_stt = time.perf_counter()

        if not user_text:
            if self.mode == "en":
                user_text = "It seems the audio was empty or could not be recognized properly."
            elif self.mode == "th":
                user_text = "ดูเหมือนว่าไฟล์เสียงว่างหรือระบบไม่สามารถถอดเสียงได้อย่างถูกต้อง"
            else:
                user_text = "Похоже, звук был пустой или плохо распознан."

        # 2) LLM
        messages = self._build_messages(user_text)

        if use_llm_stream:
            if cancel_token is None:
                cancel_token = llm_client.CancelToken(False)
            parts: list[str] = []
            for chunk in llm_client.chat_with_alina_stream(messages, cancel=cancel_token):
                parts.append(chunk)
            answer_text = "".join(parts)
        else:
            answer_text = llm_client.chat_with_alina(messages)

        answer_text = (answer_text or "").strip()
        t_llm = time.perf_counter()

        # 3) History
        self._update_history(user_text, answer_text)

        # 4) TTS (batch пока)
        audio_base64: str = tts_client.text_to_speech_base64(answer_text)
        t_tts = time.perf_counter()

        return {
            "transcript": user_text,
            "answer": answer_text,
            "audio_base64": audio_base64,
            "audio_mime": "audio/mpeg",
            "history": self.history,
            "timings": {
                "stt_ms": int((t_stt - t0) * 1000),
                "llm_ms": int((t_llm - t_stt) * 1000),
                "tts_ms": int((t_tts - t_llm) * 1000),
                "total_ms": int((t_tts - t0) * 1000),
            },
        }
