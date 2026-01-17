"""
Alina Voice Assistant (FastAPI)
STT → LLM → TTS (RU / EN / TH)

Railway start command:
uvicorn backend.alina_server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import base64
import os
import uuid
import traceback
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# =========================
# Cancel token (safe)
# =========================
try:
    from assistant.llm_client import CancelToken  # type: ignore
except Exception:
    class CancelToken:
        def __init__(self, cancelled: bool = False):
            self.cancelled = cancelled

        def cancel(self):
            self.cancelled = True


# =========================
# Try full assistant
# =========================
assistant_import_error = None
try:
    from assistant.alina import AlinaAssistant  # type: ignore

    assistant_ru = AlinaAssistant(mode="ru")
    assistant_en = AlinaAssistant(mode="en")
    assistant_th = AlinaAssistant(mode="th")
except Exception as e:
    assistant_import_error = str(e)
    assistant_ru = assistant_en = assistant_th = None  # type: ignore

    # Fallback pipeline
    from assistant.stt_client import transcribe  # async, Deepgram
    from assistant.llm_client import chat_with_alina
    from assistant.elevenlabs_client import tts_elevenlabs


# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Alina Voice Assistant",
    version="1.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_cancels: Dict[str, CancelToken] = {}


# =========================
# Helpers
# =========================
def _pick_assistant(lang: str):
    if lang == "en":
        return assistant_en
    if lang == "th":
        return assistant_th
    return assistant_ru


def _system_prompt(lang: str) -> str:
    if lang == "th":
        return "คุณคือ Alina ผู้ช่วยเสียง ตอบเป็นภาษาไทย สุภาพ กระชับ"
    if lang == "en":
        return "You are Alina, a helpful voice assistant. Reply in English."
    return "Ты — Алина, полезный голосовой ассистент. Отвечай по-русски."


# =========================
# Health
# =========================
@app.get("/health")
async def health():
    return {"status": "ok"}


# =========================
# Cancel
# =========================
@app.post("/alina/cancel")
async def cancel(session_id: str = Form(...)):
    token = active_cancels.get(session_id)
    if token:
        token.cancel()
        return {"status": "cancelled"}
    return {"status": "not_found"}


# =========================
# Fallback pipeline
# =========================
async def fallback_pipeline(
    audio_bytes: bytes,
    filename: str,
    lang: str,
    cancel_token: CancelToken,
    mimetype: Optional[str],
) -> Dict[str, Any]:

    # --- STT (Deepgram) ---
    transcript = await transcribe(
        audio_bytes=audio_bytes,
        filename=filename,
        mimetype=mimetype,
    )

    if cancel_token.cancelled:
        return {"transcript": transcript, "answer": "", "audio_base64": ""}

    # --- LLM ---
    messages = [
        {"role": "system", "content": _system_prompt(lang)},
        {"role": "user", "content": transcript or ""},
    ]
    answer = chat_with_alina(messages)

    if cancel_token.cancelled:
        return {"transcript": transcript, "answer": answer, "audio_base64": ""}

    # --- TTS ---
    audio_mp3 = tts_elevenlabs(answer)
    audio_b64 = base64.b64encode(audio_mp3).decode("utf-8")

    return {
        "transcript": transcript,
        "answer": answer,
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg",
        "history": messages + [{"role": "assistant", "content": answer}],
    }


# =========================
# MAIN VOICE ENDPOINT
# =========================
@app.post("/alina/voice")
async def alina_voice(
    audio: UploadFile = File(...),
    lang: str = Form("ru"),
    session_id: str = Form(""),
):
    if not session_id:
        session_id = str(uuid.uuid4())

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio")

    cancel_token = CancelToken(False)
    active_cancels[session_id] = cancel_token

    try:
        filename = audio.filename or "audio.wav"
        mimetype = audio.content_type

        # ===== Primary assistant =====
        if assistant_ru is not None:
            assistant = _pick_assistant(lang)
            if assistant is None:
                raise RuntimeError("Assistant not initialised")

            result = assistant.handle_user_audio(
                audio_bytes,
                filename,
                cancel_token=cancel_token,
                use_llm_stream=False,
            )

        # ===== Fallback =====
        else:
            result = await fallback_pipeline(
                audio_bytes,
                filename,
                lang,
                cancel_token,
                mimetype,
            )
            result["assistant_import_error"] = assistant_import_error

        result["session_id"] = session_id
        return JSONResponse(result)

    except Exception as e:
        debug = os.getenv("DEBUG_ERRORS", "1") == "1"
        if debug:
            raise HTTPException(
                status_code=500,
                detail=f"{e}\n{traceback.format_exc()}",
            )
        raise HTTPException(500, "Internal error")

    finally:
        active_cancels.pop(session_id, None)


# =========================
# UI (unchanged)
# =========================
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("<h1>Alina Voice Assistant</h1>")
