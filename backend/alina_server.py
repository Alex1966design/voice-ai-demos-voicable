# backend/alina_server.py
"""
Alina Voice Assistant (FastAPI)
STT (Deepgram async) -> LLM -> TTS (ElevenLabs)
RU / EN / TH
"""

from __future__ import annotations

import base64
import os
import uuid
import traceback
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

# ============================================================
# CONFIG
# ============================================================

DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "1") == "1"

# ============================================================
# Cancel token (safe)
# ============================================================

class CancelToken:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


active_cancels: Dict[str, CancelToken] = {}

# ============================================================
# Import pipeline modules (STRICT)
# ============================================================

try:
    from assistant.stt_client import transcribe  # async Deepgram
    from assistant.llm_client import chat_with_alina
    from assistant.elevenlabs_client import tts_elevenlabs
except Exception as e:
    raise RuntimeError(f"CRITICAL IMPORT ERROR: {e}")

# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Alina Voice Assistant",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Health
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok"}

# ============================================================
# Cancel
# ============================================================

@app.post("/alina/cancel")
async def cancel(session_id: str = Form(...)):
    tok = active_cancels.get(session_id)
    if tok:
        tok.cancel()
        return {"status": "cancelled"}
    return {"status": "not_found"}

# ============================================================
# Helpers
# ============================================================

def system_prompt(lang: str) -> str:
    if lang == "th":
        return "คุณคือ Alina ผู้ช่วยเสียง ตอบเป็นภาษาไทย แบบเป็นมิตร กระชับ และชัดเจน"
    if lang == "en":
        return "You are Alina, a helpful voice assistant. Reply in English."
    return "Ты — Алина, полезный голосовой ассистент. Отвечай по-русски."

# ============================================================
# MAIN VOICE ENDPOINT
# ============================================================

@app.post("/alina/voice")
async def alina_voice(
    audio: UploadFile = File(...),
    lang: str = Form("ru"),
    session_id: str = Form(""),
):
    if not audio:
        raise HTTPException(400, "No audio file")

    audio_bytes = await audio.read()_
