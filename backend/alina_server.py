"""
Alina Voice Assistant (FastAPI) — DEMO MODE (TH / EN)

Pipeline:
  Browser Mic (webm/opus) or Audio File
    -> STT (Deepgram, language-aware)
    -> LLM (OpenAI)
    -> TTS (ElevenLabs)

Routes:
  - GET  /health
  - GET  /            -> HTML UI
  - POST /alina/voice -> Full voice pipeline
"""

from __future__ import annotations

import base64
import uuid
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# --- STT / LLM / TTS (fallback, stable) ---
from assistant.stt_client import transcribe
from assistant.llm_client import chat_with_alina
from assistant.elevenlabs_client import tts_elevenlabs


# --- Cancel token (simple, demo-safe) ---
class CancelToken:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


app = FastAPI(
    title="Alina Voice Assistant (Demo)",
    version="demo-th-en",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

active_cancels: Dict[str, CancelToken] = {}


# -------------------------
# Health
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "alina-demo"}


# -------------------------
# Cancel (barge-in)
# -------------------------
@app.post("/alina/cancel")
async def alina_cancel(session_id: str = Form(...)):
    tok = active_cancels.get(session_id)
    if tok:
        tok.cancel()
        return {"status": "cancelled", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# -------------------------
# Fallback pipeline (STABLE)
# -------------------------
def run_fallback_pipeline(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    lang: str,
    cancel_token: CancelToken,
) -> Dict[str, Any]:
    """
    STT -> LLM -> TTS
    """

    # 1) STT (Deepgram recommended)
    transcript = transcribe(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        lang=lang,
    )

    if cancel_token.cancelled:
        return {"transcript": transcript, "answer": "", "audio_base64": "", "audio_mime": "audio/mpeg"}

    # 2) LLM
    if lang == "th":
        system_prompt = (
            "You are Alina, a professional AI voice assistant for business demos in Thailand. "
            "Always reply in Thai language only. "
            "Keep answers short (1–2 sentences), polite and confident."
        )
    else:
        system_prompt = (
            "You are Alina, a professional AI voice assistant. "
            "Reply in English. Keep answers short and clear."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript or ""},
    ]

    answer = chat_with_alina(messages=messages)

    if cancel_token.cancelled:
        return {"transcript": transcript, "answer": answer, "audio_base64": "", "audio_mime": "audio/mpeg"}

    # 3) TTS (ElevenLabs, Thai supported)
    audio_mp3 = tts_elevenlabs(answer)
    audio_b64 = base64.b64encode(audio_mp3).decode("utf-8")

    return {
        "transcript": transcript,
        "answer": answer,
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg",
        "history": messages + [{"role": "assistant", "content": answer}],
    }


# -------------------------
# Main voice endpoint
# -------------------------
@app.post("/alina/voice")
async def alina_voice(
    audio: UploadFile = File(...),
    lang: str = Form("th"),          # th | en
    session_id: str = Form(""),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio")

    if not session_id:
        session_id = str(uuid.uuid4())

    cancel_token = CancelToken()
    active_cancels[session_id] = cancel_token

    try:
        filename = audio.filename or "audio.webm"
        content_type = audio.content_type or "application/octet-stream"

        result = run_fallback_pipeline(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
            lang=lang,
            cancel_token=cancel_token,
        )

        result["session_id"] = session_id
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alina error: {e}")

    finally:
        active_cancels.pop(session_id, None)


# -------------------------
# HTML UI (TH / EN only)
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8" />
<title>Alina – Thai Voice Assistant</title>
</head>
<body>
<h2>Alina – Thai Voice Assistant (Demo)</h2>

<input type="file" id="audio" accept="audio/*" /><br><br>

<label><input type="radio" name="lang" value="th" checked> 🇹🇭 TH</label>
<label><input type="radio" name="lang" value="en"> 🇬🇧 EN</label>
<br><br>

<button onclick="send()">Send to Alina</button>

<pre id="out"></pre>
<audio id="player" controls></audio>

<script>
async function send() {
  const f = document.getElementById("audio").files[0];
  if (!f) { alert("Choose audio file"); return; }

  const lang = document.querySelector('input[name="lang"]:checked').value;

  const fd = new FormData();
  fd.append("audio", f);
  fd.append("lang", lang);

  const r = await fetch("/alina/voice", { method: "POST", body: fd });
  const j = await r.json();

  document.getElementById("out").textContent =
    "Transcript:\\n" + j.transcript + "\\n\\nAnswer:\\n" + j.answer;

  if (j.audio_base64) {
    document.getElementById("player").src =
      "data:audio/mpeg;base64," + j.audio_base64;
  }
}
</script>
</body>
</html>
"""
    )
