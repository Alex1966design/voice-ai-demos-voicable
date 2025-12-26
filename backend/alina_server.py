# backend/alina_server.py

from __future__ import annotations

import uuid
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from assistant.alina import AlinaAssistant
from assistant.llm_client import CancelToken

app = FastAPI(
    title="Alina Voice Assistant",
    version="1.2.0",
    description="STT → LLM → TTS (RU / EN / TH)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Assistants ---
assistant_ru = AlinaAssistant(mode="ru")
assistant_en = AlinaAssistant(mode="en")
assistant_th = AlinaAssistant(mode="th")

active_cancels: Dict[str, CancelToken] = {}


# ---------- Health ----------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "alina"}


# ---------- Cancel ----------
@app.post("/alina/cancel")
async def alina_cancel(session_id: str = Form(...)):
    tok = active_cancels.get(session_id)
    if tok:
        tok.cancel()
        return {"status": "cancelled"}
    return {"status": "not_found"}


# ---------- Voice ----------
@app.post("/alina/voice")
async def alina_voice(
    audio: UploadFile = File(...),
    lang: str = Form("ru"),
    session_id: str = Form(""),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio")

    if not session_id:
        session_id = str(uuid.uuid4())

    assistant = {
        "ru": assistant_ru,
        "en": assistant_en,
        "th": assistant_th,
    }.get(lang, assistant_ru)

    cancel_token = CancelToken(False)
    active_cancels[session_id] = cancel_token

    try:
        result = assistant.handle_user_audio(
            audio_bytes,
            audio.filename or "audio.wav",
            cancel_token=cancel_token,
            use_llm_stream=True,
        )
        result["session_id"] = session_id
        return result
    finally:
        active_cancels.pop(session_id, None)


# ---------- UI ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Alina – Voice Assistant</title>
<style>
body { font-family: system-ui; background:#f5f6f7; padding:20px }
.card { background:#fff; padding:20px; border-radius:12px; margin-bottom:20px }
.btn { padding:8px 16px; border-radius:8px; border:1px solid #ccc; cursor:pointer }
.btn-primary { background:#2563eb; color:#fff; border:none }
</style>
</head>
<body>

<h1>Alina – голосовой ассистент</h1>
<p>STT → LLM → TTS (RU / EN / TH)</p>

<div class="card">
  <h3>Шаг 1</h3>
  <input type="file" id="audio" accept="audio/*" />
</div>

<div class="card">
  <h3>Шаг 2</h3>

  <label><input type="radio" name="lang" value="ru" checked> 🇷🇺 RU</label>
  <label><input type="radio" name="lang" value="en"> 🇬🇧 EN</label>
  <label><input type="radio" name="lang" value="th"> 🇹🇭 TH</label>

  <br><br>
  <button class="btn btn-primary" onclick="send()">Отправить Алине</button>
</div>

<div class="card">
  <h3>Ответ Алины</h3>
  <audio id="player" controls style="width:100%"></audio>
  <pre id="text"></pre>
</div>

<script>
async function send() {
  const file = document.getElementById("audio").files[0];
  if (!file) return alert("No audio");

  const lang = document.querySelector("input[name=lang]:checked").value;

  const fd = new FormData();
  fd.append("audio", file);
  fd.append("lang", lang);

  const r = await fetch("/alina/voice", { method:"POST", body:fd });
  const d = await r.json();

  document.getElementById("text").textContent = d.answer || "";
  if (d.audio_base64) {
    document.getElementById("player").src =
      "data:audio/mpeg;base64," + d.audio_base64;
  }
}
</script>

</body>
</html>
""")


# ---------- Local ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("alina_server:app", host="0.0.0.0", port=8000)
