# backend/alina_server.py
from __future__ import annotations

import uuid
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from assistant.alina import AlinaAssistant
from assistant.llm_client import CancelToken

app = FastAPI(
    title="Alina Voice Assistant",
    description="STT → LLM → TTS (RU / EN / TH)",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Assistants ===
assistant_ru = AlinaAssistant(mode="ru")
assistant_en = AlinaAssistant(mode="en")
assistant_th = AlinaAssistant(mode="th")

# session_id → cancel token (barge-in)
active_cancels: Dict[str, CancelToken] = {}


# =========================
# Healthcheck (Railway)
# =========================
@app.get("/health")
async def health():
    return {"status": "ok", "service": "alina"}


# =========================
# Cancel generation
# =========================
@app.post("/alina/cancel")
async def alina_cancel(session_id: str = Form(...)):
    tok = active_cancels.get(session_id)
    if tok:
        tok.cancel()
        return {"status": "cancelled", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# =========================
# Voice pipeline
# =========================
@app.post("/alina/voice")
async def alina_voice(
    audio: UploadFile = File(...),
    lang: str = Form("ru"),        # ru | en | th
    session_id: str = Form(""),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if not session_id:
        session_id = str(uuid.uuid4())

    if lang == "en":
        assistant = assistant_en
    elif lang == "th":
        assistant = assistant_th
    else:
        assistant = assistant_ru

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
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alina error: {e}")
    finally:
        active_cancels.pop(session_id, None)


# =========================
# UI (INDEX)
# =========================
@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <title>Alina – голосовой ассистент</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f5f5f7; margin:0; padding:20px; }
    h1 { margin-bottom:4px; }
    .subtitle { color:#777; margin-bottom:20px; }
    .card { background:#fff; border-radius:12px; padding:20px; box-shadow:0 2px 6px rgba(0,0,0,0.05); margin-bottom:20px; }
    .btn { padding:8px 16px; border-radius:8px; border:1px solid #ccc; cursor:pointer; background:#fff; font-size:14px; }
    .btn-primary { background:#1a73e8; color:#fff; border-color:#1a73e8; }
    .btn:disabled { opacity:0.5; cursor:default; }
    .status-ok { color:#1a7f37; font-size:14px; margin-left:8px; }
    .status-error { color:#d93025; font-size:14px; margin-left:8px; }
  </style>
</head>
<body>

<h1>Alina – голосовой ассистент</h1>
<div class="subtitle">Отдельный сервер: STT → LLM → TTS (RU / EN / TH)</div>

<div class="card">
  <h3>Шаг 1. Запиши или выбери аудиофайл</h3>

  <input type="file" id="audio-file" accept="audio/*" />
  <br><br>

  <button class="btn" id="btn-start">🎤 Начать запись</button>
  <button class="btn" id="btn-stop" disabled>⏹ Остановить запись</button>
  <span id="record-status"></span>

  <h3 style="margin-top:20px;">Шаг 2. Отправь запрос Алине</h3>

  <!-- === LANGUAGE SWITCH === -->
  <div style="margin-bottom:10px;">
    <label style="margin-right:10px;">
      <input type="radio" name="lang" value="ru" checked />
      🇷🇺 RU
    </label>

    <label style="margin-right:10px;">
      <input type="radio" name="lang" value="en" />
      🇬🇧 EN
    </label>

    <label>
      <input type="radio" name="lang" value="th" />
      🇹🇭 TH
    </label>
  </div>

  <button class="btn btn-primary" id="btn-send">Отправить Алине</button>
  <span id="send-status"></span>
</div>

<div class="card">
  <h3>Ответ Алины</h3>
  <audio id="reply-audio" controls style="width:100%;"></audio>
  <div id="reply-text" style="margin-top:10px;"></div>
</div>

<script>
let mediaRecorder = null;
let recordedChunks = [];
let sessionId = crypto.randomUUID();

const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnSend = document.getElementById("btn-send");
const recordStatus = document.getElementById("record-status");
const sendStatus = document.getElementById("send-status");
const replyAudio = document.getElementById("reply-audio");
const replyText = document.getElementById("reply-text");
const audioFileInput = document.getElementById("audio-file");

btnStart.onclick = async () => {
  recordedChunks = [];
  const stream = await navigator.mediaDevices.getUserMedia({ audio:true });
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);
  mediaRecorder.start();
  btnStart.disabled = true;
  btnStop.disabled = false;
  recordStatus.textContent = "Запись…";
};

btnStop.onclick = () => {
  mediaRecorder.stop();
  btnStart.disabled = false;
  btnStop.disabled = true;
  recordStatus.textContent = "Запись завершена";
};

btnSend.onclick = async () => {
  let audioBlob;
  let filename;

  if (recordedChunks.length > 0) {
    audioBlob = new Blob(recordedChunks, { type:"audio/webm" });
    filename = "recording.webm";
  } else {
    if (!audioFileInput.files[0]) {
      alert("Выберите или запишите аудио");
      return;
    }
    audioBlob = audioFileInput.files[0];
    filename = audioBlob.name;
  }

  const lang = document.querySelector('input[name="lang"]:checked').value;

  const fd = new FormData();
  fd.append("audio", audioBlob, filename);
  fd.append("lang", lang);
  fd.append("session_id", sessionId);

  sendStatus.textContent = "Отправка…";

  const resp = await fetch("/alina/voice", { method:"POST", body:fd });
  const data = await resp.json();

  if (data.audio_base64) {
    replyAudio.src = "data:audio/mpeg;base64," + data.audio_base64;
    replyAudio.load();
  }
  replyText.textContent = data.answer || "";
  sendStatus.textContent = "Готово ✔";
};
</script>

</body>
</html>
    """
    return HTMLResponse(content=html)


# =========================
# Local run
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.alina_server:app", host="0.0.0.0", port=8001, reload=True)
