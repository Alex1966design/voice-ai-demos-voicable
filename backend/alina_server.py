# backend/alina_server.py

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import uuid
from typing import Dict

from assistant.alina import AlinaAssistant
from assistant.llm_client import CancelToken

app = FastAPI(
    title="Alina Voice Assistant",
    description="Отдельный сервер Алины: STT → LLM → TTS",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant_ru = AlinaAssistant(mode="ru")
assistant_en = AlinaAssistant(mode="en")
assistant_th = AlinaAssistant(mode="th")

# Активные токены отмены по session_id (для barge-in)
active_cancels: Dict[str, CancelToken] = {}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "alina"}


@app.post("/alina/cancel")
async def alina_cancel(session_id: str = Form(...)):
    tok = active_cancels.get(session_id)
    if tok:
        tok.cancel()
        return {"status": "cancelled", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@app.post("/alina/voice")
async def alina_voice(
    audio: UploadFile = File(...),
    lang: str = Form("ru"),        # "ru"|"en"|"th"
    session_id: str = Form(""),    # приходит с фронта
):
    """
    Полный голосовой цикл Алины (RU/EN/TH):
    1) STT → текст пользователя
    2) LLM → ответ Алины с учётом истории (LLM streaming внутри, для cancel)
    3) TTS → озвучка ответа (base64)
    """

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

    # Создаём cancel token на этот запрос
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alina error: {e}")
    finally:
        # На данном этапе удаляем токен после завершения запроса.
        # (Далее можно будет держать активный токен только пока SPEAKING/THINKING)
        active_cancels.pop(session_id, None)

    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <title>Alina – голосовой ассистент</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f7; margin: 0; padding: 20px; }
    h1 { margin-bottom: 4px; }
    .subtitle { color: #777; margin-bottom: 20px; }
    .card { background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .btn { padding: 8px 16px; border-radius: 8px; border: 1px solid #ccc; cursor: pointer; background: #fff; font-size: 14px; }
    .btn-primary { background: #1a73e8; color: #fff; border-color: #1a73e8; }
    .btn-primary:disabled, .btn:disabled { opacity: 0.5; cursor: default; }
    .status-ok { color: #1a7f37; font-size: 14px; margin-left: 8px; }
    .status-error { color: #d93025; font-size: 14px; margin-left: 8px; }
    #reply-chat div.bubble { margin-bottom: 10px; }
    .bubble-header { font-size: 13px; color: #666; margin-bottom: 2px; }
    .bubble-user { display: inline-block; background: #e8f0fe; border-radius: 12px; padding: 8px 12px; max-width: 100%; }
    .bubble-alina { display: inline-block; background: #f1f3f4; border-radius: 12px; padding: 8px 12px; max-width: 100%; }
    pre { background: #f6f6f6; border-radius: 8px; padding: 10px; font-size: 12px; overflow-x: auto; }
    .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .pill { font-size:12px; background:#f1f3f4; padding:6px 10px; border-radius:999px; color:#333; }
  </style>
</head>
<body>
  <h1 id="ui-title">Alina – голосовой ассистент</h1>
  <div class="subtitle" id="ui-subtitle">Отдельный сервер: STT → LLM → TTS (RU / EN / TH)</div>

  <div class="card">
    <div class="row" style="justify-content:space-between;">
      <h3 id="ui-step1" style="margin:0;">Шаг 1. Запиши или выбери аудиофайл</h3>
      <span class="pill" id="ui-session">session: —</span>
    </div>

    <div style="margin: 12px 0 10px;">
      <input type="file" id="audio-file" accept="audio/*" />
      <span id="ui-hint" style="font-size: 12px; color:#777; margin-left:8px;">
        Можно выбрать готовый аудиофайл или записать голос с микрофона прямо в браузере.
      </span>
    </div>

    <div style="margin-bottom: 10px;">
      <button class="btn" id="btn-start">🎤 Начать запись</button>
      <button class="btn" id="btn-stop" disabled>⏹ Остановить запись</button>
      <span id="record-status" style="margin-left: 8px; font-size: 14px; color: #555;"></span>
    </div>

    <h3 id="ui-step2">Шаг 2. Отправь запрос Алине</h3>

    <div style="margin-bottom: 10px;">
      <label style="margin-right: 10px;">
        <input type="radio" name="lang" value="ru" checked />
        🇷🇺 RU
      </label>
      <label style="margin-right: 10px;">
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

    <div style="margin-top: 12px;">
      <pre id="ui-timings" style="display:none;"></pre>
    </div>
  </div>

  <div class="card">
    <h3 id="ui-answer">Ответ Алины</h3>
    <audio id="reply-audio" controls style="width: 100%; margin-bottom: 10px;"></audio>

    <div id="reply-chat" style="margin-bottom: 12px;"></div>

    <pre id="reply-history" style="display:none;"></pre>
  </div>

  <script>
    // --- i18n ---
    const I18N = {
      ru: {
        title: "Alina – голосовой ассистент",
        subtitle: "Отдельный сервер: STT → LLM → TTS (RU / EN / TH)",
        step1: "Шаг 1. Запиши или выбери аудиофайл",
        hint: "Можно выбрать готовый аудиофайл или записать голос с микрофона прямо в браузере.",
        step2: "Шаг 2. Отправь запрос Алине",
        send: "Отправить Алине",
        answer: "Ответ Алины",
        rec: "Запись идёт…",
        recDone: "Запись завершена. Теперь можно отправить Алине.",
        micErr: "Не удалось получить доступ к микрофону.",
        sending: "Отправка…",
        done: "Готово ✔",
        err: "Ошибка ✖",
      },
      en: {
        title: "Alina – voice assistant",
        subtitle: "Standalone server: STT → LLM → TTS (RU / EN / TH)",
        step1: "Step 1. Record or choose an audio file",
        hint: "You can select an audio file or record from the microphone directly in the browser.",
        step2: "Step 2. Send a request to Alina",
        send: "Send to Alina",
        answer: "Alina's reply",
        rec: "Recording…",
        recDone: "Recording finished. You can now send it to Alina.",
        micErr: "Microphone access error.",
        sending: "Sending…",
        done: "Done ✔",
        err: "Error ✖",
      },
      th: {
        title: "Alina – ผู้ช่วยเสียง",
        subtitle: "เซิร์ฟเวอร์เดี่ยว: STT → LLM → TTS (RU / EN / TH)",
        step1: "ขั้นตอนที่ 1 บันทึกเสียงหรือเลือกไฟล์เสียง",
        hint: "คุณสามารถเลือกไฟล์เสียง หรือบันทึกเสียงจากไมโครโฟนในเบราว์เซอร์ได้",
        step2: "ขั้นตอนที่ 2 ส่งคำถามให้ Alina",
        send: "ส่งให้ Alina",
        answer: "คำตอบของ Alina",
        rec: "กำลังบันทึก…",
        recDone: "บันทึกเสร็จแล้ว พร้อมส่งให้ Alina",
        micErr: "ไม่สามารถเข้าถึงไมโครโฟนได้",
        sending: "กำลังส่ง…",
        done: "เสร็จสิ้น ✔",
        err: "เกิดข้อผิดพลาด ✖",
      }
    };

    function getUILang() {
      // UI язык берём из выбранного режима
      return document.querySelector('input[name="lang"]:checked').value || "ru";
    }

    function applyUI(lang) {
      const t = I18N[lang] || I18N.ru;
      document.getElementById("ui-title").textContent = t.title;
      document.getElementById("ui-subtitle").textContent = t.subtitle;
      document.getElementById("ui-step1").textContent = t.step1;
      document.getElementById("ui-hint").textContent = t.hint;
      document.getElementById("ui-step2").textContent = t.step2;
      document.getElementById("btn-send").textContent = t.send;
      document.getElementById("ui-answer").textContent = t.answer;
    }

    // --- State ---
    let mediaRecorder = null;
    let recordedChunks = [];
    let sessionId = (crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now());

    const btnStart = document.getElementById("btn-start");
    const btnStop = document.getElementById("btn-stop");
    const recordStatus = document.getElementById("record-status");
    const btnSend = document.getElementById("btn-send");
    const sendStatus = document.getElementById("send-status");
    const audioFileInput = document.getElementById("audio-file");

    const replyAudio = document.getElementById("reply-audio");
    const replyChat = document.getElementById("reply-chat");
    const replyHistory = document.getElementById("reply-history");
    const uiTimings = document.getElementById("ui-timings");
    const uiSession = document.getElementById("ui-session");

    uiSession.textContent = "session: " + sessionId;
    applyUI(getUILang());

    document.querySelectorAll('input[name="lang"]').forEach(r => {
      r.addEventListener("change", () => applyUI(getUILang()));
    });

    async function cancelServerIfNeeded() {
      // server-side barge-in: отменяем текущую генерацию
      const fd = new FormData();
      fd.append("session_id", sessionId);
      try { await fetch("/alina/cancel", { method: "POST", body: fd }); } catch (e) {}
    }

    // --- Recording ---
    btnStart.onclick = async () => {
      recordedChunks = [];
      recordStatus.textContent = "";

      // barge-in (client): остановить проигрывание немедленно
      try {
        replyAudio.pause();
        replyAudio.currentTime = 0;
        replyAudio.src = "";
      } catch (e) {}

      // barge-in (server): отменить текущую генерацию
      await cancelServerIfNeeded();

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) recordedChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
          const t = I18N[getUILang()] || I18N.ru;
          recordStatus.textContent = t.recDone;
        };

        mediaRecorder.start();
        btnStart.disabled = true;
        btnStop.disabled = false;

        const t = I18N[getUILang()] || I18N.ru;
        recordStatus.textContent = t.rec;
      } catch (err) {
        console.error(err);
        const t = I18N[getUILang()] || I18N.ru;
        recordStatus.textContent = t.micErr;
      }
    };

    btnStop.onclick = () => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        btnStart.disabled = false;
        btnStop.disabled = true;
      }
    };

    // --- Send ---
    btnSend.onclick = async () => {
      const t = I18N[getUILang()] || I18N.ru;

      sendStatus.textContent = "";
      sendStatus.className = "";
      uiTimings.style.display = "none";
      uiTimings.textContent = "";

      let audioBlob = null;
      let filename = "audio.wav";

      if (recordedChunks.length > 0) {
        audioBlob = new Blob(recordedChunks, { type: "audio/webm" });
        filename = "recording.webm";
      } else {
        const file = audioFileInput.files[0];
        if (!file) {
          alert(t.hint);
          return;
        }
        audioBlob = file;
        filename = file.name || "audio.wav";
      }

      const formData = new FormData();
      formData.append("audio", audioBlob, filename);

      const lang = document.querySelector('input[name="lang"]:checked').value;
      formData.append("lang", lang);

      // session_id для cancel/barge-in
      formData.append("session_id", sessionId);

      btnSend.disabled = true;
      sendStatus.textContent = t.sending;
      sendStatus.className = "";

      try {
        const resp = await fetch("/alina/voice", {
          method: "POST",
          body: formData,
        });

        if (!resp.ok) {
          const errData = await resp.json().catch(() => ({}));
          throw new Error(errData.detail || ("HTTP " + resp.status));
        }

        const data = await resp.json();

        if (data.session_id) {
          sessionId = data.session_id;
          uiSession.textContent = "session: " + sessionId;
        }

        // Audio
        if (data.audio_base64) {
          const mime = data.audio_mime || "audio/mpeg";
          replyAudio.src = `data:${mime};base64,${data.audio_base64}`;
          replyAudio.load();
        }

        // Chat bubbles
        replyChat.innerHTML = "";
        if (data.transcript) {
          const div = document.createElement("div");
          div.className = "bubble";
          div.innerHTML = `
            <div class="bubble-header">👤</div>
            <div class="bubble-user">
              ${String(data.transcript).replace(/\\n/g, "<br>")}
            </div>
          `;
          replyChat.appendChild(div);
        }

        if (data.answer) {
          const div = document.createElement("div");
          div.className = "bubble";
          div.innerHTML = `
            <div class="bubble-header">🤖</div>
            <div class="bubble-alina">
              ${String(data.answer).replace(/\\n/g, "<br>")}
            </div>
          `;
          replyChat.appendChild(div);
        }

        // History
        replyHistory.style.display = "block";
        replyHistory.textContent = "История диалога (history):\\n" + JSON.stringify(data.history, null, 2);

        // Timings
        if (data.timings) {
          uiTimings.style.display = "block";
          uiTimings.textContent = "Latency breakdown (ms):\\n" + JSON.stringify(data.timings, null, 2);
        }

        sendStatus.textContent = t.done;
        sendStatus.className = "status-ok";
      } catch (err) {
        console.error(err);
        sendStatus.textContent = t.err;
        sendStatus.className = "status-error";
      } finally {
        btnSend.disabled = false;
      }
    };
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "alina_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
