# backend/alina_server.py
"""
Alina Voice Assistant (FastAPI)
Routes:
  - GET  /health         -> JSON healthcheck
  - GET  /               -> HTML UI (RU / EN / TH)
  - POST /alina/voice    -> STT -> LLM -> TTS pipeline
  - POST /alina/cancel   -> cancel in-flight generation for a session_id (best-effort)

Railway start command (Root Directory = backend):
  uvicorn alina_server:app --host 0.0.0.0 --port $PORT
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

# --- Cancel token (safe fallback if not present elsewhere) ---
try:
    from assistant.llm_client import CancelToken  # type: ignore
except Exception:
    class CancelToken:
        def __init__(self, cancelled: bool = False):
            self.cancelled = cancelled

        def cancel(self):
            self.cancelled = True


# --- Optional: try to use your existing assistant class (but NEVER let it break demo) ---
assistant_import_error = None
assistant_ru = assistant_en = assistant_th = None
try:
    from assistant.alina import AlinaAssistant  # type: ignore

    assistant_ru = AlinaAssistant(mode="ru")
    assistant_en = AlinaAssistant(mode="en")
    assistant_th = AlinaAssistant(mode="th")
except Exception as e:
    assistant_import_error = e
    assistant_ru = assistant_en = assistant_th = None

# --- Fallback pipeline (used always when assistant is missing OR crashes) ---
from assistant.stt_client import transcribe  # async (we will await)
from assistant.llm_client import chat_with_alina  # sync
from assistant.elevenlabs_client import tts_elevenlabs  # sync


app = FastAPI(
    title="Alina Voice Assistant",
    description="Standalone server: STT → LLM → TTS (RU / EN / TH)",
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active cancels by session_id (barge-in / cancel)
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


def _pick_lang_assistant(lang: str):
    if lang == "en":
        return assistant_en
    if lang == "th":
        return assistant_th
    return assistant_ru


def _fallback_system_prompt(lang: str) -> str:
    # Быстрый демо-режим: просим отвечать строго на выбранном языке
    if lang == "th":
        return (
            "You are Alina, a helpful voice assistant. Reply in Thai language only. "
            "Be concise, structured, and friendly. If user asks about Phuket food, give practical suggestions."
        )
    if lang == "en":
        return (
            "You are Alina, a helpful voice assistant. Reply in English. "
            "Be concise, structured, and friendly."
        )
    return (
        "Ты — Алина, полезный голосовой ассистент. Отвечай на русском. "
        "Коротко, структурно и дружелюбно."
    )


async def _fallback_pipeline(
    audio_bytes: bytes,
    filename: str,
    content_type: Optional[str],
    lang: str,
    cancel_token: CancelToken,
) -> Dict[str, Any]:
    """
    Fallback pipeline:
      STT -> LLM -> TTS (ElevenLabs)
    """
    # 1) STT
    transcript = await transcribe(
        audio_bytes=audio_bytes,
        filename=filename,
        lang=lang,
        content_type=content_type,
    )

    if cancel_token.cancelled:
        return {
            "transcript": transcript,
            "answer": "",
            "audio_base64": "",
            "audio_mime": "audio/mpeg",
            "history": [],
            "timings": {"cancelled": True},
        }

    # 2) LLM
    messages = [
        {"role": "system", "content": _fallback_system_prompt(lang)},
        {"role": "user", "content": transcript or ""},
    ]
    answer = chat_with_alina(messages=messages)

    if cancel_token.cancelled:
        return {
            "transcript": transcript,
            "answer": answer,
            "audio_base64": "",
            "audio_mime": "audio/mpeg",
            "history": messages + [{"role": "assistant", "content": answer}],
            "timings": {"cancelled": True},
        }

    # 3) TTS
    audio_mp3 = tts_elevenlabs(answer)
    audio_b64 = base64.b64encode(audio_mp3).decode("utf-8")

    timings: Dict[str, Any] = {}
    if assistant_import_error is not None:
        timings["assistant_import_error"] = str(assistant_import_error)

    return {
        "transcript": transcript,
        "answer": answer,
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg",
        "history": messages + [{"role": "assistant", "content": answer}],
        "timings": timings,
    }


@app.post("/alina/voice")
async def alina_voice(
    audio: UploadFile = File(...),
    lang: str = Form("th"),        # "ru" | "en" | "th"
    session_id: str = Form(""),
):
    """
    Full voice cycle:
      STT -> LLM -> TTS
    Returns JSON:
      { transcript, answer, audio_base64, audio_mime, history, timings, session_id }
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if not session_id:
        session_id = str(uuid.uuid4())

    cancel_token = CancelToken(False)
    active_cancels[session_id] = cancel_token

    filename = audio.filename or "audio.wav"
    content_type = audio.content_type  # IMPORTANT for Deepgram (webm/ogg/wav)
    # Для webm иногда content_type пустой — подстрахуемся расширением
    if not content_type and filename.lower().endswith(".webm"):
        content_type = "audio/webm"

    try:
        # 1) Попытка primary assistant (если есть) — но без права ломать демо
        result: Dict[str, Any]
        if assistant_ru is not None:
            try:
                assistant = _pick_lang_assistant(lang)
                if assistant is None:
                    raise RuntimeError("Assistant not initialised")

                # Expect assistant.handle_user_audio(...) to return dict
                maybe = assistant.handle_user_audio(
                    audio_bytes,
                    filename,
                    cancel_token=cancel_token,
                    use_llm_stream=True,
                )
                if not isinstance(maybe, dict):
                    raise RuntimeError("assistant.handle_user_audio must return dict")
                result = maybe
            except Exception as e:
                # ВАЖНО: не падаем — уходим в fallback и добавляем причину
                fb = await _fallback_pipeline(audio_bytes, filename, content_type, lang, cancel_token)
                fb.setdefault("timings", {})
                fb["timings"]["assistant_failed_fallback"] = str(e)
                result = fb
        else:
            # 2) Fallback pipeline (надёжный демо-режим)
            result = await _fallback_pipeline(audio_bytes, filename, content_type, lang, cancel_token)

        result["session_id"] = session_id
        return JSONResponse(content=result)

    except Exception as e:
        # Логируем traceback в Railway logs — это ключ к любым оставшимся 500
        print("ERROR in /alina/voice:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Alina error: {e}")

    finally:
        active_cancels.pop(session_id, None)


@app.get("/", response_class=HTMLResponse)
async def index():
    # UI оставляем; можешь позже скрыть RU ради демо
    html = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
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
        <input type="radio" name="lang" value="ru" />
        🇷🇺 RU
      </label>
      <label style="margin-right: 10px;">
        <input type="radio" name="lang" value="en" />
        🇬🇧 EN
      </label>
      <label>
        <input type="radio" name="lang" value="th" checked />
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
    const I18N = {
      ru: { title:"Alina – голосовой ассистент", subtitle:"Отдельный сервер: STT → LLM → TTS (RU / EN / TH)",
        step1:"Шаг 1. Запиши или выбери аудиофайл", hint:"Можно выбрать готовый аудиофайл или записать голос с микрофона прямо в браузере.",
        step2:"Шаг 2. Отправь запрос Алине", send:"Отправить Алине", answer:"Ответ Алины",
        rec:"Запись идёт…", recDone:"Запись завершена. Теперь можно отправить Алине.", micErr:"Не удалось получить доступ к микрофону.",
        sending:"Отправка…", done:"Готово ✔", err:"Ошибка ✖"
      },
      en: { title:"Alina – voice assistant", subtitle:"Standalone server: STT → LLM → TTS (RU / EN / TH)",
        step1:"Step 1. Record or choose an audio file", hint:"You can select an audio file or record from the microphone directly in the browser.",
        step2:"Step 2. Send a request to Alina", send:"Send to Alina", answer:"Alina's reply",
        rec:"Recording…", recDone:"Recording finished. You can now send it to Alina.", micErr:"Microphone access error.",
        sending:"Sending…", done:"Done ✔", err:"Error ✖"
      },
      th: { title:"Alina – ผู้ช่วยเสียง", subtitle:"เซิร์ฟเวอร์เดี่ยว: STT → LLM → TTS (RU / EN / TH)",
        step1:"ขั้นตอนที่ 1 บันทึกเสียงหรือเลือกไฟล์เสียง", hint:"คุณสามารถเลือกไฟล์เสียง หรือบันทึกเสียงจากไมโครโฟนในเบราว์เซอร์ได้",
        step2:"ขั้นตอนที่ 2 ส่งคำถามให้ Alina", send:"ส่งให้ Alina", answer:"คำตอบของ Alina",
        rec:"กำลังบันทึก…", recDone:"บันทึกเสร็จแล้ว พร้อมส่งให้ Alina", micErr:"ไม่สามารถเข้าถึงไมโครโฟนได้",
        sending:"กำลังส่ง…", done:"เสร็จสิ้น ✔", err:"เกิดข้อผิดพลาด ✖"
      }
    };

    function getUILang(){ return document.querySelector('input[name="lang"]:checked').value || "th"; }
    function applyUI(lang){
      const t = I18N[lang] || I18N.th;
      document.getElementById("ui-title").textContent = t.title;
      document.getElementById("ui-subtitle").textContent = t.subtitle;
      document.getElementById("ui-step1").textContent = t.step1;
      document.getElementById("ui-hint").textContent = t.hint;
      document.getElementById("ui-step2").textContent = t.step2;
      document.getElementById("btn-send").textContent = t.send;
      document.getElementById("ui-answer").textContent = t.answer;
    }

    let mediaRecorder=null, recordedChunks=[];
    let sessionId=(crypto&&crypto.randomUUID)?crypto.randomUUID():String(Date.now());

    const btnStart=document.getElementById("btn-start");
    const btnStop=document.getElementById("btn-stop");
    const recordStatus=document.getElementById("record-status");
    const btnSend=document.getElementById("btn-send");
    const sendStatus=document.getElementById("send-status");
    const audioFileInput=document.getElementById("audio-file");

    const replyAudio=document.getElementById("reply-audio");
    const replyChat=document.getElementById("reply-chat");
    const replyHistory=document.getElementById("reply-history");
    const uiTimings=document.getElementById("ui-timings");
    const uiSession=document.getElementById("ui-session");

    uiSession.textContent="session: "+sessionId;
    applyUI(getUILang());

    document.querySelectorAll('input[name="lang"]').forEach(r=>r.addEventListener("change",()=>applyUI(getUILang())));

    async function cancelServerIfNeeded(){
      const fd=new FormData();
      fd.append("session_id",sessionId);
      try{ await fetch("/alina/cancel",{method:"POST",body:fd}); }catch(e){}
    }

    btnStart.onclick=async()=>{
      recordedChunks=[];
      recordStatus.textContent="";

      try{ replyAudio.pause(); replyAudio.currentTime=0; replyAudio.src=""; }catch(e){}
      await cancelServerIfNeeded();

      try{
        const stream=await navigator.mediaDevices.getUserMedia({audio:true});
        mediaRecorder=new MediaRecorder(stream);
        mediaRecorder.ondataavailable=(e)=>{ if(e.data.size>0) recordedChunks.push(e.data); };
        mediaRecorder.onstop=()=>{ const t=I18N[getUILang()]||I18N.th; recordStatus.textContent=t.recDone; };
        mediaRecorder.start();
        btnStart.disabled=true; btnStop.disabled=false;
        const t=I18N[getUILang()]||I18N.th; recordStatus.textContent=t.rec;
      }catch(err){
        console.error(err);
        const t=I18N[getUILang()]||I18N.th; recordStatus.textContent=t.micErr;
      }
    };

    btnStop.onclick=()=>{
      if(mediaRecorder && mediaRecorder.state!=="inactive"){
        mediaRecorder.stop();
        btnStart.disabled=false;
        btnStop.disabled=true;
      }
    };

    btnSend.onclick=async()=>{
      const t=I18N[getUILang()]||I18N.th;

      sendStatus.textContent="";
      sendStatus.className="";
      uiTimings.style.display="none"; uiTimings.textContent="";

      let audioBlob=null;
      let filename="audio.wav";

      if(recordedChunks.length>0){
        audioBlob=new Blob(recordedChunks,{type:"audio/webm"});
        filename="recording.webm";
      }else{
        const file=audioFileInput.files[0];
        if(!file){ alert(t.hint); return; }
        audioBlob=file; filename=file.name||"audio.wav";
      }

      const formData=new FormData();
      formData.append("audio",audioBlob,filename);
      const lang=document.querySelector('input[name="lang"]:checked').value;
      formData.append("lang",lang);
      formData.append("session_id",sessionId);

      btnSend.disabled=true;
      sendStatus.textContent=t.sending;

      try{
        const resp=await fetch("/alina/voice",{method:"POST",body:formData});
        if(!resp.ok){
          const errData=await resp.json().catch(()=>({}));
          throw new Error(errData.detail || ("HTTP "+resp.status));
        }

        const data=await resp.json();
        if(data.session_id){ sessionId=data.session_id; uiSession.textContent="session: "+sessionId; }

        if(data.audio_base64){
          const mime=data.audio_mime||"audio/mpeg";
          replyAudio.src=`data:${mime};base64,${data.audio_base64}`;
          replyAudio.load();
        }

        replyChat.innerHTML="";
        if(data.transcript){
          const div=document.createElement("div");
          div.className="bubble";
          div.innerHTML=`<div class="bubble-header">👤</div><div class="bubble-user">${String(data.transcript).replace(/\\n/g,"<br>")}</div>`;
          replyChat.appendChild(div);
        }
        if(data.answer){
          const div=document.createElement("div");
          div.className="bubble";
          div.innerHTML=`<div class="bubble-header">🤖</div><div class="bubble-alina">${String(data.answer).replace(/\\n/g,"<br>")}</div>`;
          replyChat.appendChild(div);
        }

        replyHistory.style.display="block";
        replyHistory.textContent="history:\\n"+JSON.stringify(data.history,null,2);

        if(data.timings){
          uiTimings.style.display="block";
          uiTimings.textContent="timings:\\n"+JSON.stringify(data.timings,null,2);
        }

        sendStatus.textContent=t.done;
        sendStatus.className="status-ok";
      }catch(err){
        console.error(err);
        sendStatus.textContent=t.err;
        sendStatus.className="status-error";
      }finally{
        btnSend.disabled=false;
      }
    };
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
