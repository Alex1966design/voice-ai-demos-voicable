# backend/alina_server.py
"""
Alina Voice Assistant (FastAPI)

Routes:
  - GET  /health         -> JSON healthcheck
  - GET  /               -> HTML UI (RU / EN / TH)
  - POST /alina/voice    -> STT -> LLM -> TTS pipeline
  - POST /alina/cancel   -> cancel in-flight generation for a session_id (best-effort)

Railway start command (repo root):
  uvicorn backend.alina_server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import base64
import logging
import os
import traceback
import uuid
import inspect
from typing import Dict, Any, Optional, Callable

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# -------------------------
# Logging (production-safe)
# -------------------------
logger = logging.getLogger("alina")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DEBUG_ERRORS = (os.getenv("DEBUG_ERRORS", "0").strip() in ("1", "true", "yes", "on"))


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _safe_exc_detail(e: Exception) -> str:
    # No secrets, short message
    return f"{type(e).__name__}: {str(e)[:400]}"


# -------------------------
# Cancel token compatibility
# -------------------------
try:
    from assistant.llm_client import CancelToken  # type: ignore
except Exception:
    class CancelToken:
        def __init__(self, cancelled: bool = False):
            self.cancelled = cancelled

        def cancel(self):
            self.cancelled = True


# -------------------------
# Try primary assistant path
# -------------------------
assistant_import_error = None
assistant_ru = assistant_en = assistant_th = None

try:
    from assistant.alina import AlinaAssistant  # type: ignore

    assistant_ru = AlinaAssistant(mode="ru")
    assistant_en = AlinaAssistant(mode="en")
    assistant_th = AlinaAssistant(mode="th")
    logger.info("Primary assistant.alina loaded successfully.")
except Exception as e:
    assistant_import_error = e
    logger.warning("Primary assistant.alina import failed. Will use fallback pipeline. error=%s", _safe_exc_detail(e))

    # Fallback pipeline imports
    from assistant.stt_client import transcribe  # type: ignore
    from assistant.llm_client import chat_with_alina  # type: ignore
    from assistant.elevenlabs_client import tts_elevenlabs  # type: ignore


def _pick_lang_assistant(lang: str):
    lang = (lang or "ru").lower().strip()
    if lang == "en":
        return assistant_en
    if lang == "th":
        return assistant_th
    return assistant_ru


def _fallback_system_prompt(lang: str) -> str:
    lang = (lang or "ru").lower().strip()
    if lang == "th":
        return (
            "You are Alina, a helpful voice assistant. Reply in Thai. "
            "Be concise, structured, and friendly."
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


def _call_with_supported_kwargs(fn: Callable[..., Any], **kwargs) -> Any:
    """
    A+B hardening: call function with only the kwargs it actually supports.
    This prevents crashes like: got an unexpected keyword argument 'lang'
    even if function versions drift.
    """
    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        # If we can't inspect, be conservative: call without kwargs
        return fn()

    supported = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            supported[k] = v

    # If function supports **kwargs, pass everything
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return fn(**kwargs)

    return fn(**supported)


app = FastAPI(
    title="Alina Voice Assistant",
    description="Standalone server: STT → LLM → TTS (RU / EN / TH)",
    version="1.4.0",
)

# CORS: keep demo-wide by default, but allow tightening via env
allow_origins = os.getenv("ALLOW_ORIGINS", "*").strip()
origins = ["*"] if allow_origins == "*" else [o.strip() for o in allow_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active cancels by session_id
active_cancels: Dict[str, CancelToken] = {}


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or _new_request_id()
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception("Unhandled error request_id=%s path=%s", request_id, request.url.path)
        raise e
    response.headers["x-request-id"] = request_id
    return response


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


async def _fallback_pipeline(
    *,
    audio_bytes: bytes,
    filename: str,
    lang: str,
    cancel_token: CancelToken,
    mimetype: Optional[str],
    request_id: str,
) -> Dict[str, Any]:
    """
    Fallback pipeline:
      STT (Deepgram) -> LLM -> TTS (ElevenLabs)
    Hardened against signature drift in transcribe() via _call_with_supported_kwargs.
    """
    # 1) STT — async
    # A+B: call with only supported kwargs (or pass through if transcribe has **kwargs)
    transcript = await _call_with_supported_kwargs(
        transcribe,  # type: ignore
        audio_bytes=audio_bytes,
        filename=filename,
        lang=lang,
        mimetype=mimetype,
        request_id=request_id,
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
    answer = chat_with_alina(messages=messages)  # type: ignore

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
    audio_mp3 = tts_elevenlabs(answer)  # type: ignore
    audio_b64 = base64.b64encode(audio_mp3).decode("utf-8")

    return {
        "transcript": transcript,
        "answer": answer,
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg",
        "history": messages + [{"role": "assistant", "content": answer}],
        "timings": {},
    }


@app.post("/alina/voice")
async def alina_voice(
    request: Request,
    audio: UploadFile = File(...),
    lang: str = Form("ru"),        # "ru" | "en" | "th"
    session_id: str = Form(""),    # comes from frontend
):
    """
    Returns JSON:
      { transcript, answer, audio_base64, audio_mime, history, timings, session_id }
    """
    request_id = getattr(request.state, "request_id", _new_request_id())

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if not session_id:
        session_id = str(uuid.uuid4())

    cancel_token = CancelToken(False)
    active_cancels[session_id] = cancel_token

    filename = audio.filename or "audio.wav"
    mimetype = audio.content_type  # important for webm/mp3/wav

    logger.info(
        "VOICE request start request_id=%s session_id=%s lang=%s filename=%s content_type=%s bytes=%d",
        request_id, session_id, lang, filename, mimetype, len(audio_bytes)
    )

    try:
        # Primary assistant path
        if assistant_ru is not None:
            assistant = _pick_lang_assistant(lang)
            if assistant is None:
                raise RuntimeError("Assistant not initialised")

            # Note: keep this sync unless your assistant is async
            result = assistant.handle_user_audio(
                audio_bytes,
                filename,
                cancel_token=cancel_token,
                use_llm_stream=True,
            )
            if not isinstance(result, dict):
                raise RuntimeError("assistant.handle_user_audio must return dict")

        # Fallback path
        else:
            result = await _fallback_pipeline(
                audio_bytes=audio_bytes,
                filename=filename,
                lang=lang,
                cancel_token=cancel_token,
                mimetype=mimetype,
                request_id=request_id,
            )
            result.setdefault("timings", {})
            result["timings"]["assistant_import_error"] = str(assistant_import_error)

        result["session_id"] = session_id
        result.setdefault("timings", {})
        result["timings"]["request_id"] = request_id

        logger.info("VOICE request done request_id=%s session_id=%s", request_id, session_id)
        return JSONResponse(content=result)

    except HTTPException:
        raise

    except Exception as e:
        # Production-safe: return error_id to client, full trace in logs
        error_id = _new_request_id()
        logger.exception(
            "VOICE request failed request_id=%s error_id=%s session_id=%s err=%s",
            request_id, error_id, session_id, _safe_exc_detail(e)
        )

        if DEBUG_ERRORS:
            tb = traceback.format_exc()
            # In DEBUG mode we still avoid secrets; stacktrace is okay for your private demo
            raise HTTPException(
                status_code=500,
                detail=f"Alina error_id={error_id} request_id={request_id} :: {e}\n---\n{tb}",
            )

        raise HTTPException(
            status_code=500,
            detail=f"Alina internal error. error_id={error_id} request_id={request_id}",
        )

    finally:
        active_cancels.pop(session_id, None)


@app.get("/", response_class=HTMLResponse)
async def index():
    # UI: unchanged (your current HTML). Keep as-is.
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
    const I18N = {
      ru: { title:"Alina – голосовой ассистент", subtitle:"Отдельный сервер: STT → LLM → TTS (RU / EN / TH)",
        step1:"Шаг 1. Запиши или выбери аудиофайл",
        hint:"Можно выбрать готовый аудиофайл или записать голос с микрофона прямо в браузере.",
        step2:"Шаг 2. Отправь запрос Алине", send:"Отправить Алине", answer:"Ответ Алины",
        rec:"Запись идёт…", recDone:"Запись завершена. Теперь можно отправить Алине.",
        micErr:"Не удалось получить доступ к микрофону.", sending:"Отправка…", done:"Готово ✔", err:"Ошибка
