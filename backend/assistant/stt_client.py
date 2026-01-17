# backend/assistant/stt_client.py
"""
STT client (Deepgram) — production-safe, async-friendly.

Key goals:
- Stable function contract: transcribe(audio_bytes, filename, lang, mimetype, **kwargs)
- Never crash on unexpected kwargs (A+B “неубиваемость”)
- Do not block FastAPI event loop: run requests in threadpool
- Strong debug logging WITHOUT leaking secrets

ENV:
  DEEPGRAM_API_KEY (required)
  DEEPGRAM_MODEL (optional, default: nova-2)
  DEEPGRAM_TIER (optional, default: standard)
  DEEPGRAM_PUNCTUATE (optional, default: true)
  DEEPGRAM_SMART_FORMAT (optional, default: true)
  DEEPGRAM_ENDPOINT (optional, default: https://api.deepgram.com/v1/listen)
  DEBUG_ERRORS (optional, default: 0) -> adds extra debug logs
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional, Dict, Any

import anyio
import requests

logger = logging.getLogger("alina.stt")

_FILE_VERSION = "stt_client.py@2026-01-17.v2"
_THIS_FILE = os.path.abspath(__file__)


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _lang_to_deepgram(lang: str) -> str:
    # Deepgram expects ISO language codes like "en", "ru", "th".
    # Keep conservative mapping.
    lang = (lang or "ru").strip().lower()
    if lang in ("ru", "rus", "ru-ru"):
        return "ru"
    if lang in ("en", "eng", "en-us", "en-gb"):
        return "en"
    if lang in ("th", "tha", "th-th"):
        return "th"
    # fallback: pass through first two letters if plausible
    return lang[:2] if len(lang) >= 2 else "en"


def _guess_mimetype(filename: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit

    fn = (filename or "").lower().strip()
    if fn.endswith(".mp3"):
        return "audio/mpeg"
    if fn.endswith(".wav"):
        return "audio/wav"
    if fn.endswith(".m4a"):
        return "audio/mp4"
    if fn.endswith(".mp4"):
        return "audio/mp4"
    if fn.endswith(".webm"):
        return "audio/webm"
    if fn.endswith(".ogg"):
        return "audio/ogg"
    # default safe-ish
    return "application/octet-stream"


def _deepgram_request(
    *,
    audio_bytes: bytes,
    filename: str,
    lang: str,
    mimetype: Optional[str],
    request_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Blocking call. Must be executed in a worker thread.
    Returns transcript (string).
    """
    api_key = (os.getenv("DEEPGRAM_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")

    endpoint = (os.getenv("DEEPGRAM_ENDPOINT", "") or "").strip() or "https://api.deepgram.com/v1/listen"
    model = (os.getenv("DEEPGRAM_MODEL", "") or "").strip() or "nova-2"
    tier = (os.getenv("DEEPGRAM_TIER", "") or "").strip() or "standard"

    punctuate = _bool_env("DEEPGRAM_PUNCTUATE", True)
    smart_format = _bool_env("DEEPGRAM_SMART_FORMAT", True)

    dg_lang = _lang_to_deepgram(lang)
    content_type = _guess_mimetype(filename, mimetype)

    params = {
        "model": model,
        "tier": tier,
        "language": dg_lang,
        "punctuate": str(punctuate).lower(),
        "smart_format": str(smart_format).lower(),
    }

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": content_type,
    }

    # Log minimal safe diagnostics
    audio_sha1 = _safe_sha1(audio_bytes[:200000])  # hash prefix only (privacy + perf)
    logger.info(
        "STT Deepgram request: request_id=%s file=%s bytes=%d mimetype=%s lang=%s dg_lang=%s model=%s tier=%s sha1(prefix)=%s version=%s path=%s",
        request_id,
        filename,
        len(audio_bytes),
        content_type,
        lang,
        dg_lang,
        model,
        tier,
        audio_sha1,
        _FILE_VERSION,
        _THIS_FILE,
    )

    resp = requests.post(endpoint, params=params, headers=headers, data=audio_bytes, timeout=60)
    if resp.status_code >= 400:
        # Attempt to parse error JSON
        try:
            err_json = resp.json()
        except Exception:
            err_json = {"raw": resp.text[:500]}
        raise RuntimeError(f"Deepgram STT failed: HTTP {resp.status_code}: {json.dumps(err_json, ensure_ascii=False)}")

    data = resp.json()

    # Deepgram typical response:
    # data["results"]["channels"][0]["alternatives"][0]["transcript"]
    transcript = ""
    try:
        transcript = (
            data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
        )
    except Exception:
        transcript = ""

    transcript = (transcript or "").strip()
    if not transcript:
        # Not fatal: return empty string, but log diagnostics
        logger.warning("STT Deepgram returned empty transcript: request_id=%s file=%s", request_id, filename)

    # Optional debug dump (never includes api_key)
    if _bool_env("DEBUG_ERRORS", False):
        logger.info("STT Deepgram response keys: request_id=%s keys=%s", request_id, list(data.keys()))

    return transcript


async def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    lang: str = "ru",
    mimetype: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Async STT wrapper (НЕУБИВАЕМЫЙ):
    - accepts lang/mimetype/request_id
    - accepts **kwargs to survive contract drift
    """
    # Preserve extra debug context but do not enforce schema
    extra = {}
    if kwargs:
        # Keep only JSON-serializable primitives for safe logging
        for k, v in kwargs.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                extra[k] = v
            else:
                extra[k] = str(type(v))
        if extra and _bool_env("DEBUG_ERRORS", False):
            logger.info("STT transcribe got extra kwargs: request_id=%s extra=%s", request_id, extra)

    # Run blocking request in threadpool
    return await anyio.to_thread.run_sync(
        _deepgram_request,
        audio_bytes=audio_bytes,
        filename=filename,
        lang=lang,
        mimetype=mimetype,
        request_id=request_id,
        extra=extra,
    )
