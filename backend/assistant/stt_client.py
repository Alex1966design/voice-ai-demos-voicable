# assistant/stt_client.py
"""
STT client (Deepgram) - production-safe

Goals:
- Keep API stable for callers: transcribe(audio_bytes, filename, lang=..., mimetype=...)
- Accept extra kwargs without crashing (A: compatibility layer)
- Use Deepgram SDK if available, otherwise REST fallback (B: resilience)
- Provide actionable errors and minimal logging hooks

ENV:
- DEEPGRAM_API_KEY (required)
Optional:
- STT_TIMEOUT_SECS (default 45)
- STT_MAX_BYTES (default 25_000_000)  # 25MB
- STT_DEBUG (default 0)              # if "1", prints short debug info
"""

from __future__ import annotations

import os
import json
import mimetypes
from typing import Optional, Any, Dict, Tuple

import httpx

# -----------------------------
# Config
# -----------------------------
DEEPGRAM_API_KEY = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
STT_TIMEOUT_SECS = float(os.getenv("STT_TIMEOUT_SECS", "45"))
STT_MAX_BYTES = int(os.getenv("STT_MAX_BYTES", str(25_000_000)))
STT_DEBUG = os.getenv("STT_DEBUG", "0") == "1"


# -----------------------------
# Helpers
# -----------------------------
def _dbg(msg: str) -> None:
    if STT_DEBUG:
        print(f"[stt_client] {msg}")


def _normalize_lang(lang: Optional[str]) -> str:
    """
    Map UI lang to Deepgram language codes.
    UI sends: ru | en | th
    Deepgram expects: ru | en | th (these are OK), but keep mapping explicit.
    """
    if not lang:
        return "ru"
    lang = lang.lower().strip()
    if lang in ("ru", "ru-ru"):
        return "ru"
    if lang in ("en", "en-us", "en-gb"):
        return "en"
    if lang in ("th", "th-th"):
        return "th"
    return lang


def _guess_mimetype(filename: str, provided: Optional[str]) -> str:
    if provided and "/" in provided:
        return provided.split(";")[0].strip()
    mt, _ = mimetypes.guess_type(filename)
    if mt:
        return mt
    # common fallbacks
    if filename.lower().endswith(".webm"):
        return "audio/webm"
    if filename.lower().endswith(".wav"):
        return "audio/wav"
    if filename.lower().endswith(".mp3"):
        return "audio/mpeg"
    if filename.lower().endswith(".m4a"):
        return "audio/mp4"
    return "application/octet-stream"


def _pick_extension(filename: str, mimetype: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext:
        return ext
    # reverse guess
    if mimetype == "audio/webm":
        return ".webm"
    if mimetype in ("audio/wav", "audio/x-wav"):
        return ".wav"
    if mimetype in ("audio/mpeg", "audio/mp3"):
        return ".mp3"
    if mimetype in ("audio/mp4", "audio/m4a"):
        return ".m4a"
    return ".bin"


def _deepgram_query_params(lang: str) -> Dict[str, Any]:
    """
    Reasonable defaults for general speech.
    You can tune later (model, punctuation, diarize, etc.).
    """
    return {
        "model": "nova-2",
        "language": _normalize_lang(lang),
        "smart_format": "true",
        "punctuate": "true",
        "profanity_filter": "false",
    }


def _parse_deepgram_transcript(payload: Dict[str, Any]) -> str:
    """
    Parse both SDK-like and REST-like response shapes.
    Deepgram prerecorded typically returns:
      results.channels[0].alternatives[0].transcript
    """
    try:
        results = payload.get("results") or {}
        channels = results.get("channels") or []
        if not channels:
            return ""
        alts = (channels[0].get("alternatives") or [])
        if not alts:
            return ""
        transcript = (alts[0].get("transcript") or "").strip()
        return transcript
    except Exception:
        return ""


# -----------------------------
# Public API
# -----------------------------
async def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    lang: Optional[str] = None,
    mimetype: Optional[str] = None,
    language: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Production-safe transcription.

    Compatibility:
      - accepts lang=..., language=..., mimetype=...
      - accepts extra kwargs and ignores them (prevents "unexpected keyword" crashes)

    Returns:
      transcript string (may be empty)
    Raises:
      RuntimeError with details (will appear in FastAPI 500 if you bubble it)
    """
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")

    if not audio_bytes:
        raise RuntimeError("Empty audio_bytes")

    if len(audio_bytes) > STT_MAX_BYTES:
        raise RuntimeError(f"Audio too large: {len(audio_bytes)} bytes > STT_MAX_BYTES={STT_MAX_BYTES}")

    # allow caller to pass either lang or language
    use_lang = language or lang or "ru"

    mime = _guess_mimetype(filename, mimetype)
    ext = _pick_extension(filename, mime)

    _dbg(f"transcribe() len={len(audio_bytes)} filename={filename} mime={mime} lang={use_lang} ext={ext}")

    # 1) Try Deepgram SDK (if installed and compatible)
    sdk_err: Optional[Exception] = None
    try:
        # Deepgram SDK v3 style import
        from deepgram import DeepgramClient, PrerecordedOptions  # type: ignore

        client = DeepgramClient(DEEPGRAM_API_KEY)

        options_dict = _deepgram_query_params(use_lang)

        # SDK expects booleans not strings sometimes
        options = PrerecordedOptions(
            model=options_dict["model"],
            language=options_dict["language"],
            smart_format=True,
            punctuate=True,
            profanity_filter=False,
        )

        # Deepgram SDK accepts "buffer" source
        source = {"buffer": audio_bytes, "mimetype": mime}

        # v3 has: client.listen.prerecorded.v("1").transcribe_file(source, options)
        # But different minor versions vary. We'll try common signatures.
        dg_resp = None
        try:
            dg_resp = await client.listen.prerecorded.v("1").transcribe_file(source, options)  # type: ignore
        except TypeError:
            # fallback: sync call
            dg_resp = client.listen.prerecorded.v("1").transcribe_file(source, options)  # type: ignore

        # response can be object with .to_dict()
        if hasattr(dg_resp, "to_dict"):
            payload = dg_resp.to_dict()  # type: ignore
        elif isinstance(dg_resp, dict):
            payload = dg_resp
        else:
            # last resort
            payload = json.loads(str(dg_resp))

        transcript = _parse_deepgram_transcript(payload)
        if transcript:
            return transcript

        # If empty, still return empty (no exception)
        return transcript

    except Exception as e:
        sdk_err = e
        _dbg(f"Deepgram SDK failed, fallback to REST. Error: {repr(e)}")

    # 2) REST fallback (robust)
    url = "https://api.deepgram.com/v1/listen"
    params = _deepgram_query_params(use_lang)

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": mime,
    }

    timeout = httpx.Timeout(STT_TIMEOUT_SECS, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, params=params, content=audio_bytes, headers=headers)
        except Exception as e:
            # include sdk error too
            raise RuntimeError(f"Deepgram REST request failed: {e}. SDK error was: {repr(sdk_err)}") from e

    if resp.status_code >= 400:
        # return a short body snippet for debugging
        body_snip = resp.text[:600]
        raise RuntimeError(
            f"Deepgram REST error {resp.status_code}: {body_snip}. SDK error was: {repr(sdk_err)}"
        )

    try:
        payload = resp.json()
    except Exception as e:
        raise RuntimeError(f"Deepgram REST returned non-JSON: {resp.text[:600]}") from e

    transcript = _parse_deepgram_transcript(payload)
    return transcript.strip()
