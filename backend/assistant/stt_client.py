# backend/assistant/stt_client.py
from __future__ import annotations

import os
from typing import Optional, Any

from dotenv import load_dotenv
from deepgram import AsyncDeepgramClient

load_dotenv()


def _get_env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return (v or "").strip()


DEEPGRAM_API_KEY = _get_env("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set in .env or environment")

# Model: nova-2 / nova-3 etc.
DEEPGRAM_MODEL = _get_env("DEEPGRAM_MODEL", "nova-3")

# Optional: default language; for Thai we will force "th" by lang param
DEEPGRAM_LANGUAGE_DEFAULT = _get_env("DEEPGRAM_LANGUAGE", "")  # e.g. "en" or ""


client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)


def _lang_to_deepgram(lang: str) -> Optional[str]:
    """
    Map UI lang -> Deepgram language code.
    If None, Deepgram may auto-detect (depending on plan/model).
    """
    lang = (lang or "").lower().strip()
    if lang == "th":
        return "th"
    if lang == "en":
        return "en"
    if lang == "ru":
        return "ru"
    # fallback to env default or None
    return DEEPGRAM_LANGUAGE_DEFAULT or None


def _guess_mimetype(filename: str, provided: Optional[str]) -> Optional[str]:
    if provided:
        return provided
    fn = (filename or "").lower()
    if fn.endswith(".webm"):
        return "audio/webm"
    if fn.endswith(".mp3"):
        return "audio/mpeg"
    if fn.endswith(".wav"):
        return "audio/wav"
    if fn.endswith(".m4a"):
        return "audio/mp4"
    if fn.endswith(".ogg"):
        return "audio/ogg"
    return None


def _safe_extract_transcript(resp: Any) -> str:
    """
    Deepgram SDK response formats can vary slightly by version.
    We defensively extract transcript.
    """
    try:
        # Common: resp.results.channels[0].alternatives[0].transcript
        transcript = resp.results.channels[0].alternatives[0].transcript
        return (transcript or "").strip()
    except Exception:
        pass

    try:
        # Sometimes dict-like
        results = resp.get("results") or {}
        channels = results.get("channels") or []
        if channels:
            alts = channels[0].get("alternatives") or []
            if alts:
                return (alts[0].get("transcript") or "").strip()
    except Exception:
        pass

    return ""


async def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    lang: str = "ru",
    mimetype: Optional[str] = None,
) -> str:
    """
    STT via Deepgram (async). Returns recognised text.
    Works with bytes coming from browser (often audio/webm).
    """
    if not audio_bytes:
        return ""

    dg_lang = _lang_to_deepgram(lang)
    mt = _guess_mimetype(filename, mimetype)

    # Options
    options: dict[str, Any] = {
        "model": DEEPGRAM_MODEL,
        "smart_format": True,
    }
    # If language is set, pass it explicitly (good for Thai).
    if dg_lang:
        options["language"] = dg_lang

    # Important: content-type helps Deepgram decode webm/mp3/wav
    # Depending on SDK version, parameter name can be mimetype/content_type.
    # We use 'mimetype' in the call below where supported.
    try:
        resp = await client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            mimetype=mt,   # critical for audio/webm
            **options,
        )
    except TypeError:
        # Fallback for SDK variants that use content_type instead of mimetype
        resp = await client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            content_type=mt,
            **options,
        )
    except Exception as e:
        raise RuntimeError(f"Deepgram STT failed: {e}")

    transcript = _safe_extract_transcript(resp)
    return transcript
