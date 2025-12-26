# assistant/__init__.py

from . import stt_client, llm_client, tts_client
from .alina import AlinaAssistant

__all__ = ["stt_client", "llm_client", "tts_client", "AlinaAssistant"]
