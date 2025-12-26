from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"

    # Deepgram
    deepgram_api_key: str | None = None
    deepgram_model: str = "nova-3"
    deepgram_mimetype: str = "audio/webm"

    # ElevenLabs
    elevenlabs_api_key: str | None = None
    elevenlabs_voice: str | None = None  # имя или ID голоса
    elevenlabs_model: str = "eleven_multilingual_v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
