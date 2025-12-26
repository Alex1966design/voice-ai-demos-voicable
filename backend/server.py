# alina_server.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from assistant.alina import AlinaAssistant


# ----------------------------------
# ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ FASTAPI
# ----------------------------------

app = FastAPI(
    title="Alina Voice Assistant",
    description="Отдельный сервер Алины: STT → LLM → TTS",
    version="1.0.0",
)

# Разрешаем фронтенд подключаться
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Один экземпляр Алины на всё приложение
assistant = AlinaAssistant()


# ----------------------------------
# HEALTHCHECK
# ----------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "alina"}


# ----------------------------------
# ГОЛОСОВОЙ ЭНДПОИНТ
# ----------------------------------

@app.post("/alina/voice")
async def alina_voice(audio: UploadFile = File(...)):
    """
    Полный голосовой цикл Алины:

    1) STT → текст пользователя
    2) LLM → ответ Алины с учётом истории
    3) TTS → озвучка ответа (base64)

    Фронт отправляет multipart/form-data:
        audio=<файл>
    """

    # читаем файл
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        result = assistant.handle_user_audio(
            audio_bytes,
            audio.filename or "audio.wav",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alina error: {e}")

    return result


# ----------------------------------
# ЛОКАЛЬНЫЙ ЗАПУСК UVICORN
# ----------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "alina_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
