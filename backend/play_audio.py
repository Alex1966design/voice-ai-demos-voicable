import base64

# читаем JSON
import json
with open("alina_response.json", "r", encoding="utf-8") as f:
    data = json.load(f)

audio_b64 = data["audio_base64"]
audio_bytes = base64.b64decode(audio_b64)

# сохраняем в mp3
with open("alina_answer.mp3", "wb") as f:
    f.write(audio_bytes)

print("Готово! Файл alina_answer.mp3 создан.")
