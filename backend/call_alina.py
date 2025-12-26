import json
import requests

URL = "http://127.0.0.1:8001/alina/voice"

# Открываем наш тестовый wav
with open("test.wav", "rb") as f:
    files = {
        "audio": ("test.wav", f, "audio/wav"),  # имя поля ДОЛЖНО быть "audio"
    }
    resp = requests.post(URL, files=files)

print("Status:", resp.status_code)
print("Raw response:", resp.text[:500], "..." if len(resp.text) > 500 else "")

# Сохраним ответ в файл
with open("alina_response.json", "w", encoding="utf-8") as out:
    out.write(resp.text)

print("Saved to alina_response.json")
