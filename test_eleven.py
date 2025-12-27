import os
import requests
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")
voice_id = os.getenv("ELEVENLABS_VOICE")

if not api_key:
    raise RuntimeError("ELEVENLABS_API_KEY not found in .env")
if not voice_id:
    raise RuntimeError("ELEVENLABS_VOICE not found in .env")

print("Using voice_id:", voice_id)

url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

headers = {
    "xi-api-key": api_key,
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
}

payload = {
    "text": "Hello Alex! This is Rachel from ElevenLabs. Your voice demo is working perfectly.",
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.8,
    },
}

print("Sending request to ElevenLabs...")
resp = requests.post(url, headers=headers, json=payload)

print("Status:", resp.status_code)

if resp.status_code != 200:
    print("Response text:", resp.text[:500])
    raise SystemExit("ElevenLabs returned an error")

with open("rachel_test.mp3", "wb") as f:
    f.write(resp.content)

print("Saved file: rachel_test.mp3")
