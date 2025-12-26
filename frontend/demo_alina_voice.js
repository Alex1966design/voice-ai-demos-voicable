const API_URL = "http://127.0.0.1:8001/alina/voice";

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const playBtn = document.getElementById("playBtn");
const statusLabel = document.getElementById("statusLabel");
const chatLog = document.getElementById("chatLog");
const errorBox = document.getElementById("errorBox");
const answerAudio = document.getElementById("answerAudio");

let mediaRecorder = null;
let audioChunks = [];
let lastAnswerAudioURL = null;

function setStatus(text) {
    statusLabel.textContent = text;
}

function setError(text) {
    errorBox.textContent = text || "";
}

function addMessage(role, text) {
    const msg = document.createElement("div");
    msg.className = "msg " + (role === "user" ? "msg-user" : "msg-bot");

    const bubble = document.createElement("div");
    bubble.className = "bubble " + (role === "user" ? "bubble-user" : "bubble-bot");
    bubble.textContent = text;

    msg.appendChild(bubble);
    chatLog.appendChild(msg);
    chatLog.scrollTop = chatLog.scrollHeight;
}

async function startRecording() {
    setError("");

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setError("Браузер не поддерживает запись с микрофона.");
            return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];

        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            setStatus("Обработка записи...");
            recordBtn.disabled = false;
            stopBtn.disabled = true;

            const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
            await sendAudioToServer(blob);
        };

        mediaRecorder.start();
        setStatus("Запись... говорите");
        recordBtn.disabled = true;
        stopBtn.disabled = false;
    } catch (e) {
        console.error(e);
        setError("Ошибка доступа к микрофону: " + e.message);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        setStatus("Останавливаем запись...");
    }
}

async function sendAudioToServer(blob) {
    try {
        setError("");
        setStatus("Отправляем аудио Алине...");

        const formData = new FormData();
        // имя поля ДОЛЖНО быть "audio" — так ждёт FastAPI
        formData.append("audio", blob, "input.webm");

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const text = await response.text();
            throw new Error("Ошибка сервера: " + response.status + " " + text);
        }

        const data = await response.json();

        const transcript = data.transcript || "(распознавание не дало текста)";
        const answer = data.answer || "(нет ответа)";

        addMessage("user", transcript);
        addMessage("bot", answer);

        if (data.audio_base64) {
            const mime = data.audio_mime || "audio/mpeg";
            const audioBlob = base64ToBlob(data.audio_base64, mime);

            if (lastAnswerAudioURL) {
                URL.revokeObjectURL(lastAnswerAudioURL);
            }

            lastAnswerAudioURL = URL.createObjectURL(audioBlob);
            answerAudio.src = lastAnswerAudioURL;
            playBtn.disabled = false;

            // Можно сразу авто-плей:
            // answerAudio.play().catch(() => {});
        } else {
            playBtn.disabled = true;
            answerAudio.removeAttribute("src");
        }

        setStatus("Готово. Можно записывать новый запрос.");
    } catch (e) {
        console.error(e);
        setError(e.message || "Ошибка при запросе к Алине");
        setStatus("Ошибка");
    }
}

function base64ToBlob(base64, mimeType) {
    const byteChars = atob(base64);
    const byteNumbers = new Array(byteChars.length);

    for (let i = 0; i < byteChars.length; i++) {
        byteNumbers[i] = byteChars.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

recordBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);
playBtn.addEventListener("click", () => {
    if (answerAudio.src) {
        answerAudio.play().catch((e) => {
            console.error(e);
            setError("Не удалось проиграть аудио: " + e.message);
        });
    }
});
