let mediaRecorder = null;
let chunks = [];
const recordBtn = document.getElementById("recordBtn");
const log = document.getElementById("log");

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

  chunks = [];

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) {
      chunks.push(e.data);
    }
  };

  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: "audio/webm" });
    log.innerText += "▶ Sending audio to server...\n";

    const formData = new FormData();
    formData.append("file", blob, "recording.webm");

    const response = await fetch("http://127.0.0.1:8000/api/demo1/voice", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      log.innerText += `❌ Error: ${response.status}\n`;
      return;
    }

    const audioData = await response.arrayBuffer();
    const audioBlob = new Blob([audioData], { type: "audio/mpeg" });
    const audioUrl = URL.createObjectURL(audioBlob);

    const audio = new Audio(audioUrl);
    audio.play();

    log.innerText += "✅ Reply audio received and playing.\n";
  };

  mediaRecorder.start();
  log.innerText += "🎙 Recording...\n";
}

recordBtn.onmousedown = () => {
  recordBtn.innerText = "Recording... Release to send";
  startRecording().catch(err => {
    console.error(err);
    log.innerText += `Error: ${err}\n`;
  });
};

recordBtn.onmouseup = () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    recordBtn.innerText = "🎤 Hold to record";
  }
};
