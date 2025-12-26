// frontend/demo1.js

const ws = new WebSocket("ws://localhost:8000/ws/demo1");

const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const log = document.getElementById("log");

ws.onopen = () => {
  log.innerText += "Connected to /ws/demo1\n";
};

ws.onmessage = (event) => {
  const reply = event.data;
  log.innerText += `Assistant: ${reply}\n\n`;
};

ws.onclose = () => {
  log.innerText += "Disconnected.\n";
};

sendBtn.onclick = () => {
  const text = input.value.trim();
  if (!text) return;
  log.innerText += `You: ${text}\n`;
  ws.send(text);
  input.value = "";
};
