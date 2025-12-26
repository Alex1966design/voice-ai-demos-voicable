// frontend/demo2.js

const ws2 = new WebSocket("ws://localhost:8000/ws/demo2");

const input2 = document.getElementById("input");
const sendBtn2 = document.getElementById("send");
const log2 = document.getElementById("log");

ws2.onopen = () => {
  log2.innerText += "Connected to /ws/demo2\n";
};

ws2.onmessage = (event) => {
  log2.innerText += `Server: ${event.data}\n`;
};

ws2.onclose = () => {
  log2.innerText += "Disconnected.\n";
};

sendBtn2.onclick = () => {
  const text = input2.value.trim();
  if (!text) return;
  log2.innerText += `You: ${text}\n`;
  ws2.send(text);
  input2.value = "";
};
