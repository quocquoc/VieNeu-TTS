# AI Voice Assistant Demo - Step-by-Step Guide

## Architecture Overview

```
[Browser Microphone] --> WebSocket --> [FastAPI Server on RTX 5080]
                                            |
                                    1. Qwen3-ASR (Speech-to-Text)
                                            |
                                    2. Gemini API (LLM Response)
                                            |
                                    3. VieNeu-TTS merged model (Text-to-Speech)
                                            |
[Browser Speaker]   <-- WebSocket <-- [WAV audio response]
```

**VRAM budget (RTX 5080 = 16 GB):**

| Model | Est. VRAM |
|---|---|
| Qwen3-ASR-0.6B (bf16) | ~1.5 GB |
| VieNeu-TTS-0.3B merged (bf16) | ~1.2 GB |
| NeuCodec decoder | ~0.3 GB |
| Runtime overhead | ~2 GB |
| **Total** | **~5 GB** (plenty of headroom) |

---

## Part 1: Install Dependencies

SSH into your Vast.ai server (or use Jupyter Terminal):

```bash
cd /workspace

# 1. Core dependencies
pip install -q fastapi uvicorn websockets python-multipart

# 2. Qwen3-ASR
pip install -U qwen-asr

# 3. Gemini API client
pip install -U google-genai

# 4. VieNeu-TTS (already installed if you followed fine-tune guide)
pip install -q transformers peft torch librosa soundfile tqdm sea-g2p
pip install -q git+https://github.com/Neuphonic/NeuCodec.git

# 5. Audio processing
pip install -q numpy scipy
```

---

## Part 2: Get Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key
3. Set it as environment variable on your server:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

## Part 3: Verify Models Load

Test that both models fit in VRAM together:

```bash
cd /workspace/VieNeu-TTS-repo
```

```python
# test_models.py
import torch

# Test 1: Load ASR
print("Loading Qwen3-ASR...")
from qwen_asr import Qwen3ASRModel
asr = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
print("ASR loaded OK")

# Test 2: Load TTS
print("Loading VieNeu-TTS...")
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from vieneu import Vieneu
tts = Vieneu(backbone_repo="finetune/output/merged_model")
print("TTS loaded OK")

# Check VRAM
mem = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM used: {mem:.1f} GB")
```

```bash
python test_models.py
```

If this works, both models coexist. Delete `test_models.py` after verifying.

---

## Part 4: Create the Backend Server

Create the main server file:

```bash
nano /workspace/VieNeu-TTS-repo/demo_server.py
```

Paste the following:

```python
# demo_server.py - AI Voice Assistant Backend
import os
import sys
import json
import torch
import base64
import tempfile
import numpy as np
import soundfile as sf
from io import BytesIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ============================================================
# 1. Load Models (loaded once at startup)
# ============================================================

print("Loading Qwen3-ASR-0.6B...")
from qwen_asr import Qwen3ASRModel
asr_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_new_tokens=256,
)
print("ASR model loaded.")

print("Loading VieNeu-TTS merged model...")
from vieneu import Vieneu
tts_model = Vieneu(backbone_repo="finetune/output/merged_model")
print("TTS model loaded.")

# Check VRAM
mem_gb = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM used: {mem_gb:.1f} GB / 16 GB")

# ============================================================
# 2. Gemini API Setup
# ============================================================

from google import genai

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh, thân thiện, nói tiếng Việt.
Hãy trả lời ngắn gọn, tự nhiên như đang nói chuyện (1-3 câu).
Không dùng markdown, emoji, hay ký tự đặc biệt trong câu trả lời.
Không dùng dấu ngoặc kép hay gạch đầu dòng."""

conversation_history = {}


def chat_with_gemini(user_text: str, session_id: str) -> str:
    """Send text to Gemini and get response."""
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    conversation_history[session_id].append({
        "role": "user",
        "parts": [{"text": user_text}]
    })

    # Keep last 20 turns to avoid token limit
    history = conversation_history[session_id][-20:]

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=200,
        ),
    )

    assistant_text = response.text.strip()

    conversation_history[session_id].append({
        "role": "model",
        "parts": [{"text": assistant_text}]
    })

    return assistant_text


# ============================================================
# 3. ASR: Speech-to-Text
# ============================================================

def speech_to_text(audio_bytes: bytes) -> str:
    """Convert audio bytes (WAV) to text using Qwen3-ASR."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        results = asr_model.transcribe(audio=temp_path, language="Vietnamese")
        text = results[0].text if results else ""
        return text.strip()
    finally:
        os.unlink(temp_path)


# ============================================================
# 4. TTS: Text-to-Speech
# ============================================================

def text_to_speech(text: str) -> bytes:
    """Convert text to speech WAV bytes using VieNeu-TTS."""
    audio = tts_model.infer(text)

    # Convert to WAV bytes
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    if audio.ndim > 1:
        audio = audio.squeeze()

    buffer = BytesIO()
    sf.write(buffer, audio, 24000, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


# ============================================================
# 5. FastAPI App
# ============================================================

app = FastAPI(title="AI Voice Assistant")


@app.get("/")
async def index():
    return HTMLResponse(open("demo_frontend.html", "r").read())


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"Client connected: {session_id}")

    try:
        while True:
            # Receive audio from browser (base64-encoded WAV)
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "audio":
                audio_bytes = base64.b64decode(message["data"])

                # Step 1: ASR
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Đang nhận diện giọng nói..."
                }))
                user_text = speech_to_text(audio_bytes)

                if not user_text:
                    await websocket.send_text(json.dumps({
                        "type": "error", "text": "Không nhận diện được giọng nói."
                    }))
                    continue

                await websocket.send_text(json.dumps({
                    "type": "user_text", "text": user_text
                }))

                # Step 2: Gemini LLM
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Đang suy nghĩ..."
                }))
                assistant_text = chat_with_gemini(user_text, session_id)

                await websocket.send_text(json.dumps({
                    "type": "assistant_text", "text": assistant_text
                }))

                # Step 3: TTS
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Đang tạo giọng nói..."
                }))
                audio_wav = text_to_speech(assistant_text)
                audio_b64 = base64.b64encode(audio_wav).decode("utf-8")

                await websocket.send_text(json.dumps({
                    "type": "audio", "data": audio_b64
                }))

            elif message["type"] == "reset":
                conversation_history.pop(session_id, None)
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Đã xoá lịch sử hội thoại."
                }))

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        conversation_history.pop(session_id, None)
```

---

## Part 5: Create the Frontend

```bash
nano /workspace/VieNeu-TTS-repo/demo_frontend.html
```

Paste the following:

```html
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VieNeu AI Voice Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .header {
            padding: 30px 20px 10px;
            text-align: center;
        }
        .header h1 {
            font-size: 24px;
            color: #7dd3fc;
        }
        .header p {
            font-size: 14px;
            color: #888;
            margin-top: 6px;
        }
        .chat-container {
            flex: 1;
            width: 100%;
            max-width: 700px;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 16px;
            font-size: 15px;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            align-self: flex-end;
            background: #1e40af;
            color: #dbeafe;
            border-bottom-right-radius: 4px;
        }
        .message.assistant {
            align-self: flex-start;
            background: #1e293b;
            color: #e2e8f0;
            border-bottom-left-radius: 4px;
        }
        .message.status {
            align-self: center;
            background: transparent;
            color: #666;
            font-size: 13px;
            padding: 4px;
        }
        .message.error {
            align-self: center;
            color: #f87171;
            font-size: 13px;
        }
        .controls {
            padding: 20px;
            display: flex;
            gap: 12px;
            align-items: center;
            justify-content: center;
        }
        .record-btn {
            width: 72px;
            height: 72px;
            border-radius: 50%;
            border: 3px solid #334155;
            background: #1e293b;
            color: #7dd3fc;
            font-size: 28px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .record-btn:hover { border-color: #7dd3fc; }
        .record-btn.recording {
            background: #dc2626;
            border-color: #f87171;
            color: white;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.4); }
            50% { box-shadow: 0 0 0 16px rgba(220,38,38,0); }
        }
        .reset-btn {
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid #334155;
            background: #1e293b;
            color: #94a3b8;
            cursor: pointer;
            font-size: 14px;
        }
        .reset-btn:hover { border-color: #7dd3fc; color: #7dd3fc; }
        .footer {
            padding: 10px;
            color: #444;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>VieNeu AI Voice Assistant</h1>
        <p>Qwen3-ASR + Gemini + VieNeu-TTS</p>
    </div>

    <div class="chat-container" id="chat"></div>

    <div class="controls">
        <button class="reset-btn" onclick="resetChat()">Xoa lich su</button>
        <button class="record-btn" id="recordBtn" onclick="toggleRecording()">&#127908;</button>
    </div>

    <div class="footer">Hold button or press Space to talk</div>

    <script>
        const sessionId = crypto.randomUUID();
        const wsUrl = `ws://${location.host}/ws/${sessionId}`;
        let ws = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        function connect() {
            ws = new WebSocket(wsUrl);
            ws.onopen = () => addMessage("status", "Da ket noi.");
            ws.onclose = () => {
                addMessage("status", "Mat ket noi. Dang thu ket noi lai...");
                setTimeout(connect, 2000);
            };
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                switch (msg.type) {
                    case "user_text":
                        addMessage("user", msg.text);
                        break;
                    case "assistant_text":
                        addMessage("assistant", msg.text);
                        break;
                    case "status":
                        addMessage("status", msg.text);
                        break;
                    case "error":
                        addMessage("error", msg.text);
                        break;
                    case "audio":
                        playAudio(msg.data);
                        break;
                }
            };
        }

        function addMessage(type, text) {
            const chat = document.getElementById("chat");
            // Remove previous status messages
            if (type === "status") {
                chat.querySelectorAll(".message.status").forEach(el => el.remove());
            }
            const div = document.createElement("div");
            div.className = `message ${type}`;
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function playAudio(base64Data) {
            const bytes = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
            const blob = new Blob([bytes], { type: "audio/wav" });
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.play();
            audio.onended = () => URL.revokeObjectURL(url);
        }

        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: { sampleRate: 16000, channelCount: 1 }
                });
                audioChunks = [];
                mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) audioChunks.push(e.data);
                };

                mediaRecorder.onstop = async () => {
                    stream.getTracks().forEach(t => t.stop());
                    const webmBlob = new Blob(audioChunks, { type: "audio/webm" });
                    const wavBytes = await convertToWav(webmBlob);
                    const b64 = arrayBufferToBase64(wavBytes);
                    ws.send(JSON.stringify({ type: "audio", data: b64 }));
                };

                mediaRecorder.start();
                isRecording = true;
                document.getElementById("recordBtn").classList.add("recording");
                document.getElementById("recordBtn").innerHTML = "&#9632;";
            } catch (err) {
                addMessage("error", "Khong the truy cap microphone: " + err.message);
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
            }
            isRecording = false;
            document.getElementById("recordBtn").classList.remove("recording");
            document.getElementById("recordBtn").innerHTML = "&#127908;";
        }

        // Convert WebM to WAV using AudioContext
        async function convertToWav(webmBlob) {
            const arrayBuffer = await webmBlob.arrayBuffer();
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            audioCtx.close();

            const numSamples = audioBuffer.length;
            const sampleRate = audioBuffer.sampleRate;
            const buffer = new ArrayBuffer(44 + numSamples * 2);
            const view = new DataView(buffer);
            const channelData = audioBuffer.getChannelData(0);

            // WAV header
            writeString(view, 0, "RIFF");
            view.setUint32(4, 36 + numSamples * 2, true);
            writeString(view, 8, "WAVE");
            writeString(view, 12, "fmt ");
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(view, 36, "data");
            view.setUint32(40, numSamples * 2, true);

            // PCM samples
            for (let i = 0; i < numSamples; i++) {
                let s = Math.max(-1, Math.min(1, channelData[i]));
                view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            return buffer;
        }

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        function arrayBufferToBase64(buffer) {
            let binary = "";
            const bytes = new Uint8Array(buffer);
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            return btoa(binary);
        }

        function resetChat() {
            ws.send(JSON.stringify({ type: "reset" }));
            document.getElementById("chat").innerHTML = "";
        }

        // Spacebar to record
        document.addEventListener("keydown", (e) => {
            if (e.code === "Space" && !e.repeat && !isRecording) {
                e.preventDefault();
                startRecording();
            }
        });
        document.addEventListener("keyup", (e) => {
            if (e.code === "Space" && isRecording) {
                e.preventDefault();
                stopRecording();
            }
        });

        connect();
    </script>
</body>
</html>
```

---

## Part 6: Run the Server

```bash
cd /workspace/VieNeu-TTS-repo

# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Start the server
python -m uvicorn demo_server:app --host 0.0.0.0 --port 8000
```

**Expected startup output:**
```
Loading Qwen3-ASR-0.6B...
ASR model loaded.
Loading VieNeu-TTS merged model...
TTS model loaded.
VRAM used: ~3.5 GB / 16 GB
Uvicorn running on http://0.0.0.0:8000
```

---

## Part 7: Access the Demo

### Option A: Vast.ai Port Forwarding

1. In Vast.ai dashboard, find your instance
2. Look for the **Open Ports** section, map port `8000`
3. Access via the provided URL: `http://<vast-host>:<mapped-port>/`

### Option B: SSH Tunnel (more secure)

From your local machine:

```bash
ssh -L 8000:localhost:8000 -p <PORT> root@<HOST>
```

Then open: `http://localhost:8000/` in your browser.

---

## Part 8: Usage

1. Open the web page in your browser
2. Click the microphone button (or **hold Space**)
3. Speak in Vietnamese
4. Release the button / Space
5. Wait for the pipeline: ASR -> Gemini -> TTS
6. The assistant's voice plays automatically

---

## Part 9: Customize the Assistant

### Change the System Prompt

Edit `SYSTEM_PROMPT` in `demo_server.py`:

```python
SYSTEM_PROMPT = """Ban la tro ly cua cong ty ABC.
Tra loi ngan gon, than thien.
Chi tra loi cac cau hoi lien quan den san pham cua cong ty."""
```

### Use a Different Gemini Model

Change the model in `chat_with_gemini()`:

```python
model="gemini-2.5-pro"      # smarter, slower
model="gemini-2.5-flash"    # fast, default
```

### Use the Larger ASR Model (better accuracy)

If VRAM allows, switch to the 1.7B model:

```python
asr_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",  # ~4 GB VRAM instead of ~1.5 GB
    ...
)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GEMINI_API_KEY` not found | Run `export GEMINI_API_KEY="your-key"` before starting |
| Microphone not working | Browser requires HTTPS or localhost for mic access. Use SSH tunnel |
| CUDA OOM when both models load | Use Qwen3-ASR-0.6B (smaller). Check no other processes use GPU: `nvidia-smi` |
| Audio not playing in browser | Check browser allows autoplay. Try clicking the page first |
| WebSocket disconnects | Check Vast.ai port mapping. Ensure port 8000 is open |
| `qwen-asr` import error | Run `pip install -U qwen-asr` |
| `google.genai` import error | Run `pip install -U google-genai` |
| ASR returns empty text | Ensure you speak clearly and audio is at least 1 second |
| Slow response | Pipeline has 3 sequential steps. Gemini API latency is the main variable |

---

## Quick Reference: All Commands

```bash
# === INSTALL ===
pip install -q fastapi uvicorn websockets python-multipart
pip install -U qwen-asr
pip install -U google-genai

# === SET API KEY ===
export GEMINI_API_KEY="your-key"

# === RUN ===
cd /workspace/VieNeu-TTS-repo
python -m uvicorn demo_server:app --host 0.0.0.0 --port 8000

# === ACCESS ===
# Via SSH tunnel: ssh -L 8000:localhost:8000 -p <PORT> root@<HOST>
# Then open: http://localhost:8000/
```
