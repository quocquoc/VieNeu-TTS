# AI Voice Assistant Demo - Step-by-Step Guide

## Architecture Overview

```
[Client Browser] --HTTPS/WSS--> [Cloudflare Tunnel] --> [FastAPI Server on RTX 5080]
                                                                |
                                                        1. Qwen3-ASR (Speech-to-Text)
                                                                |
                                                        2. Gemini API (LLM Response)
                                                                |
                                                        3. VieNeu-TTS merged model (Text-to-Speech)
                                                                |
[Client Browser] <--HTTPS/WSS-- [Cloudflare Tunnel] <-- [WAV audio response]
```

**VRAM budget (RTX 3090 Ti = 24 GB):**

| Model | Est. VRAM |
|---|---|
| Qwen3-ASR-0.6B (bf16) | ~1.5 GB |
| VieNeu-TTS-0.3B merged (LMDeploy, bf16) | ~1.2 GB |
| NeuCodec decoder (Triton-compiled) | ~0.3 GB |
| LMDeploy KV cache + runtime | ~3 GB |
| **Total** | **~6 GB** (18 GB free for headroom) |

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

# 4. VieNeu-TTS with GPU-optimized inference (LMDeploy)
pip install -q transformers peft torch librosa soundfile tqdm sea-g2p
pip install -q git+https://github.com/Neuphonic/NeuCodec.git
pip install -q lmdeploy triton

# 5. Audio processing
pip install -q numpy scipy
```

---

## Part 2: Upload the Fine-Tuned TTS Model

You need the VieNeu-TTS repo and your fine-tuned merged model on the server.

### Step 1: Clone the VieNeu-TTS repo on the server

```bash
cd /workspace
git clone https://github.com/quocquoc/VieNeu-TTS VieNeu-TTS-repo
cd VieNeu-TTS-repo
```

### Step 2: Upload the merged model from your local machine

The merged model file is `merged_model.tar.gz` (~431 MB). From your **local machine**, upload it to the server:

```bash
# From your local machine (adjust SSH port and host for your server)
scp -P <PORT> finetune/finetune_results/merged_model.tar.gz root@<HOST>:/workspace/VieNeu-TTS-repo/
```

### Step 3: Extract the model on the server

```bash
cd /workspace/VieNeu-TTS-repo

# Extract — this creates finetune/output/merged_model/ with model files
tar -xzf merged_model.tar.gz

# Verify the model files are in place
ls finetune/output/merged_model/
# Expected: config.json  generation_config.json  model.safetensors  tokenizer.json  tokenizer_config.json

# Clean up the archive to save disk space
rm merged_model.tar.gz
```

---

## Part 3: Get Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key
3. Set it as environment variable on your server:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

## Part 4: Verify Models Load

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
tts = Vieneu(mode="fast", backbone_repo="finetune/output/merged_model")
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

## Part 5: Create the Backend Server

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

print("Loading VieNeu-TTS merged model (LMDeploy fast mode)...")
from vieneu import Vieneu
tts_model = Vieneu(
    mode="fast",
    backbone_repo="finetune/output/merged_model",
    memory_util=0.4,
    enable_prefix_caching=True,
    enable_triton=True,
)
print("TTS model loaded.")

# Check VRAM
mem_gb = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM used: {mem_gb:.1f} GB / 24 GB")

# ============================================================
# 2. Gemini API Setup
# ============================================================

from google import genai

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """You are Mobi, a friendly English speaking coach. You speak like a real person in a voice conversation — warm, natural, and brief. Never use bullet points, lists, tables, or any text formatting.

---

RESPONSE STYLE
Keep every response short — ideally 2 to 4 sentences. Say one thing at a time, then wait. Never pile up explanations, questions, or feedback in one message.

LANGUAGE
Communicate in natural Vietnamese. Use English only for examples, target vocabulary, and speaking practice. Switch naturally as needed.

---

FLOW

1. INTENT CHECK
If the user has not expressed intent to learn English, ask once: "Bạn có muốn luyện nói tiếng Anh cùng Mobi không?"
Do not begin teaching until they say yes.

2. INTRODUCTION (after they agree)
Say exactly: "Tuyệt! Mình sẽ kiểm tra trình độ, chọn chủ đề, học từ vựng, rồi luyện nói từng bước. Bạn đang ở trình độ nào — mới bắt đầu, trung cấp, hay nâng cao?"
Wait for their answer.

3. TOPIC SELECTION
After level is confirmed, ask: "Bạn muốn luyện chủ đề gì hôm nay?"
Wait for their choice.

4. VOCABULARY (5–7 words)
Introduce one word at a time. Wait for acknowledgment before moving to the next.
For each word, say: the English word → simple Vietnamese meaning → one short example sentence. No symbols. No parentheses. Speak like a teacher, not a textbook.

5. SPEAKING PRACTICE
Ask one question in English suited to their CEFR level (A1–C2). Then wait.

---

EVALUATING ANSWERS

Step 1 — Language Check
If the answer contains significant Vietnamese, do not evaluate it.
Simply say: "Bạn thử trả lời bằng tiếng Anh nhé. Câu hỏi là..." then repeat the question. Stop there.
Note: grammar mistakes and awkward phrasing still count as English — evaluate those normally.

Step 2 — Feedback (English answers only)
Give feedback in one or two sentences maximum.
- Correct and natural → brief praise + slightly harder follow-up question.
- Grammar error → gently correct it, explain briefly in Vietnamese, ask them to try again.
- Off-topic → note it kindly, repeat the original question.
- No answer → give a simple model sentence, encourage them to repeat it.

---

TONE
Always encouraging. Never overwhelm with too much at once. One step, one message, one question."""

conversation_history = {}


def chat_with_gemini(user_text: str, session_id: str) -> str:
    """Send text to Gemini and get response."""
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    conversation_history[session_id].append({
        "role": "user",
        "parts": [{"text": user_text}]
    })

    # Keep last 10 turns to avoid token limit
    history = conversation_history[session_id][-10:]

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=history,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
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

def audio_chunk_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert a numpy audio chunk to WAV bytes."""
    if audio.ndim > 1:
        audio = audio.squeeze()
    buffer = BytesIO()
    sf.write(buffer, audio, 24000, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def text_to_speech(text: str) -> bytes:
    """Convert text to speech WAV bytes using VieNeu-TTS (non-streaming fallback)."""
    audio = tts_model.infer(text)
    return audio_chunk_to_wav_bytes(audio)


def text_to_speech_stream(text: str):
    """Stream TTS audio chunks for faster time-to-first-audio."""
    for audio_chunk in tts_model.infer_stream(text):
        yield audio_chunk_to_wav_bytes(audio_chunk)


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

                # Step 3: TTS (streaming — sends audio chunks as they're generated)
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Đang tạo giọng nói..."
                }))
                chunk_idx = 0
                for audio_chunk in text_to_speech_stream(assistant_text):
                    audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
                    await websocket.send_text(json.dumps({
                        "type": "audio_chunk",
                        "data": audio_b64,
                        "index": chunk_idx,
                    }))
                    chunk_idx += 1
                # Signal end of audio stream
                await websocket.send_text(json.dumps({
                    "type": "audio_end"
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

## Part 6: Create the Frontend

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
        const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${wsProtocol}//${location.host}/ws/${sessionId}`;
        let ws = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        // Streaming audio playback queue
        let audioQueue = [];
        let isPlaying = false;

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
                    case "audio_chunk":
                        queueAudioChunk(msg.data);
                        break;
                    case "audio_end":
                        // All chunks received; queue will drain naturally
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

        function queueAudioChunk(base64Data) {
            audioQueue.push(base64Data);
            if (!isPlaying) playNextChunk();
        }

        function playNextChunk() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                return;
            }
            isPlaying = true;
            const base64Data = audioQueue.shift();
            const bytes = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
            const blob = new Blob([bytes], { type: "audio/wav" });
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.play();
            audio.onended = () => {
                URL.revokeObjectURL(url);
                playNextChunk();
            };
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

## Part 7: Run the Server

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
VRAM used: ~3.5 GB / 24 GB
Uvicorn running on http://0.0.0.0:8000
```

---

## Part 8: Expose the App with Cloudflare Tunnel (Public HTTPS)

Cloudflare Tunnel gives you a **public HTTPS URL** for free — no domain, no port forwarding, no firewall rules. This also enables **microphone access** in the browser (which requires HTTPS on non-localhost origins).

### Step 1: Install `cloudflared` on the server

```bash
# Download the latest cloudflared binary
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# Verify installation
cloudflared --version
```

### Step 2: Start the tunnel

In a **separate terminal** (or use `tmux`/`screen`), run:

```bash
cloudflared tunnel --url http://localhost:8000
```

After a few seconds you will see output like:

```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://random-name-1234.trycloudflare.com                                                |
+--------------------------------------------------------------------------------------------+
```

Copy the `https://....trycloudflare.com` URL — this is your **public address**.

### Step 3: Share with clients

Send the URL to anyone. They can open it in any browser (desktop or mobile) and use the voice assistant immediately. No installation needed on the client side.

> **Tip:** The quick tunnel URL changes every time you restart `cloudflared`. For a **permanent subdomain**, see the "Persistent Tunnel" section below.

### (Optional) Run both server and tunnel in one terminal with tmux

```bash
# Install tmux if not available
apt-get install -y tmux

# Start a tmux session
tmux new -s demo

# Pane 1: Start the server
cd /workspace/VieNeu-TTS-repo
export GEMINI_API_KEY="your-api-key-here"
python -m uvicorn demo_server:app --host 0.0.0.0 --port 8000

# Press Ctrl+B then % to split pane

# Pane 2: Start the tunnel
cloudflared tunnel --url http://localhost:8000

# Press Ctrl+B then D to detach (keeps both running)
# Re-attach later with: tmux attach -t demo
```

### (Optional) Persistent Tunnel with Custom Domain

If you want a **fixed URL** that doesn't change between restarts:

```bash
# 1. Authenticate with your Cloudflare account (one-time)
cloudflared tunnel login

# 2. Create a named tunnel
cloudflared tunnel create vieneu-assistant

# 3. Route your domain/subdomain to the tunnel
#    (requires a domain managed by Cloudflare)
cloudflared tunnel route dns vieneu-assistant assistant.yourdomain.com

# 4. Run the tunnel
cloudflared tunnel run --url http://localhost:8000 vieneu-assistant
```

Now `https://assistant.yourdomain.com` will always point to your server.

---

### Alternative Access Methods

#### Option A: Vast.ai Port Forwarding (LAN / testing only)

1. In Vast.ai dashboard, find your instance
2. Look for the **Open Ports** section, map port `8000`
3. Access via the provided URL: `http://<vast-host>:<mapped-port>/`

> **Note:** Microphone will NOT work over plain HTTP from a non-localhost origin. Use Cloudflare Tunnel or SSH tunnel instead.

#### Option B: SSH Tunnel (local access only)

From your local machine:

```bash
ssh -L 8000:localhost:8000 -p <PORT> root@<HOST>
```

Then open: `http://localhost:8000/` in your browser.

---

## Part 9: Usage

1. Open the web page in your browser
2. Click the microphone button (or **hold Space**)
3. Speak in Vietnamese
4. Release the button / Space
5. Wait for the pipeline: ASR -> Gemini -> TTS
6. The assistant's voice plays automatically

---

## Part 10: Customize the Assistant

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
| Microphone not working | Browser requires HTTPS or localhost for mic access. Use Cloudflare Tunnel (provides HTTPS) or SSH tunnel |
| `cloudflared` connection refused | Make sure uvicorn is running on port 8000 before starting the tunnel |
| Tunnel URL not reachable | Wait 10-15 seconds after starting. If still down, restart `cloudflared` |
| WebSocket fails over tunnel | Cloudflare supports WebSocket by default. Check browser console for errors |
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
# === UPLOAD MERGED MODEL (from local machine) ===
scp -P <PORT> finetune/finetune_results/merged_model.tar.gz root@<HOST>:/workspace/VieNeu-TTS-repo/

# === EXTRACT MODEL (on server) ===
cd /workspace/VieNeu-TTS-repo
tar -xzf merged_model.tar.gz && rm merged_model.tar.gz

# === INSTALL ===
pip install -q fastapi uvicorn websockets python-multipart
pip install -U qwen-asr
pip install -U google-genai
pip install -q lmdeploy triton

# === SET API KEY ===
export GEMINI_API_KEY="your-key"

# === RUN SERVER ===
cd /workspace/VieNeu-TTS-repo
python -m uvicorn demo_server:app --host 0.0.0.0 --port 8000

# === INSTALL CLOUDFLARE TUNNEL ===
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# === EXPOSE TO PUBLIC (in another terminal) ===
cloudflared tunnel --url http://localhost:8000
# -> Share the https://....trycloudflare.com URL with clients

# === ALTERNATIVE: LOCAL ACCESS ONLY ===
# Via SSH tunnel: ssh -L 8000:localhost:8000 -p <PORT> root@<HOST>
# Then open: http://localhost:8000/
```
