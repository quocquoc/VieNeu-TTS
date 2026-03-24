# demo_server.py - AI Voice Assistant Backend
import os
import sys
import json
import torch
import base64
import asyncio
import tempfile
import numpy as np
import soundfile as sf
from io import BytesIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

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

# print("Loading VieNeu-TTS merged model...")
print("Loading VieNeu-TTS base model + LoRA adapter...")
from vieneu import Vieneu
tts_model = Vieneu(
    mode="standard",
    backbone_repo="finetune/output/merged_model",
    # backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",
    backbone_device="cuda",
    codec_device="cuda",
)
# print("TTS model loaded.")

# If you want to apply LoRA adapter for your custom voice
# tts_model.load_lora_adapter("finetune/output/VieNeu-TTS-0.3B-LoRA")
# print("TTS model + LoRA loaded.")

# Load the default preset voice (ref_codes + ref_text) once at startup
try:
    default_voice = tts_model.get_preset_voice()
    print(f"Default voice loaded OK")
except Exception as e:
    print(f"WARNING: No preset voice found: {e}")
    print("Falling back to encoding reference audio directly...")
    ref_codes = tts_model.encode_reference("finetune/output/en_0006.wav")
    ref_text = "My teacher Tuan told us an amazing story about a brave little rabbit in the forest."
    default_voice = {"codes": ref_codes, "text": ref_text}
    print("Reference voice encoded OK")

# Check VRAM
mem_gb = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM used: {mem_gb:.1f} GB / 24 GB")

# ============================================================
# Startup diagnostic: generate a test audio and save to disk
# ============================================================
print("Running TTS startup test...")
try:
    test_audio = tts_model.infer(
        "Xin chào, đây là bài kiểm tra.",
        voice=default_voice,
        temperature=0.8,
    )
    print(f"  Test audio: {len(test_audio)} samples, {len(test_audio)/24000:.1f}s")
    sf.write("test_startup.wav", test_audio, 24000)
    print("  Saved test_startup.wav — play this file to verify TTS quality")
    if len(test_audio) / 24000 > 10:
        print("  WARNING: Audio too long for a short sentence! Model may not be generating properly.")
except Exception as e:
    print(f"  TTS startup test FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 2. Gemini API Setup
# ============================================================

from google import genai
from google.genai import types

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
            thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
            ),
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
    """Convert text to speech WAV bytes using VieNeu-TTS."""
    print(f"TTS generating for: {text[:80]}...")
    audio = tts_model.infer(text, voice=default_voice, temperature=0.8)
    duration = len(audio) / 24000
    print(f"TTS done: {len(audio)} samples, {duration:.1f}s")
    # Save latest TTS output to disk for debugging
    sf.write("last_tts_output.wav", audio, 24000)
    return audio_chunk_to_wav_bytes(audio)


# ============================================================
# 5. FastAPI App
# ============================================================

app = FastAPI(title="AI Voice Assistant")


@app.get("/")
async def index():
    return HTMLResponse(open("demo_frontend.html", "r").read())


@app.get("/test")
async def test_tts():
    """Direct TTS test — visit /test in browser to download a WAV file."""
    from fastapi.responses import Response
    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(
        None, text_to_speech, "Xin chào, đây là bài kiểm tra giọng nói."
    )
    return Response(content=wav_bytes, media_type="audio/wav")


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

                loop = asyncio.get_event_loop()

                # Step 1: ASR (run in thread to not block event loop)
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Đang nhận diện giọng nói..."
                }))
                user_text = await loop.run_in_executor(None, speech_to_text, audio_bytes)

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
                assistant_text = await loop.run_in_executor(
                    None, chat_with_gemini, user_text, session_id
                )

                await websocket.send_text(json.dumps({
                    "type": "assistant_text", "text": assistant_text
                }))

                # Step 3: TTS (run in thread to not block event loop)
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Đang tạo giọng nói..."
                }))
                try:
                    audio_wav = await loop.run_in_executor(
                        None, text_to_speech, assistant_text
                    )
                    audio_b64 = base64.b64encode(audio_wav).decode("utf-8")
                    await websocket.send_text(json.dumps({
                        "type": "audio", "data": audio_b64
                    }))
                except Exception as e:
                    print(f"TTS ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_text(json.dumps({
                        "type": "error", "text": f"Lỗi TTS: {str(e)}"
                    }))

            elif message["type"] == "reset":
                conversation_history.pop(session_id, None)
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Đã xoá lịch sử hội thoại."
                }))

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        conversation_history.pop(session_id, None)
