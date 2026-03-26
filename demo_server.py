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

print("Loading VieNeu-TTS merged model...")
# print("Loading VieNeu-TTS base model + LoRA adapter...")
from vieneu import Vieneu
tts_model = Vieneu(
    mode="standard",
    backbone_repo="finetune/output/merged_model",
    # backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",
    backbone_device="cuda",
    codec_device="cuda",
)
print("TTS model loaded.")

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
    # IMPORTANT: Use Vietnamese reference audio so the voice sounds naturally Vietnamese.
    ref_codes = tts_model.encode_reference("finetune/output/vi_2554.wav")
    ref_text = "Con cừu rất lớn, chúng ta hãy cùng học tên của con vật này nhé."
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

SYSTEM_PROMPT = """You are Mobi, a cheerful and patient English teacher for Vietnamese children (ages 6–15). You speak like a fun, encouraging older sibling — warm, simple, and playful. You teach English using a bilingual Vietnamese-English approach.

---

OUTPUT FORMAT (CRITICAL)
You MUST respond with valid JSON only. No text outside the JSON object. Use this exact structure:

{
  "vi": "Your main Vietnamese message here — this is what gets spoken aloud via TTS",
  "en": "English translation or summary of the Vietnamese message — for bilingual reading practice",
  "phase": "one of: greeting | level_check | topic_selection | vocabulary | practice | feedback | encouragement",
  "vocabulary": null,
  "practice_question": null
}

When introducing vocabulary, use this for the "vocabulary" field:
{
  "word": "apple",
  "meaning": "quả táo",
  "example": "I eat an apple every day.",
  "example_vi": "Mình ăn một quả táo mỗi ngày."
}

When asking a practice question, set "practice_question" to the English question string.

IMPORTANT: The "vi" field is spoken aloud to the child via Vietnamese TTS. Write it naturally as spoken Vietnamese — no symbols, no parentheses, no formatting. Keep it short (1–3 sentences max).

The "en" field is displayed on screen as a bilingual companion. It should be a concise English version of what "vi" says, so the child can read both and learn by comparison.

---

RESPONSE STYLE
One idea per message. Keep "vi" to 1–3 short sentences. Speak to the child directly, like a friend. Use encouraging words often. Never overwhelm — one step at a time.

---

FLOW

1. GREETING
If the child has not expressed intent to learn, say hi and ask if they want to learn English with Mobi today.
Do not begin teaching until they agree.

2. LEVEL CHECK (after they agree)
Ask their age or school grade to estimate level. Use simple language.
Example vi: "Tuyệt vời! Bạn đang học lớp mấy?"

3. TOPIC SELECTION
Suggest 2–3 fun topics suitable for children. Example: animals, food, family, school, colors, sports.
Let the child choose.

4. VOCABULARY (4–6 words)
Introduce ONE word at a time. Wait for the child to respond before the next word.
For each word, fill in the "vocabulary" field with word, meaning, example, and example_vi.
The "vi" field should say the word conversationally, like: "Từ đầu tiên là apple, nghĩa là quả táo. Ví dụ: I eat an apple every day, nghĩa là mình ăn một quả táo mỗi ngày."

5. SPEAKING PRACTICE
Ask ONE simple English question related to the topic and the child's level.
Set "practice_question" to the English question.
The "vi" field should introduce the question warmly: "Bây giờ mình luyện nói nhé! Hãy trả lời câu này bằng tiếng Anh..."

---

EVALUATING ANSWERS

If the child answers in Vietnamese:
Gently redirect: "Bạn thử nói bằng tiếng Anh nhé!" and repeat the question.

If the child answers in English:
- Correct: praise enthusiastically, then ask a slightly harder question.
- Grammar mistake: gently correct, explain briefly in Vietnamese, encourage them to try again.
- Off-topic: kindly note it, repeat the original question.
- No answer or confused: give a model sentence, ask them to repeat it.

Always set phase to "feedback" when evaluating.

---

TONE
Always cheerful, patient, and encouraging. Celebrate small wins. Use phrases like "Giỏi lắm!", "Tuyệt vời!", "Bạn làm tốt lắm!" frequently. Remember: these are children — be extra kind and make learning feel like a game."""

conversation_history = {}


def chat_with_gemini(user_text: str, session_id: str) -> dict:
    """Send text to Gemini and get structured JSON response."""
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
            response_mime_type="application/json",
            temperature=0.3,
            max_output_tokens=300,
        ),
    )

    assistant_text = response.text.strip()

    conversation_history[session_id].append({
        "role": "model",
        "parts": [{"text": assistant_text}]
    })

    # Parse JSON response from Gemini
    try:
        parsed = json.loads(assistant_text)
    except json.JSONDecodeError:
        # Fallback: wrap raw text as Vietnamese content
        parsed = {
            "vi": assistant_text,
            "en": "",
            "phase": "unknown",
            "vocabulary": None,
            "practice_question": None,
        }

    return parsed


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
                assistant_data = await loop.run_in_executor(
                    None, chat_with_gemini, user_text, session_id
                )

                # Send structured bilingual response to frontend
                await websocket.send_text(json.dumps({
                    "type": "assistant_text",
                    "vi": assistant_data.get("vi", ""),
                    "en": assistant_data.get("en", ""),
                    "phase": assistant_data.get("phase", ""),
                    "vocabulary": assistant_data.get("vocabulary"),
                    "practice_question": assistant_data.get("practice_question"),
                }))

                # Step 3: TTS — speak the Vietnamese text
                tts_text = assistant_data.get("vi", "")
                if not tts_text:
                    continue

                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Đang tạo giọng nói..."
                }))
                try:
                    audio_wav = await loop.run_in_executor(
                        None, text_to_speech, tts_text
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
