# demo2_server.py - AI Voice Assistant Backend (Dual TTS Mode)
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

# --- ASR Model ---
print("Loading Qwen3-ASR-0.6B...")
from qwen_asr import Qwen3ASRModel
asr_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_new_tokens=256,
)
print("ASR model loaded.")

# --- VieNeu-TTS (Vietnamese + bilingual single-model mode) ---
print("Loading VieNeu-TTS merged model...")
from vieneu import Vieneu
vieneu_tts = Vieneu(
    mode="standard",
    backbone_repo="finetune/output/merged_model",
    backbone_device="cuda",
    codec_device="cuda",
)
print("VieNeu-TTS model loaded.")

# IMPORTANT: Use Vietnamese reference audio directly for voice cloning.
# Do NOT use get_preset_voice() — the merged model's voices.json may contain
# English reference audio, which makes the model sound like an American speaking Vietnamese.
VIENEU_REF_AUDIO = "finetune/output/vi_2554.wav"
VIENEU_REF_TEXT = "Con cừu rất lớn, chúng ta hãy cùng học tên của con vật này nhé."
print(f"VieNeu-TTS will use Vietnamese reference: {VIENEU_REF_AUDIO}")

# --- VieNeu-TTS 0.5B base model (for "dual2" mode comparison) ---
print("Loading VieNeu-TTS-0.5B base model...")
vieneu_tts_05b = Vieneu(
    mode="standard",
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device="cuda",
    codec_device="cuda",
)
print("VieNeu-TTS-0.5B base model loaded.")

# --- Qwen3-TTS (English, for dual-model mode) ---
print("Loading Qwen3-TTS-12Hz-0.6B-Base...")
from qwen_tts import Qwen3TTSModel
qwen_tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",  # optional, requires flash-attn installed
)
print("Qwen3-TTS model loaded.")

# Reference audio for Qwen3-TTS voice cloning (English)
# Generate a short VieNeu-TTS clip at startup and use it as reference for Qwen3-TTS.
# This avoids hardcoding a WAV path that may not exist on the server.
# You can override these with your own English reference audio path + transcript.
QWEN_TTS_REF_AUDIO = "finetune/output/en_0176.wav"
QWEN_TTS_REF_TEXT = "The tiger woke up early every morning and walked all the way to the zoo."

# Check VRAM
mem_gb = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM used: {mem_gb:.1f} GB")

# ============================================================
# Startup diagnostic
# ============================================================
print("Running TTS startup tests...")

# Test VieNeu-TTS and save output as reference audio for Qwen3-TTS
try:
    test_audio_vi = vieneu_tts.infer(
        "Xin chao, day la bai kiem tra.",
        ref_audio=VIENEU_REF_AUDIO,
        ref_text=VIENEU_REF_TEXT,
        temperature=0.8,
        silence_p=0.05,
    )
    print(f"  VieNeu-TTS test: {len(test_audio_vi)} samples, {len(test_audio_vi)/24000:.1f}s")
    sf.write("test_startup_vieneu.wav", test_audio_vi, 24000)
    QWEN_TTS_REF_AUDIO = "test_startup_vieneu.wav"
    print("  Saved test_startup_vieneu.wav (also used as Qwen3-TTS reference audio)")
except Exception as e:
    print(f"  VieNeu-TTS startup test FAILED: {e}")
    import traceback; traceback.print_exc()

# Test Qwen3-TTS
try:
    test_wavs, test_sr = qwen_tts.generate_voice_clone(
        text="Hello, this is a test.",
        language="English",
        ref_audio=QWEN_TTS_REF_AUDIO,
        ref_text=QWEN_TTS_REF_TEXT,
        x_vector_only_mode=True,
    )
    print(f"  Qwen3-TTS test: {len(test_wavs[0])} samples, sr={test_sr}, {len(test_wavs[0])/test_sr:.1f}s")
    sf.write("test_startup_qwen_tts.wav", test_wavs[0], test_sr)
    print("  Saved test_startup_qwen_tts.wav")
except Exception as e:
    print(f"  Qwen3-TTS startup test FAILED: {e}")
    import traceback; traceback.print_exc()


# ============================================================
# 2. Gemini API Setup
# ============================================================

from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """You are Mobi, a cheerful and patient English teacher for Vietnamese children (ages 6-15). You speak like a fun, encouraging older sibling - warm, simple, and playful. You teach English using a bilingual Vietnamese-English approach.

---

OUTPUT FORMAT (CRITICAL)
You MUST respond with valid JSON only. No text outside the JSON object. Use this exact structure:

{
  "vi": "Your main Vietnamese message here - this is what gets spoken aloud via TTS",
  "en": "English translation or summary of the Vietnamese message - for bilingual reading practice",
  "phase": "one of: greeting | level_check | topic_selection | vocabulary | practice | feedback | encouragement",
  "vocabulary": null,
  "practice_question": null
}

When introducing vocabulary, use this for the "vocabulary" field:
{
  "word": "apple",
  "meaning": "qua tao",
  "example": "I eat an apple every day.",
  "example_vi": "Minh an mot qua tao moi ngay."
}

When asking a practice question, set "practice_question" to the English question string.

IMPORTANT: The "vi" field is spoken aloud to the child via Vietnamese TTS. Write it naturally as spoken Vietnamese - no symbols, no parentheses, no formatting. Keep it short (1-3 sentences max).

The "en" field is displayed on screen as a bilingual companion. It should be a concise English version of what "vi" says, so the child can read both and learn by comparison.

---

RESPONSE STYLE
One idea per message. Keep "vi" to 1-3 short sentences. Speak to the child directly, like a friend. Use encouraging words often. Never overwhelm - one step at a time.

---

FLOW

1. GREETING
If the child has not expressed intent to learn, say hi and ask if they want to learn English with Mobi today.
Do not begin teaching until they agree.

2. LEVEL CHECK (after they agree)
Ask their age or school grade to estimate level. Use simple language.
Example vi: "Tuyet voi! Ban dang hoc lop may?"

3. TOPIC SELECTION
Suggest 2-3 fun topics suitable for children. Example: animals, food, family, school, colors, sports.
Let the child choose.

4. VOCABULARY (4-6 words)
Introduce ONE word at a time. Wait for the child to respond before the next word.
For each word, fill in the "vocabulary" field with word, meaning, example, and example_vi.
The "vi" field should say the word conversationally.

5. SPEAKING PRACTICE
Ask ONE simple English question related to the topic and the child's level.
Set "practice_question" to the English question.

---

EVALUATING ANSWERS

If the child answers in Vietnamese:
Gently redirect: "Ban thu noi bang tieng Anh nhe!" and repeat the question.

If the child answers in English:
- Correct: praise enthusiastically, then ask a slightly harder question.
- Grammar mistake: gently correct, explain briefly in Vietnamese, encourage them to try again.
- Off-topic: kindly note it, repeat the original question.
- No answer or confused: give a model sentence, ask them to repeat it.

Always set phase to "feedback" when evaluating.

---

TONE
Always cheerful, patient, and encouraging. Celebrate small wins. Use phrases like "Gioi lam!", "Tuyet voi!", "Ban lam tot lam!" frequently. Remember: these are children - be extra kind and make learning feel like a game."""

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

    try:
        parsed = json.loads(assistant_text)
    except json.JSONDecodeError:
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
# 4. TTS: Text-to-Speech (both engines)
# ============================================================

def normalize_volume(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize audio peak volume to target_db (dBFS)."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    target_peak = 10 ** (target_db / 20.0)
    return audio * (target_peak / peak)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy audio array to WAV bytes, normalized to consistent volume."""
    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = normalize_volume(audio)
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def tts_vieneu(text: str) -> bytes:
    """Generate speech with VieNeu-TTS (24kHz). Works for both Vietnamese and English."""
    print(f"[VieNeu-TTS] Generating: {text[:80]}...")
    audio = vieneu_tts.infer(
        text,
        ref_audio=VIENEU_REF_AUDIO,
        ref_text=VIENEU_REF_TEXT,
        temperature=0.8,
        silence_p=0.0,   # no extra silence between chunks (model generates its own pauses)
    )
    duration = len(audio) / 24000
    print(f"[VieNeu-TTS] Done: {len(audio)} samples, {duration:.1f}s")
    sf.write("last_tts_vieneu.wav", audio, 24000)
    return audio_to_wav_bytes(audio, 24000)


def tts_vieneu_05b(text: str) -> bytes:
    """Generate speech with VieNeu-TTS 0.5B base model (24kHz)."""
    print(f"[VieNeu-TTS-0.5B] Generating: {text[:80]}...")
    audio = vieneu_tts_05b.infer(
        text,
        ref_audio=VIENEU_REF_AUDIO,
        ref_text=VIENEU_REF_TEXT,
        temperature=0.8,
        silence_p=0.0,
    )
    duration = len(audio) / 24000
    print(f"[VieNeu-TTS-0.5B] Done: {len(audio)} samples, {duration:.1f}s")
    sf.write("last_tts_vieneu_05b.wav", audio, 24000)
    return audio_to_wav_bytes(audio, 24000)


def tts_qwen_en(text: str) -> bytes:
    """Generate English speech with Qwen3-TTS (12kHz)."""
    print(f"[Qwen3-TTS] Generating: {text[:80]}...")

    # Use x_vector_only_mode=True to avoid reference audio bleeding into output.
    # In default mode (ICL), the library prepends reference codes to generated codes
    # for decoding, then tries to trim them — but the proportional trim is imprecise,
    # causing reference audio to leak (1st call) or first sentence to be cut (2nd+ call).
    # x_vector_only_mode uses only speaker embeddings, so output is clean target speech only.
    wavs, sr = qwen_tts.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=QWEN_TTS_REF_AUDIO,
        ref_text=QWEN_TTS_REF_TEXT,
        x_vector_only_mode=True,
    )
    audio = wavs[0]
    duration = len(audio) / sr
    print(f"[Qwen3-TTS] Done: {len(audio)} samples, sr={sr}, {duration:.1f}s")
    sf.write("last_tts_qwen.wav", audio, sr)
    return audio_to_wav_bytes(audio, sr)


# ============================================================
# 5. FastAPI App
# ============================================================

app = FastAPI(title="AI Voice Assistant v2")


@app.get("/")
async def index():
    return HTMLResponse(open("demo2_frontend.html", "r").read())


@app.get("/test")
async def test_tts():
    """Direct VieNeu-TTS test."""
    from fastapi.responses import Response
    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(
        None, tts_vieneu, "Xin chao, day la bai kiem tra giong noi."
    )
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/test-en")
async def test_tts_en():
    """Direct Qwen3-TTS English test."""
    from fastapi.responses import Response
    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(
        None, tts_qwen_en, "Hello, this is a voice test."
    )
    return Response(content=wav_bytes, media_type="audio/wav")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"Client connected: {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "audio":
                audio_bytes = base64.b64decode(message["data"])
                tts_mode = message.get("tts_mode", "single")  # "single", "dual", or "dual2"

                loop = asyncio.get_event_loop()

                # Step 1: ASR
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Dang nhan dien giong noi..."
                }))
                user_text = await loop.run_in_executor(None, speech_to_text, audio_bytes)

                if not user_text:
                    await websocket.send_text(json.dumps({
                        "type": "error", "text": "Khong nhan dien duoc giong noi."
                    }))
                    continue

                await websocket.send_text(json.dumps({
                    "type": "user_text", "text": user_text
                }))

                # Step 2: Gemini LLM
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Dang suy nghi..."
                }))
                assistant_data = await loop.run_in_executor(
                    None, chat_with_gemini, user_text, session_id
                )

                await websocket.send_text(json.dumps({
                    "type": "assistant_text",
                    "vi": assistant_data.get("vi", ""),
                    "en": assistant_data.get("en", ""),
                    "phase": assistant_data.get("phase", ""),
                    "vocabulary": assistant_data.get("vocabulary"),
                    "practice_question": assistant_data.get("practice_question"),
                }))

                vi_text = assistant_data.get("vi", "")
                en_text = assistant_data.get("en", "")

                # Step 3: TTS
                if tts_mode in ("dual", "dual2"):
                    # --- DUAL MODE ---
                    # dual  = merged_model (VieNeu-TTS 0.3B fine-tuned) + Qwen3-TTS
                    # dual2 = VieNeu-TTS 0.5B base model + Qwen3-TTS
                    vi_tts_fn = tts_vieneu_05b if tts_mode == "dual2" else tts_vieneu
                    vi_tts_label = "VieNeu-TTS-0.5B" if tts_mode == "dual2" else "VieNeu-TTS merged"

                    # Generate Vietnamese audio first
                    if vi_text:
                        await websocket.send_text(json.dumps({
                            "type": "status", "text": f"Dang tao giong noi tieng Viet ({vi_tts_label})..."
                        }))
                        try:
                            vi_wav = await loop.run_in_executor(None, vi_tts_fn, vi_text)
                            vi_b64 = base64.b64encode(vi_wav).decode("utf-8")
                            await websocket.send_text(json.dumps({
                                "type": "audio", "data": vi_b64, "lang": "vi"
                            }))
                        except Exception as e:
                            print(f"{vi_tts_label} ERROR: {e}")
                            import traceback; traceback.print_exc()
                            await websocket.send_text(json.dumps({
                                "type": "error", "text": f"Loi TTS tieng Viet: {str(e)}"
                            }))

                    # Then generate English audio
                    if en_text:
                        await websocket.send_text(json.dumps({
                            "type": "status", "text": "Generating English voice..."
                        }))
                        try:
                            en_wav = await loop.run_in_executor(None, tts_qwen_en, en_text)
                            en_b64 = base64.b64encode(en_wav).decode("utf-8")
                            await websocket.send_text(json.dumps({
                                "type": "audio", "data": en_b64, "lang": "en"
                            }))
                        except Exception as e:
                            print(f"Qwen3-TTS ERROR: {e}")
                            import traceback; traceback.print_exc()
                            await websocket.send_text(json.dumps({
                                "type": "error", "text": f"English TTS error: {str(e)}"
                            }))

                else:
                    # --- SINGLE MODE: VieNeu-TTS for Vietnamese only ---
                    # VieNeu-TTS is a Vietnamese model and cannot pronounce English well.
                    # Only speak the Vietnamese text; English is shown on screen for reading.
                    if vi_text:
                        await websocket.send_text(json.dumps({
                            "type": "status", "text": "Dang tao giong noi..."
                        }))
                        try:
                            wav = await loop.run_in_executor(None, tts_vieneu, vi_text)
                            b64 = base64.b64encode(wav).decode("utf-8")
                            await websocket.send_text(json.dumps({
                                "type": "audio", "data": b64
                            }))
                        except Exception as e:
                            print(f"TTS ERROR: {e}")
                            import traceback; traceback.print_exc()
                            await websocket.send_text(json.dumps({
                                "type": "error", "text": f"Loi TTS: {str(e)}"
                            }))

            elif message["type"] == "reset":
                conversation_history.pop(session_id, None)
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Da xoa lich su hoi thoai."
                }))

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        conversation_history.pop(session_id, None)
