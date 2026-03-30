# demo3_server.py - AI Voice Assistant Backend (Optimized for RTX 5080 Blackwell)
#
# Key optimizations over demo2_server.py:
#   1. torch.set_float32_matmul_precision('high') - TF32 on Tensor Cores
#   2. VieNeu-TTS mode="fast" (LMDeploy TurbomindEngine) - 2-3x speedup
#   3. faster-qwen3-tts with CUDA graphs - 5-6x speedup for Qwen3-TTS
#   4. vLLM backend for Qwen3-ASR - 2-3x throughput
#   5. SDPA / FlashAttention 2 attention implementations
#   6. Warmup all compiled models at startup

import os
import sys
import json
import time
import base64
import asyncio
import tempfile
import numpy as np
import soundfile as sf
from io import BytesIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ============================================================
# 0. Blackwell / CUDA Optimizations (BEFORE any model loading)
# ============================================================

# vLLM needs 'spawn' multiprocessing to avoid "Cannot re-initialize CUDA in forked subprocess"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Force Triton attention backend — vLLM's bundled flash-attn PTX doesn't support Blackwell (SM 12.0)
# Triton JIT-compiles for the current GPU, so it works on any architecture
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

import torch
torch.set_float32_matmul_precision('high')  # Enable TF32 on Tensor Cores

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Detect GPU compute capability
GPU_SM = (0, 0)
if torch.cuda.is_available():
    GPU_SM = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, SM {GPU_SM[0]}.{GPU_SM[1]}")

# Detect if FlashAttention 2 is available
ATTN_IMPL = "sdpa"  # safe default
try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
    print(f"FlashAttention 2 detected (v{flash_attn.__version__}), using flash_attention_2")
except ImportError:
    print("FlashAttention 2 not installed, using SDPA (still fast)")

# Check if LMDeploy supports this GPU
# lmdeploy 0.12.1 does NOT have CUDA kernels for SM 100+ (Blackwell RTX 5080/5090)
# It will crash with "no kernel image is available" — skip it entirely on Blackwell
LMDEPLOY_SUPPORTED = GPU_SM[0] < 10  # SM 80 (Ampere), 89 (Ada), 90 (Hopper) are supported
if not LMDEPLOY_SUPPORTED:
    print(f"LMDeploy skipped: no SM {GPU_SM[0]}{GPU_SM[1]} kernel support in lmdeploy 0.12.x")

# ============================================================
# 1. Load Models (optimized for RTX 5080)
# ============================================================

load_start = time.time()

# --- ASR Model (vLLM backend if available, otherwise standard) ---
print("Loading Qwen3-ASR-0.6B...")
from qwen_asr import Qwen3ASRModel

USE_VLLM_ASR = False
try:
    asr_model = Qwen3ASRModel.LLM(
        "Qwen/Qwen3-ASR-0.6B",
        gpu_memory_utilization=0.15,
        max_inference_batch_size=16,
        max_new_tokens=256,
    )
    USE_VLLM_ASR = True
    print("ASR model loaded (vLLM backend - optimized).")
except Exception as e:
    print(f"vLLM ASR backend not available ({e}), falling back to standard...")
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=256,
    )
    print("ASR model loaded (standard backend).")

# --- VieNeu-TTS (LMDeploy fast mode if supported, otherwise standard with torch.compile) ---
from vieneu import Vieneu

USE_FAST_VIENEU = False
if LMDEPLOY_SUPPORTED:
    print("Loading VieNeu-TTS merged model (LMDeploy fast mode)...")
    try:
        vieneu_tts = Vieneu(
            mode="fast",
            backbone_repo="finetune/output/merged_model",
            backbone_device="cuda",
            codec_device="cuda",
            memory_util=0.2,
            quant_policy=8,              # INT8 KV cache quantization
            enable_prefix_caching=True,
        )
        USE_FAST_VIENEU = True
        print("VieNeu-TTS model loaded (LMDeploy fast mode - optimized).")
    except Exception as e:
        print(f"LMDeploy failed ({e}), falling back to standard mode...")

if not USE_FAST_VIENEU:
    print("Loading VieNeu-TTS merged model (standard mode + torch.compile)...")
    vieneu_tts = Vieneu(
        mode="standard",
        backbone_repo="finetune/output/merged_model",
        backbone_device="cuda",
        codec_device="cuda",
    )
    print("VieNeu-TTS model loaded (standard mode).")

VIENEU_REF_AUDIO = "finetune/output/vi_2554.wav"
VIENEU_REF_TEXT = "Con cừu rất lớn, chúng ta hãy cùng học tên của con vật này nhé."
print(f"VieNeu-TTS will use Vietnamese reference: {VIENEU_REF_AUDIO}")

# --- VieNeu-TTS 0.5B base model (for "dual2" mode) ---
USE_FAST_VIENEU_05B = False

if LMDEPLOY_SUPPORTED:
    print("Loading VieNeu-TTS-0.5B base model (LMDeploy fast mode)...")
    try:
        vieneu_tts_05b = Vieneu(
            mode="fast",
            backbone_repo="pnnbao-ump/VieNeu-TTS",
            backbone_device="cuda",
            codec_device="cuda",
            memory_util=0.2,
            quant_policy=8,
            enable_prefix_caching=True,
        )
        USE_FAST_VIENEU_05B = True
        print("VieNeu-TTS-0.5B base model loaded (LMDeploy fast mode).")
    except Exception as e:
        print(f"LMDeploy failed for 0.5B ({e}), falling back to standard mode...")

if not USE_FAST_VIENEU_05B:
    print("Loading VieNeu-TTS-0.5B base model (standard mode + torch.compile)...")
    vieneu_tts_05b = Vieneu(
        mode="standard",
        backbone_repo="pnnbao-ump/VieNeu-TTS",
        backbone_device="cuda",
        codec_device="cuda",
    )
    print("VieNeu-TTS-0.5B base model loaded (standard mode).")

# --- Qwen3-TTS with CUDA Graphs (5-6x faster) ---
print("Loading Qwen3-TTS-12Hz-0.6B-Base...")
USE_FASTER_QWEN_TTS = False
try:
    from faster_qwen3_tts import FasterQwen3TTS
    qwen_tts = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    USE_FASTER_QWEN_TTS = True
    print("Qwen3-TTS loaded (faster-qwen3-tts with CUDA graphs - optimized).")
except Exception as e:
    print(f"faster-qwen3-tts not available ({e}), falling back to standard...")
    from qwen_tts import Qwen3TTSModel
    qwen_tts = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
    )
    print(f"Qwen3-TTS loaded (standard, attn={ATTN_IMPL}).")

# Reference audio for Qwen3-TTS voice cloning (English)
QWEN_TTS_REF_AUDIO = "finetune/output/en_0176.wav"
QWEN_TTS_REF_TEXT = "The tiger woke up early every morning and walked all the way to the zoo."

# Check VRAM
mem_gb = torch.cuda.memory_allocated() / 1024**3
load_elapsed = time.time() - load_start
print(f"VRAM used: {mem_gb:.1f} GB | Models loaded in {load_elapsed:.1f}s")

# ============================================================
# Startup diagnostic + warmup
# ============================================================
print("Running TTS startup tests (also warms up compiled models)...")

# Test VieNeu-TTS and save output as reference audio for Qwen3-TTS
try:
    t0 = time.time()
    test_audio_vi = vieneu_tts.infer(
        "Xin chao, day la bai kiem tra.",
        ref_audio=VIENEU_REF_AUDIO,
        ref_text=VIENEU_REF_TEXT,
        temperature=0.8,
        silence_p=0.05,
    )
    t_vi = time.time() - t0
    duration_vi = len(test_audio_vi) / 24000
    rtf_vi = t_vi / duration_vi if duration_vi > 0 else 0
    print(f"  VieNeu-TTS test: {len(test_audio_vi)} samples, {duration_vi:.1f}s audio, {t_vi:.1f}s wall, RTF={rtf_vi:.2f}")
    sf.write("test_startup_vieneu.wav", test_audio_vi, 24000)
    QWEN_TTS_REF_AUDIO = "test_startup_vieneu.wav"
    print("  Saved test_startup_vieneu.wav (also used as Qwen3-TTS reference audio)")
except Exception as e:
    print(f"  VieNeu-TTS startup test FAILED: {e}")
    import traceback; traceback.print_exc()

# Test Qwen3-TTS
try:
    t0 = time.time()
    # FasterQwen3TTS uses xvec_only; standard Qwen3TTSModel uses x_vector_only_mode
    clone_kwargs = dict(
        text="Hello, this is a test.",
        language="English",
        ref_audio=QWEN_TTS_REF_AUDIO,
        ref_text=QWEN_TTS_REF_TEXT,
    )
    if USE_FASTER_QWEN_TTS:
        clone_kwargs["xvec_only"] = True
    else:
        clone_kwargs["x_vector_only_mode"] = True
    test_wavs, test_sr = qwen_tts.generate_voice_clone(**clone_kwargs)
    t_en = time.time() - t0
    duration_en = len(test_wavs[0]) / test_sr
    rtf_en = t_en / duration_en if duration_en > 0 else 0
    print(f"  Qwen3-TTS test: {len(test_wavs[0])} samples, sr={test_sr}, {duration_en:.1f}s audio, {t_en:.1f}s wall, RTF={rtf_en:.2f}")
    sf.write("test_startup_qwen_tts.wav", test_wavs[0], test_sr)
    print("  Saved test_startup_qwen_tts.wav")
except Exception as e:
    print(f"  Qwen3-TTS startup test FAILED: {e}")
    import traceback; traceback.print_exc()

# Print optimization summary
print("\n" + "=" * 60)
print("OPTIMIZATION SUMMARY")
print("=" * 60)
print(f"  TF32 matmul precision:  enabled")
print(f"  Attention impl:         {ATTN_IMPL}")
print(f"  ASR backend:            {'vLLM (optimized)' if USE_VLLM_ASR else 'standard'}")
print(f"  VieNeu-TTS engine:      {'LMDeploy fast' if USE_FAST_VIENEU else 'standard'}")
print(f"  VieNeu-TTS-0.5B engine: {'LMDeploy fast' if USE_FAST_VIENEU_05B else 'standard'}")
print(f"  Qwen3-TTS engine:       {'faster-qwen3-tts (CUDA graphs)' if USE_FASTER_QWEN_TTS else 'standard'}")
print(f"  VRAM usage:             {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print("=" * 60 + "\n")


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
# 4. TTS: Text-to-Speech (optimized engines)
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
    t0 = time.time()
    audio = vieneu_tts.infer(
        text,
        ref_audio=VIENEU_REF_AUDIO,
        ref_text=VIENEU_REF_TEXT,
        temperature=0.8,
        silence_p=0.0,
    )
    elapsed = time.time() - t0
    duration = len(audio) / 24000
    rtf = elapsed / duration if duration > 0 else 0
    print(f"[VieNeu-TTS] Done: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f})")
    sf.write("last_tts_vieneu.wav", audio, 24000)
    return audio_to_wav_bytes(audio, 24000)


def tts_vieneu_05b(text: str) -> bytes:
    """Generate speech with VieNeu-TTS 0.5B base model (24kHz)."""
    print(f"[VieNeu-TTS-0.5B] Generating: {text[:80]}...")
    t0 = time.time()
    audio = vieneu_tts_05b.infer(
        text,
        ref_audio=VIENEU_REF_AUDIO,
        ref_text=VIENEU_REF_TEXT,
        temperature=0.8,
        silence_p=0.0,
    )
    elapsed = time.time() - t0
    duration = len(audio) / 24000
    rtf = elapsed / duration if duration > 0 else 0
    print(f"[VieNeu-TTS-0.5B] Done: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f})")
    sf.write("last_tts_vieneu_05b.wav", audio, 24000)
    return audio_to_wav_bytes(audio, 24000)


def tts_qwen_en(text: str) -> bytes:
    """Generate English speech with Qwen3-TTS."""
    print(f"[Qwen3-TTS] Generating: {text[:80]}...")
    t0 = time.time()

    # FasterQwen3TTS uses xvec_only; standard Qwen3TTSModel uses x_vector_only_mode
    clone_kwargs = dict(
        text=text,
        language="English",
        ref_audio=QWEN_TTS_REF_AUDIO,
        ref_text=QWEN_TTS_REF_TEXT,
    )
    if USE_FASTER_QWEN_TTS:
        clone_kwargs["xvec_only"] = True
    else:
        clone_kwargs["x_vector_only_mode"] = True
    wavs, sr = qwen_tts.generate_voice_clone(**clone_kwargs)
    audio = wavs[0]
    elapsed = time.time() - t0
    duration = len(audio) / sr
    rtf = elapsed / duration if duration > 0 else 0
    print(f"[Qwen3-TTS] Done: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f})")
    sf.write("last_tts_qwen.wav", audio, sr)
    return audio_to_wav_bytes(audio, sr)


# ============================================================
# 5. FastAPI App
# ============================================================

app = FastAPI(title="AI Voice Assistant v3 (Optimized)")


@app.get("/")
async def index():
    return HTMLResponse(open("demo3_frontend.html", "r").read())


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


@app.get("/stats")
async def stats():
    """Return optimization stats and VRAM usage."""
    return {
        "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "gpu_name": torch.cuda.get_device_name(0),
        "optimizations": {
            "tf32_matmul": True,
            "attention": ATTN_IMPL,
            "asr_backend": "vLLM" if USE_VLLM_ASR else "standard",
            "vieneu_engine": "LMDeploy fast" if USE_FAST_VIENEU else "standard",
            "vieneu_05b_engine": "LMDeploy fast" if USE_FAST_VIENEU_05B else "standard",
            "qwen_tts_engine": "faster-qwen3-tts (CUDA graphs)" if USE_FASTER_QWEN_TTS else "standard",
        },
    }


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
                timings = {}

                # Step 1: ASR
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Dang nhan dien giong noi..."
                }))
                t0 = time.time()
                user_text = await loop.run_in_executor(None, speech_to_text, audio_bytes)
                timings["asr"] = round(time.time() - t0, 2)

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
                t0 = time.time()
                assistant_data = await loop.run_in_executor(
                    None, chat_with_gemini, user_text, session_id
                )
                timings["llm"] = round(time.time() - t0, 2)

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
                    vi_tts_fn = tts_vieneu_05b if tts_mode == "dual2" else tts_vieneu
                    vi_tts_label = "VieNeu-TTS-0.5B" if tts_mode == "dual2" else "VieNeu-TTS merged"

                    if vi_text:
                        await websocket.send_text(json.dumps({
                            "type": "status", "text": f"Dang tao giong noi tieng Viet ({vi_tts_label})..."
                        }))
                        try:
                            t0 = time.time()
                            vi_wav = await loop.run_in_executor(None, vi_tts_fn, vi_text)
                            timings["tts_vi"] = round(time.time() - t0, 2)
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

                    if en_text:
                        await websocket.send_text(json.dumps({
                            "type": "status", "text": "Generating English voice..."
                        }))
                        try:
                            t0 = time.time()
                            en_wav = await loop.run_in_executor(None, tts_qwen_en, en_text)
                            timings["tts_en"] = round(time.time() - t0, 2)
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
                    if vi_text:
                        await websocket.send_text(json.dumps({
                            "type": "status", "text": "Dang tao giong noi..."
                        }))
                        try:
                            t0 = time.time()
                            wav = await loop.run_in_executor(None, tts_vieneu, vi_text)
                            timings["tts_vi"] = round(time.time() - t0, 2)
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

                # Send timing info to frontend
                total = sum(timings.values())
                timings["total"] = round(total, 2)
                await websocket.send_text(json.dumps({
                    "type": "timings", **timings
                }))
                print(f"[Timings] {timings}")

            elif message["type"] == "reset":
                conversation_history.pop(session_id, None)
                await websocket.send_text(json.dumps({
                    "type": "status", "text": "Da xoa lich su hoi thoai."
                }))

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        conversation_history.pop(session_id, None)
