# AI Voice Assistant Demo v2 - Dual TTS Mode Guide

## Architecture Overview

```
[Client Browser] --HTTPS/WSS--> [Cloudflare Tunnel] --> [FastAPI Server on GPU]
                                                                |
                                                        1. Qwen3-ASR (Speech-to-Text)
                                                                |
                                                        2. Gemini API (LLM Response -> JSON with "vi" + "en")
                                                                |
                                                        3. TTS (two modes):
                                                           Mode A: VieNeu-TTS merged model (vi + en combined)
                                                           Mode B: VieNeu-TTS (vi) + Qwen3-TTS (en) separately
                                                                |
[Client Browser] <--HTTPS/WSS-- [Cloudflare Tunnel] <-- [WAV audio response(s)]
```

### TTS Modes

| Mode | TTS Engine(s) | How it works |
|---|---|---|
| **Single TTS** | VieNeu-TTS-0.3B merged model | Generates Vietnamese audio only (EN shown on screen) |
| **Dual TTS** | VieNeu-TTS-0.3B merged (VI) + Qwen3-TTS (EN) | Vietnamese audio first, then English audio |
| **Dual TTS 2** | VieNeu-TTS-0.5B base (VI) + Qwen3-TTS (EN) | Same as Dual TTS but uses the 0.5B base model for comparison |

### VRAM Budget (RTX 3090 Ti = 24 GB)

| Model | Est. VRAM |
|---|---|
| Qwen3-ASR-0.6B (bf16) | ~1.5 GB |
| VieNeu-TTS-0.3B merged (bf16) | ~1.2 GB |
| VieNeu-TTS-0.5B base (bf16) | ~1.5 GB |
| NeuCodec decoder (shared) | ~0.3 GB |
| Qwen3-TTS-12Hz-0.6B-Base (bf16) | ~1.5 GB |
| KV cache + runtime | ~3 GB |
| **Total** | **~9.0 GB** (15 GB free for headroom) |

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

# 4. VieNeu-TTS dependencies
pip install -q transformers peft librosa soundfile tqdm sea-g2p

# 4a. Install torch + torchaudio for CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# If error with torchaudio:
# pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install transformers==4.57.6

pip install -q git+https://github.com/Neuphonic/NeuCodec.git

# 5. Qwen3-TTS
pip install -U qwen-tts

# Optional: FlashAttention 2 for faster Qwen3-TTS inference
# SKIP this if you get CUDA version mismatch errors — it is NOT required
# pip install -U flash-attn --no-build-isolation

# 6. Audio processing
pip install -q numpy scipy
```

---

## Part 2: Upload the Fine-Tuned TTS Model

Same as demo v1 — you need the VieNeu-TTS repo and your fine-tuned merged model on the server.

### Step 1: Clone the VieNeu-TTS repo on the server

```bash
cd /workspace
git clone https://github.com/quocquoc/VieNeu-TTS VieNeu-TTS-repo
cd VieNeu-TTS-repo
```

### Step 2: Upload the merged model from your local machine

```bash
# From your local machine
scp -P <PORT> finetune/finetune_results/merged_model.tar.gz root@<HOST>:/workspace/VieNeu-TTS-repo/
```

### Step 3: Re-merge the model with correct precision (bfloat16)

```bash
cd /workspace/VieNeu-TTS-repo

# Re-merge with bfloat16 precision
python finetune/merge_lora.py \
  --base_model pnnbao-ump/VieNeu-TTS-0.3B \
  --adapter finetune/output/VieNeu-TTS-0.3B-LoRA \
  --output finetune/output/merged_model

# Create voices.json for the merged model
# IMPORTANT: Use a VIETNAMESE reference audio so the voice sounds naturally Vietnamese.
# Using English reference audio will make the model sound like an American speaking Vietnamese.
python finetune/create_voices_json.py \
  --audio finetune/output/vi_2554.wav \
  --text "Con cừu rất lớn, chúng ta hãy cùng học tên của con vật này nhé." \
  --name my_custom_voice \
  --description "My fine-tuned Vietnamese voice" \
  --output finetune/output/merged_model/voices.json

# Verify all files are in place
ls finetune/output/merged_model/
# Expected: config.json  generation_config.json  model.safetensors  tokenizer.json  tokenizer_config.json  voices.json

rm merged_model.tar.gz
```

Or copy available merged_model to server and extract
# === EXTRACT MODEL (on server) ===
cd /workspace/VieNeu-TTS-repo
tar -xzf merged_model.tar.gz && rm merged_model.tar.gz
---

## Part 3: Get Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key
3. Set it as environment variable on your server:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

## Part 4: Verify All Models Load Together

Test that all three models fit in VRAM:

```bash
cd /workspace/VieNeu-TTS-repo
```

```python
# test_models_v2.py
import torch
import soundfile as sf

# Test 1: Load ASR
print("Loading Qwen3-ASR...")
from qwen_asr import Qwen3ASRModel
asr = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
print("ASR loaded OK")

# Test 2: Load VieNeu-TTS (base model from HuggingFace)
print("Loading VieNeu-TTS base model...")
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from vieneu import Vieneu
vieneu_base = Vieneu(
    mode="standard",
    backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",
    backbone_device="cuda",
    codec_device="cuda",
)
print("VieNeu-TTS base loaded OK")

# Test 2a: Inference test with VieNeu-TTS base (voice cloning)
print("Testing VieNeu-TTS base inference with voice cloning...")
test_audio_base = vieneu_base.infer(
    "Xin chao, day la bai kiem tra giong noi.",
    ref_audio="finetune/output/vi_2554.wav",
    ref_text="Con cừu rất lớn, chúng ta hãy cùng học tên của con vật này nhé.",
    temperature=0.8,
)
sf.write("test_vieneu_base.wav", test_audio_base, 24000)
print(f"  VieNeu-TTS base: {len(test_audio_base)} samples, {len(test_audio_base)/24000:.1f}s -> test_vieneu_base.wav")

# Free base model VRAM before loading merged model
del vieneu_base
torch.cuda.empty_cache()

# Test 3: Load VieNeu-TTS (merged/fine-tuned model)
print("Loading VieNeu-TTS merged model...")
vieneu_tts = Vieneu(
    mode="standard",
    backbone_repo="finetune/output/merged_model",
    backbone_device="cuda",
    codec_device="cuda",
)
print("VieNeu-TTS merged loaded OK")

# Test 3a: Inference test with VieNeu-TTS merged
print("Testing VieNeu-TTS merged inference...")
try:
    merged_voice = vieneu_tts.get_preset_voice()
except Exception:
    ref_codes = vieneu_tts.encode_reference("finetune/output/vi_2554.wav")
    merged_voice = {"codes": ref_codes, "text": "Con cừu rất lớn, chúng ta hãy cùng học tên của con vật này nhé."}
test_audio_merged = vieneu_tts.infer(
    "Xin chao, day la bai kiem tra giong noi.",
    voice=merged_voice,
    temperature=0.8,
)
sf.write("test_vieneu_merged.wav", test_audio_merged, 24000)
print(f"  VieNeu-TTS merged: {len(test_audio_merged)} samples, {len(test_audio_merged)/24000:.1f}s -> test_vieneu_merged.wav")

# Test 4: Load Qwen3-TTS
print("Loading Qwen3-TTS...")
from qwen_tts import Qwen3TTSModel
qwen_tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
print("Qwen3-TTS loaded OK")

# Test 4a: Inference test with Qwen3-TTS
# Reuse the VieNeu-TTS merged output as reference audio for voice cloning
print("Testing Qwen3-TTS inference...")
print("  Using test_vieneu_merged.wav as reference audio for Qwen3-TTS voice cloning")
wavs, sr = qwen_tts.generate_voice_clone(
    text="Hello, this is a voice test.",
    language="English",
    ref_audio="finetune/output/en_0176.wav",
    ref_text="The tiger woke up early every morning and walked all the way to the zoo.",
)
sf.write("test_qwen_tts.wav", wavs[0], sr)
print(f"  Qwen3-TTS: {len(wavs[0])} samples, sr={sr}, {len(wavs[0])/sr:.1f}s -> test_qwen_tts.wav")

# Check VRAM
mem = torch.cuda.memory_allocated() / 1024**3
print(f"\nVRAM used: {mem:.1f} GB")
print("\nAll tests passed! Play the WAV files to verify audio quality:")
print("  - test_vieneu_base.wav   (VieNeu-TTS base model)")
print("  - test_vieneu_merged.wav (VieNeu-TTS fine-tuned model)")
print("  - test_qwen_tts.wav      (Qwen3-TTS English)")
```

```bash
python test_models_v2.py
```

Delete `test_models_v2.py` and test WAV files after verifying.

---

## Part 5: Prepare Reference Audio for Qwen3-TTS

Qwen3-TTS uses voice cloning with a reference audio clip. By default, the server **automatically generates** a VieNeu-TTS test clip at startup (`test_startup_vieneu.wav`) and uses it as the Qwen3-TTS reference audio. No manual WAV file needed.

To use a **custom** English reference voice instead, edit these lines in `demo2_server.py`:

```python
QWEN_TTS_REF_AUDIO = "finetune/output/en_0176.wav"
QWEN_TTS_REF_TEXT = "The tiger woke up early every morning and walked all the way to the zoo."
```

> **Tip:** Use a clear 3-10 second audio clip with its exact transcript for best voice cloning results.

---

## Part 6: Server and Frontend Files

The demo v2 consists of two files:

- **`demo2_server.py`** — FastAPI backend with ASR, Gemini, and dual TTS engines
- **`demo2_frontend.html`** — Browser UI with a mode toggle (Single TTS / Dual TTS)

Both files are already in the repo root. No additional setup needed.

### Key differences from demo v1:

| Feature | demo v1 | demo v2 |
|---|---|---|
| TTS models | VieNeu-TTS only | VieNeu-TTS + Qwen3-TTS |
| Audio output | Single audio (vi text only) | Single mode: vi+en combined. Dual mode: vi then en separately |
| Mode toggle | None | Frontend toggle between Single/Dual |
| Audio playback | Immediate | Queue-based (plays in order for dual mode) |

---

## Part 7: Run the Server

```bash
cd /workspace/VieNeu-TTS-repo

# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Start the server
python -m uvicorn demo2_server:app --host 0.0.0.0 --port 8000
```

**Expected startup output:**
```
Loading Qwen3-ASR-0.6B...
ASR model loaded.
Loading VieNeu-TTS merged model...
VieNeu-TTS model loaded.
VieNeu-TTS default voice loaded OK
Loading Qwen3-TTS-12Hz-0.6B-Base...
Qwen3-TTS model loaded.
VRAM used: ~4.5 GB
Running TTS startup tests...
  VieNeu-TTS test: ... samples, ...s
  Saved test_startup_vieneu.wav
  Qwen3-TTS test: ... samples, sr=12000, ...s
  Saved test_startup_qwen_tts.wav
Uvicorn running on http://0.0.0.0:8000
```

### Test endpoints

- `http://localhost:8000/` — Main web UI
- `http://localhost:8000/test` — Download a VieNeu-TTS test WAV
- `http://localhost:8000/test-en` — Download a Qwen3-TTS English test WAV

---

## Part 8: Expose the App with Cloudflare Tunnel

Same as demo v1.

### Step 1: Install `cloudflared` on the server

```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
cloudflared --version
```

### Step 2: Start the tunnel

In a **separate terminal** (or use `tmux`/`screen`):

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the `https://....trycloudflare.com` URL.

### (Optional) Run both in one terminal with tmux

```bash
apt-get install -y tmux
tmux new -s demo2

# Pane 1: Start the server
cd /workspace/VieNeu-TTS-repo
export GEMINI_API_KEY="your-api-key-here"
python -m uvicorn demo2_server:app --host 0.0.0.0 --port 8000

# Press Ctrl+B then % to split pane

# Pane 2: Start the tunnel
cloudflared tunnel --url http://localhost:8000

# Press Ctrl+B then D to detach
# Re-attach later with: tmux attach -t demo2
```

---

## Part 9: Usage

1. Open the web page in your browser
2. **Choose TTS mode** at the top:
   - **Single TTS**: VieNeu-TTS speaks both Vietnamese and English text combined in one audio
   - **Dual TTS**: VieNeu-TTS speaks Vietnamese first, then Qwen3-TTS speaks English second
3. Click the microphone button (or **hold Space**)
4. Speak in Vietnamese
5. Release the button / Space
6. Wait for the pipeline: ASR -> Gemini -> TTS
7. The assistant's voice plays automatically

---

## Part 10: Customize the Assistant

### Change the System Prompt

Edit `SYSTEM_PROMPT` in `demo2_server.py`.

### Use a Different Gemini Model

Change the model in `chat_with_gemini()`:

```python
model="gemini-2.5-pro"      # smarter, slower
model="gemini-2.5-flash"    # fast, default
```

### Change the Qwen3-TTS Reference Voice

Edit these variables at the top of `demo2_server.py`:

```python
QWEN_TTS_REF_AUDIO = "path/to/english_speaker.wav"
QWEN_TTS_REF_TEXT = "Transcript of the reference audio."
```

### Use the Smaller ASR Model (save VRAM)

```python
asr_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",  # already using the smaller one
    ...
)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GEMINI_API_KEY` not found | Run `export GEMINI_API_KEY="your-key"` before starting |
| Microphone not working | Browser requires HTTPS or localhost. Use Cloudflare Tunnel |
| `qwen-tts` import error | Run `pip install -U qwen-tts` |
| Qwen3-TTS returns empty audio | Check reference audio exists and transcript matches |
| Dual mode: English audio doesn't play | Check browser console. Ensure Qwen3-TTS loaded without errors |
| Audio plays out of order | The frontend uses a queue — wait for VI to finish before EN plays |
| CUDA OOM | Check `nvidia-smi`. Three models need ~7.5 GB. Free up GPU memory |
| `libcudart.so` / torchaudio error | CUDA mismatch. Reinstall: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| `cloudflared` connection refused | Make sure uvicorn is running on port 8000 before starting the tunnel |
| Slow response | Pipeline has 3-4 sequential steps. Dual mode adds an extra TTS call |

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
pip install -q transformers peft librosa soundfile tqdm sea-g2p
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -q git+https://github.com/Neuphonic/NeuCodec.git
pip install -U qwen-tts
# pip install -U flash-attn --no-build-isolation  # optional, skip if CUDA mismatch
pip install -q numpy scipy

# === SET API KEY ===
export GEMINI_API_KEY="your-key"

# === RUN SERVER ===
cd /workspace/VieNeu-TTS-repo
python -m uvicorn demo2_server:app --host 0.0.0.0 --port 8000

# === EXPOSE TO PUBLIC (in another terminal) ===
cloudflared tunnel --url http://localhost:8000
```
