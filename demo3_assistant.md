# AI Voice Assistant Demo v3 - Optimized for RTX 5080 Blackwell

## What's New in v3 (vs v2)

| Optimization | v2 (baseline) | v3 (optimized) | Speedup |
|---|---|---|---|
| VieNeu-TTS engine | HuggingFace `generate()` | LMDeploy TurbomindEngine + INT8 KV cache | **2-3x** |
| Qwen3-TTS engine | Standard `qwen-tts` | `faster-qwen3-tts` with CUDA graphs | **5-6x** |
| Qwen3-ASR backend | Standard transformers | vLLM with PagedAttention | **2-3x** |
| Float32 matmul | Default precision | TF32 on Tensor Cores | 5-15% |
| Attention | Default | SDPA / FlashAttention 2 auto-detect | 10-30% |
| Pipeline timing | No visibility | Per-step timing displayed in UI | -- |
| Graceful fallback | Hard fail | Auto-fallback if optimized lib missing | -- |

All optimizations auto-detect: if a library (LMDeploy, faster-qwen3-tts, vLLM, flash-attn) is not installed, demo3 falls back to the standard engine automatically.

## Architecture Overview

```
[Client Browser] --HTTPS/WSS--> [Cloudflare Tunnel] --> [FastAPI Server on RTX 5080]
                                                                |
                                                        1. Qwen3-ASR (vLLM backend)
                                                                |
                                                        2. Gemini API (LLM -> JSON with "vi" + "en")
                                                                |
                                                        3. TTS (optimized engines):
                                                           VieNeu-TTS: LMDeploy TurbomindEngine
                                                           Qwen3-TTS: faster-qwen3-tts (CUDA graphs)
                                                                |
[Client Browser] <--HTTPS/WSS-- [Cloudflare Tunnel] <-- [WAV audio + timing stats]
```

### TTS Modes

| Mode | TTS Engine(s) | How it works |
|---|---|---|
| **Single TTS** | VieNeu-TTS-0.3B merged (LMDeploy) | Vietnamese audio only (EN shown on screen) |
| **Dual TTS** | VieNeu-TTS-0.3B merged (LMDeploy) + Qwen3-TTS (CUDA graphs) | Vietnamese first, then English |
| **Dual TTS 2** | VieNeu-TTS-0.5B base (LMDeploy) + Qwen3-TTS (CUDA graphs) | Same but uses 0.5B base model |

### VRAM Budget (RTX 5080 = 16 GB)

| Model | Est. VRAM |
|---|---|
| Qwen3-ASR-0.6B (vLLM, bf16) | ~1.5 GB |
| VieNeu-TTS-0.3B merged (LMDeploy) | ~1.2 GB |
| VieNeu-TTS-0.5B base (LMDeploy) | ~1.5 GB |
| NeuCodec decoder (shared) | ~0.3 GB |
| Qwen3-TTS (faster-qwen3-tts) | ~1.5 GB |
| KV cache + CUDA graphs | ~3 GB |
| **Total** | **~9.0 GB** (7 GB free) |

---

## Part 1: Install Dependencies

SSH into your server (RTX 5080 GPU):

```bash
cd /workspace

# 1. Core dependencies
pip install -q fastapi uvicorn websockets python-multipart

# 2. PyTorch with CUDA 12.8 (REQUIRED for RTX 5080 Blackwell)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify Blackwell is detected
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'CUDA: {torch.version.cuda}'); print(f'Compute cap: {torch.cuda.get_device_capability(0)}')"
# Expected: NVIDIA GeForce RTX 5080, CUDA: 12.8, Compute cap: (10, 0)

# 3. Qwen3-ASR + vLLM backend (2-3x faster ASR)
pip install -U qwen-asr
pip install vllm --pre "vllm[audio]"

# 4. Gemini API client
pip install -U google-genai

# 5. VieNeu-TTS dependencies
pip install -q transformers peft librosa soundfile tqdm sea-g2p
pip install transformers==4.57.6
pip install -q git+https://github.com/Neuphonic/NeuCodec.git

# 6. LMDeploy (2-3x faster VieNeu-TTS inference)
pip install lmdeploy

# 7. faster-qwen3-tts (5-6x faster Qwen3-TTS with CUDA graphs)
pip install faster-qwen3-tts

# 8. FlashAttention 2 for Blackwell (optional but recommended)
# Pre-built wheel for RTX 5080:
pip install flash-attn==2.7.4.post1
# If the above fails, try building from source:
# pip install flash-attn --no-build-isolation
# Or skip — demo3 auto-falls back to SDPA which is also fast

# 9. Audio processing
pip install -q numpy scipy
```

### Minimal install (without optimized packages)

If you can't install LMDeploy, vLLM, or faster-qwen3-tts, demo3 still works — it falls back to standard engines automatically:

```bash
# Minimal install (same as demo2)
pip install -q fastapi uvicorn websockets python-multipart
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U qwen-asr google-genai qwen-tts
pip install -q transformers==4.57.6 peft librosa soundfile tqdm sea-g2p
pip install -q git+https://github.com/Neuphonic/NeuCodec.git
pip install -q numpy scipy
```

---

## Part 2: Upload the Fine-Tuned TTS Model

Same as demo v2 — you need the VieNeu-TTS repo and your fine-tuned merged model on the server.

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

### Step 3: Extract (or re-merge) the model

```bash
cd /workspace/VieNeu-TTS-repo

# Option A: Extract pre-built merged model
tar -xzf merged_model.tar.gz && rm merged_model.tar.gz

# Option B: Re-merge with bfloat16 precision
python finetune/merge_lora.py \
  --base_model pnnbao-ump/VieNeu-TTS-0.3B \
  --adapter finetune/output/VieNeu-TTS-0.3B-LoRA \
  --output finetune/output/merged_model

# Create voices.json (use Vietnamese reference audio)
python finetune/create_voices_json.py \
  --audio finetune/output/vi_2554.wav \
  --text "Con cừu rất lớn, chúng ta hãy cùng học tên của con vật này nhé." \
  --name my_custom_voice \
  --description "My fine-tuned Vietnamese voice" \
  --output finetune/output/merged_model/voices.json

# Verify
ls finetune/output/merged_model/
```

---

## Part 3: Get Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key
3. Set it as environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

## Part 4: Run the Optimized Server

```bash
cd /workspace/VieNeu-TTS-repo

export GEMINI_API_KEY="your-api-key-here"

python -m uvicorn demo3_server:app --host 0.0.0.0 --port 8000
```

**Expected startup output:**
```
FlashAttention 2 detected (v2.7.4.post1), using flash_attention_2
Loading Qwen3-ASR-0.6B...
ASR model loaded (vLLM backend - optimized).
Loading VieNeu-TTS merged model (LMDeploy fast mode)...
VieNeu-TTS model loaded (LMDeploy fast mode - optimized).
Loading VieNeu-TTS-0.5B base model (LMDeploy fast mode)...
VieNeu-TTS-0.5B base model loaded (LMDeploy fast mode).
Loading Qwen3-TTS-12Hz-0.6B-Base...
Qwen3-TTS loaded (faster-qwen3-tts with CUDA graphs - optimized).
VRAM used: ~5.5 GB | Models loaded in 45.2s
Running TTS startup tests (also warms up compiled models)...
  VieNeu-TTS test: ... samples, 2.1s audio, 0.8s wall, RTF=0.38
  Qwen3-TTS test: ... samples, sr=12000, 1.5s audio, 0.3s wall, RTF=0.20

============================================================
OPTIMIZATION SUMMARY
============================================================
  TF32 matmul precision:  enabled
  Attention impl:         flash_attention_2
  ASR backend:            vLLM (optimized)
  VieNeu-TTS engine:      LMDeploy fast
  VieNeu-TTS-0.5B engine: LMDeploy fast
  Qwen3-TTS engine:       faster-qwen3-tts (CUDA graphs)
  VRAM usage:             5.5 GB
============================================================
```

### Test endpoints

- `http://localhost:8000/` — Main web UI with optimization badges
- `http://localhost:8000/test` — Download a VieNeu-TTS test WAV
- `http://localhost:8000/test-en` — Download a Qwen3-TTS English test WAV
- `http://localhost:8000/stats` — JSON with optimization stats and VRAM usage

---

## Part 5: Expose the App with Cloudflare Tunnel

```bash
# Install cloudflared
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# Start the tunnel (in a separate terminal or tmux pane)
cloudflared tunnel --url http://localhost:8000
```

### Run both in one terminal with tmux

```bash
apt-get install -y tmux
tmux new -s demo3

# Pane 1: Start the server
cd /workspace/VieNeu-TTS-repo
export GEMINI_API_KEY="your-api-key-here"
python -m uvicorn demo3_server:app --host 0.0.0.0 --port 8000

# Press Ctrl+B then % to split pane

# Pane 2: Start the tunnel
cloudflared tunnel --url http://localhost:8000

# Press Ctrl+B then D to detach
# Re-attach later with: tmux attach -t demo3
```

---

## Part 6: Usage

1. Open the web page in your browser
2. Check the **optimization badges** at the top — green = optimized, red = standard fallback
3. **Choose TTS mode**:
   - **Single TTS**: VieNeu-TTS speaks Vietnamese only (EN shown on screen)
   - **Dual TTS**: Vietnamese audio first, then English audio
   - **Dual TTS 2**: Same as Dual but uses 0.5B base model
4. Click the microphone button (or **hold Space**)
5. Speak in Vietnamese
6. Release the button / Space
7. The assistant's voice plays automatically
8. Check the **timing bar** below each response to see per-step latency

---

## Part 7: Understanding the Timing Bar

After each response, a timing bar appears showing:

```
ASR: 0.4s  |  LLM: 0.6s  |  TTS-VI: 0.8s  |  TTS-EN: 0.3s  |  Total: 2.1s
```

| Metric | What it measures | Target (RTX 5080) |
|---|---|---|
| ASR | Speech recognition latency | < 1.0s with vLLM |
| LLM | Gemini API response time | 0.3-1.0s (network dependent) |
| TTS-VI | Vietnamese speech generation | < 1.5s with LMDeploy |
| TTS-EN | English speech generation | < 0.5s with CUDA graphs |
| Total | End-to-end pipeline | < 3.0s for dual mode |

---

## Part 8: Customize

### Change the System Prompt

Edit `SYSTEM_PROMPT` in `demo3_server.py`.

### Use a Different Gemini Model

Change the model in `chat_with_gemini()`:

```python
model="gemini-2.5-pro"      # smarter, slower
model="gemini-2.5-flash"    # fast, default
```

### Change the Qwen3-TTS Reference Voice

Edit these variables at the top of `demo3_server.py`:

```python
QWEN_TTS_REF_AUDIO = "path/to/english_speaker.wav"
QWEN_TTS_REF_TEXT = "Transcript of the reference audio."
```

### Tune LMDeploy memory usage

If you see OOM errors, reduce `memory_util`:

```python
vieneu_tts = Vieneu(
    mode="fast",
    memory_util=0.15,  # reduce from 0.2
    ...
)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GEMINI_API_KEY` not found | Run `export GEMINI_API_KEY="your-key"` before starting |
| Microphone not working | Browser requires HTTPS or localhost. Use Cloudflare Tunnel |
| All badges show red/standard | Optimized packages not installed. Follow Part 1 full install |
| `lmdeploy` import error | `pip install lmdeploy` — requires Linux + CUDA 12.8 |
| `faster_qwen3_tts` import error | `pip install faster-qwen3-tts` |
| `vllm` import error | `pip install vllm --pre "vllm[audio]"` — requires Linux |
| "no kernel image available" | Wrong CUDA version. Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| CUDA OOM | Reduce `memory_util` in Vieneu() calls, or reduce `gpu_memory_utilization` for ASR |
| FlashAttention install fails | Skip it — demo3 auto-falls back to SDPA (built into PyTorch) |
| Slow first inference | Normal — models warm up on first call (CUDA graphs, torch.compile). 2nd+ calls are fast |
| `cloudflared` connection refused | Make sure uvicorn is running on port 8000 before starting the tunnel |

---

## Quick Reference: All Commands

```bash
# === UPLOAD MERGED MODEL (from local machine) ===
scp -P <PORT> finetune/finetune_results/merged_model.tar.gz root@<HOST>:/workspace/VieNeu-TTS-repo/

# === EXTRACT MODEL (on server) ===
cd /workspace/VieNeu-TTS-repo
tar -xzf merged_model.tar.gz && rm merged_model.tar.gz

# === FULL INSTALL (optimized) ===
pip install -q fastapi uvicorn websockets python-multipart
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U qwen-asr google-genai
pip install -q transformers==4.57.6 peft librosa soundfile tqdm sea-g2p
pip install -q git+https://github.com/Neuphonic/NeuCodec.git
pip install lmdeploy                              # VieNeu-TTS fast mode
pip install faster-qwen3-tts                      # Qwen3-TTS CUDA graphs
pip install vllm --pre "vllm[audio]"              # ASR vLLM backend
pip install flash-attn==2.7.4.post1               # FlashAttention 2 for Blackwell
pip install -q numpy scipy

# === SET API KEY ===
export GEMINI_API_KEY="your-key"

# === RUN OPTIMIZED SERVER ===
cd /workspace/VieNeu-TTS-repo
python -m uvicorn demo3_server:app --host 0.0.0.0 --port 8000

# === EXPOSE TO PUBLIC (in another terminal) ===
cloudflared tunnel --url http://localhost:8000

# === CHECK STATS ===
curl http://localhost:8000/stats | python -m json.tool
```
