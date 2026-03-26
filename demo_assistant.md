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
| Qwen3-ASR-1.7B (bf16) | ~4 GB |
| VieNeu-TTS-0.3B merged (LMDeploy, bf16) | ~1.2 GB |
| NeuCodec decoder (Triton-compiled) | ~0.3 GB |
| LMDeploy KV cache + runtime | ~3 GB |
| **Total** | **~8.5 GB** (15.5 GB free for headroom) |

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
pip install -q transformers peft librosa soundfile tqdm sea-g2p

# 4a. Install torch + torchaudio for CUDA 12.8 (RTX 3090 Ti server)
#     (neucodec requires torchaudio — must match the CUDA runtime on the server)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# if error: OSError: /venv/main/lib/python3.12/site-packages/torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl
# pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128


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

### Step 3: Re-merge the model with correct precision (bfloat16)

> **Important:** The merge script previously used `float16` which caused audio quality issues.
> You must re-merge with `bfloat16` on the server (requires the LoRA adapter).

```bash
cd /workspace/VieNeu-TTS-repo

# Re-merge with bfloat16 precision (fixes audio quality)
python finetune/merge_lora.py \
  --base_model pnnbao-ump/VieNeu-TTS-0.3B \
  --adapter finetune/output/VieNeu-TTS-0.3B-LoRA \
  --output finetune/output/merged_model

# Create voices.json for the merged model (encodes your reference audio as voice preset)
python finetune/create_voices_json.py \
  --audio finetune/output/en_0176.wav \
  --text "The tiger woke up early every morning and walked all the way to the zoo." \
  --name my_custom_voice \
  --description "My fine-tuned voice" \
  --output finetune/output/merged_model/voices.json

# Verify all files are in place
ls finetune/output/merged_model/
# Expected: config.json  generation_config.json  model.safetensors  tokenizer.json  tokenizer_config.json  voices.json

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
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
print("ASR loaded OK")

# Test 2: Load TTS
print("Loading VieNeu-TTS...")
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from vieneu import Vieneu
tts = Vieneu(mode="standard", backbone_repo="finetune/output/merged_model", backbone_device="cuda", codec_device="cuda")
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

File demo_server.py

---

## Part 6: Create the Frontend

File demo_frontend.html

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
Loading Qwen3-ASR-1.7B...
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

### Use the Smaller ASR Model (save VRAM)

If you need to free up VRAM, switch to the 0.6B model:

```python
asr_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",  # ~1.5 GB VRAM instead of ~4 GB
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
| CUDA OOM when both models load | Check no other processes use GPU: `nvidia-smi`. If needed, switch to Qwen3-ASR-0.6B (~1.5 GB) to save VRAM |
| Audio not playing in browser | Check browser allows autoplay. Try clicking the page first |
| WebSocket disconnects | Check Vast.ai port mapping. Ensure port 8000 is open |
| `libcudart.so` / torchaudio error | CUDA version mismatch. Reinstall: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128` (replace `cu124` with your CUDA version) |
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
