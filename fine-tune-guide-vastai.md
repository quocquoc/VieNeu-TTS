# Fine-tune VieNeu-TTS on Vast.ai (RTX 5080) - Step-by-Step Guide

## Overview

This guide walks you through fine-tuning the VieNeu-TTS model using LoRA on a Vast.ai GPU server (RTX 5080), then hosting the fine-tuned model for TTS inference.

**Your dataset:**
- 7,619 audio samples in `finetune/dataset/raw_audio/` (~3.5 GB)
- Metadata file: `finetune/dataset/metadata.csv` (format: `filename|text`)

**What you'll get:** A LoRA adapter that can be merged into the base model for custom voice TTS.

---

## Part 1: Rent and Setup Vast.ai Server

### 1.1 Choose a Server on Vast.ai

- Go to [vast.ai](https://vast.ai/) and search for an **RTX 5080** instance
- Recommended template: **PyTorch 2.x** (with CUDA 12.x)
- Minimum requirements:
  - GPU: RTX 5080 (16 GB VRAM)
  - RAM: 32 GB+
  - Disk: 50 GB+ (your dataset is ~3.5 GB, model weights ~2 GB, plus output)
- Select **Jupyter** as the launch mode

### 1.2 Connect to Jupyter

After the instance starts, click the **Open** button to access Jupyter in your browser. You'll use both the Jupyter file browser (for uploads) and the **Terminal** (for running commands).

---

## Part 2: Upload Dataset to Server

### 2.1 Upload via Jupyter File Browser

1. In Jupyter, navigate to your workspace directory (usually `/workspace/` or `/root/`)
2. Create the project structure:

Open a **Terminal** in Jupyter and run:

```bash
mkdir -p VieNeu-TTS/finetune/dataset/raw_audio
```

3. **Upload `metadata.csv`:**
   - In Jupyter file browser, navigate to `VieNeu-TTS/finetune/dataset/`
   - Click **Upload** and select your `metadata.csv`

4. **Upload audio files:**
   - Since you have ~3.5 GB of audio (7,619 .wav files), uploading one-by-one via Jupyter is impractical
   - **Recommended:** Compress and upload as a single archive

### 2.2 Upload Audio Files (Recommended: Archive Method)

**On your local machine**, compress the audio folder:

```bash
cd /path/to/VieNeu-TTS/finetune/dataset
tar -czf raw_audio.tar.gz raw_audio/
```

This will create `raw_audio.tar.gz` (~2-3 GB depending on compression).

**Upload options (choose one):**

**Option A: Upload via Jupyter**
- In Jupyter file browser, navigate to `VieNeu-TTS/finetune/dataset/`
- Click Upload and select `raw_audio.tar.gz`
- Then in Terminal:

```bash
cd /workspace/VieNeu-TTS/finetune/dataset
tar -xzf raw_audio.tar.gz
rm raw_audio.tar.gz  # cleanup to save disk space
```

**Option B: Upload via SCP (faster for large files)**
- In Vast.ai dashboard, find your server's SSH info (host, port)
- From your local machine:

```bash
scp -P <PORT> raw_audio.tar.gz root@<HOST>:/workspace/VieNeu-TTS/finetune/dataset/
```

Then SSH in or use Terminal to extract:

```bash
cd /workspace/VieNeu-TTS/finetune/dataset
tar -xzf raw_audio.tar.gz
rm raw_audio.tar.gz
```

**Option C: Upload to cloud storage first**
- Upload `raw_audio.tar.gz` to Google Drive, Dropbox, or a cloud storage
- Then download on the server using `wget` or `gdown` (for Google Drive):

```bash
# Example with Google Drive
pip install gdown
gdown "https://drive.google.com/uc?id=YOUR_FILE_ID" -O /workspace/VieNeu-TTS/finetune/dataset/raw_audio.tar.gz
cd /workspace/VieNeu-TTS/finetune/dataset
tar -xzf raw_audio.tar.gz
rm raw_audio.tar.gz
```

### 2.3 Verify Upload

```bash
ls /workspace/VieNeu-TTS/finetune/dataset/raw_audio/ | head -20
wc -l /workspace/VieNeu-TTS/finetune/dataset/metadata.csv
ls /workspace/VieNeu-TTS/finetune/dataset/raw_audio/ | wc -l
```

Expected: `metadata.csv` has 7,619 lines, and `raw_audio/` has 7,619 .wav files.

---

## Part 3: Install Dependencies and Clone Repository

Open **Terminal** in Jupyter and run:

### 3.1 Clone the VieNeu-TTS Repository

```bash
cd /workspace
git clone https://github.com/pnnbao97/VieNeu-TTS.git VieNeu-TTS-repo
```

### 3.2 Copy Your Dataset into the Cloned Repo

```bash
# Create the dataset directory first (it doesn't exist in the cloned repo)
mkdir -p /workspace/VieNeu-TTS-repo/finetune/dataset

# Copy your uploaded dataset into the repo structure
cp /workspace/VieNeu-TTS/finetune/dataset/metadata.csv /workspace/VieNeu-TTS-repo/finetune/dataset/metadata.csv
# If you uploaded raw_audio directly:
ln -s /workspace/VieNeu-TTS/finetune/dataset/raw_audio /workspace/VieNeu-TTS-repo/finetune/dataset/raw_audio
# Or if you want to copy (uses more disk):
# cp -r /workspace/VieNeu-TTS/finetune/dataset/raw_audio /workspace/VieNeu-TTS-repo/finetune/dataset/raw_audio
```

### 3.3 Install Python Dependencies

```bash
cd /workspace/VieNeu-TTS-repo

# Install pip dependencies (no uv needed on Vast.ai)
pip install -q transformers peft torch datasets librosa soundfile tqdm sea-g2p
pip install -q git+https://github.com/Neuphonic/NeuCodec.git
```

### 3.4 Verify GPU is Available

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output: `CUDA available: True` and `GPU: NVIDIA GeForce RTX 5080`

---

## Part 4: Data Preprocessing

### 4.1 Filter Data

This removes audio files that are too short (<3s), too long (>15s), or have invalid text (contains numbers, acronyms, etc.).

```bash
cd /workspace/VieNeu-TTS-repo
python finetune/data_scripts/filter_data.py
```

**Output:** Creates `finetune/dataset/metadata_cleaned.csv`

Check the results:

```bash
wc -l finetune/dataset/metadata_cleaned.csv
```

### 4.2 Encode Audio to VQ Codes

This converts audio files into NeuCodec vector-quantized codes that the LLM can learn from. This step requires GPU and takes some time.

```bash
python finetune/data_scripts/encode_data.py
```

**Note:** By default, `encode_data.py` processes a maximum of 2,000 randomly sampled audio files. If you want to use more samples (recommended for better quality), edit the script:

```bash
# To process all samples, edit encode_data.py or run with a larger max:
python -c "
import sys
sys.path.insert(0, '.')
from finetune.data_scripts.encode_data import encode_dataset
encode_dataset(dataset_dir='finetune/dataset', max_samples=7000)
"
```

**Output:** Creates `finetune/dataset/metadata_encoded.csv`

Verify:

```bash
wc -l finetune/dataset/metadata_encoded.csv
head -1 finetune/dataset/metadata_encoded.csv
```

You should see lines in the format: `filename|text|[code1, code2, ...]`

---

## Part 5: Configure Training

### 5.1 Review Training Config

The config file is at `finetune/configs/lora_config.py`. The defaults are good for RTX 5080:

| Parameter | Default | Notes |
|---|---|---|
| `model` | `pnnbao-ump/VieNeu-TTS-0.3B` | Base model |
| `max_steps` | `5000` | Good for a single voice. Increase for more data |
| `learning_rate` | `2e-4` | Standard LoRA learning rate |
| `per_device_train_batch_size` | `1` | RTX 5080 (16GB); effective batch=2 via gradient accumulation |
| `bf16` | `True` | Uses BFloat16 for memory efficiency |

### 5.2 (Optional) Adjust Config

If you want to change settings:

```bash
# Example: increase batch size for RTX 5080's 16GB VRAM
nano finetune/configs/lora_config.py
```

Suggested adjustments for RTX 5080:
- `per_device_train_batch_size`: Keep at `1` (use `gradient_accumulation_steps` to increase effective batch size)
- `max_steps`: With ~7000 samples, `5000-8000` steps is reasonable
- `save_steps`: `500` (saves checkpoints every 500 steps)

---

## Part 6: Start Training

### 6.1 Run Training

```bash
cd /workspace/VieNeu-TTS-repo
python finetune/train.py
```

**What to expect:**
- First, it downloads the base model `pnnbao-ump/VieNeu-TTS-0.3B` from HuggingFace (~600 MB)
- Then applies LoRA adapters (only ~0.5% of parameters are trainable)
- Training logs print every 50 steps showing the loss
- Checkpoints are saved every 500 steps to `finetune/output/VieNeu-TTS-0.3B-LoRA/`
- With RTX 5080 and batch_size=2, expect ~1-2 hours for 5000 steps

### 6.2 Monitor Training

Watch the training loss in the terminal output. A healthy training should show:
- Loss decreasing over time (e.g., from ~5.0 to ~2.0 or lower)
- No OOM (Out of Memory) errors

If you get an **OOM error**, reduce batch size:
```bash
# Edit the config to use batch_size=1
sed -i 's/per_device_train_batch_size.*: 2/per_device_train_batch_size: 1/' finetune/configs/lora_config.py
python finetune/train.py
```

### 6.3 Training Output

After training completes, the LoRA adapter files are saved at:

```
finetune/output/VieNeu-TTS-0.3B-LoRA/
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

---

## Part 7: Download Results from Server

### 7.1 Compress Output

```bash
cd /workspace/VieNeu-TTS-repo
tar -czf lora_output.tar.gz finetune/output/VieNeu-TTS-0.3B-LoRA/
```

### 7.2 Download

**Option A: Download via Jupyter**
- In Jupyter file browser, navigate to `/workspace/VieNeu-TTS-repo/`
- Right-click on `lora_output.tar.gz` and select **Download**

**Option B: Download via SCP**
```bash
# From your local machine:
scp -P <PORT> root@<HOST>:/workspace/VieNeu-TTS-repo/lora_output.tar.gz ./
```

---

## Part 8: Merge LoRA into Base Model (Recommended for Production)

Merging creates a standalone model that doesn't require loading the adapter separately. You can do this on the Vast.ai server before downloading, or on your own GPU server.

### 8.1 Merge on Vast.ai Server

```bash
cd /workspace/VieNeu-TTS-repo
python finetune/merge_lora.py \
  --base_model pnnbao-ump/VieNeu-TTS-0.3B \
  --adapter finetune/output/VieNeu-TTS-0.3B-LoRA \
  --output finetune/output/merged_model
```

### 8.2 Download Merged Model

```bash
tar -czf merged_model.tar.gz finetune/output/merged_model/
# Then download via Jupyter or SCP
```

---

## Part 9: Host Fine-tuned Model on Your GPU Server

Now you have your fine-tuned model. Here's how to host it on your own GPU server for TTS inference.

### 9.1 Setup Your GPU Server

```bash
# On your GPU server
pip install vieneu
# or
pip install git+https://github.com/pnnbao97/VieNeu-TTS.git
```

### 9.2 Option A: Use Merged Model (Recommended)

Copy the `merged_model/` folder to your GPU server, then:

```python
from vieneu import Vieneu

# Load from local merged model
tts = Vieneu(backbone_repo="/workspace/VieNeu-TTS-repo/finetune/output/merged_model")

# Generate speech (requires a reference audio for voice cloning)
audio = tts.infer(
    "Trong tiếng Anh, “biển” là sea hoặc beach. Ví dụ: I go to the beach (Tôi đi tắm biển). Khi ở biển, chúng ta có thể nói: I swim (Tôi bơi) hoặc I play with sand (Tôi chơi cát).",
    ref_audio="/workspace/VieNeu-TTS-repo/test_result_model/en_0006.wav",
    ref_text="My teacher Tuan told us an amazing story about a brave little rabbit in the forest."
)

tts.save(audio, "long_mix_output.wav")
```

### 9.3 Option B: Use LoRA Adapter

If you didn't merge, you can load the base model + LoRA adapter:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "pnnbao-ump/VieNeu-TTS-0.3B",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "/path/to/VieNeu-TTS-0.3B-LoRA")
```

Or using the VieNeu SDK:

```python
from vieneu import Vieneu

tts = Vieneu()
tts.load_lora_adapter("/path/to/VieNeu-TTS-0.3B-LoRA")

audio = tts.infer(
    "Xin chao!",
    ref_audio="reference.wav",
    ref_text="Text of the reference audio."
)
tts.save(audio, "output.wav")
```

### 9.4 Option C: Upload to HuggingFace and Use Remotely

Upload the merged model or LoRA adapter to HuggingFace:

```bash
pip install huggingface-cli
huggingface-cli login

# Upload merged model
huggingface-cli upload your-username/your-model-name finetune/output/merged_model

# Or upload LoRA adapter
huggingface-cli upload your-username/your-lora-name finetune/output/VieNeu-TTS-0.3B-LoRA
```

Then use it from anywhere:

```python
from vieneu import Vieneu

# Merged model
tts = Vieneu(backbone_repo="your-username/your-model-name")

# Or LoRA adapter
tts = Vieneu()
tts.load_lora_adapter("your-username/your-lora-name")
```

### 9.5 (Optional) Create voices.json for No-Reference-Audio Usage

To make your model usable without needing a reference audio every time:

```bash
cd /workspace/VieNeu-TTS-repo
python finetune/create_voices_json.py \
  --audio finetune/output/en_0006.wav \
  --text "My teacher Tuan told us an amazing story about a brave little rabbit in the forest." \
  --name my_custom_voice \
  --description "My fine-tuned voice"
```

Copy `voices.json` into your model folder:

```bash
cp voices.json finetune/output/merged_model/
```

Then users can use the model without reference audio:

```python
from vieneu import Vieneu
tts = Vieneu(backbone_repo="your-username/your-model-name")
audio = tts.infer("Xin chao!")  # No ref_audio needed
tts.save(audio, "output.wav")
```

---

## Part 10: Host as TTS API Service (Optional)

If you want to serve TTS as an API on your GPU server:

### 10.1 Use the Built-in Gradio Web UI

```bash
cd /path/to/VieNeu-TTS
# Run Gradio web interface
vieneu-web
# Or: python -m apps.gradio_main
```

This launches a web UI where you can:
- Enter text to generate speech
- Load your LoRA adapter via the UI
- Listen to and download generated audio

### 10.2 Simple FastAPI Server

Create a simple API server:

```python
# tts_server.py
from fastapi import FastAPI, Response
from pydantic import BaseModel
from vieneu import Vieneu
import soundfile as sf
import io

app = FastAPI()
tts = Vieneu(backbone_repo="/path/to/merged_model")

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def generate_speech(req: TTSRequest):
    audio = tts.infer(req.text)
    buffer = io.BytesIO()
    sf.write(buffer, audio.numpy(), 24000, format="WAV")
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")
```

Run with:

```bash
pip install fastapi uvicorn
uvicorn tts_server:app --host 0.0.0.0 --port 8000
```

---

## Quick Reference: Complete Command Sequence

Here's every command in order for copy-pasting into Vast.ai Terminal:

```bash
# === SETUP ===
cd /workspace
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
pip install -q transformers peft torch datasets librosa soundfile tqdm sea-g2p
pip install -q git+https://github.com/Neuphonic/NeuCodec.git

# === UPLOAD DATASET ===
# (upload metadata.csv and raw_audio/ to finetune/dataset/ via Jupyter or SCP)

# === VERIFY ===
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
wc -l finetune/dataset/metadata.csv
ls finetune/dataset/raw_audio/ | wc -l

# === PREPROCESS ===
python finetune/data_scripts/filter_data.py
python finetune/data_scripts/encode_data.py

# === TRAIN ===
python finetune/train.py

# === MERGE (optional) ===
python finetune/merge_lora.py \
  --base_model pnnbao-ump/VieNeu-TTS-0.3B \
  --adapter finetune/output/VieNeu-TTS-0.3B-LoRA \
  --output finetune/output/merged_model

# === PACKAGE FOR DOWNLOAD ===
tar -czf lora_output.tar.gz finetune/output/
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Already optimized: batch_size=1, gradient checkpointing, dynamic padding. If still OOM, try adding `'use_4bit': True` in config |
| `FileNotFoundError: metadata_cleaned.csv` | Run `filter_data.py` first |
| `FileNotFoundError: metadata_encoded.csv` | Run `encode_data.py` first |
| `encode_data.py` is slow | It's processing on GPU. With 2000 samples it takes ~10-30 minutes |
| Training loss not decreasing | Try lowering `learning_rate` to `1e-4` |
| Training loss goes to NaN | Reduce learning rate or check dataset for corrupted audio |
| Upload is too slow via Jupyter | Use SCP or cloud storage method (see Part 2) |
| `espeak-ng` not found | Run `apt install espeak-ng -y` (may not be needed with sea-g2p) |
