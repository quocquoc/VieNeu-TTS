# Fine-tune VieNeu-TTS on GPU Server (NVIDIA RTX 5080)
# Use Case: AI English Teacher for Vietnamese Children

## Overview

This guide walks you through renting a GPU server with an NVIDIA RTX 5080 (16 GB VRAM) and fine-tuning VieNeu-TTS using LoRA to create an **AI English teaching assistant** that can:
- Speak **clear, slow English** for teaching pronunciation
- Speak **natural Vietnamese** for explanations
- **Mix English and Vietnamese smoothly in one sentence** for bilingual teaching

**Prerequisites:** You have prepared your voice data following `prepare-voice-data.md`.

**Data split reminder:** ~7h pure EN (slow) + ~7h pure VI + ~6h mixed EN-VI = ~20h total.

---

## 1. RTX 5080 Specifications

| Spec | RTX 5080 |
|---|---|
| VRAM | 16 GB GDDR7 |
| CUDA Cores | 10,752 |
| Architecture | Blackwell |
| Suitable? | Yes — LoRA on 0.3B model fits comfortably |

> Per the official VieNeu-TTS guide: GPU with 12 GB VRAM or more is recommended (RTX 3060, 4060 Ti, etc.). RTX 5080 with 16 GB is more than enough.

---

## 2. Rent a GPU Server

### Recommended Providers

| Provider | Website | Approx. Cost |
|---|---|---|
| Vast.ai | https://vast.ai | $0.15–0.40/hr |
| RunPod | https://runpod.io | $0.30–0.50/hr |
| Lambda Labs | https://lambdalabs.com | $0.50–0.75/hr |
| Google Colab Pro+ | https://colab.google | $50/month |

### Rent on Vast.ai (Step-by-Step Example)

1. Go to https://vast.ai and create an account
2. Add credits ($20–50 is enough for initial testing)
3. Click "Search" and filter:
   - GPU: RTX 5080 (or RTX 4090 / A100 if unavailable)
   - VRAM: >= 16 GB
   - Disk: >= 100 GB (you need space for data + model)
   - Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (or similar)
4. Click "Rent" on the cheapest machine
5. Connect via SSH: `ssh -p PORT root@IP_ADDRESS`

### Rent on RunPod (Step-by-Step Example)

1. Go to https://runpod.io and create an account
2. Click "GPU Cloud" then "Deploy"
3. Select template: **RunPod PyTorch 2.x**
4. Choose GPU: RTX 5080 or equivalent
5. Set disk space to **100 GB** minimum
6. Deploy and connect via web terminal or SSH

---

## 3. Server Setup

Run all commands below on the GPU server.

### 3.1 Install System Dependencies

```bash
apt update && apt upgrade -y
apt install -y git ffmpeg sox nano htop

# Install uv (package manager used by VieNeu-TTS)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 3.2 Clone the Project

```bash
cd ~
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
```

### 3.3 Install Dependencies

```bash
# Sync all dependencies (includes torch, torchaudio, peft, transformers, etc.)
uv sync
```

> The `train.py` script imports from `peft` and `transformers` which are already included via `uv sync`. If you encounter missing packages, install them with: `uv pip install peft accelerate bitsandbytes`

### 3.4 Verify GPU Access

```bash
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5080
VRAM: 16.0 GB
```

---

## 4. Upload Your Data to the Server

### Option A: Using SCP (from your local machine)

```bash
# From your LOCAL machine
scp -P PORT -r finetune/dataset/raw_audio/ root@IP_ADDRESS:~/VieNeu-TTS/finetune/dataset/
scp -P PORT finetune/dataset/metadata.csv root@IP_ADDRESS:~/VieNeu-TTS/finetune/dataset/
```

### Option B: Using rsync (faster for large datasets)

```bash
# From your LOCAL machine
rsync -avz --progress -e "ssh -p PORT" \
  finetune/dataset/ root@IP_ADDRESS:~/VieNeu-TTS/finetune/dataset/
```

### Option C: Upload to HuggingFace first, then download on server

```bash
# On the server
pip install huggingface-hub
huggingface-cli download your-username/your-dataset --local-dir ~/VieNeu-TTS/finetune/dataset/
```

### Verify the upload

```bash
ls ~/VieNeu-TTS/finetune/dataset/raw_audio/ | wc -l    # Should show ~10,000 files
wc -l ~/VieNeu-TTS/finetune/dataset/metadata.csv        # Should show ~10,000 lines
```

---

## 5. Data Preprocessing

Run these on the server, from the project root:

```bash
cd ~/VieNeu-TTS
```

### 5.1 Filter Data

The `filter_data.py` script removes clips that are too short (< 3s), too long (> 15s), or have invalid text (contains digits, acronyms, or missing end punctuation):

```bash
uv run python finetune/data_scripts/filter_data.py
```

**Expected output:**
```
🧹 Bắt đầu lọc dữ liệu...

🦜 KẾT QUẢ LỌC DỮ LIỆU:
   - Tổng ban đầu: ~10000
   - Hợp lệ: ~9000+ (90%+)
   - Bị loại: ~1000
     + Không tìm thấy audio: 0
     + Lỗi file audio: ~50
     + Thời lượng không hợp lệ (3-15s): ~500
     + Text rác/chứa số: ~450

✅ Đã lưu metadata sạch tại: finetune/dataset/metadata_cleaned.csv
```

> If too many clips are rejected, review your data against the rules in `prepare-voice-data.md` Section 5.

### 5.2 Encode Audio with NeuCodec

This converts WAV audio into neural codec tokens (the format the LLM model learns from):

```bash
uv run python finetune/data_scripts/encode_data.py
```

> **Important:** The default `encode_data.py` processes a maximum of **2,000 samples** (see line 10: `max_samples=2000`). For your 20h dataset (~10,000 samples), you **must** increase this limit.

**Edit the file before running:**

```bash
nano finetune/data_scripts/encode_data.py
```

Find this line (line 10):
```python
def encode_dataset(dataset_dir="finetune/dataset", max_samples=2000):
```

Change to:
```python
def encode_dataset(dataset_dir="finetune/dataset", max_samples=12000):
```

Then run:
```bash
uv run python finetune/data_scripts/encode_data.py
```

**This takes ~1–2 hours** on GPU for ~10,000 clips. The script shows a progress bar and uses `metadata_cleaned.csv` (output of Step 5.1). It randomly shuffles and takes up to `max_samples`.

**Expected output:**
```
🦜 Đang tải NeuCodec model...
🦜 Bắt đầu encode metadata: finetune/dataset/metadata_cleaned.csv
🦜 Đang lấy ngẫu nhiên tối đa 12000 mẫu để xử lý
100%|████████████████████| 9500/9500 [1:23:00<00:00]

🦜 Hoàn tất! Đã lưu file mã hóa tại: finetune/dataset/metadata_encoded.csv
   - Tổng file xử lý thành công: ~9300
   - Số file lỗi/bỏ qua: ~200
```

The result `metadata_encoded.csv` has format: `filename|text|[codec_codes_json]`

---

## 6. Configure Training

### 6.1 Edit LoRA Configuration

Open the config file:

```bash
nano finetune/configs/lora_config.py
```

**Replace the entire content with these settings optimized for 20h bilingual teaching data on RTX 5080:**

```python
import os
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

# LoRA Configuration
lora_config = LoraConfig(
    r=16,                   # Rank 16 — good balance of quality vs memory
    lora_alpha=32,          # Alpha = 2x rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

training_config = {
    'model': "pnnbao-ump/VieNeu-TTS-0.3B",       # 0.3B fits in 16GB VRAM
    'run_name': "VieNeu-TTS-EN-Teacher",           # Descriptive name
    'output_dir': os.path.join("finetune", "output"),

    'per_device_train_batch_size': 2,              # Max 2 for 16GB VRAM
    'gradient_accumulation_steps': 4,              # Effective batch = 2 x 4 = 8

    'learning_rate': 1e-4,        # Lower than default 2e-4 for larger dataset
    'max_steps': 15000,           # More steps for 20h data (default is 5000 for 2-4h)
    'logging_steps': 50,
    'save_steps': 1000,           # Save checkpoint every 1000 steps
    'eval_steps': 1000,

    'warmup_ratio': 0.05,
    'bf16': True,                 # RTX 5080 supports bfloat16

    'use_4bit': False,            # Set True only if you run out of VRAM
}

def get_training_args(config):
    return TrainingArguments(
        output_dir=os.path.join(config['output_dir'], config['run_name']),
        do_train=True,
        do_eval=False,
        max_steps=config['max_steps'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],
        bf16=config['bf16'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )
```

### 6.2 Why These Values?

| Parameter | Default (2-4h data) | Our Value (20h data) | Reason |
|---|---|---|---|
| `model` | `VieNeu-TTS-0.3B` | `VieNeu-TTS-0.3B` | Fits in 16GB; 0.5B needs more VRAM |
| `max_steps` | 5,000 | **15,000** | 5x more data needs more training steps |
| `learning_rate` | 2e-4 | **1e-4** | More data = lower learning rate for stability |
| `gradient_accumulation_steps` | 1 | **4** | Simulates larger batch without extra VRAM |
| `save_steps` | 500 | **1,000** | Save less often since training is longer |
| `r` (LoRA rank) | 16 | 16 | Default is already good |

### 6.3 If You Run Out of VRAM (CUDA OOM)

Try these fixes in order:

1. Set `'use_4bit': True` in training_config — loads base model in 4-bit quantization
2. Reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps` to 8
3. Reduce LoRA `r` from 16 to 8 (lower quality but less memory)

---

## 7. Start Training

### 7.1 Use screen/tmux (Important!)

Training takes 12–18 hours. Use `screen` so it continues if your SSH disconnects:

```bash
screen -S training
cd ~/VieNeu-TTS
```

### 7.2 Run Training

```bash
uv run python finetune/train.py
```

### 7.3 What to Expect

```
🦜 Đang tải model gốc: pnnbao-ump/VieNeu-TTS-0.3B
🦜 Đã tải 9300 mẫu dữ liệu từ finetune/dataset/metadata_encoded.csv
🦜 Đang áp dụng LoRA adapters...
trainable params: 6,815,744 || all params: 316,815,744 || trainable%: 2.15%
🦜 Bắt đầu quá trình huấn luyện! (Chúc may mắn)

Step 50/15000 | Loss: 3.2145 | LR: 5.2e-5
Step 100/15000 | Loss: 2.8734 | LR: 1.0e-4
Step 500/15000 | Loss: 1.9523 | LR: 1.0e-4
Step 1000/15000 | Loss: 1.4212 | LR: 9.8e-5
...
```

### 7.4 Detach from screen

Press `Ctrl+A` then `D` to detach. Your training continues in the background.

To reattach later: `screen -r training`

### 7.5 Training Time Estimate

| GPU | Batch Size | Est. Time for 15,000 Steps |
|---|---|---|
| RTX 5080 | 2 | ~12–18 hours |
| RTX 4090 | 2 | ~10–15 hours |
| A100 (40GB) | 4 | ~6–10 hours |

### 7.6 Monitor Training

Open another terminal:

```bash
# Watch GPU memory and utilization
watch -n 1 nvidia-smi

# Check latest training logs
ls -la finetune/output/VieNeu-TTS-EN-Teacher/
```

**Signs of good training:**
- Loss decreases steadily from ~3.0–4.0 toward ~1.0–1.5
- GPU utilization at 90–100%
- VRAM usage at 12–15 GB

**Signs of problems:**
- Loss flat or increasing → learning rate too high (try 5e-5)
- Loss drops below 0.1 → overfitting (reduce max_steps to 10,000)
- CUDA Out of Memory → see Section 6.3

---

## 8. After Training

### 8.1 Check Saved Checkpoints

```bash
ls finetune/output/VieNeu-TTS-EN-Teacher/
```

You should see:
```
checkpoint-1000/
checkpoint-2000/
...
adapter_model.safetensors    <-- Final LoRA weights
adapter_config.json
tokenizer.json
...
```

### 8.2 Test Your Model

Test all three scenarios — pure EN (slow), pure VI, and mixed code-switching:

```bash
uv run python -c "
from vieneu import Vieneu

tts = Vieneu()
tts.load_lora_adapter('finetune/output/VieNeu-TTS-EN-Teacher')

# Test 1: Slow, clear English (teaching mode)
audio = tts.infer('Hello children. Today we will learn about animals. Cat. Dog. Bird.')
tts.save(audio, 'test_en_slow.wav')

# Test 2: Natural Vietnamese
audio = tts.infer('Các con ơi, hôm nay chúng ta sẽ học về các con vật nhé.')
tts.save(audio, 'test_vi.wav')

# Test 3: Mixed - teach word and explain (Pattern 1)
audio = tts.infer('Apple, quả táo, các con đọc theo cô nhé, apple.')
tts.save(audio, 'test_mix_teach.wav')

# Test 4: Mixed - Vietnamese instruction with English words (Pattern 2)
audio = tts.infer('Hôm nay chúng ta sẽ học về các loại fruit, như apple, banana, và orange.')
tts.save(audio, 'test_mix_instruction.wav')

# Test 5: Mixed - English then Vietnamese translation (Pattern 3)
audio = tts.infer('I like to read books, tôi thích đọc sách.')
tts.save(audio, 'test_mix_translate.wav')

# Test 6: Mixed - mid-sentence switch (Pattern 4)
audio = tts.infer('Very good, giỏi lắm, bạn phát âm chuẩn lắm!')
tts.save(audio, 'test_mix_praise.wav')

print('Done! Check all test_*.wav files')
"
```

**What to listen for:**
- **English clips:** Is the speech slow and clear? Each word distinct? Warm tone?
- **Vietnamese clips:** Natural speed? Correct tones?
- **Mixed clips:** Smooth transition at switch points? No robotic pause between languages? Vietnamese tones preserved even next to English words?

Download test files to your local machine:

```bash
# From your LOCAL machine
scp -P PORT root@IP_ADDRESS:~/VieNeu-TTS/test_*.wav .
```

> **If quality is not good enough:** Try resuming training for more steps (edit `max_steps` to 20,000 and run `train.py` again — it will resume from the last checkpoint). Or try increasing LoRA rank `r` from 16 to 32.

### 8.3 Merge LoRA into Base Model (Recommended for Production)

Merging creates a standalone model that loads faster and does not require the adapter separately:

```bash
cd ~/VieNeu-TTS

uv run python finetune/merge_lora.py \
  --base_model pnnbao-ump/VieNeu-TTS-0.3B \
  --adapter finetune/output/VieNeu-TTS-EN-Teacher \
  --output finetune/output/merged_model
```

### 8.4 Create voices.json (Recommended)

This lets users use your model without providing reference audio. Pick your best reference clip (3–10 seconds, clean):

```bash
# Create voice preset from a reference audio clip
uv run python finetune/create_voices_json.py \
  --audio finetune/dataset/raw_audio/your_best_clip.wav \
  --text "The exact text matching your audio clip." \
  --name teacher_voice \
  --description "Bilingual EN-VI English teacher voice, clear and warm"

# Copy into merged model folder
cp voices.json finetune/output/merged_model/
```

### 8.5 Upload to HuggingFace

```bash
pip install huggingface-hub

# Login (paste your HuggingFace token when prompted)
huggingface-cli login

# Upload the merged model
huggingface-cli upload your-username/VieNeu-TTS-EN-Teacher \
  finetune/output/merged_model
```

---

## 9. Use Your Fine-tuned Model

### From Python SDK

```python
from vieneu import Vieneu

# Load your fine-tuned English teacher model
tts = Vieneu(backbone_repo="your-username/VieNeu-TTS-EN-Teacher")

# Teach vocabulary
audio = tts.infer("Apple, quả táo. Banana, quả chuối. Orange, quả cam.")
tts.save(audio, "lesson_vocab.wav")

# Explain grammar in Vietnamese
audio = tts.infer("Khi nói về thói quen, chúng ta dùng thì hiện tại đơn.")
tts.save(audio, "lesson_grammar.wav")

# Mixed teaching sentence
audio = tts.infer("Very good, giỏi lắm! Now repeat after me, đọc theo cô nhé, the cat is sleeping.")
tts.save(audio, "lesson_mixed.wav")
```

### From Gradio Web UI

```bash
cd ~/VieNeu-TTS
uv run vieneu-web
# Open http://127.0.0.1:7860
# Go to the LoRA Adapter tab and enter your HuggingFace repo ID
```

---

## 10. Cost Estimate

| Item | Est. Cost |
|---|---|
| GPU server: data encoding (~2h) | $0.30–0.80 |
| GPU server: training (~15h) | $2.50–6.00 |
| GPU server: testing (~3h) | $0.50–1.20 |
| **Total** | **$3.30–8.00** |

> **Tip:** Start with a small test run first — use 500 samples and 1,000 steps to verify the pipeline works. This costs under $1 and catches setup issues early.

---

## 11. Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

Fix: See Section 6.3. Most common: reduce batch_size to 1 or enable 4-bit quantization.

### Module Not Found

```
ModuleNotFoundError: No module named 'peft'
```

Fix: `uv pip install peft accelerate bitsandbytes transformers`

### Encoding Too Slow (Running on CPU)

If `encode_data.py` is slow, check CUDA:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, reinstall PyTorch with CUDA:
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Loss Not Decreasing

- Stuck above 3.0 → try higher learning rate (2e-4)
- Oscillating wildly → try lower learning rate (5e-5)
- Check data quality — run `filter_data.py` again and review rejected entries

### metadata_encoded.csv Is Empty or Too Small

- Check that `metadata_cleaned.csv` exists (output of `filter_data.py`)
- Check that `max_samples` in `encode_data.py` is set to 12000+
- Check that `raw_audio/` folder path is correct

### SSH Disconnects During Training

Always use `screen` or `tmux` (Section 7.1). If you forgot:
```bash
ps aux | grep train.py     # Check if training is still running
ls finetune/output/VieNeu-TTS-EN-Teacher/    # Check saved checkpoints
```

---

## 12. Quick Reference: Complete Command Sequence

```bash
# ==== ON GPU SERVER ====

# 1. Setup
apt update && apt install -y git ffmpeg sox
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
uv sync

# 2. Upload data (from local machine or HuggingFace)
# ... put files in finetune/dataset/raw_audio/ and metadata.csv

# 3. Filter data
uv run python finetune/data_scripts/filter_data.py

# 4. Edit encode_data.py: change max_samples=2000 to max_samples=12000
nano finetune/data_scripts/encode_data.py

# 5. Encode audio to codec tokens
uv run python finetune/data_scripts/encode_data.py

# 6. Edit training config
nano finetune/configs/lora_config.py
# Set: run_name="VieNeu-TTS-EN-Teacher", max_steps=15000,
#       gradient_accumulation_steps=4, learning_rate=1e-4

# 7. Train (inside screen)
screen -S training
uv run python finetune/train.py
# Ctrl+A then D to detach

# 8. After training — test
uv run python -c "
from vieneu import Vieneu
tts = Vieneu()
tts.load_lora_adapter('finetune/output/VieNeu-TTS-EN-Teacher')
audio = tts.infer('Hello children, hôm nay chúng ta học về animals nhé!')
tts.save(audio, 'test.wav')
print('Done!')
"

# 9. Merge LoRA
uv run python finetune/merge_lora.py \
  --base_model pnnbao-ump/VieNeu-TTS-0.3B \
  --adapter finetune/output/VieNeu-TTS-EN-Teacher \
  --output finetune/output/merged_model

# 10. Create voices.json and upload
uv run python finetune/create_voices_json.py \
  --audio finetune/dataset/raw_audio/best_clip.wav \
  --text "Exact text of the clip." \
  --name teacher_voice \
  --description "Bilingual EN-VI English teacher voice"
cp voices.json finetune/output/merged_model/

huggingface-cli login
huggingface-cli upload your-username/VieNeu-TTS-EN-Teacher \
  finetune/output/merged_model
```

---

Good luck with your English teaching AI!
