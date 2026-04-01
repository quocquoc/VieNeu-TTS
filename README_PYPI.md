# 🦜 VieNeu-TTS

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) model with **instant voice cloning** and **English-Vietnamese bilingual** support.

[![Hugging Face v2 Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2%20Turbo-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## ✨ Key Features
- **Bilingual (English-Vietnamese)**: Seamless transitions between languages (Code-switching) in version 2.0+.
- **Instant Voice Cloning**: Clone any voice with just 3-5s of reference audio (**Turbo v2** & GPU modes).
- **Ultra-Fast Turbo Mode**: Optimized for **CPU (GGUF)** and **GPU (LMDeploy)**. Extremely fast inference!
- **Production Ready**: High-fidelity 24 kHz audio generation, fully offline.
- **AI Identification**: Built-in audio watermarking for responsible AI use.

---

## 📦 Quick Install

```bash
# Minimal installation (Turbo/CPU Only)
pip install vieneu

# Optional: Pre-built llama-cpp-python for CPU (if building fails)
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/
```
---

## 🚀 Quick Start (Python SDK)

The SDK now defaults to **Turbo mode** for maximum out-of-the-box compatibility.

```python
from vieneu import Vieneu

# Initialize - Minimal dependencies required!
tts = Vieneu()

# 1. Simple synthesis (uses default 'Xuân Vĩnh')
text = "Hệ thống điện chủ sử dụng alternating current because it is more efficient."
audio = tts.infer(text=text)
tts.save(audio, "output.wav")

# 2. Using a specific Preset Voice
voices = tts.list_preset_voices()
# Select 'Bắc (Nam miền Bắc)' ID
my_voice_id = voices[1][1] if len(voices) > 1 else voices[0][1]
voice_data = tts.get_preset_voice(my_voice_id)

audio_custom = tts.infer(text="Tôi đang nói bằng giọng của anh Bình.", voice=voice_data)
tts.save(audio_custom, "output_custom.wav")
```

### 🦜 Zero-shot Voice Cloning (SDK)

```python
from vieneu import Vieneu
tts = Vieneu()

# 1. Encode reference audio (3-5s wav/mp3)
my_voice = tts.encode_reference("path/to/voice.wav")

# 2. Synthesize with cloned voice (No ref text needed for v2!)
audio = tts.infer(text="Chào bạn, đây là giọng của tôi.", voice=my_voice)
tts.save(audio, "cloned.wav")
```

---

## 🔬 Model Overview

| Model | Format | Device | Bilingual | Cloning | Speed |
|---|---|---|---|---|---|
| **VieNeu-v2-Turbo** | GGUF/ONNX | **CPU/GPU** | ✅ | ✅ Yes | **Extreme** |
| **VieNeu-TTS-v2** | PyTorch | GPU | ✅ | ✅ Yes | **Standard** |
| **VieNeu-TTS 0.3B** | PyTorch | GPU/CPU | ❌ | ✅ Yes | **Very Fast** |
| **VieNeu-TTS** | PyTorch | GPU/CPU | ❌ | ✅ Yes | **Standard** |

---

## 🤝 Support & Links
- **GitHub:** [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS)
- **Discord:** [Join our community](https://discord.gg/yJt8kzjzWZ)

**Made with ❤️ for the Vietnamese TTS community**
