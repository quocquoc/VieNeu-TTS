# Research: Maximizing AI Model Inference Speed on RTX 5080

## Target Setup

| Model | Type | Size | Current Config |
|---|---|---|---|
| Qwen3-ASR-0.6B | Speech-to-Text | 0.6B params | `torch.bfloat16`, `device_map="cuda:0"` |
| Qwen3-TTS-12Hz-0.6B-Base | Text-to-Speech | 0.6B params | `torch.bfloat16`, `device_map="cuda:0"` |
| VieNeu-TTS-0.3B (merged) | Text-to-Speech | 0.3B params | `torch.bfloat16`, standard mode (HF generate) |

**Hardware**: RTX 5080 -- Blackwell (SM 100), 16GB VRAM, 5th-gen Tensor Cores (FP4/FP8/FP16/BF16), CUDA 12.8+

**Current VRAM usage**: ~5-6 GB total (plenty of headroom on 16GB)

---

## Recommended Implementation Order

| Step | Optimization | Effort | Expected Gain | Risk |
|------|-------------|--------|---------------|------|
| 1 | `torch.set_float32_matmul_precision('high')` | 1 line | 5-15% on fp32 ops | None |
| 2 | Enable SDPA for Qwen3 models | 1 line each | 10-30% | None |
| 3 | Replace Qwen3-TTS with `faster-qwen3-tts` | Package swap | **5-6x** for Qwen3-TTS | Low |
| 4 | Switch VieNeu-TTS to `mode="fast"` (LMDeploy) | Config change | **2-3x** | Low |
| 5 | Install FlashAttention 2 for Blackwell | Package install | 20-50% on attention | Medium |
| 6 | Use vLLM backend for Qwen3-ASR | Config change | **2-3x** throughput | Low |
| 7 | `torch.compile(mode="max-autotune")` + warmup | ~20 lines | 15-30% | Medium |
| 8 | Blackwell-optimized software stack | Setup | 34%+ bandwidth gain | Low |
| 9 | KV cache quantization (LMDeploy) | Config flag | 10-20% memory savings | Low |
| 10 | Batch inference tuning | Config change | 2-3x throughput | Low |

**Combined estimated speedup**: Steps 1-7 could yield **3-8x overall** inference speed improvement.

---

## TIER 1: Easy Wins (Implement First)

### 1. `torch.set_float32_matmul_precision('high')`

Allows PyTorch to use TF32 (TensorFloat32) for float32 matrix multiplications on Tensor Cores. Even though models run in bfloat16, some operations (codec decoding, phonemization tensors) may hit float32 paths.

**Expected Speedup**: 1.3-2x on float32 operations.

**Implementation** -- add at the very top of `demo2_server.py`, before any model loading:

```python
import torch
torch.set_float32_matmul_precision('high')  # or 'medium' for even faster but less precise
```

**Compatibility**: Fully supported on RTX 5080 (Ampere+ required, Blackwell qualifies). No downsides for inference.

---

### 2. SDPA (Scaled Dot Product Attention)

PyTorch's built-in `torch.nn.functional.scaled_dot_product_attention` automatically selects the fastest attention kernel (FlashAttention, xFormers memory-efficient, or C++ math).

**Current State**: VieNeu-TTS standard engine already uses SDPA (`src/vieneu/engines/standard.py` line 123). However, Qwen3-ASR and Qwen3-TTS in `demo2_server.py` do NOT explicitly set this.

**Implementation for Qwen3 models**:

```python
# Qwen3-TTS
qwen_tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Add this
)

# Qwen3-ASR -- check if qwen_asr supports attn_implementation
asr_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",  # Try adding this
)
```

**Expected Speedup**: 10-30% vs eager attention. Longer sequences benefit more.

---

### 3. `torch.compile()` -- Extend to All Models

Compiles the model's forward pass into optimized fused kernels. Reduces Python overhead, fuses operations, and captures CUDA graphs internally.

**Current State**: VieNeu-TTS standard engine already applies `torch.compile(model, mode="reduce-overhead")` but ONLY on Linux.

**Implementation -- extend to Qwen3 models**:

```python
# After loading each model, compile it:

# Qwen3-TTS
if hasattr(qwen_tts, 'model'):
    qwen_tts.model = torch.compile(qwen_tts.model, mode="reduce-overhead")

# Qwen3-ASR
if hasattr(asr_model, 'model'):
    asr_model.model = torch.compile(asr_model.model, mode="reduce-overhead")
```

**Compilation Modes**:
- `"default"` -- balanced speed/memory
- `"reduce-overhead"` -- recommended for inference servers
- `"max-autotune"` -- slowest compilation but fastest kernels (good for production)

**Warmup**: First inference after compile is SLOW (30-120s). Add warmup calls:

```python
# After compiling Qwen3-TTS:
print("Warming up Qwen3-TTS compiled model...")
_ = qwen_tts.generate_voice_clone(
    text="warmup", language="English",
    ref_audio=QWEN_TTS_REF_AUDIO, ref_text=QWEN_TTS_REF_TEXT,
)
print("Qwen3-TTS warmup complete")
```

**Expected Speedup**: 15-30% after warmup.

**Known Issue**: `torch.compile` with PEFT/LoRA models can cause graph breaks. Compile AFTER merging LoRA weights (which you already do).

---

## TIER 2: Significant Gains (Implement Second)

### 4. `faster-qwen3-tts` -- CUDA Graphs for Qwen3-TTS (5-6x Speedup)

The `faster-qwen3-tts` package uses CUDA graphs to eliminate per-kernel CPU launch overhead. Qwen3-TTS runs ~500 small CUDA kernels per decode step; CUDA graphs replay them as a single operation.

**Benchmarks**: Reduces per-step time from 330ms to 54ms on Jetson (~6x), from ~800ms to ~156ms TTFA on RTX 4090 (~5.8x).

**Installation**:

```bash
pip install faster-qwen3-tts
```

**Implementation** -- replace Qwen3-TTS loading in `demo2_server.py`:

```python
# OLD:
from qwen_tts import Qwen3TTSModel
qwen_tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# NEW:
from faster_qwen3_tts import FasterQwen3TTS
qwen_tts = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

# Non-streaming usage (same API):
audio_list, sr = qwen_tts.generate_voice_clone(
    text="Hello, this is a test.",
    language="English",
    ref_audio=QWEN_TTS_REF_AUDIO,
    ref_text=QWEN_TTS_REF_TEXT,
)

# Streaming (even lower latency):
for audio_chunk, sr, timing in qwen_tts.generate_voice_clone_streaming(
    text="Hello world",
    language="English",
    ref_audio=QWEN_TTS_REF_AUDIO,
    ref_text=QWEN_TTS_REF_TEXT,
    chunk_size=8,
):
    # Process/send chunk immediately
    pass
```

**Expected Speedup**: **5-6x** for Qwen3-TTS inference.

**Source**: https://github.com/andimarafioti/faster-qwen3-tts

---

### 5. VieNeu-TTS `mode="fast"` (LMDeploy TurbomindEngine -- 2-3x Speedup)

Your project already has a fast inference engine using LMDeploy's TurbomindEngine (`src/vieneu/engines/fast.py`). TurboMind provides optimized C++ kernels, prefix caching, and KV cache quantization.

**Implementation** -- change one config in `demo2_server.py`:

```python
# OLD:
vieneu_tts = Vieneu(
    mode="standard",
    backbone_repo="finetune/output/merged_model",
    backbone_device="cuda",
    codec_device="cuda",
)

# NEW:
vieneu_tts = Vieneu(
    mode="fast",
    backbone_repo="finetune/output/merged_model",
    backbone_device="cuda",
    codec_device="cuda",
    memory_util=0.3,           # GPU memory fraction for KV cache
    quant_policy=8,            # INT8 KV cache quantization
    enable_prefix_caching=True,
)
```

**Installation**:

```bash
pip install lmdeploy
```

**Expected Speedup**: **2-3x** over standard HuggingFace `generate()`.

---

### 6. FlashAttention 2 for RTX 5080 (Blackwell)

FlashAttention is an IO-aware CUDA attention kernel delivering 2-4x speedup on attention operations.

**Critical Note**: FlashAttention **3** does NOT support Blackwell (Hopper H100 only). Use **FlashAttention 2 v2.7.4.post1** which has been built for Blackwell GPUs.

**Installation**:

```bash
# Option 1: Pre-built Blackwell wheel
# From: https://github.com/Zarrac/flashattention-blackwell-wheels-whl-ONLY-5090-5080-5070-5060-flash-attention-
pip install flash-attn==2.7.4.post1

# Option 2: Build from source with CUDA 12.8+
pip install flash-attn --no-build-isolation
```

**Implementation**:

```python
# VieNeu-TTS -- change attn_implementation
vieneu_tts = Vieneu(
    mode="standard",
    backbone_repo="finetune/output/merged_model",
    attn_implementation="flash_attention_2",  # Instead of "sdpa"
    # ...
)

# Qwen3-TTS
qwen_tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Qwen3-ASR
asr_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)
```

**Expected Speedup**: 1.5-3x on attention operations. More impactful on longer sequences.

---

### 7. vLLM Backend for Qwen3-ASR (2-3x Throughput)

vLLM provides PagedAttention, continuous batching, and optimized CUDA kernels. Officially supports Qwen3-ASR.

**Installation**:

```bash
pip install vllm --pre          # For latest Blackwell support
pip install "vllm[audio]"       # For audio processing
```

**Implementation** -- replace ASR loading in `demo2_server.py`:

```python
# OLD:
asr_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_new_tokens=256,
)

# NEW: vLLM backend
asr_model = Qwen3ASRModel.LLM(
    "Qwen/Qwen3-ASR-0.6B",
    gpu_memory_utilization=0.3,
    max_inference_batch_size=16,
    max_new_tokens=256,
)

# Usage remains the same:
results = asr_model.transcribe(audio=temp_path, language="Vietnamese")
```

**Expected Speedup**: **2-3x** throughput via PagedAttention. TTFT improvement ~30% via prefix caching.

**Note**: vLLM requires Linux.

**Source**: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html

---

## TIER 3: Advanced Optimizations

### 8. KV Cache Optimization

**PagedAttention** (via vLLM or LMDeploy): Reduces KV cache memory waste from 60-80% to under 4%. Already included in steps 5 and 7 above.

**Static Cache** (for torch.compile compatibility):

```python
from transformers import StaticCache

cache = StaticCache(
    config=model.config,
    max_batch_size=1,
    max_cache_len=2048,
    device=model.device,
    dtype=torch.bfloat16,
)

output = model.generate(
    input_ids,
    past_key_values=cache,
    max_length=2048,
    use_cache=True,
)
```

**KV Cache Quantization** (via LMDeploy fast mode):

```python
# quant_policy=4  -> INT4 KV cache (~75% memory reduction)
# quant_policy=8  -> INT8 KV cache (~50% memory reduction, negligible quality loss)
```

---

### 9. Batch Inference Tuning

Both VieNeu-TTS engines already support batch inference. Optimize for multi-chunk text processing:

```python
# Increase max_batch_size if VRAM allows:
vieneu_tts = Vieneu(
    mode="fast",
    max_batch_size=8,  # Increase from default 4
    # ...
)
```

**For concurrent WebSocket requests**, implement a batching queue:

```python
import asyncio

tts_queue = asyncio.Queue(maxsize=8)

async def process_tts_batch():
    while True:
        batch = []
        try:
            item = await asyncio.wait_for(tts_queue.get(), timeout=0.05)
            batch.append(item)
            while not tts_queue.empty() and len(batch) < 4:
                batch.append(tts_queue.get_nowait())
        except asyncio.TimeoutError:
            continue
        if batch:
            texts = [item['text'] for item in batch]
            wavs = tts_model.infer_batch(texts, voice=default_voice)
            for item, wav in zip(batch, wavs):
                item['future'].set_result(wav)
```

---

### 10. ONNX Runtime / TensorRT Export

**Codec Decoder** (best candidate -- already supported):

```python
# VieNeu-TTS already supports ONNX codec decoder:
vieneu_tts = Vieneu(
    mode="standard",
    codec_repo="neuphonic/neucodec-onnx-decoder-int8",
    codec_device="cpu",  # ONNX decoder currently CPU-only
)
```

**TensorRT-LLM** for backbone models (complex but up to 5-7x speedup):

```bash
pip install tensorrt-llm

trtllm-build --model_dir pnnbao-ump/VieNeu-TTS-0.3B \
    --output_dir ./trt_model \
    --gemm_plugin bfloat16 \
    --max_batch_size 4 \
    --max_input_len 1024 \
    --max_seq_len 2048
```

**Priority**: LOW -- high engineering effort, LMDeploy/vLLM achieve 80% of the benefit.

---

### 11. CUDA Streams for Concurrent Model Execution

Run ASR and TTS on separate CUDA streams for GPU-level parallelism:

```python
asr_stream = torch.cuda.Stream()
tts_stream = torch.cuda.Stream()

async def process_request(audio_bytes, session_id):
    with torch.cuda.stream(asr_stream):
        user_text = speech_to_text(audio_bytes)
    asr_stream.synchronize()

    assistant_data = chat_with_gemini(user_text, session_id)  # CPU/network

    with torch.cuda.stream(tts_stream):
        audio_wav = text_to_speech(assistant_data["vi"])
    return audio_wav
```

**Expected Speedup**: Minimal for single requests (pipeline is sequential). 20-40% throughput under concurrent multi-user load.

---

## Blackwell Architecture: What Makes RTX 5080 Different

### RTX 5080 Hardware Specs

| Spec | Value |
|---|---|
| Architecture | NVIDIA Blackwell (GB203-400, SM 100) |
| Compute Capability | 10.0 |
| CUDA Cores | 10,752 |
| Tensor Cores | 336 (5th generation) |
| VRAM | 16 GB GDDR7 |
| Memory Bandwidth | **960 GB/s** (34% faster than RTX 4080 Super's 717 GB/s) |
| Memory Bus | 256-bit |
| FP32 Performance | 56.3 TFLOPS |
| Supported Precisions | FP4, FP8, FP16, BF16, TF32, FP32, INT8 |

### Why Memory Bandwidth Matters Most for Your Models

Your models (0.3B-0.6B params) are **memory-bandwidth-bound**, not compute-bound. During autoregressive token generation, the GPU spends most time **reading model weights from VRAM**, not doing math. This means:

- The RTX 5080's **960 GB/s GDDR7 bandwidth** directly translates to ~34% faster token generation vs RTX 4080 Super
- FP8/FP4 quantization won't help much -- it accelerates compute (GEMM), which isn't your bottleneck
- The best optimizations **reduce memory traffic** (KV cache quantization, prefix caching) or **eliminate CPU overhead** (CUDA graphs, torch.compile)

### 5th-Generation Tensor Cores -- New Capabilities

| Feature | Ada Lovelace (RTX 40) | Blackwell (RTX 50) |
|---|---|---|
| FP4 (MXFP4) | Not supported | **Native hardware support** |
| FP6 (MXFP6) | Not supported | **Native hardware support** |
| FP8 (E4M3/E5M2) | Supported | Supported |
| BF16/FP16 | Supported | Supported |
| Structured Sparsity (2:4) | Supported | Supported |
| Transformer Engine | 1st gen | **2nd gen** (MXFP4/MXFP6 support) |

**FP4 is NOT recommended for your use case** -- your small models fit in 16GB at BF16, and FP4 quality degradation is severe on 0.3B-0.6B models. FP4 is useful for fitting 7B+ models into 16GB VRAM.

### Required Software Stack for Blackwell

| Software | Minimum Version | Recommended |
|---|---|---|
| NVIDIA Driver | 560+ | 580+ |
| CUDA Toolkit | 12.8 | 12.8+ |
| PyTorch | 2.7.0 | **2.7+** with cu128 wheels |
| cuDNN | 9.x | Latest 9.x |
| Triton | 3.3+ | Bundled with PyTorch 2.7 |

**Critical**: Always install PyTorch with cu128 index:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Blackwell-Specific torch.compile Optimizations

**Use `max-autotune` mode** on Blackwell -- it benchmarks CUTLASS kernels vs Triton kernels and selects the fastest for your specific GPU:

```python
# Best for Blackwell -- slower compilation but fastest runtime
model = torch.compile(model, mode="max-autotune")

# Save compiled artifacts to avoid recompilation on server restart (PyTorch 2.7+)
import torch.compiler
torch.compiler.save_cache_artifacts()  # Mega Cache feature
```

PyTorch 2.7+ Blackwell features:
- **Triton 3.3** with SM 100 architecture support
- **Inductor CUTLASS backend** for optimized GEMM operations
- **FlexAttention** with FlashAttention 4 backend on Blackwell
- **Mega Cache** for portable compiled model caching (avoids recompilation)
- **Prologue Fusion** in Inductor compiler

### FlashAttention 4 via FlexAttention (New for Blackwell)

As of March 2026, PyTorch includes a **FlashAttention 4** backend specifically for Blackwell GPUs, accessible via FlexAttention:

```python
# FlexAttention auto-selects FA4 on Blackwell
from torch.nn.attention.flex_attention import flex_attention

# For HuggingFace models, SDPA will auto-select the best backend:
model = AutoModelForCausalLM.from_pretrained(
    "pnnbao-ump/VieNeu-TTS-0.3B",
    attn_implementation="sdpa",  # Will use FA4 on Blackwell if available
)
```

Note: FlashAttention **3** does NOT support Blackwell (Hopper H100 only). Use either:
- FlashAttention **2** v2.7.4.post1 (pre-built Blackwell wheels available)
- FlashAttention **4** via FlexAttention (PyTorch 2.7+ built-in)

### Library Compatibility on Blackwell (Verified)

| Library | Blackwell Status | Notes |
|---|---|---|
| **faster-qwen3-tts** | Working | Confirmed after Qwen3-TTS issue #132 resolution |
| **LMDeploy** | Working | CUDA 12.8 deployment verified (issue #3931) |
| **vLLM** | Working | Use `pip install vllm --pre` for latest support |
| **HuggingFace Transformers** | Working | SM 100 support since issue #37824 (June 2025) |
| **FlashAttention 2** | Working | v2.7.4.post1 has Blackwell wheels |
| **FlashAttention 3** | **NOT supported** | Hopper (H100) only |
| **bitsandbytes** | Working | But not recommended for speed on small models |

### Blackwell-Specific Qwen3-TTS Issue

Qwen3-TTS has a known problem on Blackwell GPUs: **GPU utilization stuck at 4-16%** (issue #89, #132).

**Root cause**: The architecture generates 16 audio tokens per step (RVQ codec with 16 quantizers at 12Hz), creating ~500 tiny CUDA kernel launches per decode step. The CPU can't dispatch them fast enough.

**Solution**: Use `faster-qwen3-tts` which captures all kernels into a single CUDA graph, achieving ~6x speedup (RTF < 0.25 on RTX 5090). This is already listed as Step 3 in our recommendations.

### How to Maximize RTX 5080 Bandwidth Utilization

Since your small models are bandwidth-bound, focus on reducing memory traffic:

| Strategy | How It Helps |
|---|---|
| Keep all models in GPU memory | Avoids CPU-GPU PCIe transfers |
| Use BF16 (already doing this) | Halves data movement vs FP32 |
| CUDA graphs (`faster-qwen3-tts`, `torch.compile`) | Eliminates CPU kernel launch overhead |
| KV cache INT8 quantization (LMDeploy `quant_policy=8`) | Reduces cache memory traffic by 50% |
| Prefix caching (LMDeploy `enable_prefix_caching=True`) | Avoids redundant memory reads |
| Static batch sizes | Enables CUDA graph capture |

---

## Known RTX 5080 / Blackwell Issues

| Issue | Solution |
|---|---|
| Slow inference, GPU utilization 4-5% ([Qwen3-TTS #89](https://github.com/QwenLM/Qwen3-TTS/issues/89)) | Use `faster-qwen3-tts` with CUDA graphs |
| Qwen3-TTS "Inefficient on Blackwell" ([#132](https://github.com/QwenLM/Qwen3-TTS/issues/132)) | Switch to optimized fork with torch.compile + CUDA graphs |
| FlashAttention 3 incompatible with Blackwell | Use FlashAttention 2 v2.7.4.post1 or FA4 via FlexAttention |
| PyTorch fails to detect RTX 5080 | Use PyTorch 2.7+ with cu128 index |
| `torch.compile` CUDA graph capture fails | Use PyTorch 2.5.1+ (2.7+ recommended for Blackwell) |
| "no kernel image available" errors | Rebuild with CUDA 12.8+ and correct SM 100 target |
| Silent CUDA hangs under high VRAM pressure ([PyTorch #178491](https://github.com/pytorch/pytorch/issues/178491)) | Keep PyTorch updated, monitor VRAM usage |
| LMDeploy "no kernel image" on RTX 50-series | Use latest LMDeploy with CUDA 12.8 support |
| GPT-SoVITS / older TTS libs crash on RTX 50 | Requires PyTorch 2.7+ cu128 wheels |

**Correct software installation for RTX 5080**:

```bash
# PyTorch with CUDA 12.8 (REQUIRED for Blackwell)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify Blackwell is detected
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'CUDA: {torch.version.cuda}'); print(f'Compute capability: {torch.cuda.get_device_capability(0)}')"
# Expected: NVIDIA GeForce RTX 5080, CUDA: 12.8, Compute capability: (10, 0)
```

---

## Quick-Start: Minimum Changes for Maximum Impact

Apply these changes to `demo2_server.py` for the biggest speedup with minimal code changes:

```python
# === ADD AT THE VERY TOP (before any model loading) ===
import torch
torch.set_float32_matmul_precision('high')

# === ASR: Use vLLM backend (Linux only) ===
from qwen_asr import Qwen3ASRModel
asr_model = Qwen3ASRModel.LLM(
    "Qwen/Qwen3-ASR-0.6B",
    gpu_memory_utilization=0.3,
    max_inference_batch_size=16,
    max_new_tokens=256,
)

# === VieNeu-TTS: Switch to fast mode (LMDeploy) ===
from vieneu import Vieneu
vieneu_tts = Vieneu(
    mode="fast",
    backbone_repo="finetune/output/merged_model",
    backbone_device="cuda",
    codec_device="cuda",
    memory_util=0.3,
    quant_policy=8,
    enable_prefix_caching=True,
)

# === Qwen3-TTS: Use faster-qwen3-tts with CUDA graphs ===
from faster_qwen3_tts import FasterQwen3TTS
qwen_tts = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
```

**Additional pip installs required**:

```bash
pip install faster-qwen3-tts lmdeploy vllm --pre "vllm[audio]"
```

---

## Why NOT FP8/FP4 Quantization for This Project

**FP8 and FP4 are NOT recommended** for speed with your 0.3B-0.6B models:

| Reason | Detail |
|---|---|
| Models are bandwidth-bound | FP8/FP4 accelerates compute (GEMM), which isn't your bottleneck |
| Models already fit in VRAM | ~5-6GB at BF16 out of 16GB -- no memory pressure |
| Quality degradation | Proportionally worse on smaller models (fewer params to absorb quantization error) |
| Dequantization overhead | bitsandbytes 4-bit/8-bit can make inference **SLOWER** for small models |
| Realistic gain | Only 5-15% at best, vs 5-6x from CUDA graphs |

**When FP8/FP4 WOULD help**: Running 7B+ models on 16GB VRAM, or batching many concurrent requests where compute becomes the bottleneck.

---

## Sources

- [NVIDIA RTX Blackwell GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [Puget Systems RTX 5090 & 5080 AI Review](https://www.pugetsystems.com/labs/articles/nvidia-geforce-rtx-5090-amp-5080-ai-review/)
- [RTX 5080 vs RTX 4090 AI Benchmarks](https://www.bestgpusforai.com/gpu-comparison/5080-vs-4090)
- [PyTorch 2.7 Release Notes (Blackwell support)](https://pytorch.org/blog/)
- [CUDA 12.8 Blackwell Support](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/)
- [Puget Systems Blackwell GenAI Software Support](https://www.pugetsystems.com/labs/articles/nvidia-blackwell-gpu-genai-software-support/)
- [HuggingFace torch.compile Documentation](https://huggingface.co/docs/transformers/en/perf_torch_compile)
- [HuggingFace GPU Inference Optimization](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one)
- [FlashAttention Blackwell Wheels](https://github.com/Zarrac/flashattention-blackwell-wheels-whl-ONLY-5090-5080-5070-5060-flash-attention-)
- [FlashAttention Blackwell Issue #1683](https://github.com/Dao-AILab/flash-attention/issues/1683)
- [NVIDIA Transformer Engine (FP8/FP4)](https://github.com/NVIDIA/TransformerEngine)
- [faster-qwen3-tts GitHub](https://github.com/andimarafioti/faster-qwen3-tts)
- [Qwen3-TTS Slow Inference on 5090 Issue #89](https://github.com/QwenLM/Qwen3-TTS/issues/89)
- [Qwen3-TTS Inefficient on Blackwell Issue #132](https://github.com/QwenLM/Qwen3-TTS/issues/132)
- [Qwen3-ASR vLLM Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html)
- [vLLM Supported Models](https://docs.vllm.ai/en/v0.12.0/models/supported_models/)
- [LMDeploy RTX 5090 Deployment Issue #3931](https://github.com/InternLM/lmdeploy/issues/3931)
- [TensorRT-LLM Qwen3 Support](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md)
- [HuggingFace SM 100 Support Issue #37824](https://github.com/huggingface/transformers/issues/37824)
- [PyTorch Blackwell CUDA Hang Issue #178491](https://github.com/pytorch/pytorch/issues/178491)
