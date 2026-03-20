# Recording Guide for VieNeu-TTS Fine-tuning
# Platform: Windows OS

## Why Audio Quality Matters So Much

VieNeu-TTS uses **NeuCodec** to encode audio into neural tokens during training. Any noise, echo, or distortion in your recording gets encoded along with the voice — the model will learn to reproduce it. Clean recordings lead directly to a cleaner, more natural output voice.

**Key fact from the source code:**
- `encode_data.py` loads all audio at **16,000 Hz mono** via `librosa.load(audio_path, sr=16000, mono=True)`
- The TTS engine outputs at **24,000 Hz**
- This means you should record at **44,100 Hz or 48,000 Hz** — the script will downsample automatically. Recording at 16kHz directly would lose quality, since 16kHz captures only up to 8kHz in frequency range (telephone quality)

---

## 1. Technical Specifications

### Minimum Requirements

| Setting | Minimum | Recommended | Why |
|---|---|---|---|
| Sample Rate | 44,100 Hz | **48,000 Hz** | Downsampled to 16kHz by the script; higher source = better quality |
| Bit Depth | 16-bit | **24-bit** | 24-bit captures more dynamic detail before downsampling |
| Channels | Mono | **Mono** | The script forces mono; record in mono to avoid phase issues |
| Format | WAV | **WAV (PCM)** | Lossless; never use MP3 for recording |
| Peak Level | -6 dB | **-3 dB to -6 dB** | Headroom prevents clipping; too quiet = more noise floor |
| Noise Floor | < -50 dB | **< -60 dB** | Silence should be truly silent |
| File Naming | Match metadata.csv | `en_0001.wav`, `vi_0001.wav`, `mix_0001.wav` | Must match exactly |

### Room Noise Floor Test

Before any recording session, record **10 seconds of silence** in your room. Open the file in Audacity and check the waveform amplitude. It should be essentially invisible — if you can see a visible waveform during "silence," your room is too noisy.

---

## 2. Microphone Recommendations

The microphone is the single most important piece of equipment. For TTS training, you need a **condenser microphone** (not a dynamic/gaming mic) because condenser mics capture voice detail, consonant clarity, and Vietnamese tonal nuance more accurately.

### Budget-Friendly Options (Under $100)

| Microphone | Price | Connection | Why It Works |
|---|---|---|---|
| **Blue Snowball iCE** | ~$50 | USB | Cardioid pattern, clear voice capture, plug-and-play on Windows |
| **TONOR TC-777** | ~$40 | USB | Good budget option, built-in stand |
| **Fifine K669B** | ~$30 | USB | Entry-level condenser, decent clarity for the price |
| **Samson Go Mic** | ~$40 | USB | Compact, cardioid/omni switchable |

### Mid-Range Options (Best for TTS, $100–250)

| Microphone | Price | Connection | Why It Works |
|---|---|---|---|
| **Blue Yeti** | ~$110 | USB | Industry standard for voice work; cardioid mode is excellent |
| **Audio-Technica AT2020USB+** | ~$150 | USB | Studio-quality condenser, very clear high-frequency detail |
| **Rode NT-USB Mini** | ~$100 | USB | Excellent off-axis rejection, warm clear sound |
| **HyperX QuadCast S** | ~$160 | USB | Good cardioid pattern, built-in shock mount |

### Professional Options ($250+)

| Microphone | Price | Connection | Interface Needed? |
|---|---|---|---|
| **Audio-Technica AT2020** | ~$100 + interface | XLR | Yes — needs audio interface |
| **Rode NT1-A** | ~$230 + interface | XLR | Yes — very low noise floor, professional standard |
| **Shure SM7B** | ~$400 + interface | XLR | Yes — broadcast quality, excellent noise rejection |

> **Recommendation for your project:** The **Blue Yeti** or **Audio-Technica AT2020USB+** gives the best quality-to-price ratio for TTS training without needing additional equipment. USB condenser mics are simplest — plug in, configure in Windows, and record.

### What NOT to Buy

- ❌ **Headset/gaming microphones** — poor frequency response, plastic sound
- ❌ **Laptop built-in microphone** — too much room noise and keyboard noise
- ❌ **Lavalier/clip-on microphones** — inconsistent positioning, handle noise
- ❌ **Dynamic microphones** (e.g., Shure SM58) — designed for live performance, not studio voice

---

## 3. Microphone Setup

### Position

```
        [Speaker's mouth]
               |
            20–30 cm
               |
         [Microphone]        ← Slightly below mouth level, angled up ~15°
```

- Place the mic **20–30 cm (8–12 inches)** from the speaker's mouth
- Angle the microphone **slightly below the mouth**, pointing upward at ~15 degrees — this reduces plosives (the "p" and "b" air bursts)
- Never position the mic directly in front of the mouth (on-axis) — position it slightly off to the side to reduce plosive sounds
- Use a **pop filter** (foam windscreen or mesh screen) between the mouth and mic — essential for clean "p", "b", "ph" sounds in both English and Vietnamese

### Shock Mount

If your microphone did not come with a shock mount, place it on a **thick folded towel** on the table. This absorbs desk vibrations (typing, tapping) that can pollute recordings.

### USB Microphone Settings in Windows

1. Right-click the speaker icon in taskbar → "Sound settings"
2. Under "Input," select your USB microphone
3. Click "Device properties"
4. Set sample rate: **48,000 Hz, 24-bit** (or 44,100 Hz, 24-bit)
5. Disable **"Microphone Boost"** — this adds noise
6. Disable **"Echo Cancellation"** and **"Noise Suppression"** in Windows — these degrade voice quality for training; you want the raw signal

---

## 4. Room Setup for Recording

The recording room has as much impact on quality as the microphone itself.

### Find the Right Room

**Best rooms:**
- Walk-in closet (clothes absorb echo perfectly) — the single best option in a home
- Small bedroom with carpet, curtains, and soft furnishings
- A room with bookshelves full of books (books diffuse sound well)

**Rooms to avoid:**
- Large empty rooms (too much reverb)
- Kitchen or bathroom (hard reflective surfaces)
- Rooms near road noise, AC units, or foot traffic
- Any room where you can hear fan noise or electrical hum

### DIY Acoustic Treatment (Low Cost)

If you do not have a treated room, use these tricks:

**Option 1: The Blanket Fort Method (~$0)**
Hang thick blankets or duvets on three walls around the recording area. The speaker sits inside the "fort." This dramatically reduces room reflections.

**Option 2: Pillow/Blanket on Table Method**
Place the microphone on a table. Stack pillows around and behind the mic in a U-shape. The speaker faces the mic inside the pillow "cave."

**Option 3: Closet Recording**
Open a clothes closet, hang the mic from a hanger or stand, and record inside. The clothes absorb almost all reflections. Use a laptop or tablet to see the script.

### Noise Checklist Before Every Session

- [ ] Turn off ceiling fans and air conditioning
- [ ] Close windows and curtains
- [ ] Turn off refrigerators in adjacent rooms if possible (unplug for 30 min)
- [ ] Silence all phones (not just vibrate — vibrations are audible)
- [ ] Turn off computer notification sounds
- [ ] Disconnect from WiFi if your computer fan speeds up during transfers
- [ ] Tell everyone in the building not to walk heavily or slam doors
- [ ] Turn off fluorescent lights (they hum at 50–60 Hz)

---

## 5. Recording Software on Windows

### 🏆 Recommended: Audacity (Free)

**Download:** https://www.audacityteam.org/download/

Audacity is the industry standard for TTS data recording. It is free, open-source, and has everything you need. It is used by professional voice actors and TTS labs worldwide.

**Why Audacity is the best choice for this project:**
- Free and lightweight
- Real-time input level meter — see clipping before it happens
- Noise profile and noise reduction (for post-processing)
- Silence trimmer — automatically trims leading/trailing silence
- Built-in clip splitter — record long sessions and split into clips
- Labels/markers — mark each sentence during recording for easy export
- "Export Multiple" — export all labeled clips in one click with sequential filenames
- Supports 48,000 Hz / 24-bit recording natively

**Audacity Setup for VieNeu-TTS:**

1. Open Audacity
2. Edit → Preferences → Devices:
   - Recording Device: your USB microphone
   - Channels: **1 (Mono)**
3. Edit → Preferences → Quality:
   - Default Sample Rate: **48000 Hz**
   - Default Sample Format: **24-bit**
4. In the main toolbar, set:
   - Microphone level: aim for peaks between **-6 dB and -3 dB**
5. Enable the input level meter (View → Toolbars → Recording Meter Toolbar)

**Audacity Recording Workflow for metadata.csv:**

1. Open metadata.csv in a text editor (split screen)
2. In Audacity: press `R` to start recording
3. Read one sentence from metadata.csv clearly
4. Press `M` to add a label marker after each sentence (label it with the filename, e.g. `en_0001`)
5. Pause briefly between sentences (1–2 seconds)
6. After recording a batch of ~50 sentences, press `Space` to stop
7. Go to File → Export → Export Multiple:
   - Format: **WAV**
   - Split files based on: **Labels**
   - Name files: **Using Label/Track Name**
8. Files are saved automatically as `en_0001.wav`, `en_0002.wav`, etc.

---

### Alternative Option 1: Adobe Audition (Paid — $54/month via Creative Cloud)

**Best for:** Professional studios that already have Adobe subscriptions.

**Pros:**
- Excellent visual waveform editor
- Superior noise reduction tools (Adaptive Noise Reduction)
- Spectral frequency display — see noise in frequencies visually
- Auto-heal tool removes clicks and pops
- Batch processing for post-production

**Cons:** Expensive. Overkill for most beginners.

**When to choose:** If your recording environment is not ideal and you need powerful post-processing to clean up audio.

---

### Alternative Option 2: Reaper ($60 personal license, free trial)

**Download:** https://www.reaper.fm/

**Best for:** Users who want professional DAW features at low cost.

**Pros:**
- Extremely powerful routing and plugin support
- Supports ReaScript for automating repetitive tasks
- Great for recording long sessions with region markers
- Very lightweight on CPU/RAM

**Cons:** Steeper learning curve than Audacity. More features than needed for basic recording.

**When to choose:** If you are comfortable with DAW software and want more control over session management.

---

### Alternative Option 3: Ocenaudio (Free)

**Download:** https://www.ocenaudio.com/

**Best for:** Beginners who find Audacity overwhelming.

**Pros:**
- Very simple, clean interface
- Real-time preview of audio effects
- Good waveform visualization
- Free and no ads

**Cons:**
- No label/marker system for batch export
- Post-processing tools less powerful than Audacity
- No "Export Multiple" — must export clips one by one

**When to choose:** Simple spot recording where the speaker records one file per sentence (simpler but very slow for 10,000 clips).

---

### Alternative Option 4: Voice Meter + Audacity (Advanced Setup)

**Voicemeeter Banana (Free):** https://vb-audio.com/Voicemeeter/banana.htm

Voicemeeter is a virtual audio mixer for Windows. It sits between your microphone and Audacity, giving you real-time EQ, compression, and gate controls before the signal even reaches the recording software.

**Use case:** If your room has consistent low-level noise (AC hum, computer fan), use Voicemeeter to apply a noise gate that automatically silences the mic when the speaker is not talking.

---

## 6. Software Comparison Summary

| Feature | Audacity | Adobe Audition | Reaper | Ocenaudio |
|---|---|---|---|---|
| Price | **Free** | $54/month | $60 one-time | **Free** |
| Learning Curve | Low | High | Medium | Very Low |
| Label + Batch Export | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| Noise Reduction | Good | Excellent | Good (plugins) | Basic |
| 24-bit / 48kHz | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Real-time Level Meter | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Windows Support | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Recommended?** | **✅ Best choice** | Optional | Optional | Fallback |

---

## 7. Recording Workflow (Step-by-Step for This Project)

### Before You Start

```
[ ] Microphone connected and recognized by Windows
[ ] Audacity configured: Mono, 48000 Hz, 24-bit
[ ] Room is quiet (fans off, windows closed, phones silenced)
[ ] Pop filter in place, mic at 20–30 cm distance
[ ] metadata.csv open in Notepad or Excel (split screen)
[ ] Record 10 seconds of silence → check noise floor in Audacity
    (waveform should be a flat line — not a fuzzy one)
```

### During Recording

1. **Warm up** — Have the speaker talk for 2–3 minutes before recording starts. Cold voices produce inconsistent quality in the first few clips.

2. **Record in batches of 50 sentences** — do not try to record 500 clips in one go. Fatigue affects voice consistency. Aim for batches of 50, with 5-minute breaks.

3. **Speaking tips for English clips (slow teaching mode):**
   - Speak at ~100–120 words per minute (roughly half of normal speed)
   - Pause briefly at commas and periods
   - Pronounce every syllable clearly — especially word endings (-ed, -ing, -s)
   - Keep a warm, encouraging tone — like talking to a young student
   - Do NOT rush — the speaker must internalize "I am teaching a child" mindset

4. **Speaking tips for Vietnamese clips:**
   - Natural speed is fine (~160–180 wpm)
   - Vietnamese tones must be clear and accurate
   - Keep consistent energy — not too formal, not too casual

5. **Speaking tips for Mixed EN-VI clips:**
   - The switch between languages should be completely natural — no hesitation at switch points
   - The English words in Vietnamese sentences keep their English pronunciation (do not Vietnamize them)
   - Practice mixed sentences before recording them

6. **If a take is bad** — say "retake" clearly, pause 2 seconds, then re-read the sentence. The retake label helps during editing. Do NOT try to edit out mistakes mid-session.

7. **Label every clip in Audacity** — after reading each sentence, press `M` and type the filename (e.g., `en_0001`). This takes 5 seconds per clip and saves hours of manual renaming.

### After Recording

1. Export all labeled clips (File → Export → Export Multiple → WAV)
2. Spot-check 10 random files per batch — listen and verify they match the text
3. Run the quality check script (see below)
4. Move approved files into `finetune/dataset/raw_audio/`

---

## 8. Post-Processing in Audacity

After recording, do **minimal** post-processing. Over-processing damages voice quality. Only do these steps:

### Step 1: Noise Reduction (Only If Needed)

If your recordings have audible background noise:

1. Find a section of pure silence (between sentences)
2. Select 1–2 seconds of that silence
3. Effect → Noise Reduction → "Get Noise Profile"
4. Select all audio (Ctrl+A)
5. Effect → Noise Reduction → Reduction: **6–9 dB**, Sensitivity: **6**, Frequency Smoothing: **3**
6. Preview first — if voice sounds robotic, reduce the Reduction setting

> Do NOT use noise reduction if the recording is already clean. It always slightly degrades voice quality.

### Step 2: Normalize Volume

1. Select all (Ctrl+A)
2. Effect → Normalize:
   - Normalize peak amplitude to: **-1.0 dB**
   - Check "Normalize stereo channels independently": Yes

This ensures every clip has consistent volume.

### Step 3: Trim Silence

1. Analyze → Silence Finder — set minimum silence to 0.3 seconds
2. Or manually trim leading/trailing silence on each clip
3. Leave 0.2–0.3 seconds of natural room tone at the start and end (do not cut to absolute zero)

### What NOT to Do

- ❌ Do not use Equalization/EQ unless absolutely necessary
- ❌ Do not use Compression or Limiter — this flattens the natural voice dynamics
- ❌ Do not use Reverb (obviously)
- ❌ Do not use pitch correction
- ❌ Do not over-amplify — clips should peak at -3 dB to -1 dB

---

## 9. Quality Control Checklist

Run this check on every batch of recordings before moving to `raw_audio/`:

**Listen test (random 10 per batch of 50):**
- [ ] No audible background noise during speech
- [ ] No echo or reverb
- [ ] No clipping (distortion on loud syllables)
- [ ] No "pop" sounds on P/B sounds (plosives)
- [ ] Speech matches the metadata.csv text exactly
- [ ] English clips sound slow and clear (teaching pace)
- [ ] Vietnamese tones are accurate
- [ ] Mixed clips have natural language switches (no robotic pause)

**Technical check (Audacity or Windows File Explorer):**
- [ ] All files are .wav format
- [ ] File size is reasonable (~1–3 MB per clip; if under 50KB or over 10MB, something is wrong)
- [ ] Filenames match metadata.csv exactly (`en_0001.wav` not `en0001.wav` or `en_1.wav`)

**Quick Python check after uploading to server:**

```bash
# On the GPU server, after placing files in raw_audio/
uv run python finetune/data_scripts/filter_data.py
```

Look at the output. If more than 20% of your clips are rejected for `duration_out_of_range`, your pacing is off — the speaker is reading too fast or too slow.

---

## 10. Recording Session Plan for 10,000 Clips

| Session | Clips | Category | Est. Recording Time | Est. Studio Time |
|---|---|---|---|---|
| Session 1 | 700 | en short (3–5s) | 70 min | ~3h |
| Session 2 | 1,050 | en medium-short (6–8s) | 120 min | ~4h |
| Session 3 | 1,050 | en medium-long (9–11s) | 175 min | ~5h |
| Session 4 | 700 | en long (12–15s) | 145 min | ~4h |
| Session 5 | 700 | vi short (3–5s) | 60 min | ~2.5h |
| Session 6 | 1,050 | vi medium-short (6–8s) | 110 min | ~3.5h |
| Session 7 | 1,050 | vi medium-long (9–11s) | 165 min | ~5h |
| Session 8 | 700 | vi long (12–15s) | 140 min | ~4h |
| Session 9 | 600 | mix short (3–5s) | 55 min | ~2.5h |
| Session 10 | 900 | mix medium-short (6–8s) | 100 min | ~3.5h |
| Session 11 | 900 | mix medium-long (9–11s) | 145 min | ~4.5h |
| Session 12 | 600 | mix long (12–15s) | 120 min | ~4h |
| **Total** | **10,000** | | **~23h raw audio** | **~45h total** |

> "Studio time" includes setup, warm-up, breaks, retakes, and export. Expect 2x the raw recording time.
> Spread across **12–15 sessions** of 3–4 hours each over 3–4 weeks.

---

## 11. Common Problems and Fixes

| Problem | Symptom | Fix |
|---|---|---|
| Room echo | Waveform has a "shadow" after each word | Record in a closet or use blanket fort |
| Plosive pops | Sharp spikes on P/B sounds | Add pop filter, angle mic below mouth |
| Clipping | Flat tops on waveform peaks | Lower microphone gain / move mic back |
| Low volume | Waveform is tiny, lots of amplification needed | Move mic closer, increase input gain |
| Computer fan noise | Constant low hiss | Use laptop on battery, place mic away from PC |
| Inconsistent volume | Some clips quiet, some loud | Normalize in Audacity after every session |
| Windows processing audio | Weird artifacts | Disable "Exclusive Mode" in Windows sound settings and disable all audio enhancements |
| filter_data.py rejects too many | >20% rejected for duration | Speaker is reading too fast (EN) or too slow (VI); adjust pace |

---

## 12. Final Setup Checklist for Windows

```
Hardware:
[ ] Condenser USB microphone (Blue Yeti or AT2020USB+ recommended)
[ ] Pop filter or foam windscreen
[ ] Microphone stand or arm
[ ] Headphones for monitoring playback (not speakers)

Software:
[ ] Audacity installed: https://www.audacityteam.org/download/
[ ] Audacity configured: Mono, 48000 Hz, 24-bit
[ ] Windows audio enhancements disabled
[ ] Microphone Boost = 0 in Windows settings
[ ] metadata.csv open alongside Audacity (split screen)

Room:
[ ] Quiet room chosen (closet preferred)
[ ] All fans and AC turned off
[ ] Windows closed, curtains drawn
[ ] Phones silenced

Test:
[ ] Record 10 seconds silence → noise floor is flat in Audacity
[ ] Record 1 test sentence → listen back, check level -6dB to -3dB
[ ] No echo, no noise, no clipping → ready to record
```
