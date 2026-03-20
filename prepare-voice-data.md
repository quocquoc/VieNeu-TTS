# Prepare Voice Data for VieNeu-TTS Fine-tuning
# Use Case: AI English Teacher for Vietnamese Children

## Goal

Fine-tune VieNeu-TTS to create an **AI English teaching assistant** for Vietnamese children. The voice model should:
- Speak **clear, slow, well-pronounced English** (teaching mode)
- Speak **natural Vietnamese** (for explanations and instructions)
- **Mix Vietnamese and English smoothly in one sentence** (for bilingual teaching scenarios)

Total data: **~20 hours**, split into three categories.

---

## 1. Data Split Overview

| Category | Hours | % | Purpose |
|---|---|---|---|
| Pure English (slow & clear) | 7h | 35% | Teach English pronunciation, vocabulary, reading |
| Pure Vietnamese (natural) | 7h | 35% | Explain grammar, give instructions, tell stories in Vietnamese |
| Mixed EN-VI (code-switching) | 6h | 30% | Bilingual teaching: explain English words in Vietnamese mid-sentence |
| **Total** | **20h** | **100%** | |

> **Why 30% mixed?** In a real English lesson for Vietnamese children, the teacher constantly switches between languages — saying an English word then explaining it in Vietnamese, or giving Vietnamese instructions with English vocabulary. The model needs substantial code-switching data to learn natural transitions between English and Vietnamese sounds.

> **Note on data amount:** The official VieNeu-TTS guide recommends 2–4 hours for cloning a single voice. We use 20 hours because we are not just cloning a voice — we are teaching the model a new **speaking style** (slow teaching English) and **code-switching behavior** across two languages. More data is needed to learn these patterns reliably.

---

## 2. Audio Duration Distribution

Per the VieNeu-TTS `filter_data.py`, each clip must be **3–15 seconds**. Distribute your clips as follows:

| Duration | % | Pure EN Clips | Pure VI Clips | Mixed Clips | Total |
|---|---|---|---|---|---|
| 3–5s | 20% | ~700 | ~700 | ~600 | ~2,000 |
| 6–8s | 30% | ~1,050 | ~1,050 | ~900 | ~3,000 |
| 9–11s | 30% | ~1,050 | ~1,050 | ~900 | ~3,000 |
| 12–15s | 20% | ~700 | ~700 | ~600 | ~2,000 |
| **Total** | | **~3,500** | **~3,500** | **~3,000** | **~10,000** |

**Why include short 3–5s clips?**
- Short clips are great for single vocabulary words with explanation: "Apple. Quả táo."
- The model learns crisp, isolated pronunciation from short clips
- The `filter_data.py` script accepts audio from 3 seconds onward

**Why include longer 12–15s clips?**
- Longer clips are needed for reading passages and storytelling
- They teach the model to maintain consistent slow pacing over longer phrases

---

## 3. Content Design for English Teaching

### 3.1 Pure English Content (7 hours) — Slow & Clear

This is **teaching English**, not conversational English. The speaker must read slowly, clearly, with exaggerated pronunciation — like a teacher reading to a class of young students.

**Speaking style requirements:**
- Speed: **~100–120 words per minute** (normal conversation is 150–180 wpm)
- Pause briefly between phrases (0.3–0.5 seconds)
- Pronounce each syllable clearly, especially word endings (-ed, -s, -ing)
- Use a warm, encouraging, patient tone

| Content Type | % of EN Data | Duration | Example |
|---|---|---|---|
| **Vocabulary & short phrases** | 20% | 1.4h | "Cat. This is a cat. The cat is small." |
| **Simple sentences (present tense)** | 20% | 1.4h | "I go to school every day. She reads a book." |
| **Questions & answers** | 15% | ~1h | "What color is the sky? The sky is blue." |
| **Short stories & reading passages** | 20% | 1.4h | "One day, a little bird found a big red apple under the tree." |
| **Instructions & classroom phrases** | 10% | 0.7h | "Please open your book to page five. Now repeat after me." |
| **Songs & rhymes (spoken, not sung)** | 5% | ~0.35h | "Twinkle twinkle little star, how I wonder what you are." |
| **Numbers, colors, days, months** | 10% | 0.7h | "Monday, Tuesday, Wednesday, Thursday, Friday." |

### 3.2 Pure Vietnamese Content (7 hours) — Natural Speed

This is the teacher speaking Vietnamese to explain concepts, give instructions, and tell stories. Natural speaking speed, warm tone.

| Content Type | % of VI Data | Duration | Example |
|---|---|---|---|
| **Grammar explanations** | 25% | 1.75h | "Trong tiếng Anh, khi nói về thói quen hàng ngày, chúng ta dùng thì hiện tại đơn." |
| **Instructions & encouragement** | 20% | 1.4h | "Giỏi lắm! Bây giờ các con thử đọc lại câu này một lần nữa nhé." |
| **Vietnamese storytelling** | 20% | 1.4h | "Ngày xửa ngày xưa, có một chú thỏ nhỏ rất thông minh." |
| **Daily conversations** | 15% | ~1h | "Hôm nay trời đẹp quá, các con có muốn học bài ngoài sân không?" |
| **Questions to students** | 10% | 0.7h | "Bạn nào biết từ này nghĩa là gì không?" |
| **Cultural content** | 10% | 0.7h | "Tết Nguyên Đán là ngày lễ quan trọng nhất của người Việt Nam." |

### 3.3 Mixed EN-VI Content (6 hours) — Code-Switching for Teaching

This is the most critical category. It mimics how a real bilingual teacher speaks when teaching English to Vietnamese children — constantly switching between languages.

**Six code-switching patterns:**

#### Pattern 1: Teach a word — say English, explain in Vietnamese (25% = 1.5h)

The most common teaching pattern. Say the English word/phrase, then immediately explain in Vietnamese.

```
mix_0001.wav|Apple, quả táo, các con đọc theo cô nhé, apple.
mix_0002.wav|Beautiful nghĩa là đẹp, very beautiful nghĩa là rất đẹp.
mix_0003.wav|Repeat after me, lặp lại theo cô nhé, the cat is sleeping.
mix_0004.wav|Good morning nghĩa là chào buổi sáng, các con nói thử đi.
mix_0005.wav|Library là thư viện, I go to the library, tôi đi đến thư viện.
```

#### Pattern 2: Vietnamese instruction with English target words (20% = 1.2h)

Teacher gives instructions in Vietnamese but uses English for the words being taught.

```
mix_0101.wav|Các con mở sách ra trang có hình con cat nhé.
mix_0102.wav|Hôm nay chúng ta sẽ học về các loại fruit, như apple, banana, và orange.
mix_0103.wav|Bạn nào đọc được chữ elephant cho cô nghe nào?
mix_0104.wav|Bây giờ cô sẽ đọc một story, các con nghe và tìm các từ mới nhé.
mix_0105.wav|Ai biết happy nghĩa là gì, giơ tay lên nào!
```

#### Pattern 3: English sentence then Vietnamese translation (20% = 1.2h)

Full sentence in English followed by its Vietnamese meaning — a classic teaching technique.

```
mix_0201.wav|I like to read books, tôi thích đọc sách.
mix_0202.wav|The weather is very nice today, hôm nay thời tiết rất đẹp.
mix_0203.wav|My mother cooks delicious food, mẹ tôi nấu ăn rất ngon.
mix_0204.wav|Please sit down and open your notebook, hãy ngồi xuống và mở vở ra.
mix_0205.wav|How old are you, bạn bao nhiêu tuổi?
```

#### Pattern 4: Mid-sentence switch — natural teacher talk (15% = 0.9h)

Teacher starts in one language and naturally transitions to the other mid-thought.

```
mix_0301.wav|Very good, giỏi lắm, bạn Minh đọc rất hay!
mix_0302.wav|Hôm nay cô sẽ teach các con about animals, nhé!
mix_0303.wav|Now let us practice, bây giờ mình luyện tập nhé.
mix_0304.wav|Cô rất proud of you, các con học giỏi quá!
mix_0305.wav|Remember homework là bài tập về nhà, đừng quên làm nhé!
```

#### Pattern 5: Dialogue practice — teacher reads both roles (10% = 0.6h)

Teacher demonstrates both sides of a conversation, mixing explanation.

```
mix_0401.wav|Hello, how are you, xin chào bạn khỏe không, I am fine thank you, tôi khỏe cảm ơn bạn.
mix_0402.wav|What is your name, bạn tên gì, my name is Lan, tên tôi là Lan.
mix_0403.wav|Do you like ice cream, bạn có thích kem không, yes I do, có tôi thích.
mix_0404.wav|How old are you, bạn bao nhiêu tuổi, I am seven years old, tôi bảy tuổi.
mix_0405.wav|Where do you live, bạn sống ở đâu, I live in Ha Noi, tôi sống ở Hà Nội.
```

#### Pattern 6: Short code-switches — greetings, praise, exclamations (10% = 0.6h)

Quick, short switches common in classroom settings.

```
mix_0501.wav|Hello các con, hôm nay học bài mới nhé!
mix_0502.wav|Very good, giỏi lắm!
mix_0503.wav|Okay, bây giờ cô đọc trước nhé.
mix_0504.wav|Wow, excellent, bạn phát âm chuẩn lắm!
mix_0505.wav|Goodbye các con, see you tomorrow!
```

### Mixed Data Summary

| Pattern | % | Hours | ~Clips | Key Feature |
|---|---|---|---|---|
| Teach word + explain in Vietnamese | 25% | 1.5h | ~750 | Core teaching pattern |
| Vietnamese instruction + English words | 20% | 1.2h | ~600 | Classroom instructions |
| English sentence + Vietnamese translation | 20% | 1.2h | ~600 | Translation pairs |
| Mid-sentence natural switch | 15% | 0.9h | ~450 | Fluent code-switching |
| Dialogue practice | 10% | 0.6h | ~300 | Conversation demos |
| Short switches | 10% | 0.6h | ~300 | Classroom phrases |
| **Total** | **100%** | **6h** | **~3,000** | |

---

## 4. Data Source Options

### Recording Your Own (Strongly Recommended)

Mixed EN-VI teaching data does not exist in public datasets. You must record it yourself. For pure EN and VI, you can supplement with open-source data.

**Ideal speaker profile:**
- A female or male teacher with a warm, patient voice
- Fluent in both English and Vietnamese
- Comfortable with slow, clear English pronunciation
- Age range voice: 20–35 (young teacher voice appeals to children)

> **You do NOT need a child's voice.** The use case is an AI *teacher* — a clear adult voice that children can learn from. A child's voice would actually be worse for pronunciation teaching.

**Equipment needed:**
- USB condenser microphone (e.g., Blue Yeti, Audio-Technica AT2020USB) — ~$50–100
- Pop filter — ~$10
- Quiet room with soft surfaces (blankets, curtains) to reduce echo

**Recording settings:**
- Sample rate: **44,100 Hz or 48,000 Hz** (will be resampled to 16,000 Hz by `encode_data.py`)
- Bit depth: 16-bit or 24-bit
- Format: WAV (uncompressed)
- Mono channel

**Recording session plan:**

| Session | Duration | Content |
|---|---|---|
| Session 1 | ~4h | Pure English (slow reading) |
| Session 2 | ~4h | Pure English + Pure Vietnamese |
| Session 3 | ~4h | Pure Vietnamese + Mixed EN-VI |
| Session 4 | ~4h | Mixed EN-VI (all patterns) |
| Session 5 | ~4h | Mixed EN-VI + re-records |
| **Total studio time** | **~20h** | (includes breaks, retakes, setup) |

### Supplementing with Open-Source Data

For pure English and Vietnamese portions only:

| Dataset | Language | Notes |
|---|---|---|
| LibriSpeech (clean subsets) | English | Good quality; select slower speakers |
| Common Voice (Mozilla) | EN / VI | Filter by recording quality |
| VIVOS | Vietnamese | Clean Vietnamese read speech |
| HuggingFace Hub | Both | Search "tts", "speech", "audiobook" |

> **Important:** Any open-source audio must be re-recorded or speed-adjusted to match your slow, clear English teaching style. Normal-speed English datasets are too fast for teaching purposes.

### Practical Approach

| Category | Self-recorded | From Datasets | Total |
|---|---|---|---|
| Pure English (slow) | 4h | 3h (slowed down) | 7h |
| Pure Vietnamese | 3h | 4h | 7h |
| Mixed EN-VI | **6h (all self-recorded)** | 0h | 6h |
| **Total** | **13h** | **7h** | **20h** |

---

## 5. Data Quality Requirements

### Audio Quality Checklist

- [ ] **No background noise** — no music, TV, traffic, fan noise
- [ ] **No reverb/echo** — record in a treated room or use a closet
- [ ] **No clipping** — peak volume should be below -1 dB
- [ ] **Consistent volume** — normalize all clips to -3 dB to -1 dB peak
- [ ] **Single speaker** — only one voice per clip
- [ ] **No long silence** — trim leading/trailing silence to less than 0.5 seconds
- [ ] **Clean speech** — no coughs, lip smacks, or "umm/uhh" fillers

### Text Quality Checklist (Matching filter_data.py Rules)

The `filter_data.py` script in VieNeu-TTS **rejects** clips that fail these rules:

- [ ] Audio duration must be **3–15 seconds** (clips outside this range are removed)
- [ ] Text must match audio **100% exactly**
- [ ] Every sentence must **end with punctuation**: `.` `,` `?` `!`
- [ ] **No digits** in text — write them out: "5" becomes "five" or "năm"
- [ ] **No acronyms** — write them out: "ABC" becomes "A B C"
- [ ] Text must not be empty

### Special Rules for Mixed EN-VI Text

- [ ] Write the text **exactly as spoken** — do not translate or "fix" the language mixing
- [ ] Keep Vietnamese diacritics accurate (ă, â, đ, ê, ô, ơ, ư and tones)
- [ ] English words within Vietnamese sentences keep English spelling
- [ ] The switch point must match exactly where it happens in the audio

### Special Rules for Slow English

- [ ] The audio must actually be slow (~100–120 wpm), not normal speed
- [ ] Pauses between phrases are natural, not artificially inserted silence
- [ ] Pronunciation is clear but not robotic — still warm and friendly

---

## 6. Folder Structure

```
VieNeu-TTS/
  finetune/
    dataset/
      raw_audio/               <-- ALL .wav files go here
        en_0001.wav
        en_0002.wav
        ...
        vi_0001.wav
        vi_0002.wav
        ...
        mix_0001.wav
        mix_0002.wav
        ...
      metadata.csv             <-- filename|text (pipe-separated)
```

### metadata.csv Format

One line per audio clip. Format: `filename|text` (separated by pipe `|`).

```
en_0001.wav|Cat. This is a cat. The cat is small.
en_0002.wav|Please open your book to page five.
vi_0001.wav|Hôm nay chúng ta sẽ học về các con vật nhé.
vi_0002.wav|Giỏi lắm, các con đọc lại một lần nữa nào.
mix_0001.wav|Apple, quả táo, các con đọc theo cô nhé, apple.
mix_0002.wav|Very good, giỏi lắm, bạn phát âm chuẩn lắm!
```

### Naming Convention

| Prefix | Category | Range |
|---|---|---|
| `en_` | Pure English (slow, clear) | `en_0001.wav` – `en_3500.wav` |
| `vi_` | Pure Vietnamese (natural) | `vi_0001.wav` – `vi_3500.wav` |
| `mix_` | Mixed EN-VI (code-switching) | `mix_0001.wav` – `mix_3000.wav` |

---

## 7. Step-by-Step Workflow

### Step 1: Write Scripts Before Recording

Create three script files:

```
scripts/
  scripts_en.txt       # ~3,500 English sentences (slow reading style)
  scripts_vi.txt       # ~3,500 Vietnamese sentences
  scripts_mix.txt      # ~3,000 mixed EN-VI sentences
```

For mixed scripts, label each line with its pattern:

```
[P1] Apple, quả táo, các con đọc theo cô nhé, apple.
[P2] Hôm nay chúng ta sẽ học về các loại fruit, như apple, banana, và orange.
[P3] I like to read books, tôi thích đọc sách.
[P4] Very good, giỏi lắm, bạn Minh đọc rất hay!
[P5] Hello, how are you, xin chào bạn khỏe không, I am fine, tôi khỏe.
[P6] Okay, bây giờ cô đọc trước nhé.
```

> Remove the `[Px]` labels when building metadata.csv.

### Step 2: Record or Collect Raw Audio

Save all as `.wav` format. Convert other formats:

```bash
sudo apt install ffmpeg

# Convert single file
ffmpeg -i input.mp3 -ar 44100 -ac 1 output.wav

# Batch convert
for f in *.mp3; do ffmpeg -i "$f" -ar 44100 -ac 1 "${f%.mp3}.wav"; done
```

**For slowing down existing English audio (if supplementing from datasets):**

```bash
# Slow down to 80% speed without changing pitch (using sox)
sudo apt install sox
sox input.wav output_slow.wav tempo 0.8
```

Or using Python:

```python
from pydub import AudioSegment

audio = AudioSegment.from_wav("input.wav")
# Slow down by changing frame rate, then export at original rate
slow_audio = audio._spawn(audio.raw_data, overrides={
    "frame_rate": int(audio.frame_rate * 0.8)
}).set_frame_rate(audio.frame_rate)
slow_audio.export("output_slow.wav", format="wav")
```

### Step 3: Split Long Audio into Clips (3–15 seconds)

If you have long recordings, split them:

**Using Audacity (free):**

1. Open the long audio in Audacity
2. Use "Silence Finder" (Analyze menu) to detect sentence boundaries
3. "Export Multiple" (File menu) to export clips
4. Name clips sequentially

**Using Python:**

```python
from pydub import AudioSegment
from pydub.silence import split_on_silence

audio = AudioSegment.from_wav("long_recording.wav")
chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

for i, chunk in enumerate(chunks):
    duration_sec = len(chunk) / 1000
    if 3 <= duration_sec <= 15:
        chunk.export(f"en_{i:04d}.wav", format="wav")
    elif duration_sec > 15:
        print(f"Chunk {i} is {duration_sec:.1f}s - needs manual splitting")
    else:
        print(f"Chunk {i} is {duration_sec:.1f}s - too short, skipping")
```

### Step 4: Transcribe Audio

**For pure English and Vietnamese — use Whisper then manually verify:**

```bash
pip install openai-whisper

# English
for f in raw_audio/en_*.wav; do whisper "$f" --model medium --language en --output_format txt; done

# Vietnamese
for f in raw_audio/vi_*.wav; do whisper "$f" --model medium --language vi --output_format txt; done
```

**For mixed EN-VI — manually transcribe ALL of them:**

> Whisper performs poorly on code-switched speech. Do NOT rely on auto-transcription for mixed data. Listen to each clip and type exactly as spoken.

> Always review auto-transcriptions manually! The text must match audio 100%.

### Step 5: Normalize Audio Volume

```bash
sudo apt install sox

# Batch normalize all files to -1 dB peak
for f in raw_audio/*.wav; do sox "$f" "${f}.tmp" norm -1 && mv "${f}.tmp" "$f"; done
```

### Step 6: Build metadata.csv

```python
import os

audio_dir = "finetune/dataset/raw_audio"
output_file = "finetune/dataset/metadata.csv"
entries = []

for filename in sorted(os.listdir(audio_dir)):
    if filename.endswith(".wav"):
        txt_file = filename.replace(".wav", ".txt")
        txt_path = os.path.join("transcriptions", txt_file)
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            # Remove pattern labels if present
            if text.startswith("[P") and "]" in text[:5]:
                text = text[text.index("]")+2:]
            entries.append(f"{filename}|{text}")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(entries) + "\n")

print(f"Created metadata.csv with {len(entries)} entries")
en_count = sum(1 for e in entries if e.startswith("en_"))
vi_count = sum(1 for e in entries if e.startswith("vi_"))
mix_count = sum(1 for e in entries if e.startswith("mix_"))
print(f"  Pure English: {en_count}")
print(f"  Pure Vietnamese: {vi_count}")
print(f"  Mixed EN-VI: {mix_count}")
```

### Step 7: Validate Your Data

```python
import soundfile as sf
import os
import re

audio_dir = "finetune/dataset/raw_audio"
metadata_file = "finetune/dataset/metadata.csv"

issues = {"no_audio": 0, "bad_duration": 0, "has_digits": 0, "no_end_punct": 0}
cat_counts = {"en": 0, "vi": 0, "mix": 0}
cat_durations = {"en": 0.0, "vi": 0.0, "mix": 0.0}

with open(metadata_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split("|")
    if len(parts) < 2:
        continue
    filename, text = parts[0], parts[1]

    # Track category
    for prefix in ["en_", "vi_", "mix_"]:
        if filename.startswith(prefix):
            cat_counts[prefix.rstrip("_")] += 1
            break

    audio_path = os.path.join(audio_dir, filename)
    if not os.path.exists(audio_path):
        issues["no_audio"] += 1
        continue

    info = sf.info(audio_path)
    duration = info.duration

    for prefix in ["en_", "vi_", "mix_"]:
        if filename.startswith(prefix):
            cat_durations[prefix.rstrip("_")] += duration

    # filter_data.py accepts 3-15 seconds
    if not (3.0 <= duration <= 15.0):
        issues["bad_duration"] += 1

    if re.search(r"\d", text):
        issues["has_digits"] += 1

    if text[-1] not in ".,?!":
        issues["no_end_punct"] += 1

print("=== DATA VALIDATION REPORT ===")
print(f"Total entries: {len(lines)}")
print(f"\nCategory breakdown:")
for cat in ["en", "vi", "mix"]:
    h = cat_durations[cat] / 3600
    print(f"  {cat:>4}: {cat_counts[cat]:>5} clips ({h:.1f}h)")
total_h = sum(cat_durations.values()) / 3600
print(f"  TOTAL: {sum(cat_counts.values()):>5} clips ({total_h:.1f}h)")
print(f"\nIssues: {issues}")
print(f"Clean entries: {len(lines) - sum(issues.values())}")

if total_h > 0:
    for cat in ["en", "vi", "mix"]:
        pct = cat_durations[cat] / (total_h * 3600) * 100
        print(f"  {cat}: {pct:.0f}%", end="")
    print()
    if cat_durations["mix"] / (total_h * 3600) * 100 < 25:
        print("  WARNING: Mixed data below 25%! Add more code-switching clips.")
```

---

## 8. Important Notes

1. **Use the same speaker for ALL three categories.** Voice consistency across EN, VI, and mixed data is essential. The model learns one voice.

2. **The speaker must be truly bilingual.** For mixed data to sound natural, the speaker needs comfortable fluency in both languages at switch points.

3. **Slow English is intentional, not a recording artifact.** When recording English clips, the speaker should consciously slow down as if teaching young children, not just read at normal speed.

4. **The VieNeu-TTS `sea-g2p` library already supports code-switching** (Vietnamese + English phonemization). Your mixed-language text will be processed correctly by the training pipeline.

5. **Shuffle is built in.** The `encode_data.py` script randomly shuffles samples. All three categories will be mixed during training automatically.

---

## 9. Summary Checklist Before Fine-tuning

- [ ] All audio files are `.wav`, mono channel
- [ ] All clips are **3–15 seconds** long
- [ ] Data split: ~7h EN + ~7h VI + ~6h Mixed = ~20h total
- [ ] English audio is **slow and clear** (~100–120 wpm)
- [ ] Mixed data covers all six code-switching patterns
- [ ] Audio is clean: no noise, no echo, normalized volume
- [ ] All files in `finetune/dataset/raw_audio/`
- [ ] `metadata.csv` in `finetune/dataset/` with `filename|text` format
- [ ] Text matches audio 100% exactly
- [ ] No digits or acronyms in text
- [ ] Every sentence ends with `.` `,` `?` or `!`
- [ ] Validation script confirms EN ~35% / VI ~35% / Mixed ~30%
- [ ] Run `filter_data.py` and check how many clips pass

Once your data is ready, proceed to **fine-tune-guide.md** for training instructions!
