"""
Generate audio files from metadata.csv using ElevenLabs TTS API.

Usage:
    # Set your API key first
    export ELEVENLABS_API_KEY="your-api-key-here"

    # Run the script
    uv run python finetune/generate_audio.py

    # Resume from where you left off (skips existing files)
    uv run python finetune/generate_audio.py --resume

    # Process only a subset (e.g., first 100 for testing)
    uv run python finetune/generate_audio.py --limit 100

    # Use more concurrent workers (default: 5)
    uv run python finetune/generate_audio.py --workers 8
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── Configuration ──────────────────────────────────────────────
API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = "hMK7c1GPJmptCzI4bQIu"
MODEL_ID = "eleven_v3"
# OUTPUT_FORMAT = "mp3_44100_128"
OUTPUT_FORMAT = "wav_44100"

BASE_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

METADATA_CSV = Path(__file__).parent / "dataset" / "metadata_2.csv"
OUTPUT_DIR = Path(__file__).parent / "dataset" / "raw_audio"

# Voice settings tuned for clear, warm teaching voice
VOICE_SETTINGS = {
    "stability": 0.6,
    "similarity_boost": 0.75,
    "style": 0.0,
    "speed": 1.0,
    "use_speaker_boost": True,
}

# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


def generate_one(filename: str, text: str, output_dir: Path) -> tuple[str, bool, str]:
    """Generate a single audio file. Returns (filename, success, message)."""
    output_path = output_dir / filename

    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS,
    }
    url = f"{BASE_URL}?output_format={OUTPUT_FORMAT}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)

            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return (filename, True, "OK")

            if response.status_code == 429:
                # Rate limited — wait and retry
                retry_after = int(response.headers.get("Retry-After", RETRY_BACKOFF * attempt))
                time.sleep(retry_after)
                continue

            if response.status_code >= 500:
                # Server error — retry
                time.sleep(RETRY_BACKOFF * attempt)
                continue

            # Client error (4xx) — don't retry
            return (filename, False, f"HTTP {response.status_code}: {response.text[:200]}")

        except requests.exceptions.Timeout:
            time.sleep(RETRY_BACKOFF * attempt)
            continue
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                return (filename, False, f"Request error: {e}")
            time.sleep(RETRY_BACKOFF * attempt)
            continue

    return (filename, False, f"Failed after {MAX_RETRIES} retries")


def load_metadata(csv_path: Path) -> list[tuple[str, str]]:
    """Load (filename, text) pairs from metadata CSV."""
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                filename = row[0].strip()
                text = row[1].strip()
                entries.append((filename, text))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate audio via ElevenLabs API")
    parser.add_argument("--resume", action="store_true", help="Skip files that already exist")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N entries (0 = all)")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent API calls")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set ELEVENLABS_API_KEY environment variable first.")
        print("  export ELEVENLABS_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    entries = load_metadata(METADATA_CSV)
    print(f"Loaded {len(entries)} entries from {METADATA_CSV}")

    # Filter already-generated files if resuming
    if args.resume:
        existing = set(os.listdir(OUTPUT_DIR))
        before = len(entries)
        entries = [(f, t) for f, t in entries if f not in existing]
        print(f"Resume mode: skipping {before - len(entries)} existing files, {len(entries)} remaining")

    # Apply limit
    if args.limit > 0:
        entries = entries[: args.limit]
        print(f"Limited to {len(entries)} entries")

    if not entries:
        print("Nothing to generate. Done!")
        return

    # Generate audio with concurrent workers
    total = len(entries)
    success_count = 0
    fail_count = 0
    failed_files = []

    print(f"\nGenerating {total} audio files with {args.workers} workers...")
    print(f"Voice: {VOICE_ID} | Model: {MODEL_ID} | Format: {OUTPUT_FORMAT}\n")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one, filename, text, OUTPUT_DIR): filename
            for filename, text in entries
        }

        for i, future in enumerate(as_completed(futures), 1):
            filename, ok, msg = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append((filename, msg))

            # Progress update every 50 files or on failure
            if i % 50 == 0 or not ok:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (total - i) / rate / 60 if rate > 0 else 0
                status = "OK" if ok else f"FAIL: {msg}"
                print(f"[{i}/{total}] {filename} - {status}  "
                      f"({rate:.1f} files/s, ETA: {eta_min:.0f}min)")

    elapsed_total = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed_total/60:.1f} minutes")
    print(f"  Success: {success_count}/{total}")
    print(f"  Failed:  {fail_count}/{total}")

    if failed_files:
        fail_log = OUTPUT_DIR.parent / "failed_files.txt"
        with open(fail_log, "w") as f:
            for filename, msg in failed_files:
                f.write(f"{filename}|{msg}\n")
        print(f"\nFailed files logged to: {fail_log}")
        print("Re-run with --resume to retry only failed/missing files.")


if __name__ == "__main__":
    main()
