import re
import os
from typing import List, Optional
import numpy as np

# Pre-compile regex for splitting
RE_NEWLINE = re.compile(r"[\r\n]+")
RE_SENTENCE_END = re.compile(r"(?<=[\.\!\?\…])\s+")
RE_MINOR_PUNCT = re.compile(r"(?<=[\,\;\:\-\–\—])\s+")

def join_audio_chunks(chunks: List[np.ndarray], sr: int, silence_p: float = 0.0, crossfade_p: float = 0.0) -> np.ndarray:
    """
    Join audio chunks with optional silence padding and crossfading.

    Args:
        chunks: List of audio waveforms as numpy arrays.
        sr: Sample rate.
        silence_p: Duration of silence to insert between chunks (seconds).
        crossfade_p: Duration of crossfade between chunks (seconds).

    Returns:
        Joined audio waveform.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    
    silence_samples = int(sr * silence_p)
    crossfade_samples = int(sr * crossfade_p)
    
    final_wav = chunks[0]
    
    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        
        if silence_samples > 0:
            # 1. Add silence between chunks
            silence = np.zeros(silence_samples, dtype=np.float32)
            final_wav = np.concatenate([final_wav, silence, next_chunk])
        elif crossfade_samples > 0:
            # 2. Crossfade between chunks
            overlap = min(len(final_wav), len(next_chunk), crossfade_samples)
            if overlap > 0:
                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                
                blended = (final_wav[-overlap:] * fade_out + next_chunk[:overlap] * fade_in)
                final_wav = np.concatenate([
                    final_wav[:-overlap],
                    blended,
                    next_chunk[overlap:]
                ])
            else:
                final_wav = np.concatenate([final_wav, next_chunk])
        else:
            # 3. Simple concatenation
            final_wav = np.concatenate([final_wav, next_chunk])
            
    return final_wav

def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """
    Split raw text into chunks no longer than max_chars.

    Args:
        text: Input text to split.
        max_chars: Maximum number of characters per chunk.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    # 1. First split by newlines - each line/paragraph is handled independently
    paragraphs = RE_NEWLINE.split(text.strip())
    final_chunks: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 2. Split current paragraph into sentences
        sentences = RE_SENTENCE_END.split(para)
        
        buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If sentence itself is longer than max_chars, we must split it by minor punctuation or words
            if len(sentence) > max_chars:
                # Flush buffer before handling a giant sentence
                if buffer:
                    final_chunks.append(buffer)
                    buffer = ""
                
                # Split giant sentence by minor punctuation (, ; : -)
                sub_parts = RE_MINOR_PUNCT.split(sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part: continue
                    
                    if len(buffer) + 1 + len(part) <= max_chars:
                        buffer = (buffer + " " + part) if buffer else part
                    else:
                        if buffer: final_chunks.append(buffer)
                        buffer = part
                        
                        # If even a sub-part is too long, split by spaces (words)
                        if len(buffer) > max_chars:
                            words = buffer.split()
                            current = ""
                            for word in words:
                                if current and len(current) + 1 + len(word) > max_chars:
                                    final_chunks.append(current)
                                    current = word
                                else:
                                    current = (current + " " + word) if current else word
                            buffer = current
            else:
                # Normal sentence: check if it fits in current buffer
                if buffer and len(buffer) + 1 + len(sentence) > max_chars:
                    final_chunks.append(buffer)
                    buffer = sentence
                else:
                    buffer = (buffer + " " + sentence) if buffer else sentence
        
        # End of paragraph: flush whatever is in buffer
        if buffer:
            final_chunks.append(buffer)
            buffer = ""

    return [c.strip() for c in final_chunks if c.strip()]

def split_into_chunks_v2(full_phones: str, max_chunk_size: int = 256, min_chunk_size: int = 10) -> List[str]:
    """
    Phân đoạn phonemes cho VieNeu v2 (Turbo).
    1. Ưu tiên tách tối đa: Mỗi câu (.?!) là một chunk riêng.
    2. Nếu câu quá dài (>max), tách tiếp theo dấu phụ (,:;-) hoặc space.
    3. Gộp các chunk cực ngắn (<10) vào nhau để tránh bị cụt.
    """
    if not full_phones:
        return []

    # Bước 1: Tách theo đoạn (newline)
    paragraphs = RE_NEWLINE.split(full_phones.strip())
    raw_parts: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Bước 2: Luôn tách theo câu (.?!)
        # Sử dụng pattern từ app_gguf.py cho phonemes
        sentences = RE_SENTENCE_END.split(para)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Nếu câu quá dài, phải xé nhỏ tiếp
            if len(sent) > max_chunk_size:
                sub_parts = RE_MINOR_PUNCT.split(sent)
                sub_buf = ""
                for part in sub_parts:
                    part = part.strip()
                    if not part:
                        continue

                    if len(part) > max_chunk_size:
                        # Flush sub_buf trước
                        if sub_buf:
                            raw_parts.append(sub_buf)
                            sub_buf = ""
                        # Xé theo space
                        words = part.split()
                        current = ""
                        for word in words:
                            if current and len(current) + 1 + len(word) > max_chunk_size:
                                raw_parts.append(current)
                                current = word
                            else:
                                current = (current + " " + word) if current else word
                        if current:
                            raw_parts.append(current)
                    elif sub_buf and len(sub_buf) + 1 + len(part) > max_chunk_size:
                        # sub_buf đầy, flush rồi bắt đầu mới
                        raw_parts.append(sub_buf)
                        sub_buf = part
                    else:
                        sub_buf = (sub_buf + " " + part) if sub_buf else part

                if sub_buf:
                    raw_parts.append(sub_buf)
            else:
                raw_parts.append(sent)

    # Bước 3: Gộp các chunk quá ngắn (< min_chunk_size)
    if not raw_parts:
        return []

    merged: List[str] = []
    i = 0
    while i < len(raw_parts):
        current = raw_parts[i]
        
        # Nếu chunk hiện tại quá ngắn, cố gắng gộp với chunk kế tiếp
        while len(current) < min_chunk_size and i + 1 < len(raw_parts):
            next_p = raw_parts[i + 1]
            # Chỉ gộp nếu không vượt quá giới hạn tối đa
            if len(current) + 1 + len(next_p) <= max_chunk_size:
                current = (current + " " + next_p).strip()
                i += 1
            else:
                break
        
        merged.append(current)
        i += 1

    # Xử lý trường hợp chunk cuối cùng vẫn quá ngắn → gộp ngược vào trước
    if len(merged) >= 2 and len(merged[-1]) < min_chunk_size:
        last = merged.pop()
        if len(merged[-1]) + 1 + len(last) <= max_chunk_size:
            merged[-1] = (merged[-1] + " " + last).strip()
        else:
            merged.append(last)

    return [m for m in merged if m]

def get_silence_duration_v2(chunk_phones: str) -> float:
    """Trả về thời gian silence dựa trên dấu câu cho bản v2."""
    stripped = chunk_phones.strip()
    # Kiểm tra dấu câu ở cuối chunk
    if re.search(r'[\.\!\?\…]$', stripped):
        return 0.3
    elif re.search(r'[\,\;\:\-\u2013\u2014]$', stripped):
        return 0.15
    return 0.05

def env_bool(name: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")
