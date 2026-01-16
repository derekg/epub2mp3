"""EPUB parsing and TTS conversion module."""

import re
from pathlib import Path
from typing import Callable

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pocket_tts import TTSModel
import lameenc
import numpy as np


BUILTIN_VOICES = [
    "alba", "marius", "javert", "jean",
    "fantine", "cosette", "eponine", "azelma"
]


def parse_epub(epub_path: str) -> list[tuple[str, str]]:
    """
    Parse an EPUB file and extract chapters.

    Returns a list of (chapter_title, chapter_text) tuples.
    """
    book = epub.read_epub(epub_path)
    chapters = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')

            # Try to extract title from headings
            title = None
            for tag in ['h1', 'h2', 'h3', 'title']:
                heading = soup.find(tag)
                if heading:
                    title = heading.get_text(strip=True)
                    break

            if not title:
                title = item.get_name()

            # Extract and clean text
            text = soup.get_text(separator=' ')
            text = clean_text(text)

            if text and len(text) > 50:  # Skip very short sections
                chapters.append((title, text))

    return chapters


def clean_text(text: str) -> str:
    """Clean and normalize text for TTS."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', ' - ', text)
    # Strip and return
    return text.strip()


def get_voice_state(model: TTSModel, voice: str):
    """Get voice state for a given voice name or WAV path.

    Predefined voices (alba, marius, etc.) are handled automatically by pocket_tts.
    Custom WAV file paths are also supported for voice cloning.
    """
    return model.get_state_for_audio_prompt(voice)


def text_to_audio(
    model: TTSModel,
    voice_state,
    text: str,
    max_chunk_chars: int = 250
) -> np.ndarray:
    """
    Convert text to audio using Pocket TTS.

    Splits long text into chunks to avoid memory issues.
    Returns numpy array of audio samples.
    """
    # Split text into sentences for chunking
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If a single sentence is too long, split it further
        if len(sentence) > max_chunk_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Split long sentence on commas or other punctuation
            parts = re.split(r'(?<=[,;:])\s+', sentence)
            for part in parts:
                if len(part) > max_chunk_chars:
                    # Last resort: split by words
                    words = part.split()
                    sub_chunk = ""
                    for word in words:
                        if len(sub_chunk) + len(word) + 1 < max_chunk_chars:
                            sub_chunk += " " + word
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk.strip())
                            sub_chunk = word
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                else:
                    chunks.append(part.strip())
        elif len(current_chunk) + len(sentence) < max_chunk_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Generate audio for each chunk
    audio_parts = []
    for chunk in chunks:
        if chunk:
            audio = model.generate_audio(voice_state, chunk)
            audio_parts.append(audio.numpy())

    # Concatenate all audio parts
    if audio_parts:
        return np.concatenate(audio_parts)
    return np.array([])


def convert_wav_to_mp3(wav_data: np.ndarray, sample_rate: int, output_path: str):
    """Convert WAV numpy array to MP3 file using lameenc."""
    # Normalize to int16
    if wav_data.dtype != np.int16:
        wav_data = (wav_data * 32767).astype(np.int16)

    # Set up encoder
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_out_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 2 = high quality

    # Encode to MP3
    mp3_data = encoder.encode(wav_data.tobytes())
    mp3_data += encoder.flush()

    # Write to file
    with open(output_path, 'wb') as f:
        f.write(mp3_data)


def convert_epub_to_mp3(
    epub_path: str,
    output_dir: str,
    model: TTSModel,
    voice: str = "alba",
    per_chapter: bool = True,
    progress_callback: Callable[[int, int, str], None] = None
) -> list[str]:
    """
    Convert an EPUB file to MP3(s).

    Args:
        epub_path: Path to the EPUB file
        output_dir: Directory to save MP3 files
        model: Loaded TTSModel instance
        voice: Voice name or custom WAV path
        per_chapter: If True, create one MP3 per chapter; otherwise combine all
        progress_callback: Function(current, total, message) for progress updates

    Returns:
        List of paths to generated MP3 files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse EPUB
    if progress_callback:
        progress_callback(0, 100, "Parsing EPUB...")

    chapters = parse_epub(epub_path)

    if not chapters:
        raise ValueError("No readable chapters found in EPUB")

    # Get voice state
    if progress_callback:
        progress_callback(5, 100, "Loading voice...")

    voice_state = get_voice_state(model, voice)
    sample_rate = model.sample_rate

    output_files = []

    if per_chapter:
        # One MP3 per chapter
        total_chapters = len(chapters)
        for i, (title, text) in enumerate(chapters):
            progress_pct = 10 + int((i / total_chapters) * 85)
            safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
            filename = f"{i+1:03d}_{safe_title}.mp3"
            output_path = output_dir / filename

            if progress_callback:
                progress_callback(progress_pct, 100, f"Converting: {title[:30]}...")

            audio = text_to_audio(model, voice_state, text)
            if len(audio) > 0:
                convert_wav_to_mp3(audio, sample_rate, str(output_path))
                output_files.append(str(output_path))
    else:
        # Single combined MP3
        if progress_callback:
            progress_callback(10, 100, "Combining chapters...")

        all_audio = []
        total_chapters = len(chapters)

        for i, (title, text) in enumerate(chapters):
            progress_pct = 10 + int((i / total_chapters) * 80)
            if progress_callback:
                progress_callback(progress_pct, 100, f"Converting: {title[:30]}...")

            audio = text_to_audio(model, voice_state, text)
            if len(audio) > 0:
                all_audio.append(audio)
                # Add a short pause between chapters (0.5 seconds of silence)
                silence = np.zeros(int(sample_rate * 0.5), dtype=audio.dtype)
                all_audio.append(silence)

        if all_audio:
            combined_audio = np.concatenate(all_audio)
            epub_name = Path(epub_path).stem
            output_path = output_dir / f"{epub_name}.mp3"

            if progress_callback:
                progress_callback(95, 100, "Saving MP3...")

            convert_wav_to_mp3(combined_audio, sample_rate, str(output_path))
            output_files.append(str(output_path))

    if progress_callback:
        progress_callback(100, 100, "Done!")

    return output_files
