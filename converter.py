"""EPUB parsing and TTS conversion module."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pocket_tts import TTSModel
from mutagen.id3 import ID3, TIT2, TALB, TPE1, TRCK, TCON, COMM, APIC
import lameenc
import numpy as np


BUILTIN_VOICES = [
    "alba", "marius", "javert", "jean",
    "fantine", "cosette", "eponine", "azelma"
]


@dataclass
class BookMetadata:
    """Metadata extracted from an EPUB file."""
    title: str
    author: str
    chapters: list[tuple[str, str]]  # (chapter_title, chapter_text)
    cover_image: bytes | None = None  # Cover image data (JPEG/PNG)
    cover_mime: str | None = None  # MIME type of cover image


def parse_epub(epub_path: str) -> BookMetadata:
    """
    Parse an EPUB file and extract metadata and chapters.

    Returns BookMetadata with title, author, and list of chapters.
    """
    book = epub.read_epub(epub_path)

    # Extract book metadata
    book_title = "Unknown Title"
    book_author = "Unknown Author"

    # Try to get title from metadata
    title_meta = book.get_metadata('DC', 'title')
    if title_meta:
        book_title = title_meta[0][0]

    # Try to get author from metadata
    author_meta = book.get_metadata('DC', 'creator')
    if author_meta:
        book_author = author_meta[0][0]

    # Try to extract cover image
    cover_image = None
    cover_mime = None

    # Method 1: Check for cover image item
    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        cover_image = item.get_content()
        cover_mime = item.media_type
        break

    # Method 2: Look for image items with 'cover' in name
    if not cover_image:
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            name = item.get_name().lower()
            if 'cover' in name:
                cover_image = item.get_content()
                cover_mime = item.media_type
                break

    # Method 3: Check metadata for cover reference
    if not cover_image:
        cover_meta = book.get_metadata('OPF', 'cover')
        if cover_meta:
            cover_id = cover_meta[0][1].get('content') if len(cover_meta[0]) > 1 else None
            if cover_id:
                for item in book.get_items():
                    if item.get_id() == cover_id:
                        cover_image = item.get_content()
                        cover_mime = item.media_type
                        break

    # Extract chapters
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

    return BookMetadata(
        title=book_title,
        author=book_author,
        chapters=chapters,
        cover_image=cover_image,
        cover_mime=cover_mime,
    )


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


def add_id3_tags(
    mp3_path: str,
    title: str,
    album: str,
    artist: str,
    track_num: int = None,
    total_tracks: int = None,
    voice: str = None,
    cover_image: bytes = None,
    cover_mime: str = None,
):
    """Add ID3 tags to an MP3 file."""
    # Create ID3 tag (or load existing)
    try:
        tags = ID3(mp3_path)
    except Exception:
        tags = ID3()

    # Set tags
    tags["TIT2"] = TIT2(encoding=3, text=title)  # Title
    tags["TALB"] = TALB(encoding=3, text=album)  # Album
    tags["TPE1"] = TPE1(encoding=3, text=artist)  # Artist
    tags["TCON"] = TCON(encoding=3, text="Audiobook")  # Genre

    # Track number
    if track_num is not None:
        track_str = str(track_num)
        if total_tracks is not None:
            track_str = f"{track_num}/{total_tracks}"
        tags["TRCK"] = TRCK(encoding=3, text=track_str)

    # Add comment with voice info
    if voice:
        tags["COMM"] = COMM(encoding=3, lang="eng", desc="Voice", text=f"Generated with Pocket TTS ({voice})")

    # Add cover art
    if cover_image and cover_mime:
        tags["APIC"] = APIC(
            encoding=3,
            mime=cover_mime,
            type=3,  # Cover (front)
            desc="Cover",
            data=cover_image,
        )

    tags.save(mp3_path)


def convert_epub_to_mp3(
    epub_path: str,
    output_dir: str,
    model: TTSModel,
    voice: str = "alba",
    per_chapter: bool = True,
    progress_callback: Callable[[int, int, str], None] = None,
    chapter_indices: list[int] = None,
    skip_existing: bool = False,
    announce_chapters: bool = False,
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
        chapter_indices: List of chapter indices to convert (None = all)
        skip_existing: If True, skip chapters that already have output files
        announce_chapters: If True, speak chapter title at start of each chapter

    Returns:
        List of paths to generated MP3 files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse EPUB
    if progress_callback:
        progress_callback(0, 100, "Parsing EPUB...")

    book = parse_epub(epub_path)

    if not book.chapters:
        raise ValueError("No readable chapters found in EPUB")

    # Filter chapters if indices provided
    if chapter_indices is not None:
        selected_chapters = [
            (i, book.chapters[i]) for i in chapter_indices
            if 0 <= i < len(book.chapters)
        ]
    else:
        selected_chapters = list(enumerate(book.chapters))

    if not selected_chapters:
        raise ValueError("No chapters selected for conversion")

    # Get voice state
    if progress_callback:
        progress_callback(5, 100, "Loading voice...")

    voice_state = get_voice_state(model, voice)
    sample_rate = model.sample_rate

    output_files = []
    total_chapters = len(selected_chapters)
    skipped_count = 0

    if per_chapter:
        # One MP3 per chapter
        for idx, (orig_idx, (title, text)) in enumerate(selected_chapters):
            progress_pct = 10 + int((idx / total_chapters) * 85)
            safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
            filename = f"{idx+1:03d}_{safe_title}.mp3"
            output_path = output_dir / filename

            # Skip if file exists and skip_existing is enabled
            if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
                skipped_count += 1
                if progress_callback:
                    progress_callback(progress_pct, 100, f"Skipping (exists): {title[:30]}...")
                output_files.append(str(output_path))
                continue

            if progress_callback:
                progress_callback(progress_pct, 100, f"Converting: {title[:30]}...")

            # Generate chapter announcement if enabled
            chapter_audio_parts = []
            if announce_chapters:
                announcement_text = f"Chapter {idx + 1}. {title}."
                announcement = text_to_audio(model, voice_state, announcement_text)
                if len(announcement) > 0:
                    chapter_audio_parts.append(announcement)
                    # Add a brief pause after announcement
                    pause = np.zeros(int(sample_rate * 0.8), dtype=announcement.dtype)
                    chapter_audio_parts.append(pause)

            # Generate chapter content audio
            content_audio = text_to_audio(model, voice_state, text)
            if len(content_audio) > 0:
                chapter_audio_parts.append(content_audio)

            # Combine announcement and content
            if chapter_audio_parts:
                audio = np.concatenate(chapter_audio_parts)
            else:
                audio = np.array([])

            if len(audio) > 0:
                convert_wav_to_mp3(audio, sample_rate, str(output_path))
                add_id3_tags(
                    str(output_path),
                    title=title,
                    album=book.title,
                    artist=book.author,
                    track_num=idx + 1,
                    total_tracks=total_chapters,
                    voice=voice,
                    cover_image=book.cover_image,
                    cover_mime=book.cover_mime,
                )
                output_files.append(str(output_path))
    else:
        # Single combined MP3
        if progress_callback:
            progress_callback(10, 100, "Combining chapters...")

        all_audio = []

        for idx, (orig_idx, (title, text)) in enumerate(selected_chapters):
            progress_pct = 10 + int((idx / total_chapters) * 80)
            if progress_callback:
                progress_callback(progress_pct, 100, f"Converting: {title[:30]}...")

            # Generate chapter announcement if enabled
            if announce_chapters:
                announcement_text = f"Chapter {idx + 1}. {title}."
                announcement = text_to_audio(model, voice_state, announcement_text)
                if len(announcement) > 0:
                    all_audio.append(announcement)
                    # Add a brief pause after announcement
                    pause = np.zeros(int(sample_rate * 0.8), dtype=announcement.dtype)
                    all_audio.append(pause)

            audio = text_to_audio(model, voice_state, text)
            if len(audio) > 0:
                all_audio.append(audio)
                # Add a short pause between chapters (0.5 seconds of silence)
                silence = np.zeros(int(sample_rate * 0.5), dtype=audio.dtype)
                all_audio.append(silence)

        if all_audio:
            combined_audio = np.concatenate(all_audio)
            output_path = output_dir / f"{book.title}.mp3"

            if progress_callback:
                progress_callback(95, 100, "Saving MP3...")

            convert_wav_to_mp3(combined_audio, sample_rate, str(output_path))
            add_id3_tags(
                str(output_path),
                title=book.title,
                album=book.title,
                artist=book.author,
                voice=voice,
                cover_image=book.cover_image,
                cover_mime=book.cover_mime,
            )
            output_files.append(str(output_path))

    if progress_callback:
        if per_chapter and skip_existing and skipped_count > 0:
            progress_callback(100, 100, f"Done! ({skipped_count} chapters skipped)")
        else:
            progress_callback(100, 100, "Done!")

    return output_files
