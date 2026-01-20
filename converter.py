"""EPUB parsing and TTS conversion module."""

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from text_processor import process_chapter, ProcessingMode, is_gemini_available
from tts import (
    generate_speech, get_voice_list, is_tts_available,
    VOICES, DEFAULT_VOICE, SAMPLE_RATE
)

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from mutagen.id3 import ID3, TIT2, TALB, TPE1, TRCK, TCON, COMM, APIC
import lameenc
import numpy as np


# Get voice names from TTS module
BUILTIN_VOICES = list(VOICES.keys())


@dataclass
class Chapter:
    """A chapter extracted from an EPUB."""
    title: str
    text: str
    word_count: int
    is_front_matter: bool = False
    is_back_matter: bool = False
    source_file: str = ""


@dataclass
class BookMetadata:
    """Metadata extracted from an EPUB file."""
    title: str
    author: str
    chapters: list[Chapter]
    cover_image: bytes | None = None
    cover_mime: str | None = None


# Keywords for detecting front/back matter
# Note: Prologue, Introduction, Preface are kept as content (user likely wants these)
FRONT_MATTER_KEYWORDS = [
    'cover', 'title page', 'titlepage', 'copyright', 'dedication',
    'epigraph', 'contents', 'table of contents', 'toc',
    'also by', 'praise for', 'about the publisher', 'half title',
    'frontmatter', 'front matter', 'note to reader'
]

BACK_MATTER_KEYWORDS = [
    'afterword', 'acknowledgment', 'acknowledgement',
    'endnotes', 'bibliography', 'references', 'index',
    'about the author', 'about author', 'newsletter', 'also by',
    'other books', 'reading group', 'discussion questions',
    'backmatter', 'back matter', 'credits', 'further reading'
]


def is_front_matter(title: str) -> bool:
    """Check if a chapter title indicates front matter."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in FRONT_MATTER_KEYWORDS)


def is_back_matter(title: str) -> bool:
    """Check if a chapter title indicates back matter."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in BACK_MATTER_KEYWORDS)


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

    # Build a map of file names to document items
    doc_items = {}
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        doc_items[item.get_name()] = item
        # Also map without directory prefix
        short_name = item.get_name().split('/')[-1]
        if short_name not in doc_items:
            doc_items[short_name] = item

    # Try to extract chapters from TOC first (better titles)
    chapters = []
    toc = book.toc

    def flatten_toc(items, depth=0):
        """Flatten nested TOC into list of (title, href, depth)."""
        result = []
        for item in items:
            if isinstance(item, tuple):
                # Nested section: (section_link, children)
                section, children = item
                if hasattr(section, 'title') and hasattr(section, 'href'):
                    result.append((section.title, section.href, depth))
                result.extend(flatten_toc(children, depth + 1))
            elif hasattr(item, 'title') and hasattr(item, 'href'):
                result.append((item.title, item.href, depth))
        return result

    if toc:
        toc_entries = flatten_toc(toc)

        # Track which files we've seen to merge split content
        seen_files = set()
        pending_merge = None  # (title, texts, file)

        for toc_title, href, depth in toc_entries:
            # Extract file path (before #anchor)
            file_path = href.split('#')[0] if href else None

            if not file_path or file_path not in doc_items:
                # Try with just filename
                short_path = file_path.split('/')[-1] if file_path else None
                if short_path and short_path in doc_items:
                    file_path = short_path
                else:
                    continue

            item = doc_items[file_path]
            content = item.get_content().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')
            text = clean_text(soup.get_text(separator=' '))
            word_count = len(text.split())

            # Skip empty or very short sections
            if word_count < 10:
                continue

            # Detect front/back matter
            front = is_front_matter(toc_title)
            back = is_back_matter(toc_title)

            # Check if this is a continuation (same file, different anchor)
            if file_path in seen_files:
                # Skip duplicate entries for same file (already captured full content)
                continue

            seen_files.add(file_path)

            chapters.append(Chapter(
                title=toc_title,
                text=text,
                word_count=word_count,
                is_front_matter=front,
                is_back_matter=back,
                source_file=file_path,
            ))

    # Fallback: if TOC parsing yielded nothing useful, use document iteration
    if len([c for c in chapters if not c.is_front_matter and not c.is_back_matter]) < 2:
        chapters = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
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
                title = item.get_name().split('/')[-1]

            text = clean_text(soup.get_text(separator=' '))
            word_count = len(text.split())

            if word_count >= 50:  # Skip very short sections
                chapters.append(Chapter(
                    title=title,
                    text=text,
                    word_count=word_count,
                    is_front_matter=is_front_matter(title),
                    is_back_matter=is_back_matter(title),
                    source_file=item.get_name(),
                ))

    # Intelligent merging: combine small adjacent sections (Calibre split files)
    chapters = merge_small_chapters(chapters)

    return BookMetadata(
        title=book_title,
        author=book_author,
        chapters=chapters,
        cover_image=cover_image,
        cover_mime=cover_mime,
    )


def merge_small_chapters(chapters: list[Chapter], min_words: int = 500) -> list[Chapter]:
    """
    Merge small adjacent chapters that appear to be over-split.

    Detects Calibre-style splitting (index_split_XXX) and merges
    until hitting a real chapter boundary (>min_words or clear title).
    """
    if not chapters:
        return chapters

    merged = []
    pending = None  # Chapter being accumulated

    for chapter in chapters:
        # Detect if this looks like a Calibre split file
        is_split_file = 'split' in chapter.source_file.lower()

        # Check if this looks like a real chapter start
        has_chapter_title = bool(re.match(
            r'^(chapter|part|book|section|act|scene|\d+[.:]|\d+\s*[-–—]|[ivxlc]+[.:])',
            chapter.title.lower().strip()
        ))

        if pending is None:
            # Start accumulating
            pending = chapter
        elif is_split_file and not has_chapter_title and pending.word_count < min_words:
            # Merge with pending - this looks like a continuation
            pending = Chapter(
                title=pending.title,
                text=pending.text + " " + chapter.text,
                word_count=pending.word_count + chapter.word_count,
                is_front_matter=pending.is_front_matter,
                is_back_matter=pending.is_back_matter or chapter.is_back_matter,
                source_file=pending.source_file,
            )
        else:
            # This looks like a new chapter, save pending and start fresh
            merged.append(pending)
            pending = chapter

    if pending:
        merged.append(pending)

    return merged


def clean_text(text: str) -> str:
    """Clean and normalize text for TTS."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', ' - ', text)
    # Strip and return
    return text.strip()


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which('ffmpeg') is not None


def create_m4b_with_chapters(
    audio_segments: list[tuple[str, np.ndarray]],  # (chapter_title, audio_data)
    output_path: str,
    sample_rate: int,
    title: str,
    author: str,
    cover_image: bytes = None,
    cover_mime: str = None,
) -> tuple[bool, str]:
    """
    Create an M4B audiobook file with embedded chapter markers.

    Args:
        audio_segments: List of (chapter_title, audio_data) tuples
        output_path: Path for the output M4B file
        sample_rate: Audio sample rate
        title: Book title
        author: Book author
        cover_image: Optional cover image data
        cover_mime: MIME type of cover image

    Returns:
        Tuple of (success, error_message). error_message is None on success.
    """
    if not is_ffmpeg_available():
        return (False, "ffmpeg not found")

    with tempfile.TemporaryDirectory(prefix="epub2mp3_m4b_") as temp_dir:
        temp_dir = Path(temp_dir)

        # Write all audio to a single WAV file and track chapter positions
        all_audio = []
        chapters = []  # (start_ms, end_ms, title)
        current_pos_ms = 0

        for chapter_title, audio_data in audio_segments:
            if len(audio_data) == 0:
                continue

            start_ms = current_pos_ms
            duration_ms = int(len(audio_data) / sample_rate * 1000)
            end_ms = start_ms + duration_ms

            chapters.append((start_ms, end_ms, chapter_title))
            all_audio.append(audio_data)

            # Add 500ms silence between chapters
            silence = np.zeros(int(sample_rate * 0.5), dtype=audio_data.dtype)
            all_audio.append(silence)
            current_pos_ms = end_ms + 500

        if not all_audio:
            return (False, "No audio segments provided")

        # Concatenate all audio
        combined = np.concatenate(all_audio)

        # Normalize to int16
        if combined.dtype != np.int16:
            combined = (combined * 32767).astype(np.int16)

        # Write temporary WAV file
        wav_path = temp_dir / "audio.wav"
        import wave
        with wave.open(str(wav_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(combined.tobytes())

        # Create ffmpeg metadata file for chapters
        metadata_path = temp_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(";FFMETADATA1\n")
            f.write(f"title={title}\n")
            f.write(f"artist={author}\n")
            f.write(f"album={title}\n")
            f.write("genre=Audiobook\n")
            f.write("\n")

            for start_ms, end_ms, chapter_title in chapters:
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={start_ms}\n")
                f.write(f"END={end_ms}\n")
                f.write(f"title={chapter_title}\n")
                f.write("\n")

        # Build ffmpeg command - all inputs first, then output options
        cmd = [
            'ffmpeg', '-y',
            '-i', str(wav_path),
            '-i', str(metadata_path),
        ]

        # Add cover image input if available
        cover_path = None
        if cover_image and cover_mime:
            ext = 'jpg' if 'jpeg' in cover_mime else 'png'
            cover_path = temp_dir / f"cover.{ext}"
            with open(cover_path, 'wb') as f:
                f.write(cover_image)
            cmd.extend(['-i', str(cover_path)])

        # Now add output options (after all inputs)
        cmd.extend(['-map_metadata', '1'])

        if cover_image and cover_mime:
            cmd.extend(['-map', '0:a', '-map', '2:v'])
            cmd.extend(['-disposition:v:0', 'attached_pic'])
        else:
            cmd.extend(['-map', '0:a'])

        # Output settings for M4B (AAC in MP4 container)
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', str(sample_rate),
            '-ac', '1',
        ])

        # For cover art, copy the image as-is (don't re-encode to video)
        if cover_image and cover_mime:
            cmd.extend(['-c:v', 'copy'])

        cmd.append(str(output_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            if result.returncode != 0:
                return (False, f"ffmpeg error: {result.stderr}")
            return (True, None)
        except subprocess.TimeoutExpired:
            return (False, "ffmpeg timed out after 10 minutes")
        except subprocess.SubprocessError as e:
            return (False, f"ffmpeg exception: {e}")


def text_to_audio(text: str, voice: str = DEFAULT_VOICE) -> tuple[np.ndarray, int]:
    """
    Convert text to audio using Gemini TTS.

    Returns tuple of (audio numpy array, sample rate).
    """
    return generate_speech(text, voice)


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
        tags["COMM"] = COMM(encoding=3, lang="eng", desc="Voice", text=f"Generated with Gemini TTS ({voice})")

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
    voice: str = DEFAULT_VOICE,
    per_chapter: bool = True,
    progress_callback: Callable[[int, int, str, dict], None] = None,
    chapter_indices: list[int] = None,
    skip_existing: bool = False,
    announce_chapters: bool = False,
    output_format: str = "mp3",
    text_processing: str = "none",
) -> list[str]:
    """
    Convert an EPUB file to audio files (MP3 or M4B) using Gemini TTS.

    Args:
        epub_path: Path to the EPUB file
        output_dir: Directory to save audio files
        voice: Gemini TTS voice name (default: Charon)
        per_chapter: If True, create one file per chapter; otherwise combine all
        progress_callback: Function(current, total, message, details) for progress updates
            details dict can include: chapter_current, chapter_total, chapter_title,
            stage, words_processed, words_total
        chapter_indices: List of chapter indices to convert (None = all)
        skip_existing: If True, skip chapters that already have output files
        announce_chapters: If True, speak chapter title at start of each chapter
        output_format: "mp3" or "m4b" (M4B requires ffmpeg, creates single file with chapters)
        text_processing: "none", "clean", "speed", or "summary" (uses Gemini)

    Returns:
        List of paths to generated audio files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate output format
    output_format = output_format.lower()
    if output_format not in ("mp3", "m4b"):
        raise ValueError(f"Unsupported output format: {output_format}")

    # M4B requires ffmpeg
    if output_format == "m4b" and not is_ffmpeg_available():
        raise ValueError("M4B format requires ffmpeg to be installed")

    # M4B is always a single file with chapters (per_chapter is ignored)
    if output_format == "m4b":
        per_chapter = False

    # Validate text processing mode
    text_processing = text_processing.lower()
    if text_processing not in (ProcessingMode.NONE, ProcessingMode.CLEAN,
                                ProcessingMode.SPEED_READ, ProcessingMode.SUMMARY):
        text_processing = ProcessingMode.NONE

    # Check if text processing requested but Gemini not configured
    llm_available = is_gemini_available()
    if text_processing != ProcessingMode.NONE and not llm_available:
        if progress_callback:
            progress_callback(0, 100, "Note: Gemini API not configured, using basic text cleaning", {"stage": "initializing"})

    # Parse EPUB
    if progress_callback:
        progress_callback(0, 100, "Parsing EPUB...", {"stage": "parsing"})

    book = parse_epub(epub_path)

    if not book.chapters:
        raise ValueError("No readable chapters found in EPUB")

    # Filter chapters if indices provided
    if chapter_indices is not None:
        selected_chapters = [
            book.chapters[i] for i in chapter_indices
            if 0 <= i < len(book.chapters)
        ]
    else:
        selected_chapters = list(book.chapters)

    if not selected_chapters:
        raise ValueError("No chapters selected for conversion")

    # Calculate total words for progress tracking
    total_words = sum(len(ch.text.split()) for ch in selected_chapters)
    words_processed = 0

    # Validate voice
    if voice not in VOICES:
        if progress_callback:
            progress_callback(5, 100, f"Unknown voice '{voice}', using {DEFAULT_VOICE}", {
                "stage": "initializing",
                "chapter_total": len(selected_chapters),
                "words_total": total_words
            })
        voice = DEFAULT_VOICE

    if progress_callback:
        progress_callback(5, 100, f"Using voice: {voice}...", {
            "stage": "initializing",
            "chapter_total": len(selected_chapters),
            "words_total": total_words
        })

    sample_rate = SAMPLE_RATE

    output_files = []
    total_chapters = len(selected_chapters)
    skipped_count = 0

    if per_chapter:
        # One MP3 per chapter
        for idx, chapter in enumerate(selected_chapters):
            title = chapter.title
            text = chapter.text
            chapter_words = len(text.split())
            progress_pct = 10 + int((idx / total_chapters) * 85)
            safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
            filename = f"{idx+1:03d}_{safe_title}.mp3"
            output_path = output_dir / filename

            # Common details for this chapter
            chapter_details = {
                "chapter_current": idx + 1,
                "chapter_total": total_chapters,
                "chapter_title": title[:50],
                "words_processed": words_processed,
                "words_total": total_words,
            }

            # Skip if file exists and skip_existing is enabled
            if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
                skipped_count += 1
                words_processed += chapter_words
                if progress_callback:
                    progress_callback(progress_pct, 100, f"Skipping (exists): {title[:30]}...", {
                        **chapter_details,
                        "stage": "skipping",
                        "words_processed": words_processed,
                    })
                output_files.append(str(output_path))
                continue

            if progress_callback:
                progress_callback(progress_pct, 100, f"Converting: {title[:30]}...", {
                    **chapter_details,
                    "stage": "tts",
                })

            # Process text if LLM processing enabled
            if text_processing != ProcessingMode.NONE:
                if progress_callback:
                    progress_callback(progress_pct, 100, f"Processing text: {title[:30]}...", {
                        **chapter_details,
                        "stage": "text_processing",
                    })
                text = process_chapter(text, title, text_processing)

            # Generate chapter announcement if enabled
            chapter_audio_parts = []
            if announce_chapters:
                announcement_text = f"Chapter {idx + 1}. {title}."
                announcement, _ = text_to_audio(announcement_text, voice)
                if len(announcement) > 0:
                    chapter_audio_parts.append(announcement)
                    # Add a brief pause after announcement
                    pause = np.zeros(int(sample_rate * 0.8), dtype=announcement.dtype)
                    chapter_audio_parts.append(pause)

            # Generate chapter content audio
            content_audio, _ = text_to_audio(text, voice)
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

            # Update words processed after chapter completes
            words_processed += chapter_words
    else:
        # Single combined file (MP3 or M4B)
        if progress_callback:
            format_name = "M4B audiobook" if output_format == "m4b" else "combined MP3"
            progress_callback(10, 100, f"Creating {format_name}...", {
                "stage": "tts",
                "chapter_total": total_chapters,
                "words_total": total_words,
            })

        # For M4B, we need to track chapter segments separately
        chapter_segments = []  # (chapter_title, audio_data)

        for idx, chapter in enumerate(selected_chapters):
            title = chapter.title
            text = chapter.text
            chapter_words = len(text.split())
            progress_pct = 10 + int((idx / total_chapters) * 80)

            # Common details for this chapter
            chapter_details = {
                "chapter_current": idx + 1,
                "chapter_total": total_chapters,
                "chapter_title": title[:50],
                "words_processed": words_processed,
                "words_total": total_words,
            }

            if progress_callback:
                progress_callback(progress_pct, 100, f"Converting: {title[:30]}...", {
                    **chapter_details,
                    "stage": "tts",
                })

            # Process text if LLM processing enabled
            if text_processing != ProcessingMode.NONE:
                if progress_callback:
                    progress_callback(progress_pct, 100, f"Processing text: {title[:30]}...", {
                        **chapter_details,
                        "stage": "text_processing",
                    })
                text = process_chapter(text, title, text_processing)

            chapter_audio_parts = []

            # Generate chapter announcement if enabled
            if announce_chapters:
                announcement_text = f"Chapter {idx + 1}. {title}."
                announcement, _ = text_to_audio(announcement_text, voice)
                if len(announcement) > 0:
                    chapter_audio_parts.append(announcement)
                    # Add a brief pause after announcement
                    pause = np.zeros(int(sample_rate * 0.8), dtype=announcement.dtype)
                    chapter_audio_parts.append(pause)

            content_audio, _ = text_to_audio(text, voice)
            if len(content_audio) > 0:
                chapter_audio_parts.append(content_audio)

            if chapter_audio_parts:
                chapter_audio = np.concatenate(chapter_audio_parts)
                chapter_segments.append((title, chapter_audio))

            # Update words processed after chapter completes
            words_processed += chapter_words

        if chapter_segments:
            # Sanitize filename
            safe_title = re.sub(r'[^\w\s-]', '', book.title)[:100].strip()
            if not safe_title:
                safe_title = "audiobook"

            if output_format == "m4b":
                # Create M4B with embedded chapter markers
                output_path = output_dir / f"{safe_title}.m4b"

                if progress_callback:
                    progress_callback(95, 100, "Creating M4B with chapters...", {
                        "stage": "encoding",
                        "chapter_current": total_chapters,
                        "chapter_total": total_chapters,
                        "words_processed": total_words,
                        "words_total": total_words,
                    })

                success, error_msg = create_m4b_with_chapters(
                    audio_segments=chapter_segments,
                    output_path=str(output_path),
                    sample_rate=sample_rate,
                    title=book.title,
                    author=book.author,
                    cover_image=book.cover_image,
                    cover_mime=book.cover_mime,
                )

                if not success:
                    raise ValueError(f"Failed to create M4B file: {error_msg}")

                output_files.append(str(output_path))
            else:
                # Combined MP3
                all_audio = []
                for _, audio in chapter_segments:
                    all_audio.append(audio)
                    # Add a short pause between chapters
                    silence = np.zeros(int(sample_rate * 0.5), dtype=audio.dtype)
                    all_audio.append(silence)

                combined_audio = np.concatenate(all_audio)
                output_path = output_dir / f"{safe_title}.mp3"

                if progress_callback:
                    progress_callback(95, 100, "Saving MP3...", {
                        "stage": "encoding",
                        "chapter_current": total_chapters,
                        "chapter_total": total_chapters,
                        "words_processed": total_words,
                        "words_total": total_words,
                    })

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
        final_details = {
            "stage": "complete",
            "chapter_current": total_chapters,
            "chapter_total": total_chapters,
            "words_processed": total_words,
            "words_total": total_words,
        }
        if per_chapter and skip_existing and skipped_count > 0:
            progress_callback(100, 100, f"Done! ({skipped_count} chapters skipped)", final_details)
        else:
            progress_callback(100, 100, "Done!", final_details)

    return output_files
