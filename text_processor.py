"""LLM-powered text processing for cleaner TTS output."""

import re
from typing import Callable

# Try to import ollama, but make it optional
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# Default model - Gemma 2B is a good balance of speed and quality
DEFAULT_MODEL = "gemma2:2b"

# Chunk size for processing (in characters, ~750 words)
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200


def is_ollama_available() -> bool:
    """Check if Ollama is installed and running."""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        ollama.list()
        return True
    except Exception:
        return False


def is_model_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if a specific model is available in Ollama."""
    if not is_ollama_available():
        return False
    try:
        models = ollama.list()
        model_names = [m['name'] for m in models.get('models', [])]
        # Check both exact match and base name match
        return any(model in name or name.startswith(model.split(':')[0]) for name in model_names)
    except Exception:
        return False


def get_available_models() -> list[str]:
    """Get list of available Ollama models."""
    if not is_ollama_available():
        return []
    try:
        models = ollama.list()
        return [m['name'] for m in models.get('models', [])]
    except Exception:
        return []


# Text cleaning prompt
CLEAN_PROMPT = """You are preparing text for text-to-speech conversion. Clean the following text by:

1. Remove footnote markers like [1], [2], *, †, superscript numbers
2. Remove page numbers that appear randomly in text
3. Remove repeated headers or footers
4. Remove or simplify figure/table references like "See Figure 3.2"
5. Remove URLs but keep meaningful link text
6. Fix OCR artifacts and weird punctuation
7. Remove excessive whitespace and normalize formatting

IMPORTANT: Keep all actual content intact. Do not summarize or change the meaning.
Only output the cleaned text, nothing else.

Text to clean:
{text}

Cleaned text:"""


# Summarization prompt
SUMMARIZE_PROMPT = """Summarize this text for an audiobook listener.
Keep it engaging and narrative - write it as prose that sounds good when read aloud.
Target length: approximately {target_words} words ({target_percent}% of original).
Focus on key points, main ideas, and important details.

Title: {title}

Original text:
{text}

Summary:"""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for processing."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near chunk boundary
            for sep in ['. ', '! ', '? ', '\n\n', '\n']:
                last_sep = text.rfind(sep, start + chunk_size - 500, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def clean_text_with_llm(
    text: str,
    model: str = DEFAULT_MODEL,
    progress_callback: Callable[[str], None] = None,
) -> str:
    """
    Clean text using LLM to remove artifacts that don't verbalize well.

    Args:
        text: The text to clean
        model: Ollama model to use
        progress_callback: Optional callback for progress updates

    Returns:
        Cleaned text suitable for TTS
    """
    if not is_ollama_available():
        if progress_callback:
            progress_callback("Ollama not available, using basic cleaning")
        return clean_text_basic(text)

    chunks = chunk_text(text)
    cleaned_chunks = []

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Cleaning chunk {i+1}/{len(chunks)}...")

        try:
            response = ollama.generate(
                model=model,
                prompt=CLEAN_PROMPT.format(text=chunk),
                options={
                    'temperature': 0.1,  # Low temperature for consistent cleaning
                    'num_predict': len(chunk) + 500,  # Allow some expansion
                }
            )
            cleaned_chunks.append(response['response'].strip())
        except Exception as e:
            # Fall back to basic cleaning on error
            if progress_callback:
                progress_callback(f"LLM error, using basic cleaning: {e}")
            cleaned_chunks.append(clean_text_basic(chunk))

    # Merge chunks (simple join - overlap was for context, not deduplication)
    return '\n\n'.join(cleaned_chunks)


def summarize_text_with_llm(
    text: str,
    title: str = "Chapter",
    target_percent: int = 30,
    model: str = DEFAULT_MODEL,
    progress_callback: Callable[[str], None] = None,
) -> str:
    """
    Summarize text using LLM for speed-read mode.

    Args:
        text: The text to summarize
        title: Chapter/section title for context
        target_percent: Target length as percentage of original
        model: Ollama model to use
        progress_callback: Optional callback for progress updates

    Returns:
        Summarized text
    """
    if not is_ollama_available():
        if progress_callback:
            progress_callback("Ollama not available, cannot summarize")
        return text

    word_count = len(text.split())
    target_words = int(word_count * target_percent / 100)

    # For very long texts, chunk and summarize each chunk
    chunks = chunk_text(text, chunk_size=6000)
    summaries = []

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Summarizing chunk {i+1}/{len(chunks)}...")

        chunk_word_count = len(chunk.split())
        chunk_target = int(chunk_word_count * target_percent / 100)

        try:
            response = ollama.generate(
                model=model,
                prompt=SUMMARIZE_PROMPT.format(
                    text=chunk,
                    title=title,
                    target_words=chunk_target,
                    target_percent=target_percent,
                ),
                options={
                    'temperature': 0.3,  # Slightly higher for more natural summaries
                    'num_predict': chunk_target * 2,  # Allow flexibility
                }
            )
            summaries.append(response['response'].strip())
        except Exception as e:
            if progress_callback:
                progress_callback(f"Summarization error: {e}")
            summaries.append(chunk)  # Fall back to original

    return '\n\n'.join(summaries)


def clean_text_basic(text: str) -> str:
    """
    Basic regex-based text cleaning (fallback when LLM not available).
    """
    # Remove footnote markers
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[\w,\s]+\]', '', text)  # [citation needed] etc
    text = re.sub(r'[*†‡§¶]+', '', text)

    # Remove standalone numbers that look like page numbers
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Clean up figure/table references (simplify, don't remove context)
    text = re.sub(r'\(see [Ff]igure \d+[.\d]*\)', '', text)
    text = re.sub(r'\(see [Tt]able \d+[.\d]*\)', '', text)

    # Fix common OCR artifacts
    text = re.sub(r'[|l](?=[A-Z])', 'I', text)  # l before capital likely meant I
    text = re.sub(r'(?<=[a-z])0(?=[a-z])', 'o', text)  # 0 in word likely meant o

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\t+', ' ', text)

    return text.strip()


# Processing modes
class ProcessingMode:
    NONE = "none"           # No LLM processing
    CLEAN = "clean"         # Clean artifacts only
    SPEED_READ = "speed"    # Summarize to ~30%
    SUMMARY = "summary"     # Heavy summarization to ~10%


def process_chapter(
    text: str,
    title: str = "Chapter",
    mode: str = ProcessingMode.CLEAN,
    model: str = DEFAULT_MODEL,
    progress_callback: Callable[[str], None] = None,
) -> str:
    """
    Process chapter text based on mode.

    Args:
        text: Chapter text
        title: Chapter title
        mode: Processing mode (none, clean, speed, summary)
        model: Ollama model to use
        progress_callback: Optional callback for progress updates

    Returns:
        Processed text
    """
    if mode == ProcessingMode.NONE:
        return text

    if mode == ProcessingMode.CLEAN:
        return clean_text_with_llm(text, model, progress_callback)

    if mode == ProcessingMode.SPEED_READ:
        # Clean first, then summarize to 30%
        cleaned = clean_text_with_llm(text, model, progress_callback)
        return summarize_text_with_llm(cleaned, title, 30, model, progress_callback)

    if mode == ProcessingMode.SUMMARY:
        # Clean first, then heavy summarization to 10%
        cleaned = clean_text_with_llm(text, model, progress_callback)
        return summarize_text_with_llm(cleaned, title, 10, model, progress_callback)

    return text


if __name__ == "__main__":
    # Quick test
    print(f"Ollama available: {is_ollama_available()}")
    print(f"Available models: {get_available_models()}")
    print(f"Gemma2:2b available: {is_model_available('gemma2:2b')}")

    # Test basic cleaning
    test_text = """
    This is a test paragraph[1] with some footnotes[2] and a URL https://example.com
    that should be cleaned. See Figure 3.2 for more details.

    42

    Here's another paragraph with more content that should remain intact.
    """

    print("\n--- Basic cleaning test ---")
    print(clean_text_basic(test_text))
