"""Text processing for cleaner TTS output using Gemini 3 Flash.

Gemini 3 Flash has 1M+ token INPUT context and 65k token OUTPUT limit.
This handles virtually any book chapter without chunking - only exceptionally
long chapters (50k+ words) would need to be split.
"""

import os
import re
import time
from typing import Callable

# Try to import google-genai
try:
    from google import genai
    from google.genai.errors import ClientError
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    ClientError = Exception  # Fallback

# Gemini 3 Flash - 1M+ token input context, 65k output tokens
# Released Dec 2025: 3x faster than 2.5 Pro, better reasoning, $0.50/1M input
MODEL = "gemini-3-flash-preview"

# Rate limit handling
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Gemini 3 Flash output limit: 65,536 tokens (~50,000 words)
# Most chapters fit in a single call - only very long ones need chunking
MAX_OUTPUT_TOKENS = 65536
CHUNK_SIZE_CHARS = 150000  # ~37,500 words, safe margin under output limit

# Processing modes
class ProcessingMode:
    NONE = "none"           # No processing
    CLEAN = "clean"         # Clean artifacts only
    SPEED_READ = "speed"    # Summarize to ~30%
    SUMMARY = "summary"     # Heavy summarization to ~10%


def get_client():
    """Get Gemini client. API key from GEMINI_API_KEY env var."""
    if not GENAI_AVAILABLE:
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    return genai.Client(api_key=api_key)


def is_gemini_available() -> bool:
    """Check if Gemini API is configured and available."""
    return get_client() is not None


# Text cleaning prompt - optimized for TTS
CLEAN_PROMPT = """Clean this text for text-to-speech. Remove:
- Footnote markers [1], [2], *, †
- Page numbers
- Figure/table references like "See Figure 3.2"
- URLs (keep link text if meaningful)
- Repeated headers/footers
- Excessive whitespace

Keep ALL actual content. Do not summarize. Output only the cleaned text.

Text:
{text}"""


# Summarization prompt
SUMMARIZE_PROMPT = """You are summarizing a book chapter for an audiobook. Your task is to CONDENSE the text significantly.

IMPORTANT: You MUST output approximately {target_words} words (roughly {target_percent}% of the original length). Do NOT output the full text.

Requirements:
- Reduce the text to approximately {target_words} words
- Keep it as engaging prose suitable for listening
- Focus only on the most important points
- Remove all redundant details and examples

Chapter: {title}

Original text ({original_words} words):
{text}

Condensed summary (approximately {target_words} words):"""


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para) + 2  # +2 for \n\n
        if current_size + para_size > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def _clean_chunk(client, text: str, progress_callback: Callable[[str], None] = None) -> str:
    """Clean a single chunk of text."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=CLEAN_PROMPT.format(text=text),
                config={
                    "temperature": 0.1,
                    "maxOutputTokens": MAX_OUTPUT_TOKENS,
                }
            )
            return response.text.strip()
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < MAX_RETRIES - 1:
                    if progress_callback:
                        progress_callback(f"Rate limited, retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
            raise
    raise Exception("Max retries exceeded")


def clean_text_with_gemini(
    text: str,
    progress_callback: Callable[[str], None] = None,
) -> str:
    """
    Clean text using Gemini to remove artifacts that don't verbalize well.

    For long texts, chunks by paragraph boundaries to stay within output token limit.
    Each chunk uses the full 8k output tokens available.
    """
    client = get_client()
    if not client:
        if progress_callback:
            progress_callback("Gemini not configured, using basic cleaning")
        return clean_text_basic(text)

    chunks = _chunk_text(text)

    if len(chunks) == 1:
        if progress_callback:
            progress_callback("Cleaning text with Gemini...")
    else:
        if progress_callback:
            progress_callback(f"Cleaning text with Gemini ({len(chunks)} chunks)...")

    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            if len(chunks) > 1 and progress_callback:
                progress_callback(f"Cleaning chunk {i+1}/{len(chunks)}...")
            cleaned = _clean_chunk(client, chunk, progress_callback)
            cleaned_chunks.append(cleaned)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Gemini error on chunk {i+1}, using basic cleaning: {e}")
            cleaned_chunks.append(clean_text_basic(chunk))

    return '\n\n'.join(cleaned_chunks)


def summarize_text_with_gemini(
    text: str,
    title: str = "Chapter",
    target_percent: int = 30,
    progress_callback: Callable[[str], None] = None,
) -> str:
    """
    Summarize text using Gemini for speed-read mode.

    With Gemini 2.5 Flash's 1M+ token input and 65k output, we can summarize
    entire chapters (or even books) in a single call with full context.
    """
    client = get_client()
    if not client:
        if progress_callback:
            progress_callback("Gemini not configured, cannot summarize")
        return text

    word_count = len(text.split())
    target_words = int(word_count * target_percent / 100)

    if progress_callback:
        progress_callback(f"Summarizing to ~{target_words} words...")

    # With 65k output tokens, we have plenty of room - just use max
    # A 30% summary of a 100k word book = 30k words = ~40k tokens, still under limit

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=SUMMARIZE_PROMPT.format(
                    text=text,
                    title=title,
                    target_words=target_words,
                    target_percent=target_percent,
                    original_words=word_count,
                ),
                config={
                    "temperature": 0.4,  # Allow some creativity for natural summaries
                    "maxOutputTokens": MAX_OUTPUT_TOKENS,
                }
            )
            return response.text.strip()
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < MAX_RETRIES - 1:
                    if progress_callback:
                        progress_callback(f"Rate limited, retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
            if progress_callback:
                progress_callback(f"Summarization error: {e}")
            return text
        except Exception as e:
            if progress_callback:
                progress_callback(f"Summarization error: {e}")
            return text

    # If all retries failed
    if progress_callback:
        progress_callback("Rate limit exceeded, skipping summarization")
    return text


def clean_text_basic(text: str) -> str:
    """Basic regex-based text cleaning (fallback when Gemini not available)."""
    # Remove footnote markers
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[\w,\s]+\]', '', text)
    text = re.sub(r'[*†‡§¶]+', '', text)

    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Clean up figure/table references
    text = re.sub(r'\(see [Ff]igure \d+[.\d]*\)', '', text)
    text = re.sub(r'\(see [Tt]able \d+[.\d]*\)', '', text)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\t+', ' ', text)

    return text.strip()


def process_chapter(
    text: str,
    title: str = "Chapter",
    mode: str = ProcessingMode.CLEAN,
    progress_callback: Callable[[str], None] = None,
) -> str:
    """
    Process chapter text based on mode.

    Args:
        text: Chapter text
        title: Chapter title
        mode: Processing mode (none, clean, speed, summary)
        progress_callback: Optional callback for progress updates

    Returns:
        Processed text
    """
    if mode == ProcessingMode.NONE:
        return text

    if mode == ProcessingMode.CLEAN:
        return clean_text_with_gemini(text, progress_callback)

    if mode == ProcessingMode.SPEED_READ:
        # Clean first, then summarize to 30%
        cleaned = clean_text_with_gemini(text, progress_callback)
        return summarize_text_with_gemini(cleaned, title, 30, progress_callback)

    if mode == ProcessingMode.SUMMARY:
        # Clean first, then heavy summarization to 10%
        cleaned = clean_text_with_gemini(text, progress_callback)
        return summarize_text_with_gemini(cleaned, title, 10, progress_callback)

    return text


if __name__ == "__main__":
    print(f"Gemini available: {is_gemini_available()}")

    if is_gemini_available():
        test_text = """
        This is a test paragraph[1] with some footnotes[2] and a URL https://example.com
        that should be cleaned. See Figure 3.2 for more details.

        42

        Here's another paragraph with more content that should remain intact.
        """
        print("\n--- Gemini cleaning test ---")
        print(clean_text_with_gemini(test_text))
    else:
        print("Set GEMINI_API_KEY environment variable to test")
        print("\n--- Basic cleaning test ---")
        test_text = "Test[1] with footnote"
        print(clean_text_basic(test_text))
