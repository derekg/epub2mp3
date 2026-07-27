"""LLM text processing for narration-ready TTS output.

Backend is auto-selected from environment variables:
  1. OpenRouter (OPENROUTER_API_KEY) — preferred; any model on openrouter.ai
  2. Google Gemini API (GEMINI_API_KEY) — direct google-genai fallback

Default model: google/gemini-2.5-flash-lite. Chosen for this workload
deliberately — clean mode re-emits the whole chapter, so output-token price
dominates cost, and the model must reproduce ~13k-token chunks verbatim
minus artifacts. Flash-Lite is ~7x cheaper than Gemini 3 Flash
($0.10/$0.40 vs $0.50/$3.00 per 1M tokens in/out), supports 65k-token
outputs, and has the same instruction-following lineage this feature has
been reliable on. Override with LLM_MODEL (OpenRouter slug, e.g.
"openai/gpt-5-nano"; the "google/" prefix is stripped automatically for
the direct Gemini backend).
"""

import os
import re
import time
from typing import Callable

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Default model, as an OpenRouter slug.
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Rate limit handling
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# gemini-2.5-flash-lite supports 65k output tokens; 32k leaves headroom for
# other models a user may select via LLM_MODEL while comfortably covering a
# full chunk (~13k tokens of output per CHUNK_SIZE_CHARS of input).
MAX_OUTPUT_TOKENS = 32768
CHUNK_SIZE_CHARS = 40000  # ~10,000 words; conservative to stay under model output limits

# If cleaned output is below this fraction of the original word count, treat as truncated
MIN_RETENTION_RATIO = 0.85


# Processing modes
class ProcessingMode:
    NONE = "none"           # No processing
    CLEAN = "clean"         # Clean artifacts only
    SPEED_READ = "speed"    # Summarize to ~30%
    SUMMARY = "summary"     # Heavy summarization to ~10%


# Cached API clients, keyed by backend name
_clients: dict = {}


def get_backend() -> str | None:
    """Return the active backend name: 'openrouter', 'gemini', or None."""
    if OPENAI_SDK_AVAILABLE and os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if GENAI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return None


def get_model() -> str:
    """Return the model id for the active backend."""
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    if get_backend() == "gemini" and model.startswith("google/"):
        # google-genai wants bare model names, not OpenRouter slugs
        model = model.removeprefix("google/")
    return model


def _get_client():
    """Get (and cache) the API client for the active backend."""
    backend = get_backend()
    if backend is None:
        return None
    client = _clients.get(backend)
    if client is None:
        if backend == "openrouter":
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        _clients[backend] = client
    return client


def is_llm_available() -> bool:
    """Check if an LLM backend is configured and available."""
    return get_backend() is not None


def _is_retryable(exc: Exception) -> bool:
    """Rate limits and transient overload errors are worth retrying."""
    msg = str(exc)
    return any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "rate limit", "503", "overloaded"))


def _generate(prompt: str, temperature: float,
              progress_callback: Callable[[str], None] = None) -> tuple[str, bool]:
    """Run one LLM completion on the active backend.

    Returns (text, truncated) where truncated means the model hit its output
    token limit.  Raises on unrecoverable API errors.
    """
    backend = get_backend()
    client = _get_client()
    model = get_model()

    for attempt in range(MAX_RETRIES):
        try:
            if backend == "openrouter":
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/derekg/epub2mp3",
                        "X-Title": "Inkvoice",
                    },
                )
                choice = response.choices[0]
                text = (choice.message.content or "").strip()
                truncated = choice.finish_reason in ("length", "max_tokens")
                return text, truncated

            # gemini
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "maxOutputTokens": MAX_OUTPUT_TOKENS,
                },
            )
            text = (response.text or "").strip()
            truncated = False
            try:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason and str(finish_reason) in ("MAX_TOKENS", "2"):
                    truncated = True
            except (AttributeError, IndexError):
                pass
            return text, truncated

        except Exception as e:
            if _is_retryable(e) and attempt < MAX_RETRIES - 1:
                if progress_callback:
                    progress_callback(f"Rate limited, retrying in {RETRY_DELAY * (attempt + 1)}s...")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise
    raise Exception("Max retries exceeded")


# Text cleaning prompt - optimized for TTS
CLEAN_PROMPT = """You are preparing a book chapter for text-to-speech narration.

YOUR ONLY JOB IS DELETION. Delete the following artifact types and nothing else:
- Footnote markers: [1], [2], [3], *, †, ‡, §
- Standalone page numbers (a number on its own line, e.g. "42" or "Page 12")
- Figure/table callouts: "See Figure 3.2", "Table 4", "(Fig. 1)", etc.
- URLs and bare hyperlinks (https://..., www....)
- Repeated running headers or footers (same text block appearing multiple times)
- Lines that are only whitespace, dashes, or asterisks used as dividers

STRICT RULES — violations are not acceptable:
- DO NOT rephrase, rewrite, or tighten any sentence
- DO NOT remove redundant phrasing or repeated ideas
- DO NOT summarize or condense anything
- DO NOT correct grammar or spelling
- DO NOT add any words that were not in the original
- Every prose sentence must be reproduced WORD FOR WORD

Output only the cleaned text with no commentary.

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


def _is_truncated(original: str, cleaned: str) -> bool:
    """Return True if cleaned output looks truncated relative to original."""
    orig_words = len(original.split())
    clean_words = len(cleaned.split())
    if orig_words == 0:
        return False
    return (clean_words / orig_words) < MIN_RETENTION_RATIO


def _clean_chunk(text: str, progress_callback: Callable[[str], None] = None) -> str:
    """Clean a single chunk of text, retrying with smaller sub-chunks on truncation."""
    result, truncated = _generate(
        CLEAN_PROMPT.format(text=text), temperature=0.1,
        progress_callback=progress_callback,
    )

    if not truncated:
        truncated = _is_truncated(text, result)

    if truncated:
        # Split into two halves and clean each independently
        paragraphs = text.split('\n\n')
        mid = len(paragraphs) // 2
        if mid == 0:
            # Single paragraph too long — fall back to basic cleaning
            if progress_callback:
                progress_callback("Chunk too long for model output limit, using basic cleaning")
            return clean_text_basic(text)
        if progress_callback:
            progress_callback("Output truncated, splitting chunk and retrying...")
        half_a = '\n\n'.join(paragraphs[:mid])
        half_b = '\n\n'.join(paragraphs[mid:])
        return _clean_chunk(half_a, progress_callback) + '\n\n' + _clean_chunk(half_b, progress_callback)

    return result


def clean_text_with_llm(
    text: str,
    progress_callback: Callable[[str], None] = None,
) -> str:
    """
    Clean text using the configured LLM to remove artifacts that don't verbalize well.

    For long texts, chunks by paragraph boundaries to stay within output token limit.
    """
    if not is_llm_available():
        if progress_callback:
            progress_callback("No LLM configured, using basic cleaning")
        return clean_text_basic(text)

    model = get_model()
    chunks = _chunk_text(text)

    if progress_callback:
        if len(chunks) == 1:
            progress_callback(f"Cleaning text with {model}...")
        else:
            progress_callback(f"Cleaning text with {model} ({len(chunks)} chunks)...")

    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            if len(chunks) > 1 and progress_callback:
                progress_callback(f"Cleaning chunk {i+1}/{len(chunks)}...")
            cleaned = _clean_chunk(chunk, progress_callback)
            cleaned_chunks.append(cleaned)
        except Exception as e:
            if progress_callback:
                progress_callback(f"LLM error on chunk {i+1}, using basic cleaning: {e}")
            cleaned_chunks.append(clean_text_basic(chunk))

    return '\n\n'.join(cleaned_chunks)


def summarize_text_with_llm(
    text: str,
    title: str = "Chapter",
    target_percent: int = 30,
    progress_callback: Callable[[str], None] = None,
) -> str:
    """Summarize text using the configured LLM for speed-read/summary modes."""
    if not is_llm_available():
        if progress_callback:
            progress_callback("No LLM configured, cannot summarize")
        return text

    word_count = len(text.split())
    target_words = int(word_count * target_percent / 100)

    if progress_callback:
        progress_callback(f"Summarizing to ~{target_words} words...")

    try:
        result, _ = _generate(
            SUMMARIZE_PROMPT.format(
                text=text,
                title=title,
                target_words=target_words,
                target_percent=target_percent,
                original_words=word_count,
            ),
            temperature=0.4,  # Allow some creativity for natural summaries
            progress_callback=progress_callback,
        )
        return result or text
    except Exception as e:
        if progress_callback:
            progress_callback(f"Summarization error: {e}")
        return text


def clean_text_basic(text: str) -> str:
    """Basic regex-based text cleaning (fallback when no LLM is available)."""
    # Remove footnote markers only — NOT editorial bracket insertions like [sic] or [that had]
    text = re.sub(r'\[\d+(?:\s*[,\-–]\s*\d+)*\]', '', text)  # numeric: [1], [1,2], [1-3]
    text = re.sub(r'\[\*+\]', '', text)           # asterisk: [*], [**]
    text = re.sub(r'(?<!\w)[*†‡§¶]+(?!\w)', '', text)  # standalone symbols only

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
        return clean_text_with_llm(text, progress_callback)

    if mode == ProcessingMode.SPEED_READ:
        # Clean first, then summarize to 30%
        cleaned = clean_text_with_llm(text, progress_callback)
        return summarize_text_with_llm(cleaned, title, 30, progress_callback)

    if mode == ProcessingMode.SUMMARY:
        # Clean first, then heavy summarization to 10%
        cleaned = clean_text_with_llm(text, progress_callback)
        return summarize_text_with_llm(cleaned, title, 10, progress_callback)

    return text


if __name__ == "__main__":
    backend = get_backend()
    print(f"LLM backend: {backend or 'none'}")
    if backend:
        print(f"Model: {get_model()}")
        test_text = """
        This is a test paragraph[1] with some footnotes[2] and a URL https://example.com
        that should be cleaned. See Figure 3.2 for more details.

        42

        Here's another paragraph with more content that should remain intact.
        """
        print("\n--- LLM cleaning test ---")
        print(clean_text_with_llm(test_text))
    else:
        print("Set OPENROUTER_API_KEY (or GEMINI_API_KEY) to test LLM cleaning")
        print("\n--- Basic cleaning test ---")
        print(clean_text_basic("Test[1] with footnote"))
