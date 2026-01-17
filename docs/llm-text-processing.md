# LLM-Powered Text Processing for Inkvoice

## Problem

EPUB text often contains artifacts that don't verbalize well:

1. **Footnote markers** - `[1]`, `[2]`, superscript numbers
2. **Page numbers** - leftover from PDF conversions
3. **Headers/footers** - chapter titles repeated on every "page"
4. **Figure references** - "See Figure 3.2" when there's no figure
5. **Tables/lists** - don't flow well as speech
6. **URLs** - embedded links that get read out
7. **OCR artifacts** - weird characters from scanned books
8. **Formatting cruft** - excessive whitespace, weird punctuation

Additionally, users may want:
- **Summarization** - condensed "speed read" version
- **Key points extraction** - just the highlights

## Proposed Solution

Use a small local LLM (Gemma 2B or Phi-3-mini) for:

### 1. Text Cleaning Mode (default: on)
Clean each chapter before TTS:
- Remove footnote markers and references
- Strip page numbers and headers/footers
- Convert tables to prose descriptions
- Remove or describe figure references
- Fix OCR artifacts and weird punctuation

### 2. Summarize Mode (optional)
Create condensed versions:
- **Chapter summaries** - 1-2 paragraphs per chapter
- **Speed read** - ~30% of original length, key points only
- **Executive summary** - entire book in 5-10 minutes

## Model Options

| Model | Size | Speed | Quality | Notes |
|-------|------|-------|---------|-------|
| **Gemma 2B** | 1.5GB | Fast | Good | Google, Apache 2.0 |
| **Phi-3-mini** | 2.3GB | Fast | Good | Microsoft, MIT |
| **Qwen2-1.5B** | 1.1GB | Fastest | OK | Alibaba, good for cleaning |
| **Llama-3.2-1B** | 1.3GB | Fast | OK | Meta, newest small model |

**Recommendation**: Start with **Gemma 2B** via Ollama - good balance of quality and speed.

## Implementation Options

### Option A: Ollama (Recommended)
- Simple API, handles model management
- User installs Ollama separately
- We call `ollama.generate()` or HTTP API
- Graceful fallback if not available

```python
import ollama

def clean_text_with_llm(text: str) -> str:
    response = ollama.generate(
        model='gemma2:2b',
        prompt=f"""Clean this text for text-to-speech. Remove:
- Footnote markers like [1], [2]
- Page numbers
- Figure/table references
- URLs
Keep the content intact, just remove artifacts that would sound bad when read aloud.

Text:
{text}

Cleaned text:"""
    )
    return response['response']
```

### Option B: llama-cpp-python
- No external dependencies
- We bundle or download the model
- More control, but more complexity

### Option C: Transformers + GGUF
- Use HuggingFace transformers with quantized models
- Good middle ground

## Processing Pipeline

```
EPUB → Parse chapters → [LLM Clean] → [LLM Summarize?] → TTS → MP3
                            ↑               ↑
                        (optional)      (optional)
```

## API Design

### CLI
```bash
# Enable text cleaning (default)
inkvoice convert book.epub --clean

# Disable cleaning
inkvoice convert book.epub --no-clean

# Speed read mode (summarize)
inkvoice convert book.epub --speed-read

# Chapter summaries only
inkvoice convert book.epub --summaries-only
```

### Web UI
- Checkbox: "Clean text for speech" (default: on)
- Dropdown: "Mode" → Full / Speed Read / Summaries Only
- Show word count change after processing

## Prompts

### Text Cleaning Prompt
```
You are preparing text for text-to-speech conversion. Clean the following text by:

1. Removing footnote markers like [1], [2], *, †, etc.
2. Removing page numbers
3. Removing repeated headers/footers
4. Converting "See Figure X" to just describing what would be there, or removing if not important
5. Removing URLs (keep the link text if meaningful)
6. Fixing OCR artifacts and weird punctuation
7. Keeping all actual content intact

Do not summarize or change the meaning. Just clean for audio.

Text:
{text}
```

### Summarization Prompt
```
Summarize this chapter for an audiobook listener who wants the key points.
Keep it engaging and narrative, not bullet points.
Target length: ~{target_percent}% of original.

Chapter: {title}

{text}
```

## Chunking Strategy

LLMs have context limits. For a 2B model with 8K context:
- ~6000 tokens for input
- ~2000 tokens for output
- Roughly 4000 words input per chunk

Strategy:
1. Split chapter into ~3000 word chunks with overlap
2. Process each chunk
3. Merge results, deduplicating overlap

## Performance Estimates

On Apple M-series:
- Gemma 2B: ~30-50 tokens/sec
- Cleaning 5000 words: ~10-15 seconds
- Full 80,000 word book: ~3-5 minutes of LLM processing

This adds modest time to the overall conversion (which is dominated by TTS anyway).

## Graceful Degradation

If Ollama/LLM not available:
1. Check for Ollama on startup
2. If not available, disable LLM features in UI
3. Show message: "Install Ollama for text cleaning features"
4. Fall back to regex-based cleaning (current approach)

## Next Steps

1. [ ] Add Ollama integration with availability check
2. [ ] Implement text cleaning prompt
3. [ ] Add --clean / --no-clean CLI flags
4. [ ] Add cleaning toggle to web UI
5. [ ] Implement summarization prompt
6. [ ] Add speed-read mode
7. [ ] Benchmark performance on real books
8. [ ] Test quality on various EPUB sources
