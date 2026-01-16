# AGENTS.md

Guidelines for AI agents working on this codebase.

## Project Overview

epub2mp3 is a web application that converts EPUB ebooks to MP3 audiobooks using Kyutai's Pocket TTS model.

## Architecture

- **app.py**: FastAPI server with REST API endpoints and SSE for progress updates
- **converter.py**: Core conversion logic - EPUB parsing, text chunking, TTS generation, MP3 encoding
- **templates/index.html**: Single-page web UI with drag-drop upload

## Key Technical Details

### Text Chunking
The TTS model has input length limits. Text is chunked at ~250 characters, splitting on sentence boundaries, then commas, then words as fallback. This is in `text_to_audio()`.

### Audio Processing
- Pocket TTS outputs float32 PCM at 24kHz
- Converted to int16 for pydub compatibility
- Exported as 192kbps MP3 via ffmpeg

### Async Pattern
Conversion runs in a background thread via `asyncio.to_thread()` to avoid blocking the event loop. Progress is reported via callback and streamed to the client via SSE.

## Development Commands

```bash
# Run the server
python app.py

# Server runs at http://localhost:8000
```

## Known Limitations

1. **CPU only**: Pocket TTS doesn't support MPS/CUDA due to hardcoded device placement
2. **Memory**: Large EPUBs with many chapters use significant memory
3. **No persistence**: Jobs are stored in memory, lost on restart

## Future Improvements

- Parallel chapter processing for multi-core speedup
- Persistent job storage (SQLite)
- GPU support if Kyutai adds it upstream
- CLI mode for batch processing
