# AGENTS.md

Guidelines for AI agents working in this codebase.

## What this is

Inkvoice converts EPUB ebooks into audiobooks (MP3 or M4B) using Kokoro TTS running locally on the machine. It has a FastAPI web UI and a Typer CLI. The primary target is Apple Silicon Macs where Kokoro runs via MLX at 13–18× real-time; ONNX is the fallback for other platforms.

## Key files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server — all HTTP endpoints, job management, SSE progress streaming, persistent library |
| `converter.py` | EPUB parsing, chapter extraction, text-to-audio pipeline, MP3/M4B encoding |
| `tts.py` | Public TTS API — `generate_speech()`, speed resampling, text chunking at sentence boundaries |
| `kokoro_tts.py` | Kokoro model loading (MLX and ONNX), voice catalogue, `generate_speech_kokoro()` |
| `text_processor.py` | Gemini-powered text cleaning (deletion-only) and summarization |
| `setup_kokoro.py` | Downloads ONNX model weights from GitHub releases on first run |
| `templates/index.html` | Entire web UI — single HTML file, vanilla JS, no build step |
| `cli.py` | Typer CLI, thin wrapper over the same conversion pipeline |

## Architecture

```
Browser / CLI
     │
     ▼
app.py  (FastAPI)
  ├── POST /api/upload    → parse EPUB, return chapter list
  ├── POST /api/convert   → create job, start background task
  ├── GET  /api/progress  → SSE stream of job progress
  ├── GET  /api/library   → list completed audiobooks from OUTPUT_DIR
  └── GET  /api/download/{job_id}/{filename}
           │
           ▼
     run_conversion()  (asyncio background task)
           │
           ├── text_processor.process_chapter()   [optional Gemini]
           └── converter.convert_epub_to_mp3()
                    │
                    └── tts.generate_speech()
                             │
                             └── kokoro_tts.generate_speech_kokoro()
                                      │
                                      ├── MLX engine (Apple Silicon)
                                      └── ONNX engine (fallback)
```

## Job lifecycle

1. `POST /api/convert` creates a job dict in the in-memory `jobs` dict, writes a checkpoint dir to `~/epub2mp3_output/checkpoints/{job_id}/`, and starts `run_conversion()` as an asyncio task.
2. `run_conversion()` holds `CONVERSION_SEMAPHORE` (only 1 job at a time), then calls the converter.
3. On completion, output files are copied to `~/epub2mp3_output/{job_id}/` and `_save_jobs_manifest()` writes `~/epub2mp3_output/jobs.json`.
4. On crash or cancel, checkpoint files remain so the job can be resumed.
5. `_recover_crashed_jobs()` and `_recover_orphaned_output_dirs()` run at startup to rebuild state from disk.

## TTS threading model

MLX uses a single Metal GPU command queue and is not thread-safe across OS threads. Two guards prevent Metal assertion failures:

- `TTS_GPU_SEMAPHORE` (asyncio.Semaphore(1)) — prevents concurrent TTS coroutines
- `TTS_EXECUTOR` (ThreadPoolExecutor(max_workers=1)) — pins all MLX calls to one OS thread

All TTS work runs via `loop.run_in_executor(TTS_EXECUTOR, ...)`.

## Text chunking

The TTS model has practical limits per call. `tts._split_text_into_chunks()` splits at sentence boundaries (`.!?;`), targeting ~500 words per chunk, with word-boundary fallback for sentences that exceed the limit.

`text_processor.clean_text_with_gemini()` also chunks by paragraph for chapters that exceed Gemini's effective output limit (~2000 words per call in practice). If a chunk's output is less than 85% of input words, it retries with the chunk halved.

## Persistent storage

```
~/epub2mp3_output/
├── jobs.json                    # manifest of completed/failed/cancelled jobs
├── checkpoints/
│   └── {job_id}/
│       ├── input.epub           # copy of source EPUB for resume
│       └── checkpoint.json      # per-chapter completion status
└── {job_id}/
    ├── Chapter 01.mp3
    ├── Chapter 02.mp3
    └── cover.jpg
```

## Testing

```bash
pytest                           # run all tests
pytest test_api.py               # specific file
pytest -x                        # stop on first failure
```

Test files use FastAPI `TestClient`. Tests that exercise TTS are skipped when the model isn't loaded. Imports resolve from the repo root — run pytest from there.

## Development commands

```bash
# Start server
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Download Kokoro model (first time only)
python setup_kokoro.py

# CLI
python cli.py convert book.epub --voice george --format m4b
```

## What not to do

- Don't add features beyond what's asked — the codebase is intentionally minimal.
- Don't mock the TTS model in tests — use the real one or skip.
- Don't touch `TTS_EXECUTOR` or `TTS_GPU_SEMAPHORE` without understanding the Metal threading constraint.
- Don't write output files outside `OUTPUT_DIR` or temp dirs created by `tempfile.mkdtemp`.
