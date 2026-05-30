# Inkvoice

Turn any EPUB into a beautifully narrated audiobook using on-device AI text-to-speech.

Uses [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) (82M parameter model) for fast, high-quality local synthesis and [Gemini](https://ai.google.dev/) for optional text cleaning and summarization.

## Features

- **Web interface** — Clean, responsive UI with drag-and-drop; works on mobile and tablet
- **CLI** — Batch convert from the command line
- **8 voices** — American and British, male and female
- **Voice preview** — Audition each voice before committing to a long conversion
- **Chapter selection** — Choose exactly which chapters to include
- **Estimated listening time** — Updates live as you select or deselect chapters
- **Narration speed** — 0.75×, 1×, 1.25×, 1.5×, 2×
- **Bitrate selection** — 64 / 128 / 192 kbps
- **M4B audiobook format** — Single file with embedded chapter markers (requires ffmpeg)
- **MP3 with metadata** — ID3 tags with title, author, chapter number, and cover art
- **Multi-job queue** — Start another book while the first is still converting
- **Resume interrupted jobs** — Per-chapter checkpoint system; pick up where you left off after a crash or restart
- **Persistent library** — Browse all completed audiobooks with cover art, duration, and one-click download
- **Cancel conversion** — Stop any in-progress job at any time
- **Chapter announcements** — Optionally speak chapter titles before each chapter
- **Auto cleanup** — Temp files removed automatically after 1 hour
- **Browser notifications** — Get notified when a book finishes, even with the tab in the background
- **AI text processing** — Optional Gemini-powered modes:
  - **Narration-ready** — Remove footnotes, URLs, figure captions, page numbers
  - **Condensed** — ~30% shorter while preserving key information
  - **Key points** — ~10% summary of main ideas

## Voices

| Name | Character | Gender | Accent |
|------|-----------|--------|--------|
| George ⭐ | Classic British | Male | British |
| Emma | Warm British | Female | British |
| Heart | Warm American | Female | American |
| Bella | Expressive American | Female | American |
| Sarah | Clear American | Female | American |
| Adam | Confident American | Male | American |
| Michael | Deep American | Male | American |
| Nova | Bright & Upbeat | Female | American |

## Requirements

- Python 3.11+
- Apple Silicon Mac recommended (uses MLX for fast on-device inference; ONNX fallback works on any platform but is slower)
- ffmpeg — optional, for M4B format (`brew install ffmpeg`)
- Gemini API key — optional, for text cleaning and summarization

## Installation

```bash
git clone https://github.com/derekg/epub2mp3.git
cd epub2mp3
pip install -r requirements.txt

# Download Kokoro model weights (first run only, ~180 MB)
python setup_kokoro.py

# Optional: M4B support
brew install ffmpeg

# Optional: Gemini text processing
echo "GEMINI_API_KEY=your_key_here" > .env
```

## Usage

### Web Interface

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000:

1. Drop an EPUB onto the page
2. Select chapters (front/back matter auto-deselected)
3. Choose a voice and preview it
4. Pick format, bitrate, speed, and text processing mode
5. Click **Convert** — start another book immediately while the first runs
6. Browse completed books in the **Library** tab

### Command Line

```bash
# Basic conversion
python cli.py convert book.epub

# Choose voice and format
python cli.py convert book.epub --voice george --format m4b

# Resume an interrupted conversion
python cli.py convert book.epub --resume

# AI text cleaning (remove footnotes, artifacts)
python cli.py convert book.epub --clean

# Condensed version (~30% of original length)
python cli.py convert book.epub --speed-read

# Key points summary (~10%)
python cli.py convert book.epub --summary

# List available voices
python cli.py voices
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output directory (default: same as input) |
| `--voice` | `-v` | Voice name |
| `--format` | `-f` | Output format: `mp3` or `m4b` |
| `--single-file` | `-s` | Combine chapters into one MP3 |
| `--resume` | `-r` | Skip chapters with existing output files |
| `--announce` | `-a` | Speak chapter title at start of each chapter |
| `--clean` | `-c` | Narration-ready text (remove artifacts) |
| `--speed-read` | | Condensed ~30% version |
| `--summary` | | Key points ~10% summary |

## Performance

On Apple Silicon (M-series) via MLX, Kokoro runs at ~13–18× real-time — a 10-hour audiobook typically takes 45 minutes to 1.5 hours to generate. On non-Apple hardware via ONNX the same job takes several hours.

Only one book is processed at a time to avoid GPU/NPU contention; additional jobs queue automatically and start as soon as the active one finishes.

## Project Structure

```
inkvoice/
├── app.py              # FastAPI web server and job management
├── cli.py              # Command-line interface (Typer)
├── converter.py        # EPUB parsing and audio encoding pipeline
├── tts.py              # TTS engine wrapper, speed resampling, text chunking
├── kokoro_tts.py       # Kokoro MLX/ONNX model interface and voice catalogue
├── text_processor.py   # Gemini text cleaning and summarization
├── setup_kokoro.py     # First-run model download script
├── templates/
│   └── index.html      # Web UI (single-page app)
├── static/
│   └── favicon.svg
├── test_*.py           # pytest test suite
└── requirements.txt
```

## License

MIT
