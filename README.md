# Inkvoice

Turn any EPUB into a beautifully narrated audiobook using local AI text-to-speech.

Uses [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) for on-device speech synthesis and [Gemini](https://ai.google.dev/) for optional text processing.

## Features

- **Web interface** — Clean, responsive UI with drag-and-drop; works on mobile and tablet
- **CLI** — Batch convert from the command line
- **8 voices** — American and British, male and female (see below)
- **Voice preview** — Audition each voice before committing to a long conversion
- **Chapter selection** — Choose exactly which chapters to include
- **Estimated listening time** — Updates live as you select or deselect chapters
- **Narration speed** — 0.75×, 1×, 1.25×, 1.5×, 2×
- **Bitrate selection** — 64 / 128 / 192 kbps
- **M4B audiobook format** — Single file with embedded chapter markers (requires ffmpeg)
- **MP3 with metadata** — ID3 tags with title, author, chapter number, and cover art
- **Multi-job queue** — Convert multiple books simultaneously; track all jobs in a sidebar
- **Resume interrupted jobs** — Checkpoint system saves per-chapter progress; pick up where you left off after a failure or restart
- **Persistent library** — Browse all completed audiobooks with cover art, duration, and one-click download
- **Cancel conversion** — Stop any in-progress job at any time
- **Chapter announcements** — Optionally speak chapter titles
- **Auto cleanup** — Temp files cleaned up automatically
- **Browser notifications** — Get notified when a book finishes, even with the tab in the background
- **AI text processing** — Powered by Gemini:
  - **Narration-ready** — Remove footnotes, URLs, figure captions
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
- ffmpeg (optional, for M4B format)
- Gemini API key (optional, for text processing)

## Installation

```bash
git clone https://github.com/derekg/epub2mp3.git
cd epub2mp3
pip install -r requirements.txt

# Optional: M4B support
brew install ffmpeg       # macOS
apt install ffmpeg        # Linux

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
4. Pick format, bitrate, speed, and text processing
5. Click **Convert** — queue another book immediately while the first runs
6. Browse completed books any time in the **Library** tab

### Command Line

```bash
# Basic conversion
python cli.py convert book.epub

# Specify voice and format
python cli.py convert book.epub --voice george --format m4b

# Resume an interrupted conversion
python cli.py convert book.epub --resume

# AI text cleaning (remove footnotes, artifacts)
python cli.py convert book.epub --clean

# Condensed version (~30%)
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

Kokoro TTS runs on Apple Silicon via MLX. On an M-series Mac a 10-hour audiobook typically takes 1–2 hours to generate. Only one book is processed at a time to avoid GPU contention; additional jobs queue automatically.

## Project Structure

```
inkvoice/
├── app.py              # FastAPI web server
├── cli.py              # Command-line interface
├── converter.py        # EPUB parsing and conversion pipeline
├── tts.py              # TTS engine wrapper + speed resampling
├── kokoro_tts.py       # Kokoro MLX/ONNX model interface
├── text_processor.py   # Gemini text processing
├── templates/
│   └── index.html      # Web UI (single-page)
├── test_*.py           # pytest test suite
└── requirements.txt
```

## License

MIT
