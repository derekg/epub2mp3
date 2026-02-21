# Inkvoice

Convert any EPUB into a high-quality audiobook using AI-powered text-to-speech.

Uses [Pocket TTS](https://kyutai.org/blog/2026-01-13-pocket-tts) for local speech synthesis and [Gemini](https://ai.google.dev/) for intelligent text processing.

![Inkvoice UI](docs/screenshots/inkvoice-ui.png)

## Features

- **Web interface** - Clean, modern UI with drag-and-drop
- **CLI** - Batch convert from the command line
- **8 built-in voices** - Alba, Marius, Javert, Jean, Fantine, Cosette, Eponine, Azelma
- **Voice preview** - Listen to voices before converting
- **Custom voice cloning** - Use any WAV file to clone a voice (CLI only)
- **Chapter selection** - Choose which chapters to convert
- **Estimated listening time** - See duration estimate before converting, updates per chapter selection
- **Narration speed** - 0.75×, 1×, 1.25×, 1.5×, 2× playback speed
- **Bitrate selection** - 64 / 128 / 192 kbps output with estimated file size
- **Multi-job queue** - Convert multiple books in parallel, track all jobs in a sidebar
- **Cancel conversion** - Stop an in-progress job at any time
- **Job history** - Completed downloads persist across page refreshes via localStorage
- **Smart EPUB parsing** - TOC-first parsing with intelligent chapter detection
- **Front/back matter detection** - Auto-deselects title pages, copyright, acknowledgements
- **Cover art display** - Shows book cover in the web UI
- **M4B audiobook format** - Single file with embedded chapter markers (requires ffmpeg)
- **MP3 with metadata** - ID3 tags with title, author, chapter, and cover art
- **Chapter announcements** - Optionally speak chapter titles
- **Auto cleanup** - Completed jobs and temp files cleaned up automatically after 1 hour
- **AI text processing** - Powered by Gemini 3 Flash:
  - **Narration-ready** - Remove footnotes, URLs, figure references
  - **Condensed** - ~30% shorter while preserving key information
  - **Key points** - ~10% summary of main ideas

## Requirements

- Python 3.11+
- ffmpeg (optional, for M4B format)
- Gemini API key (optional, for text processing)

## Installation

```bash
# Clone the repo
git clone https://github.com/derekg/epub2mp3.git
cd epub2mp3

# Install dependencies
pip install -r requirements.txt

# Optional: Install ffmpeg for M4B support
brew install ffmpeg  # macOS
# or: apt install ffmpeg  # Linux

# Optional: Set up Gemini for text processing
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Usage

### Web Interface

```bash
python app.py
```

Open http://localhost:8000 in your browser:

1. Drop an EPUB file (or click to browse)
2. Select chapters to include (front/back matter auto-deselected)
3. See estimated listening time update as you select/deselect chapters
4. Choose a voice and click the play button to preview
5. Pick format: MP3 (per-chapter) or M4B (single file with chapters)
6. Set narration speed, output bitrate, and text processing mode
7. Click "Convert" — the form resets so you can queue another book while it runs
8. Download files when complete; history persists across page refreshes

### Command Line

```bash
# Convert an EPUB to MP3 (one file per chapter)
python cli.py convert book.epub

# Use a different voice
python cli.py convert book.epub --voice marius

# Create M4B audiobook with chapter markers
python cli.py convert book.epub --format m4b

# Combine all chapters into a single MP3
python cli.py convert book.epub --single-file

# Announce chapter titles at the start of each chapter
python cli.py convert book.epub --announce

# Resume an interrupted conversion (skip existing files)
python cli.py convert book.epub --resume

# Clean text with Gemini (removes footnotes, artifacts)
python cli.py convert book.epub --clean

# Create condensed version (~30%)
python cli.py convert book.epub --speed-read

# Create key points summary (~10%)
python cli.py convert book.epub --summary

# Specify output directory
python cli.py convert book.epub --output ./audiobooks

# List available voices
python cli.py voices
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output directory (default: same as input) |
| `--voice` | `-v` | Voice name or path to WAV file |
| `--format` | `-f` | Output format: `mp3` or `m4b` |
| `--single-file` | `-s` | Combine chapters into one MP3 |
| `--resume` | `-r` | Skip chapters with existing output files |
| `--announce` | `-a` | Speak chapter title at start of each chapter |
| `--clean` | `-c` | Narration-ready text (remove artifacts) |
| `--speed-read` | | Create ~30% condensed version |
| `--summary` | | Create ~10% key points summary |

## Performance

Pocket TTS runs at ~6x realtime on CPU (Apple M4). A 5-hour audiobook takes roughly 50 minutes to generate.

The TTS model (~225MB) is downloaded automatically on first run and cached locally.

## Project Structure

```
inkvoice/
├── cli.py              # Command-line interface (typer)
├── app.py              # FastAPI web server
├── converter.py        # EPUB parsing and conversion orchestration
├── tts.py              # Pocket TTS speech synthesis + speed resampling
├── text_processor.py   # Gemini-powered text processing
├── templates/
│   └── index.html      # Web UI
├── test_*.py           # pytest test suite (124 tests)
└── requirements.txt    # Python dependencies
```

## License

MIT
