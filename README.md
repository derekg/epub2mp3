# Inkvoice

Turn your ebooks into audiobooks using [Kyutai's Pocket TTS](https://kyutai.org/blog/2026-01-13-pocket-tts) - a lightweight, high-quality text-to-speech model that runs locally on CPU.

## Features

- **Web interface** - Modern dark theme with drag-and-drop
- **CLI** - Batch convert from the command line
- **8 built-in voices** - Alba, Marius, Javert, Jean, Fantine, Cosette, Eponine, Azelma
- **Voice preview** - Listen to voices before converting
- **Custom voice cloning** - Use any WAV file to clone a voice (CLI only)
- **Chapter selection** - Choose which chapters to convert
- **Smart EPUB parsing** - TOC-first parsing with intelligent chapter detection
- **Front/back matter detection** - Auto-deselects title pages, copyright, acknowledgements, etc.
- **M4B audiobook format** - Single file with embedded chapter markers (requires ffmpeg)
- **MP3 with metadata** - ID3 tags with title, author, chapter, and cover art
- **Resume support** - Skip already-converted chapters
- **Chapter announcements** - Optionally speak chapter titles

## Requirements

- Python 3.11+
- ffmpeg (optional, for M4B format)

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
```

## Usage

### Web Interface

```bash
python cli.py serve
```

Open http://localhost:8000 in your browser:

1. Drop an EPUB file (or click to browse)
2. Select chapters to include (front/back matter auto-deselected)
3. Choose a voice and click the play button to preview
4. Pick format: MP3 (per-chapter or combined) or M4B (single file with chapters)
5. Click "Convert" and download when complete

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

## Performance

Pocket TTS runs at ~6x realtime on CPU (Apple M4). A 5-hour audiobook takes roughly 50 minutes to generate.

The TTS model (~225MB) is downloaded automatically on first run and cached locally.

## Project Structure

```
inkvoice/
├── cli.py              # Command-line interface (typer)
├── app.py              # FastAPI web server
├── converter.py        # EPUB parsing and TTS conversion
├── templates/
│   └── index.html      # Web UI
└── requirements.txt    # Python dependencies
```

## License

MIT
