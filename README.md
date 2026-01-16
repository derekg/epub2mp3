# epub2mp3

Convert EPUB ebooks to MP3 audiobooks using [Kyutai's Pocket TTS](https://kyutai.org/blog/2026-01-13-pocket-tts) - a lightweight, high-quality text-to-speech model that runs on CPU.

## Features

- **Web interface** - Drag-and-drop in your browser
- **CLI** - Batch convert from the command line
- 8 built-in voices (alba, marius, javert, jean, fantine, cosette, eponine, azelma)
- Custom voice cloning via WAV file upload
- Per-chapter or single combined MP3 output
- Real-time progress tracking

## Requirements

- Python 3.11+

## Installation

```bash
# Clone the repo
git clone https://github.com/derekg/epub2mp3.git
cd epub2mp3

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Convert an EPUB to MP3 (one file per chapter)
python cli.py convert book.epub

# Use a different voice
python cli.py convert book.epub --voice marius

# Combine all chapters into a single MP3
python cli.py convert book.epub --single-file

# Specify output directory
python cli.py convert book.epub --output ./audiobooks

# List available voices
python cli.py voices
```

### Web Interface

```bash
python cli.py serve
# or
python app.py
```

Open http://localhost:8000 in your browser, then:

1. Drag and drop an EPUB file (or click to browse)
2. Select a voice from the dropdown
3. Optionally upload a WAV file for voice cloning
4. Toggle "One MP3 per chapter" on/off
5. Click "Convert to MP3"
6. Download the generated files

## Performance

Pocket TTS runs at ~6x realtime on CPU (Apple M4). A 5-hour audiobook takes roughly 50 minutes to generate.

The TTS model (~225MB) is downloaded automatically on first run and cached locally.

## Project Structure

```
epub2mp3/
├── cli.py              # Command-line interface
├── app.py              # FastAPI web server
├── converter.py        # EPUB parsing and TTS conversion
├── templates/
│   └── index.html      # Web UI
└── requirements.txt    # Python dependencies
```

## License

MIT
