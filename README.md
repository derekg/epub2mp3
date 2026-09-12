# Inkvoice

Turn any EPUB into a beautifully narrated audiobook using on-device AI text-to-speech.

Uses [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) (82M parameter model) for fast, high-quality local synthesis and an LLM (via [OpenRouter](https://openrouter.ai/) or the Gemini API) for optional text cleaning and summarization.

## Features

- **Web interface** — Clean, responsive UI with drag-and-drop; works on mobile and tablet
- **CLI** — Batch convert from the command line
- **8 voices** — American and British, male and female
- **Voice preview** — Audition each voice before committing to a long conversion
- **Chapter selection** — Choose exactly which chapters to include
- **Estimated listening time** — Updates live as you select or deselect chapters
- **Narration speed** — 0.75×, 1×, 1.25×, 1.5×, 2×
- **Bitrate selection** — 64 / 128 / 192 kbps (default 64: Kokoro outputs 24 kHz mono speech, for which 64 kbps is transparent — a 10-hour book is ~290 MB instead of ~865 MB at 192)
- **M4B audiobook format** — Single file with embedded chapter markers (requires ffmpeg)
- **MP3 with metadata** — ID3 tags with title, author, chapter number, and cover art
- **Multi-job queue** — Start another book while the first is still converting
- **Resume interrupted jobs** — Per-chapter checkpoint system; pick up where you left off after a crash or restart
- **Persistent library** — Browse all completed audiobooks with cover art, duration, and one-click download
- **Cancel conversion** — Stop any in-progress job at any time
- **Chapter announcements** — Optionally speak chapter titles before each chapter
- **Auto cleanup** — Temp files removed automatically after 1 hour
- **Browser notifications** — Get notified when a book finishes, even with the tab in the background
- **AI text processing** — Optional LLM-powered modes (OpenRouter or Gemini API):
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
- OpenRouter API key (or Gemini API key) — optional, for text cleaning and summarization

## Installation

```bash
git clone https://github.com/derekg/epub2mp3.git
cd epub2mp3
pip install -r requirements.txt

# Download Kokoro model weights (first run only; ~160 MB MLX, ~340 MB ONNX)
python setup_kokoro.py

# Optional: M4B support
brew install ffmpeg

# Optional: LLM text processing (clean / speed-read / summary modes)
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

### Text processing model

Text cleaning defaults to **`google/gemini-2.5-flash-lite`** via OpenRouter.
This default is deliberate: clean mode re-emits the entire chapter (minus
footnotes, page numbers, URLs, and other non-narration artifacts), so output
tokens dominate cost, and the model must reproduce long passages word for
word without paraphrasing. Flash-Lite does this reliably, supports 65k-token
outputs, is fast, and costs about **$0.07 per 100k-word book** — roughly 7×
cheaper than Gemini 3 Flash ($0.10/$0.40 vs $0.50/$3.00 per 1M tokens).

Configuration (in `.env` or the environment):

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | Preferred backend — any model on [openrouter.ai](https://openrouter.ai/models) |
| `GEMINI_API_KEY` | Fallback backend — direct Gemini API (used when no OpenRouter key is set) |
| `LLM_MODEL` | Override the model, e.g. `LLM_MODEL=openai/gpt-5-nano` (default: `google/gemini-2.5-flash-lite`) |

To verify cleaning quality after changing models, run `python test_clean_diff.py your.epub`
— it diffs original vs cleaned text so you can confirm only artifacts were removed.

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

### Adaptive throttling

TTS generation is CPU/GPU-heavy. To avoid making the rest of the machine
sluggish while you're using it, the server checks (via macOS's HID idle
time) whether anyone is actively at the keyboard/mouse and pauses briefly
between chunks if so — full speed resumes automatically once the machine
goes idle. Current state is visible at `GET /api/stats` under
`activity_throttle`. Tune it with env vars (or add them to `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `EPUB2MP3_ADAPTIVE_THROTTLE` | `1` | Set to `0` to disable and always run at full speed |
| `EPUB2MP3_IDLE_THRESHOLD_SECONDS` | `120` | Seconds of no input before the machine is considered idle |
| `EPUB2MP3_ACTIVE_THROTTLE_SLEEP` | `0.4` | Pause (seconds) inserted between chunks while the machine is actively in use |

To measure synthesis speed on your machine (e.g. before and after a dependency upgrade):

```bash
python benchmark_tts.py            # default voice
python benchmark_tts.py heart      # specific voice
```

Notes from benchmarking the ONNX engine on x86 CPU: the quantized model
variants that kokoro-onnx publishes did not pay off there (fp16 was within
noise of fp32; int8 was several times *slower*), and parallel chunk synthesis
gains nothing because ONNX Runtime already saturates all cores. If you have
an NVIDIA GPU, `pip install kokoro-onnx[gpu]` enables CUDA, which is several
times faster than CPU.

## Project Structure

```
inkvoice/
├── app.py              # FastAPI web server and job management
├── cli.py              # Command-line interface (Typer)
├── converter.py        # EPUB parsing and audio encoding pipeline
├── activity_monitor.py # Adaptive throttling based on macOS idle time
├── tts.py              # TTS engine wrapper, speed resampling, text chunking
├── kokoro_tts.py       # Kokoro MLX/ONNX model interface and voice catalogue
├── text_processor.py   # LLM text cleaning and summarization (OpenRouter / Gemini)
├── setup_kokoro.py     # First-run model download script
├── benchmark_tts.py    # TTS speed benchmark (run before/after upgrades)
├── templates/
│   └── index.html      # Web UI (single-page app)
├── static/
│   └── favicon.svg
├── test_*.py           # pytest test suite
└── requirements.txt
```

## License

MIT
