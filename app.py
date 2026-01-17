"""FastAPI server for EPUB to MP3 conversion."""

import asyncio
import json
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pocket_tts import TTSModel
import lameenc

from converter import convert_epub_to_mp3, parse_epub, BUILTIN_VOICES, is_ffmpeg_available
from text_processor import ProcessingMode, is_ollama_available

# Store for uploaded EPUBs awaiting conversion
uploads: dict = {}

# Global model instance (loaded once at startup)
tts_model: TTSModel = None

# Store for conversion jobs
jobs: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TTS model on startup."""
    global tts_model
    print("Loading TTS model...")
    tts_model = TTSModel.load_model()
    print("TTS model loaded!")
    yield


app = FastAPI(title="Inkvoice", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text()


@app.get("/api/voices")
async def get_voices():
    """Return list of available voices."""
    return {"voices": BUILTIN_VOICES}


# Cache for voice previews
voice_preview_cache: dict[str, bytes] = {}


@app.get("/api/voice-preview/{voice}")
async def get_voice_preview(voice: str):
    """Generate a short audio preview for a voice."""
    from fastapi.responses import Response
    import numpy as np

    if voice not in BUILTIN_VOICES:
        raise HTTPException(status_code=404, detail="Voice not found")

    # Check cache first
    if voice in voice_preview_cache:
        return Response(
            content=voice_preview_cache[voice],
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400"}
        )

    # Generate preview
    preview_text = f"Hello, I'm {voice}. I'll be reading your book."

    try:
        voice_state = tts_model.get_state_for_audio_prompt(voice)
        audio = tts_model.generate_audio(voice_state, preview_text)
        audio_np = audio.numpy()

        # Convert to MP3
        if audio_np.dtype != np.int16:
            audio_np = (audio_np * 32767).astype(np.int16)

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(128)
        encoder.set_in_sample_rate(tts_model.sample_rate)
        encoder.set_out_sample_rate(tts_model.sample_rate)
        encoder.set_channels(1)
        encoder.set_quality(2)

        mp3_data = encoder.encode(audio_np.tobytes())
        mp3_data += encoder.flush()

        # Convert to bytes and cache
        mp3_bytes = bytes(mp3_data)
        voice_preview_cache[voice] = mp3_bytes

        return Response(
            content=mp3_bytes,
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {e}")


@app.get("/api/capabilities")
async def get_capabilities():
    """Return server capabilities (e.g., available formats)."""
    return {
        "formats": {
            "mp3": True,
            "m4b": is_ffmpeg_available(),
        },
        "ffmpeg_available": is_ffmpeg_available(),
        "ollama_available": is_ollama_available(),
        "text_processing_modes": [
            {"value": "none", "label": "None", "description": "No text processing"},
            {"value": "clean", "label": "Clean", "description": "Remove footnotes, artifacts"},
            {"value": "speed", "label": "Speed Read", "description": "~30% summary"},
            {"value": "summary", "label": "Summary", "description": "~10% brief summary"},
        ],
    }


@app.post("/api/upload")
async def upload_epub(epub_file: UploadFile = File(...)):
    """Upload an EPUB and get chapter list for selection."""
    upload_id = str(uuid.uuid4())

    # Create temp directory
    upload_dir = Path(tempfile.mkdtemp(prefix=f"epub2mp3_upload_{upload_id}_"))
    epub_path = upload_dir / "input.epub"

    # Save uploaded file
    with open(epub_path, "wb") as f:
        shutil.copyfileobj(epub_file.file, f)

    # Parse EPUB to get chapters
    try:
        book = parse_epub(str(epub_path))
    except Exception as e:
        shutil.rmtree(upload_dir)
        raise HTTPException(status_code=400, detail=f"Failed to parse EPUB: {e}")

    # Store upload info
    uploads[upload_id] = {
        "epub_path": str(epub_path),
        "upload_dir": upload_dir,
        "book": book,
    }

    # Return chapter list with word counts and front/back matter flags
    chapters = []
    total_words = 0
    for i, ch in enumerate(book.chapters):
        total_words += ch.word_count
        chapters.append({
            "index": i,
            "title": ch.title,
            "length": len(ch.text),
            "words": ch.word_count,
            "is_front_matter": ch.is_front_matter,
            "is_back_matter": ch.is_back_matter,
        })

    return {
        "upload_id": upload_id,
        "title": book.title,
        "author": book.author,
        "chapters": chapters,
        "total_words": total_words,
        "has_cover": book.cover_image is not None,
    }


@app.post("/api/convert")
async def start_conversion(
    upload_id: str = Form(None),
    selected_chapters: str = Form(None),  # JSON array of indices
    voice: str = Form("alba"),
    per_chapter: bool = Form(True),
    skip_existing: bool = Form(False),  # Resume support
    announce_chapters: bool = Form(False),  # Chapter announcements
    output_format: str = Form("mp3"),  # mp3 or m4b
    text_processing: str = Form("none"),  # none, clean, speed, summary
    custom_voice: UploadFile = File(None),
    epub_file: UploadFile = File(None),  # For backwards compatibility
):
    """Start EPUB to audio conversion (MP3 or M4B)."""
    # Validate output format
    output_format = output_format.lower()
    if output_format not in ("mp3", "m4b"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {output_format}")

    if output_format == "m4b" and not is_ffmpeg_available():
        raise HTTPException(status_code=400, detail="M4B format requires ffmpeg (not installed on server)")

    job_id = str(uuid.uuid4())
    job_dir = Path(tempfile.mkdtemp(prefix=f"epub2mp3_{job_id}_"))

    # Get EPUB path from upload_id or direct upload
    if upload_id and upload_id in uploads:
        upload = uploads[upload_id]
        epub_path = upload["epub_path"]
        # Parse selected chapters
        chapter_indices = None
        if selected_chapters:
            try:
                chapter_indices = json.loads(selected_chapters)
            except json.JSONDecodeError:
                pass
    elif epub_file and epub_file.filename:
        # Direct upload (backwards compatible)
        epub_path = job_dir / "input.epub"
        with open(epub_path, "wb") as f:
            shutil.copyfileobj(epub_file.file, f)
        epub_path = str(epub_path)
        chapter_indices = None
    else:
        raise HTTPException(status_code=400, detail="No EPUB file provided")

    # Handle custom voice if provided
    voice_to_use = voice
    if custom_voice and custom_voice.filename:
        custom_voice_path = job_dir / "custom_voice.wav"
        with open(custom_voice_path, "wb") as f:
            shutil.copyfileobj(custom_voice.file, f)
        voice_to_use = str(custom_voice_path)

    # Initialize job state
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting...",
        "output_dir": job_dir / "output",
        "job_dir": job_dir,
        "files": []
    }

    # Validate text processing mode
    if text_processing not in ("none", "clean", "speed", "summary"):
        text_processing = "none"

    # Run conversion in background
    asyncio.create_task(run_conversion(
        job_id, epub_path, voice_to_use, per_chapter, chapter_indices, skip_existing, announce_chapters, output_format, text_processing
    ))

    return {"job_id": job_id}


async def run_conversion(
    job_id: str,
    epub_path: str,
    voice: str,
    per_chapter: bool,
    chapter_indices: list[int] = None,
    skip_existing: bool = False,
    announce_chapters: bool = False,
    output_format: str = "mp3",
    text_processing: str = "none",
):
    """Run the conversion in the background."""
    job = jobs[job_id]

    def progress_callback(current: int, total: int, message: str):
        job["progress"] = current
        job["message"] = message

    try:
        output_files = await asyncio.to_thread(
            convert_epub_to_mp3,
            epub_path,
            str(job["output_dir"]),
            tts_model,
            voice,
            per_chapter,
            progress_callback,
            chapter_indices,
            skip_existing,
            announce_chapters,
            output_format,
            text_processing,
        )

        job["status"] = "complete"
        job["files"] = [Path(f).name for f in output_files]
        job["progress"] = 100
        job["message"] = "Conversion complete!"

    except Exception as e:
        job["status"] = "error"
        job["message"] = str(e)
        job["progress"] = 0


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a conversion job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "files": job.get("files", [])
    }


@app.get("/api/progress/{job_id}")
async def stream_progress(job_id: str):
    """Stream progress updates via SSE."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        job = jobs[job_id]
        last_progress = -1

        while True:
            if job["progress"] != last_progress or job["status"] in ["complete", "error"]:
                last_progress = job["progress"]
                data = json.dumps({
                    "status": job["status"],
                    "progress": job["progress"],
                    "message": job["message"],
                    "files": job.get("files", [])
                })
                yield f"data: {data}\n\n"

                if job["status"] in ["complete", "error"]:
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a generated audio file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    file_path = job["output_dir"] / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine MIME type based on extension
    if filename.lower().endswith('.m4b'):
        media_type = "audio/x-m4b"
    else:
        media_type = "audio/mpeg"

    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=filename
    )


@app.get("/api/download-all/{job_id}")
async def download_all(job_id: str):
    """Download all MP3 files as a ZIP."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")

    # Create ZIP file
    zip_path = job["job_dir"] / "audiobook.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in job["files"]:
            file_path = job["output_dir"] / filename
            if file_path.exists():
                zf.write(file_path, filename)

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename="audiobook.zip"
    )


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    try:
        shutil.rmtree(job["job_dir"])
    except Exception:
        pass

    del jobs[job_id]
    return {"status": "cleaned"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
