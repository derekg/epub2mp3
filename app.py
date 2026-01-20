"""FastAPI server for EPUB to MP3 conversion using Gemini TTS."""

import asyncio
import json
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

# Load environment variables from .env file (in same directory as this script)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import lameenc
import numpy as np

from converter import convert_epub_to_mp3, parse_epub, BUILTIN_VOICES, is_ffmpeg_available
from text_processor import ProcessingMode, is_gemini_available
from tts import (
    get_voice_list, generate_preview, is_tts_available, load_model,
    VOICES, DEFAULT_VOICE, SAMPLE_RATE
)

# Store for uploaded EPUBs awaiting conversion
uploads: dict = {}

# Store for conversion jobs
jobs: dict = {}

# Load TTS model at startup
if is_tts_available():
    load_model()

app = FastAPI(title="Inkvoice")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text()


@app.get("/api/voices")
async def get_voices():
    """Return list of available Gemini TTS voices with metadata."""
    return {
        "voices": get_voice_list(),
        "default": DEFAULT_VOICE,
    }


# Cache for voice previews
voice_preview_cache: dict[str, bytes] = {}


@app.get("/api/voice-preview/{voice}")
async def get_voice_preview(voice: str):
    """Generate a short audio preview for a voice using Gemini TTS."""
    from fastapi.responses import Response

    if voice not in VOICES:
        raise HTTPException(status_code=404, detail="Voice not found")

    # Check cache first
    if voice in voice_preview_cache:
        return Response(
            content=voice_preview_cache[voice],
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400"}
        )

    try:
        # Generate preview using Gemini TTS
        audio_np, sample_rate = await asyncio.to_thread(generate_preview, voice)

        # Convert to MP3
        if audio_np.dtype != np.int16:
            audio_np = (audio_np * 32767).astype(np.int16)

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(128)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_out_sample_rate(sample_rate)
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
        "gemini_available": is_gemini_available(),
        "tts_available": is_tts_available(),
        "tts_engine": "Pocket TTS (local)",
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


@app.get("/api/cover/{upload_id}")
async def get_cover(upload_id: str):
    """Return the cover image for an uploaded EPUB."""
    from fastapi.responses import Response

    if upload_id not in uploads:
        raise HTTPException(status_code=404, detail="Upload not found")

    upload = uploads[upload_id]
    book = upload["book"]

    if book.cover_image is None:
        raise HTTPException(status_code=404, detail="No cover image")

    # Determine media type from cover data
    cover_data = book.cover_image
    media_type = "image/jpeg"  # Default
    if cover_data[:8] == b'\x89PNG\r\n\x1a\n':
        media_type = "image/png"
    elif cover_data[:2] == b'\xff\xd8':
        media_type = "image/jpeg"
    elif cover_data[:4] == b'GIF8':
        media_type = "image/gif"

    return Response(content=cover_data, media_type=media_type)


@app.post("/api/convert")
async def start_conversion(
    upload_id: str = Form(None),
    selected_chapters: str = Form(None),  # JSON array of indices
    voice: str = Form(DEFAULT_VOICE),
    per_chapter: bool = Form(True),
    skip_existing: bool = Form(False),  # Resume support
    announce_chapters: bool = Form(False),  # Chapter announcements
    output_format: str = Form("mp3"),  # mp3 or m4b
    text_processing: str = Form("none"),  # none, clean, speed, summary
    epub_file: UploadFile = File(None),  # For backwards compatibility
):
    """Start EPUB to audio conversion (MP3 or M4B) using Gemini TTS."""
    # Check TTS availability
    if not is_tts_available():
        raise HTTPException(status_code=503, detail="Gemini TTS not configured. Set GEMINI_API_KEY environment variable.")

    # Validate output format
    output_format = output_format.lower()
    if output_format not in ("mp3", "m4b"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {output_format}")

    if output_format == "m4b" and not is_ffmpeg_available():
        raise HTTPException(status_code=400, detail="M4B format requires ffmpeg (not installed on server)")

    # Validate voice
    if voice not in VOICES:
        voice = DEFAULT_VOICE

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

    # Initialize job state with detailed progress tracking
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting...",
        "output_dir": job_dir / "output",
        "job_dir": job_dir,
        "files": [],
        # Detailed progress info
        "details": {
            "chapter_current": 0,
            "chapter_total": 0,
            "chapter_title": "",
            "stage": "initializing",  # initializing, parsing, text_processing, tts, encoding, finalizing
            "start_time": time.time(),
            "words_processed": 0,
            "words_total": 0,
        }
    }

    # Validate text processing mode
    if text_processing not in ("none", "clean", "speed", "summary"):
        text_processing = "none"

    # Run conversion in background
    asyncio.create_task(run_conversion(
        job_id, epub_path, voice, per_chapter, chapter_indices, skip_existing, announce_chapters, output_format, text_processing
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
    """Run the conversion in the background using Gemini TTS."""
    job = jobs[job_id]

    def progress_callback(current: int, total: int, message: str, details: dict = None):
        job["progress"] = current
        job["message"] = message
        if details:
            job["details"].update(details)

    try:
        output_files = await asyncio.to_thread(
            convert_epub_to_mp3,
            epub_path,
            str(job["output_dir"]),
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
        last_message = ""

        while True:
            # Update on progress change, message change, or status change
            if (job["progress"] != last_progress or
                job["message"] != last_message or
                job["status"] in ["complete", "error"]):

                last_progress = job["progress"]
                last_message = job["message"]

                # Calculate elapsed and estimated remaining time
                details = job.get("details", {})
                start_time = details.get("start_time", time.time())
                elapsed = time.time() - start_time

                # Estimate remaining time based on progress
                estimated_remaining = None
                if job["progress"] > 5:  # Only estimate after some progress
                    total_estimated = elapsed / (job["progress"] / 100)
                    estimated_remaining = max(0, total_estimated - elapsed)

                data = json.dumps({
                    "status": job["status"],
                    "progress": job["progress"],
                    "message": job["message"],
                    "files": job.get("files", []),
                    "details": {
                        **details,
                        "elapsed_seconds": int(elapsed),
                        "estimated_remaining_seconds": int(estimated_remaining) if estimated_remaining else None,
                    }
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
