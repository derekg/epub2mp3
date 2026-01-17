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

from converter import convert_epub_to_mp3, parse_epub, BUILTIN_VOICES

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


app = FastAPI(title="EPUB to MP3 Converter", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text()


@app.get("/api/voices")
async def get_voices():
    """Return list of available voices."""
    return {"voices": BUILTIN_VOICES}


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

    # Return chapter list with word counts
    chapters = []
    total_words = 0
    for i, (title, text) in enumerate(book.chapters):
        word_count = len(text.split())
        total_words += word_count
        chapters.append({
            "index": i,
            "title": title,
            "length": len(text),
            "words": word_count,
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
    custom_voice: UploadFile = File(None),
    epub_file: UploadFile = File(None),  # For backwards compatibility
):
    """Start EPUB to MP3 conversion."""
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

    # Run conversion in background
    asyncio.create_task(run_conversion(
        job_id, epub_path, voice_to_use, per_chapter, chapter_indices, skip_existing, announce_chapters
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
    """Download a generated MP3 file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    file_path = job["output_dir"] / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        str(file_path),
        media_type="audio/mpeg",
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
