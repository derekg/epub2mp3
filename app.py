"""FastAPI server for EPUB to MP3 conversion."""

import asyncio
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pocket_tts import TTSModel

from converter import convert_epub_to_mp3, BUILTIN_VOICES

app = FastAPI(title="EPUB to MP3 Converter")

# Global model instance (loaded once at startup)
tts_model: TTSModel = None

# Store for conversion jobs
jobs: dict = {}


@app.on_event("startup")
async def startup_event():
    """Load the TTS model at startup."""
    global tts_model
    print("Loading TTS model...")
    tts_model = TTSModel.load_model()
    print("TTS model loaded!")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text()


@app.get("/api/voices")
async def get_voices():
    """Return list of available voices."""
    return {"voices": BUILTIN_VOICES}


@app.post("/api/convert")
async def start_conversion(
    epub_file: UploadFile = File(...),
    voice: str = Form("alba"),
    per_chapter: bool = Form(True),
    custom_voice: UploadFile = File(None)
):
    """Start EPUB to MP3 conversion."""
    job_id = str(uuid.uuid4())

    # Create temp directory for this job
    job_dir = Path(tempfile.mkdtemp(prefix=f"epub2mp3_{job_id}_"))

    # Save uploaded EPUB
    epub_path = job_dir / "input.epub"
    with open(epub_path, "wb") as f:
        shutil.copyfileobj(epub_file.file, f)

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
        job_id, str(epub_path), voice_to_use, per_chapter
    ))

    return {"job_id": job_id}


async def run_conversion(job_id: str, epub_path: str, voice: str, per_chapter: bool):
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
            progress_callback
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
