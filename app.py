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
from fastapi.staticfiles import StaticFiles
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

# Persistent output directory — survives server restarts
OUTPUT_DIR = Path.home() / "epub2mp3_output"
JOBS_MANIFEST = OUTPUT_DIR / "jobs.json"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# Load TTS model at startup
if is_tts_available():
    load_model()

# ---------------------------------------------------------------------------
# Cleanup constants
# ---------------------------------------------------------------------------
CLEANUP_INTERVAL_SECONDS = 10 * 60          # Run cleanup loop every 10 minutes
JOB_TTL_SECONDS = 60 * 60                   # Remove completed/error jobs after 1 hour
ORPHAN_TTL_SECONDS = 2 * 60 * 60           # Remove orphaned temp dirs after 2 hours
EPUB2MP3_TMP_PREFIX = "epub2mp3_"           # Prefix used by tempfile.mkdtemp calls

# MLX runs on a single Metal GPU command queue and is not thread-safe for
# concurrent inference. Serialize all TTS work behind a semaphore so multiple
# simultaneous jobs queue up rather than racing on the GPU.
TTS_GPU_SEMAPHORE = asyncio.Semaphore(1)


def _cleanup_orphaned_tmp_dirs(max_age_seconds: int = ORPHAN_TTL_SECONDS) -> int:
    """Delete orphaned /tmp/epub2mp3_* directories older than *max_age_seconds*.

    Skips directories that belong to in-progress jobs so long-running
    conversions (e.g. 10-hour audiobooks) are not interrupted.

    Returns the number of directories removed.
    """
    # Collect dirs actively in use by processing jobs
    active_dirs: set[str] = set()
    for job in jobs.values():
        if job.get("status") == "processing":
            jd = job.get("job_dir")
            if jd:
                active_dirs.add(str(jd))

    tmp = Path(tempfile.gettempdir())
    now = time.time()
    removed = 0
    for entry in tmp.iterdir():
        if not entry.name.startswith(EPUB2MP3_TMP_PREFIX):
            continue
        if str(entry) in active_dirs:
            continue  # Never delete an active job's working directory
        try:
            age = now - entry.stat().st_mtime
            if age > max_age_seconds:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except Exception:
            pass
    return removed


def _cleanup_stale_jobs(max_age_seconds: int = JOB_TTL_SECONDS) -> int:
    """Remove jobs whose status is complete/error/cancelled and whose
    completion time is older than *max_age_seconds*.

    Returns the number of jobs removed.
    """
    now = time.time()
    to_delete = []
    for job_id, job in jobs.items():
        if job.get("status") not in ("complete", "error", "cancelled"):
            continue
        completed_at = job.get("completed_at")
        if completed_at is None:
            continue
        if now - completed_at > max_age_seconds:
            to_delete.append(job_id)

    for job_id in to_delete:
        job = jobs.pop(job_id)
        job_dir = job.get("job_dir")
        if job_dir:
            shutil.rmtree(job_dir, ignore_errors=True)

    return len(to_delete)


async def cleanup_loop():
    """Background task: periodically clean up old jobs and orphaned temp dirs."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            jobs_removed = _cleanup_stale_jobs()
            dirs_removed = _cleanup_orphaned_tmp_dirs()
            if jobs_removed or dirs_removed:
                print(
                    f"[cleanup] Cleaned up {jobs_removed} completed jobs, "
                    f"{dirs_removed} orphaned temp dirs"
                )
        except Exception as exc:
            print(f"[cleanup] Error during periodic cleanup: {exc}")


def _save_jobs_manifest():
    """Write all completed/failed/cancelled jobs to ~/epub2mp3_output/jobs.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for job_id, job in jobs.items():
        if job.get("status") not in ("complete", "error", "cancelled"):
            continue
        output_dir = job.get("output_dir")
        if output_dir is None:
            continue
        # Check whether cover art was saved alongside the audio files
        has_cover = (Path(str(output_dir)) / "cover.jpg").exists()
        entries.append({
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress", 100),
            "message": job.get("message", ""),
            "files": job.get("files", []),
            "completed_at": job.get("completed_at"),
            "output_dir": str(output_dir),
            "title": job.get("title", ""),
            "author": job.get("author", ""),
            "checkpoint_dir": str(job["checkpoint_dir"]) if job.get("checkpoint_dir") else None,
            "can_resume": job.get("can_resume", False),
            "settings": job.get("settings", {}),
            "has_cover": has_cover,
            "summary": job.get("summary"),
        })
    JOBS_MANIFEST.write_text(json.dumps(entries, indent=2))


def _load_persisted_jobs():
    """Restore completed/failed/cancelled jobs from ~/epub2mp3_output/jobs.json on startup."""
    if not JOBS_MANIFEST.exists():
        return
    try:
        data = json.loads(JOBS_MANIFEST.read_text())
        loaded = 0
        for entry in data:
            job_id = entry.get("job_id")
            if not job_id or job_id in jobs:
                continue
            output_dir = Path(entry["output_dir"])
            # For error/cancelled jobs output_dir may not exist — that's fine
            if not output_dir.exists() and entry.get("status") == "complete":
                continue  # Files gone for completed jobs, skip
            checkpoint_dir_str = entry.get("checkpoint_dir")
            checkpoint_dir = Path(checkpoint_dir_str) if checkpoint_dir_str else None
            jobs[job_id] = {
                "status": entry["status"],
                "progress": entry.get("progress", 0),
                "message": entry.get("message", ""),
                "files": entry.get("files", []),
                "completed_at": entry.get("completed_at"),
                "output_dir": output_dir,
                "job_dir": None,  # No temp dir for persisted jobs
                "title": entry.get("title", ""),
                "author": entry.get("author", ""),
                "checkpoint_dir": checkpoint_dir,
                "can_resume": entry.get("can_resume", False),
                "settings": entry.get("settings", {}),
                "summary": entry.get("summary"),
            }
            loaded += 1
        if loaded:
            print(f"[persist] Restored {loaded} jobs from disk")
    except Exception as e:
        print(f"[persist] Failed to load jobs manifest: {e}")


app = FastAPI(title="Inkvoice")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Run one-time stale dir cleanup and start the background cleanup loop."""
    # Ensure persistent directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Restore persisted completed jobs
    _load_persisted_jobs()

    # One-time cleanup of crash leftovers
    dirs_removed = _cleanup_orphaned_tmp_dirs(max_age_seconds=ORPHAN_TTL_SECONDS)
    if dirs_removed:
        print(f"[cleanup] Startup: removed {dirs_removed} stale epub2mp3 temp dirs")

    # Launch background cleanup loop
    asyncio.create_task(cleanup_loop())
    print("[cleanup] Background cleanup task started (interval: every 10 min, job TTL: 1 h)")


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
        "tts_engine": "Kokoro TTS (local)",
        "text_processing_modes": [
            {"value": "none", "label": "None", "description": "No text processing"},
            {"value": "clean", "label": "Clean", "description": "Remove footnotes, artifacts"},
            {"value": "speed", "label": "Speed Read", "description": "~30% summary"},
            {"value": "summary", "label": "Summary", "description": "~10% brief summary"},
        ],
    }


@app.get("/api/jobs")
async def list_jobs():
    """Return all completed and in-progress jobs."""
    result = []
    for job_id, job in jobs.items():
        if job.get("status") not in ("complete", "error", "processing"):
            continue
        result.append({
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress", 0),
            "files": job.get("files", []),
            "completed_at": job.get("completed_at"),
            "title": job.get("title", ""),
            "author": job.get("author", ""),
            "message": job.get("message", ""),
            "settings": job.get("settings", {}),
            "summary": job.get("summary"),
        })
    # Sort: processing jobs first, then by completed_at descending
    def sort_key(j):
        if j["status"] == "processing":
            return (0, 0)
        return (1, -(j.get("completed_at") or 0))
    result.sort(key=sort_key)
    return {"jobs": result}


@app.get("/api/stats")
async def get_stats():
    """Return aggregate statistics about current jobs and disk usage."""
    active_statuses = {"processing"}
    completed_statuses = {"complete", "error", "cancelled"}

    active_jobs = 0
    completed_jobs = 0
    total_bytes = 0

    for job in jobs.values():
        status = job.get("status")
        if status in active_statuses:
            active_jobs += 1
        elif status in completed_statuses:
            completed_jobs += 1

        job_dir = job.get("job_dir")
        if job_dir and Path(job_dir).exists():
            for f in Path(job_dir).rglob("*"):
                if f.is_file():
                    try:
                        total_bytes += f.stat().st_size
                    except Exception:
                        pass

    return {
        "active_jobs": active_jobs,
        "completed_jobs": completed_jobs,
        "total_disk_usage_mb": round(total_bytes / (1024 * 1024), 2),
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
    speed: float = Form(1.0),  # Playback speed multiplier
    bitrate: int = Form(192),  # MP3 bitrate: 64, 128, 192
    epub_file: UploadFile = File(None),  # For backwards compatibility
):
    """Start EPUB to audio conversion (MP3 or M4B) using Kokoro TTS."""
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
    book_title = ""
    book_author = ""
    if upload_id and upload_id in uploads:
        upload = uploads[upload_id]
        epub_path = upload["epub_path"]
        # Grab title/author from already-parsed book metadata
        book_obj = upload.get("book")
        if book_obj:
            book_title = getattr(book_obj, "title", "") or ""
            book_author = getattr(book_obj, "author", "") or ""
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

    # Create checkpoint directory and preserve a copy of the EPUB for resume
    checkpoint_dir = CHECKPOINTS_DIR / job_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epub_checkpoint_copy = checkpoint_dir / "input.epub"
    shutil.copy2(epub_path, epub_checkpoint_copy)

    # Initialize job state with detailed progress tracking
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting...",
        "output_dir": job_dir / "output",
        "job_dir": job_dir,
        "files": [],
        "completed_at": None,
        "checkpoint_dir": checkpoint_dir,
        "can_resume": False,
        "title": book_title,
        "author": book_author,
        # Conversion settings (preserved in manifest for library display)
        "settings": {
            "voice": voice,
            "output_format": output_format,
            "bitrate": bitrate,
            "speed": speed,
            "text_processing": text_processing,
        },
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

    # Validate speed — clamp to supported range
    valid_speeds = {0.75, 1.0, 1.25, 1.5, 2.0}
    allowed = sorted(valid_speeds)
    # Pick the closest valid speed
    speed = min(allowed, key=lambda s: abs(s - speed))

    # Run conversion in background
    task = asyncio.create_task(run_conversion(
        job_id, epub_path, voice, per_chapter, chapter_indices, skip_existing,
        announce_chapters, output_format, text_processing, speed, bitrate, checkpoint_dir
    ))
    jobs[job_id]["task"] = task

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
    speed: float = 1.0,
    bitrate: int = 192,
    checkpoint_dir: Path = None,
):
    """Run the conversion in the background using Kokoro TTS."""
    job = jobs[job_id]

    def progress_callback(current: int, total: int, message: str, details: dict = None):
        job["progress"] = current
        job["message"] = message
        if details:
            job["details"].update(details)
            if "book_title" in details and not job.get("title"):
                job["title"] = details["book_title"]
                job["author"] = details.get("book_author", "")

    try:
        job["message"] = "Waiting for GPU..."
        async with TTS_GPU_SEMAPHORE:
            job["message"] = "Starting..."
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
                speed,
                bitrate,
                checkpoint_dir,
            )

        job["status"] = "complete"
        job["completed_at"] = time.time()
        job["files"] = [Path(f).name for f in output_files]
        job["progress"] = 100
        job["message"] = "Conversion complete!"
        job["can_resume"] = False

        # Copy output files to persistent directory so they survive restarts
        persistent_dir = OUTPUT_DIR / job_id
        persistent_dir.mkdir(parents=True, exist_ok=True)
        for filename in job["files"]:
            src = Path(job["output_dir"]) / filename
            if src.exists():
                shutil.copy2(src, persistent_dir / filename)
        # Copy cover art if present
        cover_src = Path(job["output_dir"]) / "cover.jpg"
        if cover_src.exists():
            shutil.copy2(cover_src, persistent_dir / "cover.jpg")
        job["output_dir"] = persistent_dir

        # Build completion summary
        job["summary"] = {
            "chapter_count": len(job["files"]),
            "conversion_time_seconds": int(time.time() - job["details"]["start_time"]),
            "file_sizes": {
                f: (persistent_dir / f).stat().st_size
                for f in job["files"]
                if (persistent_dir / f).exists()
            },
        }

        # Clean up temp working dir (EPUB + intermediate files) to free space
        temp_job_dir = job.get("job_dir")
        if temp_job_dir and Path(str(temp_job_dir)).exists():
            shutil.rmtree(temp_job_dir, ignore_errors=True)
        job["job_dir"] = None

        # Checkpoint no longer needed once conversion is complete
        if checkpoint_dir and Path(str(checkpoint_dir)).exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        job["checkpoint_dir"] = None

        _save_jobs_manifest()

    except asyncio.CancelledError:
        job["status"] = "cancelled"
        job["completed_at"] = time.time()
        job["message"] = "Cancelled"
        job["progress"] = 0
        job["can_resume"] = True
        # Clean up temp output directory (keep checkpoint for resume)
        try:
            shutil.rmtree(job["output_dir"], ignore_errors=True)
        except Exception:
            pass
        _save_jobs_manifest()
        raise  # Re-raise so asyncio knows the task was cancelled

    except Exception as e:
        job["status"] = "error"
        job["completed_at"] = time.time()
        stage = job.get("details", {}).get("stage", "unknown")
        progress = job.get("progress", 0)
        context = f" (during {stage}, {progress}% complete)" if stage != "unknown" else ""
        job["message"] = f"{str(e)}{context}"
        job["progress"] = 0
        job["can_resume"] = True
        # Preserve checkpoint_dir for resume — do not clean it up
        _save_jobs_manifest()


# ---------------------------------------------------------------------------
# Library endpoints
# ---------------------------------------------------------------------------

@app.get("/api/library")
async def get_library():
    """Return all completed audiobooks from persistent storage."""
    library = []

    # Load manifest entries
    manifest_entries: dict[str, dict] = {}
    if JOBS_MANIFEST.exists():
        try:
            data = json.loads(JOBS_MANIFEST.read_text())
            for entry in data:
                job_id = entry.get("job_id")
                if job_id:
                    manifest_entries[job_id] = entry
        except Exception:
            pass

    # Also include in-memory jobs that may not have been written yet
    for job_id, job in jobs.items():
        if job.get("status") == "complete" and job_id not in manifest_entries:
            manifest_entries[job_id] = {
                "job_id": job_id,
                "status": "complete",
                "title": job.get("title", ""),
                "author": job.get("author", ""),
                "files": job.get("files", []),
                "completed_at": job.get("completed_at"),
                "output_dir": str(job.get("output_dir", OUTPUT_DIR / job_id)),
                "settings": job.get("settings", {}),
            }

    for job_id, entry in manifest_entries.items():
        if entry.get("status") != "complete":
            continue

        output_dir = Path(entry.get("output_dir", str(OUTPUT_DIR / job_id)))
        if not output_dir.exists():
            continue

        # Only include files that actually exist on disk
        existing_files = []
        file_sizes = {}
        for filename in entry.get("files", []):
            fpath = output_dir / filename
            if fpath.exists():
                size = fpath.stat().st_size
                existing_files.append(filename)
                file_sizes[filename] = size

        if not existing_files:
            continue

        total_size = sum(file_sizes.values())
        has_cover = (output_dir / "cover.jpg").exists()
        cover_url = f"/api/library/{job_id}/cover" if has_cover else None

        library.append({
            "job_id": job_id,
            "title": entry.get("title", ""),
            "author": entry.get("author", ""),
            "completed_at": entry.get("completed_at"),
            "files": existing_files,
            "file_sizes": file_sizes,
            "total_size_bytes": total_size,
            "has_cover": has_cover,
            "cover_url": cover_url,
            "settings": entry.get("settings", {}),
        })

    library.sort(key=lambda x: x.get("completed_at") or 0, reverse=True)
    return {"library": library}


@app.get("/api/library/{job_id}/cover")
async def get_library_cover(job_id: str):
    """Serve the cover art for a library entry."""
    cover_path = OUTPUT_DIR / job_id / "cover.jpg"
    if not cover_path.exists():
        raise HTTPException(status_code=404, detail="Cover not found")
    return FileResponse(str(cover_path), media_type="image/jpeg")


@app.delete("/api/library/{job_id}")
async def delete_library_entry(job_id: str):
    """Delete a library entry and all its files from disk."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists() and job_id not in jobs:
        raise HTTPException(status_code=404, detail="Library entry not found")

    # Remove files from disk
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    # Remove from in-memory jobs
    if job_id in jobs:
        del jobs[job_id]

    # Remove from manifest
    if JOBS_MANIFEST.exists():
        try:
            data = json.loads(JOBS_MANIFEST.read_text())
            data = [e for e in data if e.get("job_id") != job_id]
            JOBS_MANIFEST.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    return {"deleted": True, "job_id": job_id}


@app.delete("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel an in-progress conversion job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "processing":
        raise HTTPException(status_code=400, detail=f"Job is not in progress (status: {job['status']})")

    task = job.get("task")
    if task and not task.done():
        task.cancel()

    # Status will be set to "cancelled" by run_conversion's CancelledError handler,
    # but set it optimistically here in case the task hasn't been scheduled yet.
    job["status"] = "cancelled"
    job["message"] = "Cancelled"

    return {"status": "cancelled"}


@app.post("/api/resume/{job_id}")
async def resume_job(job_id: str):
    """Resume a failed or cancelled conversion from its last checkpoint."""
    # Look up job in memory or manifest
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") not in ("error", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Job cannot be resumed (status: {job['status']})")

    if not job.get("can_resume"):
        raise HTTPException(status_code=400, detail="Job has no checkpoint data available for resume")

    checkpoint_dir = job.get("checkpoint_dir")
    if checkpoint_dir is None:
        raise HTTPException(status_code=400, detail="Checkpoint directory not recorded for this job")

    checkpoint_dir = Path(str(checkpoint_dir))
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    if not checkpoint_file.exists():
        raise HTTPException(status_code=400, detail="Checkpoint file not found")

    # Read checkpoint to get the preserved EPUB path
    try:
        cp_data = json.loads(checkpoint_file.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read checkpoint: {e}")

    # The EPUB is preserved in the checkpoint directory
    epub_checkpoint_copy = checkpoint_dir / "input.epub"
    if not epub_checkpoint_copy.exists():
        raise HTTPException(status_code=400, detail="EPUB file not found in checkpoint — cannot resume")

    # Extract original conversion settings from checkpoint
    settings = cp_data.get("settings", {})
    voice = settings.get("voice", DEFAULT_VOICE)
    output_format = settings.get("output_format", "mp3")
    bitrate = settings.get("bitrate", 192)
    speed = settings.get("speed", 1.0)
    text_processing = settings.get("text_processing", "none")

    # Create a new job that reuses the existing checkpoint directory
    new_job_id = str(uuid.uuid4())
    new_job_dir = Path(tempfile.mkdtemp(prefix=f"epub2mp3_{new_job_id}_"))

    # Count how many chapters are already done for the toast message
    chapters_done = sum(
        1 for v in cp_data.get("chapters", {}).values()
        if v.get("status") == "done"
    )

    jobs[new_job_id] = {
        "status": "processing",
        "progress": 0,
        "message": f"Resuming from chapter {chapters_done + 1}...",
        "output_dir": new_job_dir / "output",
        "job_dir": new_job_dir,
        "files": [],
        "completed_at": None,
        "checkpoint_dir": checkpoint_dir,
        "can_resume": False,
        "title": job.get("title", ""),
        "author": job.get("author", ""),
        "details": {
            "chapter_current": chapters_done,
            "chapter_total": len(cp_data.get("chapters", {})),
            "chapter_title": "",
            "stage": "initializing",
            "start_time": time.time(),
            "words_processed": 0,
            "words_total": 0,
        },
    }

    task = asyncio.create_task(run_conversion(
        new_job_id,
        str(epub_checkpoint_copy),
        voice,
        per_chapter=True,  # checkpoint only supports per-chapter
        chapter_indices=None,
        skip_existing=False,
        announce_chapters=False,
        output_format=output_format,
        text_processing=text_processing,
        speed=speed,
        bitrate=bitrate,
        checkpoint_dir=checkpoint_dir,
    ))
    jobs[new_job_id]["task"] = task

    return {"new_job_id": new_job_id, "chapters_done": chapters_done}


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

                # Estimate remaining time based on words-per-second throughput
                words_processed = details.get("words_processed", 0)
                words_total = details.get("words_total", 0)
                estimated_remaining = None
                words_per_sec = None
                if elapsed > 30 and words_processed > 0 and words_total > 0:
                    words_per_sec = words_processed / elapsed
                    words_remaining = words_total - words_processed
                    if words_per_sec > 0:
                        estimated_remaining = max(0, words_remaining / words_per_sec)

                data = json.dumps({
                    "status": job["status"],
                    "progress": job["progress"],
                    "message": job["message"],
                    "files": job.get("files", []),
                    "can_resume": job.get("can_resume", False),
                    "details": {
                        **details,
                        "elapsed_seconds": int(elapsed),
                        "estimated_remaining_seconds": int(estimated_remaining) if estimated_remaining is not None else None,
                        "words_per_sec": round(words_per_sec, 1) if words_per_sec is not None else None,
                    }
                })
                yield f"data: {data}\n\n"

                if job["status"] in ["complete", "error", "cancelled"]:
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

    # Create ZIP file (use persistent dir if temp dir was already cleaned up)
    zip_dir = job.get("job_dir") or job.get("output_dir") or OUTPUT_DIR / job_id
    zip_path = Path(str(zip_dir)) / "audiobook.zip"
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
