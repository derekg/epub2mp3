"""Tests for the checkpoint/resume system."""

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_epub(tmp_path: Path) -> Path:
    """Create a tiny valid EPUB fixture for tests."""
    import zipfile

    epub_path = tmp_path / "test.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>'
            '<container version="1.0" xmlns="urn:oasis:schemas:container">'
            "<rootfiles>"
            '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles>"
            "</container>",
        )
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid" version="2.0">'
            "<metadata>"
            '<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Test Book</dc:title>'
            '<dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">Author</dc:creator>'
            "</metadata>"
            "<manifest>"
            '<item id="ch1" href="chapter1.html" media-type="application/xhtml+xml"/>'
            "</manifest>"
            "<spine>"
            '<itemref idref="ch1"/>'
            "</spine>"
            "</package>",
        )
        zf.writestr(
            "OEBPS/chapter1.html",
            "<?xml version='1.0'?><html><body><h1>Chapter One</h1>"
            "<p>" + ("This is test content. " * 60) + "</p>"
            "</body></html>",
        )
    return epub_path


# ---------------------------------------------------------------------------
# converter.py checkpoint tests (unit-level, no HTTP)
# ---------------------------------------------------------------------------

class TestCheckpointFunctions:
    """Test the _load_checkpoint and _save_checkpoint helpers."""

    def test_load_checkpoint_missing_file(self, tmp_path):
        from converter import _load_checkpoint

        result = _load_checkpoint(tmp_path)
        assert result == {}

    def test_load_checkpoint_corrupt_file(self, tmp_path):
        from converter import _load_checkpoint

        (tmp_path / "checkpoint.json").write_text("NOT JSON {{{{")
        result = _load_checkpoint(tmp_path)
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        from converter import _load_checkpoint, _save_checkpoint

        data = {
            "epub_path": "/some/book.epub",
            "settings": {"voice": "alba", "output_format": "mp3"},
            "chapters": {
                "0": {"status": "done", "filename": "001_Chapter_One.mp3"},
                "1": {"status": "pending", "filename": None},
            },
        }
        _save_checkpoint(tmp_path, data)
        loaded = _load_checkpoint(tmp_path)
        assert loaded == data

    def test_save_checkpoint_atomic(self, tmp_path):
        """tmp file should not linger after save."""
        from converter import _save_checkpoint

        _save_checkpoint(tmp_path, {"chapters": {}})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], "Temp file should be renamed/removed after save"


# ---------------------------------------------------------------------------
# convert_epub_to_mp3 checkpoint integration tests
# ---------------------------------------------------------------------------

class TestConvertWithCheckpoint:
    """Test that convert_epub_to_mp3 writes checkpoint.json correctly."""

    @pytest.fixture()
    def tmp_dirs(self, tmp_path):
        out = tmp_path / "output"
        cp = tmp_path / "checkpoint"
        out.mkdir()
        cp.mkdir()
        return out, cp

    def _fake_tts(self, text, voice, speed=1.0, chunk_callback=None):
        import numpy as np
        # Return a tiny silent audio buffer so we don't need a real TTS engine
        samples = max(1, len(text.split()) * 10)
        audio = np.zeros(samples, dtype=np.float32)
        if chunk_callback:
            chunk_callback(len(text.split()), len(text.split()))
        return audio, 24000

    def test_checkpoint_created_on_start(self, tmp_dirs, tmp_path):
        """checkpoint.json should exist after a (mocked) successful conversion."""
        out, cp = tmp_dirs
        epub = _make_epub(tmp_path)

        def fake_encode(audio, sr, path, bitrate=192):
            Path(path).write_bytes(b"MP3DATA")

        with (
            patch("converter.generate_speech", side_effect=self._fake_tts),
            patch("converter.convert_wav_to_mp3", side_effect=fake_encode),
            patch("converter.add_id3_tags"),
            patch("converter.parse_epub") as mock_parse,
        ):
            from converter import Chapter, BookMetadata

            mock_parse.return_value = BookMetadata(
                title="Test",
                author="Author",
                chapters=[
                    Chapter("Ch 1", "word " * 50, 50),
                    Chapter("Ch 2", "word " * 50, 50),
                ],
            )

            from converter import convert_epub_to_mp3
            convert_epub_to_mp3(
                str(epub), str(out), checkpoint_dir=cp
            )

        assert (cp / "checkpoint.json").exists()

    def test_checkpoint_marks_chapters_done(self, tmp_dirs, tmp_path):
        """Completed chapters should be marked 'done' in checkpoint.json."""
        out, cp = tmp_dirs
        epub = _make_epub(tmp_path)

        with (
            patch("converter.generate_speech", side_effect=self._fake_tts),
            patch("converter.convert_wav_to_mp3"),
            patch("converter.add_id3_tags"),
            patch("converter.parse_epub") as mock_parse,
        ):
            from converter import Chapter, BookMetadata

            mock_parse.return_value = BookMetadata(
                title="Test",
                author="Author",
                chapters=[
                    Chapter("Ch 1", "word " * 50, 50),
                    Chapter("Ch 2", "word " * 50, 50),
                ],
            )

            # Simulate that convert_wav_to_mp3 actually creates the output file
            def fake_encode(audio, sr, path, bitrate=192):
                Path(path).write_bytes(b"MP3DATA")

            with patch("converter.convert_wav_to_mp3", side_effect=fake_encode):
                from converter import convert_epub_to_mp3
                convert_epub_to_mp3(
                    str(epub), str(out), checkpoint_dir=cp
                )

        data = json.loads((cp / "checkpoint.json").read_text())
        assert data["chapters"]["0"]["status"] == "done"
        assert data["chapters"]["1"]["status"] == "done"

    def test_checkpoint_resume_skips_done_chapters(self, tmp_dirs, tmp_path):
        """Chapters marked 'done' in checkpoint.json should be skipped (not re-synthesised)."""
        out, cp = tmp_dirs
        epub = _make_epub(tmp_path)

        # Pre-populate checkpoint with chapter 0 already done
        cp_filename = "001_Ch_1.mp3"
        (cp / cp_filename).write_bytes(b"FAKEMP3")
        checkpoint_data = {
            "epub_path": str(epub),
            "settings": {"voice": "alba", "output_format": "mp3", "bitrate": 192, "speed": 1.0, "text_processing": "none"},
            "chapters": {
                "0": {"status": "done", "filename": cp_filename},
                "1": {"status": "pending", "filename": None},
            },
        }
        (cp / "checkpoint.json").write_text(json.dumps(checkpoint_data))

        tts_calls = []

        def fake_tts(text, voice, speed=1.0, chunk_callback=None):
            import numpy as np
            tts_calls.append(text)
            audio = np.zeros(100, dtype=np.float32)
            if chunk_callback:
                chunk_callback(10, 10)
            return audio, 24000

        def fake_encode(audio, sr, path, bitrate=192):
            Path(path).write_bytes(b"MP3DATA")

        with (
            patch("converter.generate_speech", side_effect=fake_tts),
            patch("converter.convert_wav_to_mp3", side_effect=fake_encode),
            patch("converter.add_id3_tags"),
            patch("converter.parse_epub") as mock_parse,
        ):
            from converter import Chapter, BookMetadata

            mock_parse.return_value = BookMetadata(
                title="Test",
                author="Author",
                chapters=[
                    Chapter("Ch 1", "word " * 50, 50),
                    Chapter("Ch 2", "word " * 50, 50),
                ],
            )

            from converter import convert_epub_to_mp3
            convert_epub_to_mp3(
                str(epub), str(out), checkpoint_dir=cp
            )

        # TTS should only have been called for chapter 2 (index 1), not chapter 1
        assert len(tts_calls) == 1, f"Expected 1 TTS call (chapter 2 only), got {len(tts_calls)}"


# ---------------------------------------------------------------------------
# app.py / HTTP endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with mocked TTS so tests run without a real model."""
    import numpy as np

    with (
        patch("tts.is_tts_available", return_value=True),
        patch("tts.load_model"),
        patch("app.is_tts_available", return_value=True),
    ):
        import app as app_module

        # Override directories to use tmp paths so we don't write to home dir
        _orig_output = app_module.OUTPUT_DIR
        _orig_cp = app_module.CHECKPOINTS_DIR
        _orig_manifest = app_module.JOBS_MANIFEST

        tmp = Path(tempfile.mkdtemp(prefix="epub2mp3_test_"))
        app_module.OUTPUT_DIR = tmp / "output"
        app_module.CHECKPOINTS_DIR = tmp / "checkpoints"
        app_module.JOBS_MANIFEST = tmp / "output" / "jobs.json"
        app_module.OUTPUT_DIR.mkdir(parents=True)
        app_module.CHECKPOINTS_DIR.mkdir(parents=True)

        with TestClient(app_module.app) as c:
            yield c

        # Restore
        app_module.OUTPUT_DIR = _orig_output
        app_module.CHECKPOINTS_DIR = _orig_cp
        app_module.JOBS_MANIFEST = _orig_manifest
        shutil.rmtree(tmp, ignore_errors=True)


class TestResumeEndpoint:
    """HTTP-level tests for the resume endpoint."""

    def _inject_failed_job(self, app_module, checkpoint_dir: Path, epub_path: Path, title="Test Book"):
        """Inject a pre-failed job with checkpoint into app.jobs."""
        job_id = str(uuid.uuid4())

        # Write a minimal checkpoint
        cp_data = {
            "epub_path": str(epub_path),
            "settings": {
                "voice": "alba",
                "output_format": "mp3",
                "bitrate": 192,
                "speed": 1.0,
                "text_processing": "none",
            },
            "chapters": {
                "0": {"status": "done", "filename": "001_Chapter_One.mp3"},
                "1": {"status": "pending", "filename": None},
            },
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "checkpoint.json").write_text(json.dumps(cp_data))
        (checkpoint_dir / "001_Chapter_One.mp3").write_bytes(b"FAKEMP3")
        # The resume endpoint needs input.epub in the checkpoint dir
        shutil.copy2(epub_path, checkpoint_dir / "input.epub")

        app_module.jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "message": "Simulated failure",
            "output_dir": Path(tempfile.mkdtemp()),
            "job_dir": None,
            "files": [],
            "completed_at": 1234567890,
            "checkpoint_dir": checkpoint_dir,
            "can_resume": True,
            "title": title,
            "author": "Author",
            "details": {},
        }
        return job_id

    def test_resume_returns_404_for_unknown_job(self, client):
        resp = client.post("/api/resume/nonexistent-job-id")
        assert resp.status_code == 404

    def test_resume_returns_400_for_processing_job(self, client):
        import app as app_module

        job_id = str(uuid.uuid4())
        app_module.jobs[job_id] = {
            "status": "processing",
            "progress": 50,
            "message": "Working…",
            "output_dir": Path(tempfile.mkdtemp()),
            "job_dir": None,
            "files": [],
            "completed_at": None,
            "checkpoint_dir": None,
            "can_resume": False,
            "title": "X",
            "author": "Y",
            "details": {},
        }
        resp = client.post(f"/api/resume/{job_id}")
        assert resp.status_code == 400

    def test_resume_creates_new_job(self, client, tmp_path):
        import app as app_module

        epub = _make_epub(tmp_path)
        cp_dir = app_module.CHECKPOINTS_DIR / str(uuid.uuid4())
        job_id = self._inject_failed_job(app_module, cp_dir, epub)

        with (
            patch("app.run_conversion", return_value=None) as mock_run,
            patch("asyncio.create_task") as mock_task,
        ):
            mock_task.return_value = MagicMock()
            resp = client.post(f"/api/resume/{job_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert "new_job_id" in body
        new_id = body["new_job_id"]
        assert new_id != job_id
        assert new_id in app_module.jobs

    def test_resume_new_job_has_correct_status(self, client, tmp_path):
        import app as app_module

        epub = _make_epub(tmp_path)
        cp_dir = app_module.CHECKPOINTS_DIR / str(uuid.uuid4())
        job_id = self._inject_failed_job(app_module, cp_dir, epub)

        with (
            patch("asyncio.create_task") as mock_task,
            patch("app.run_conversion", return_value=None),
        ):
            mock_task.return_value = MagicMock()
            resp = client.post(f"/api/resume/{job_id}")

        new_id = resp.json()["new_job_id"]
        new_job = app_module.jobs[new_id]
        assert new_job["status"] == "processing"
        assert new_job["checkpoint_dir"] == cp_dir

    def test_error_job_sets_can_resume(self, tmp_path):
        """run_conversion should set can_resume=True when the job errors."""
        import asyncio
        import app as app_module

        epub = _make_epub(tmp_path)
        cp_dir = app_module.CHECKPOINTS_DIR / str(uuid.uuid4())
        cp_dir.mkdir(parents=True, exist_ok=True)
        job_id = str(uuid.uuid4())
        out_dir = Path(tempfile.mkdtemp())

        app_module.jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting…",
            "output_dir": out_dir / "output",
            "job_dir": out_dir,
            "files": [],
            "completed_at": None,
            "checkpoint_dir": cp_dir,
            "can_resume": False,
            "title": "Test",
            "author": "Author",
            "details": {
                "chapter_current": 0,
                "chapter_total": 0,
                "chapter_title": "",
                "stage": "initializing",
                "start_time": 0,
                "words_processed": 0,
                "words_total": 0,
            },
        }

        # Make convert_epub_to_mp3 raise an exception
        with patch("app.convert_epub_to_mp3", side_effect=RuntimeError("boom")):
            asyncio.get_event_loop().run_until_complete(
                app_module.run_conversion(
                    job_id, str(epub), "alba", True,
                    checkpoint_dir=cp_dir,
                )
            )

        job = app_module.jobs[job_id]
        assert job["status"] == "error"
        assert job["can_resume"] is True


# ---------------------------------------------------------------------------
# Checkpoint JSON persistence tests
# ---------------------------------------------------------------------------

class TestManifestPersistence:
    """Verify checkpoint_dir and can_resume round-trip through the manifest."""

    def test_save_and_load_can_resume(self, tmp_path):
        import app as app_module

        _orig_output = app_module.OUTPUT_DIR
        _orig_manifest = app_module.JOBS_MANIFEST
        app_module.OUTPUT_DIR = tmp_path / "out"
        app_module.JOBS_MANIFEST = tmp_path / "out" / "jobs.json"
        app_module.OUTPUT_DIR.mkdir(parents=True)

        job_id = str(uuid.uuid4())
        out_dir = app_module.OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True)
        cp_dir = tmp_path / "cp" / job_id
        cp_dir.mkdir(parents=True)

        app_module.jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "message": "oops",
            "output_dir": out_dir,
            "job_dir": None,
            "files": [],
            "completed_at": 1234567890.0,
            "checkpoint_dir": cp_dir,
            "can_resume": True,
            "title": "Book",
            "author": "Auth",
            "details": {},
        }

        app_module._save_jobs_manifest()

        # Clear in-memory jobs and reload from manifest
        saved_jobs = dict(app_module.jobs)
        app_module.jobs.clear()
        app_module._load_persisted_jobs()

        loaded = app_module.jobs.get(job_id)
        assert loaded is not None, "Job should be restored from manifest"
        assert loaded["can_resume"] is True
        assert loaded["checkpoint_dir"] == cp_dir

        # Restore
        app_module.jobs.clear()
        app_module.jobs.update(saved_jobs)
        app_module.OUTPUT_DIR = _orig_output
        app_module.JOBS_MANIFEST = _orig_manifest
