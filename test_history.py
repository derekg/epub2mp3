"""Tests for job history feature — covering the API endpoints that the
localStorage-based history in the browser depends on.

Endpoints under test
--------------------
GET  /api/download/{job_id}/{filename}    – serve a single audio file
GET  /api/download-all/{job_id}           – ZIP archive of all files
DELETE /api/cleanup/{job_id}             – remove a job and its temp files
GET  /api/status/{job_id}               – job data structure validation
"""

import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app


@pytest.fixture(autouse=True)
def clear_jobs():
    """Ensure app.jobs is empty before every test and cleaned up after."""
    app_module.jobs.clear()
    yield
    app_module.jobs.clear()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(status: str = "complete", files: list[str] | None = None) -> tuple[str, Path, Path]:
    """Create a real temp directory structure, inject a job into app.jobs, and
    return (job_id, job_dir, output_dir)."""
    import uuid

    job_id = str(uuid.uuid4())
    job_dir = Path(tempfile.mkdtemp(prefix=f"epub2mp3_test_{job_id}_"))
    output_dir = job_dir / "output"
    output_dir.mkdir()

    app_module.jobs[job_id] = {
        "status": status,
        "progress": 100 if status == "complete" else 50,
        "message": "Conversion complete!" if status == "complete" else "Processing…",
        "output_dir": output_dir,
        "job_dir": job_dir,
        "files": files if files is not None else [],
    }

    return job_id, job_dir, output_dir


# ===========================================================================
# 1. TestDownloadEndpoint
# ===========================================================================

class TestDownloadEndpoint:
    """Tests for GET /api/download/{job_id}/{filename}."""

    def test_download_nonexistent_job_returns_404(self, client):
        """Requesting a file for a job ID that does not exist returns 404."""
        response = client.get("/api/download/fake-job-id/chapter01.mp3")
        assert response.status_code == 404

    def test_download_nonexistent_file_returns_404(self, client):
        """Job exists but the requested filename is not in its output dir → 404."""
        job_id, _, _ = _make_job()

        response = client.get(f"/api/download/{job_id}/missing.mp3")
        assert response.status_code == 404

    def test_download_valid_file_returns_200_with_audio_mpeg(self, client):
        """A real .mp3 file present in the output dir is served with audio/mpeg."""
        job_id, _, output_dir = _make_job(files=["chapter01.mp3"])

        # Create the actual file
        mp3_path = output_dir / "chapter01.mp3"
        mp3_path.write_bytes(b"ID3" + b"\x00" * 10)  # minimal fake MP3

        response = client.get(f"/api/download/{job_id}/chapter01.mp3")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/mpeg")

    def test_download_m4b_mime_type(self, client):
        """A .m4b file is served with the audio/x-m4b content type."""
        job_id, _, output_dir = _make_job(files=["audiobook.m4b"])

        m4b_path = output_dir / "audiobook.m4b"
        m4b_path.write_bytes(b"\x00" * 16)  # fake M4B bytes

        response = client.get(f"/api/download/{job_id}/audiobook.m4b")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/x-m4b")


# ===========================================================================
# 2. TestDownloadAllEndpoint
# ===========================================================================

class TestDownloadAllEndpoint:
    """Tests for GET /api/download-all/{job_id}."""

    def test_download_all_nonexistent_job_returns_404(self, client):
        """Requesting a ZIP for an unknown job ID returns 404."""
        response = client.get("/api/download-all/nonexistent-job")
        assert response.status_code == 404

    def test_download_all_incomplete_job_returns_400(self, client):
        """Requesting a ZIP while the job is still processing returns 400."""
        job_id, _, _ = _make_job(status="processing")

        response = client.get(f"/api/download-all/{job_id}")
        assert response.status_code == 400

    def test_download_all_creates_zip(self, client):
        """A completed job with real files returns a valid ZIP archive."""
        filenames = ["chapter01.mp3", "chapter02.mp3"]
        job_id, _, output_dir = _make_job(status="complete", files=filenames)

        # Write real (stub) MP3 files
        for name in filenames:
            (output_dir / name).write_bytes(b"ID3" + b"\x00" * 10)

        response = client.get(f"/api/download-all/{job_id}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        # Verify the ZIP is structurally valid and contains the expected files
        zip_bytes = response.content
        import io
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names_in_zip = zf.namelist()

        assert set(names_in_zip) == set(filenames)


# ===========================================================================
# 3. TestCleanupEndpoint
# ===========================================================================

class TestCleanupEndpoint:
    """Tests for DELETE /api/cleanup/{job_id}."""

    def test_cleanup_nonexistent_job_returns_404(self, client):
        """Cleaning up a job that does not exist returns 404."""
        response = client.delete("/api/cleanup/does-not-exist")
        assert response.status_code == 404

    def test_cleanup_removes_job_from_dict(self, client):
        """After cleanup, the job is no longer present in app.jobs."""
        job_id, _, _ = _make_job()
        assert job_id in app_module.jobs

        response = client.delete(f"/api/cleanup/{job_id}")
        assert response.status_code == 200
        assert job_id not in app_module.jobs

    def test_cleanup_returns_status_cleaned(self, client):
        """The cleanup response body contains {"status": "cleaned"}."""
        job_id, _, _ = _make_job()

        response = client.delete(f"/api/cleanup/{job_id}")
        assert response.status_code == 200
        assert response.json() == {"status": "cleaned"}

    def test_cleanup_removes_temp_directory(self, client):
        """Cleanup deletes the job's temp directory from disk."""
        job_id, job_dir, _ = _make_job()
        assert job_dir.exists()

        client.delete(f"/api/cleanup/{job_id}")
        assert not job_dir.exists()


# ===========================================================================
# 4. TestJobDataStructure
# ===========================================================================

class TestJobDataStructure:
    """Tests for the shape of entries in app.jobs (used by /api/status)."""

    def test_job_has_expected_fields(self, client):
        """A job dict exposes all fields the status endpoint returns."""
        job_id, _, _ = _make_job()
        job = app_module.jobs[job_id]

        for field in ("status", "progress", "message", "files"):
            assert field in job, f"Missing field: {field}"

    def test_completed_job_has_files_list(self, client):
        """The 'files' value on a complete job is a list."""
        filenames = ["ch01.mp3", "ch02.mp3"]
        job_id, _, _ = _make_job(status="complete", files=filenames)

        job = app_module.jobs[job_id]
        assert isinstance(job["files"], list)
        assert job["files"] == filenames

    def test_status_endpoint_returns_correct_shape(self, client):
        """GET /api/status/{job_id} returns the four expected keys."""
        job_id, _, _ = _make_job(status="complete", files=["ch01.mp3"])

        response = client.get(f"/api/status/{job_id}")
        assert response.status_code == 200

        body = response.json()
        assert "status" in body
        assert "progress" in body
        assert "message" in body
        assert "files" in body

    def test_status_endpoint_nonexistent_job_returns_404(self, client):
        """GET /api/status/{job_id} for a missing job returns 404."""
        response = client.get("/api/status/no-such-job")
        assert response.status_code == 404

    def test_processing_job_status_value(self, client):
        """A job injected with status='processing' reports that status."""
        job_id, _, _ = _make_job(status="processing")

        response = client.get(f"/api/status/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "processing"

    def test_complete_job_status_value(self, client):
        """A job injected with status='complete' reports that status."""
        job_id, _, _ = _make_job(status="complete")

        response = client.get(f"/api/status/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
