"""Unit tests for the multi-job queue feature."""

import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app, jobs


@pytest.fixture(autouse=True)
def clear_jobs():
    """Clear the jobs dict before and after each test to avoid state leakage."""
    jobs.clear()
    yield
    jobs.clear()


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(status="processing", progress=0, message="Working...", files=None, error_message=None):
    """Return a minimal job dict that the app endpoints expect."""
    job_dir = Path(tempfile.mkdtemp(prefix="test_epub2mp3_"))
    output_dir = job_dir / "output"
    output_dir.mkdir()
    return {
        "status": status,
        "progress": progress,
        "message": error_message if error_message else message,
        "output_dir": output_dir,
        "job_dir": job_dir,
        "files": files if files is not None else [],
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


# ===========================================================================
# 1. TestMultipleJobsSupport
# ===========================================================================

class TestMultipleJobsSupport:
    """Verify the jobs dict correctly supports multiple independent jobs."""

    def test_jobs_dict_supports_multiple_entries(self):
        """Inject 3 jobs with different IDs and verify all 3 are accessible."""
        ids = ["job-alpha", "job-beta", "job-gamma"]
        for jid in ids:
            jobs[jid] = _make_job(status="processing", progress=10 * (ids.index(jid) + 1))

        assert len(jobs) == 3
        for jid in ids:
            assert jid in jobs

        assert jobs["job-alpha"]["progress"] == 10
        assert jobs["job-beta"]["progress"] == 20
        assert jobs["job-gamma"]["progress"] == 30

    def test_status_endpoint_per_job(self, client):
        """Two jobs with different statuses: /api/status/{id} returns the correct one."""
        jobs["id-processing"] = _make_job(status="processing", progress=40, message="Running")
        jobs["id-complete"] = _make_job(status="complete", progress=100, message="Done", files=["a.mp3"])

        resp_p = client.get("/api/status/id-processing")
        assert resp_p.status_code == 200
        data_p = resp_p.json()
        assert data_p["status"] == "processing"
        assert data_p["progress"] == 40

        resp_c = client.get("/api/status/id-complete")
        assert resp_c.status_code == 200
        data_c = resp_c.json()
        assert data_c["status"] == "complete"
        assert data_c["progress"] == 100
        assert "a.mp3" in data_c["files"]

    def test_progress_endpoint_per_job(self, client):
        """Unknown job → 404; known job → 200."""
        jobs["known-job"] = _make_job(status="complete", progress=100, message="Done")

        resp_unknown = client.get("/api/progress/does-not-exist")
        assert resp_unknown.status_code == 404

        # For a complete job the SSE generator yields one event then stops,
        # so the response is 200 and consumable.
        resp_known = client.get("/api/progress/known-job")
        assert resp_known.status_code == 200

    def test_jobs_are_independent(self, client):
        """Mutating one job's status must not affect any other job."""
        jobs["job-one"] = _make_job(status="processing", progress=20)
        jobs["job-two"] = _make_job(status="processing", progress=50)

        # Mutate job-one in the store
        jobs["job-one"]["status"] = "complete"
        jobs["job-one"]["progress"] = 100

        # job-two should be unchanged
        assert jobs["job-two"]["status"] == "processing"
        assert jobs["job-two"]["progress"] == 50

        resp = client.get("/api/status/job-two")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processing"
        assert data["progress"] == 50


# ===========================================================================
# 2. TestStatusEndpoint
# ===========================================================================

class TestStatusEndpoint:
    """Tests for GET /api/status/{job_id}."""

    def test_status_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/status/nonexistent-id")
        assert resp.status_code == 404

    def test_status_processing_job(self, client):
        """A processing job returns status=processing and correct progress."""
        jobs["proc-job"] = _make_job(status="processing", progress=50, message="Halfway there")

        resp = client.get("/api/status/proc-job")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processing"
        assert data["progress"] == 50
        assert data["message"] == "Halfway there"

    def test_status_complete_job(self, client):
        """A complete job returns status=complete and the files list."""
        files_list = ["chapter1.mp3", "chapter2.mp3", "chapter3.mp3"]
        jobs["done-job"] = _make_job(status="complete", progress=100, message="Done!", files=files_list)

        resp = client.get("/api/status/done-job")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"
        assert data["progress"] == 100
        assert data["files"] == files_list

    def test_status_error_job(self, client):
        """An error job returns status=error and an error message."""
        jobs["err-job"] = _make_job(
            status="error",
            progress=0,
            error_message="Something went very wrong",
        )

        resp = client.get("/api/status/err-job")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "Something went very wrong" in data["message"]

    def test_status_response_structure(self, client):
        """Response always contains status, progress, message, and files fields."""
        jobs["struct-job"] = _make_job(status="processing", progress=10)

        resp = client.get("/api/status/struct-job")
        assert resp.status_code == 200
        data = resp.json()

        for field in ("status", "progress", "message", "files"):
            assert field in data, f"Missing field: {field}"

        assert isinstance(data["status"], str)
        assert isinstance(data["progress"], int)
        assert isinstance(data["message"], str)
        assert isinstance(data["files"], list)


# ===========================================================================
# 3. TestProgressSSE
# ===========================================================================

class TestProgressSSE:
    """Tests for GET /api/progress/{job_id} (Server-Sent Events)."""

    def test_progress_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/progress/ghost-job")
        assert resp.status_code == 404

    def test_progress_endpoint_exists_and_returns_event_stream(self, client):
        """A known job returns 200 with text/event-stream content type."""
        jobs["sse-job"] = _make_job(status="complete", progress=100, message="Done")

        resp = client.get("/api/progress/sse-job")
        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" in content_type

    def test_progress_sse_payload_is_valid_json(self, client):
        """The first SSE data line for a complete job contains valid JSON."""
        jobs["sse-json-job"] = _make_job(
            status="complete",
            progress=100,
            message="All done",
            files=["out.mp3"],
        )

        resp = client.get("/api/progress/sse-json-job")
        assert resp.status_code == 200

        # Parse out the first "data: ..." line from the response body
        body = resp.text
        import json
        for line in body.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                assert payload["status"] == "complete"
                assert payload["progress"] == 100
                assert "out.mp3" in payload["files"]
                break
        else:
            pytest.fail("No 'data: ...' line found in SSE response")


# ===========================================================================
# 4. TestDownloadAllEndpoint
# ===========================================================================

class TestDownloadAllEndpoint:
    """Tests for GET /api/download-all/{job_id}."""

    def test_download_all_requires_complete_status(self, client):
        """A job with status=processing should return 400."""
        jobs["not-done-job"] = _make_job(status="processing", progress=40)

        resp = client.get("/api/download-all/not-done-job")
        assert resp.status_code == 400

    def test_download_all_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/download-all/ghost-job")
        assert resp.status_code == 404

    def test_download_all_creates_valid_zip(self, client):
        """A complete job with existing output files produces a valid ZIP response."""
        job = _make_job(status="complete", progress=100, message="Done")

        # Create a real dummy file in the output directory
        dummy_file = job["output_dir"] / "chapter1.mp3"
        dummy_file.write_bytes(b"FAKE_MP3_DATA")
        job["files"] = ["chapter1.mp3"]

        jobs["zip-job"] = job

        resp = client.get("/api/download-all/zip-job")
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "application/zip"

        # Verify the response body is a valid ZIP archive
        import io
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            assert "chapter1.mp3" in names

    def test_download_all_zip_contains_all_files(self, client):
        """All files listed in job['files'] appear in the ZIP."""
        job = _make_job(status="complete", progress=100, message="Done")

        filenames = ["chapter1.mp3", "chapter2.mp3", "chapter3.mp3"]
        for name in filenames:
            (job["output_dir"] / name).write_bytes(b"FAKE_MP3_DATA_" + name.encode())
        job["files"] = filenames

        jobs["zip-all-job"] = job

        resp = client.get("/api/download-all/zip-all-job")
        assert resp.status_code == 200

        import io
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names_in_zip = set(zf.namelist())
            for expected in filenames:
                assert expected in names_in_zip, f"{expected} missing from ZIP"

    def test_download_all_zip_file_contents_are_correct(self, client):
        """The bytes inside each zipped file match what was written to disk."""
        job = _make_job(status="complete", progress=100, message="Done")

        content_map = {
            "chapter1.mp3": b"AUDIO_DATA_CHAPTER_ONE",
            "chapter2.mp3": b"AUDIO_DATA_CHAPTER_TWO",
        }
        for name, data in content_map.items():
            (job["output_dir"] / name).write_bytes(data)
        job["files"] = list(content_map.keys())

        jobs["zip-contents-job"] = job

        resp = client.get("/api/download-all/zip-contents-job")
        assert resp.status_code == 200

        import io
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name, expected_data in content_map.items():
                assert zf.read(name) == expected_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
