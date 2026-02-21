"""Unit tests for the auto cleanup feature in app.py.

Covers:
- _cleanup_stale_jobs()
- _cleanup_orphaned_tmp_dirs()
- /api/stats endpoint
- Job completed_at field lifecycle
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(status="processing", completed_at=None, job_dir=None):
    """Return a minimal job dict."""
    return {
        "status": status,
        "progress": 0,
        "message": "",
        "output_dir": None,
        "job_dir": job_dir,
        "files": [],
        "completed_at": completed_at,
        "details": {
            "chapter_current": 0,
            "chapter_total": 0,
            "chapter_title": "",
            "stage": "initializing",
            "start_time": time.time(),
            "words_processed": 0,
            "words_total": 0,
        },
    }


# ---------------------------------------------------------------------------
# TestCleanupStaleJobs
# ---------------------------------------------------------------------------

class TestCleanupStaleJobs:
    """Tests for _cleanup_stale_jobs()."""

    def setup_method(self):
        """Clear app.jobs before each test to prevent cross-test pollution."""
        import app as app_module
        app_module.jobs.clear()

    def teardown_method(self):
        """Clear app.jobs after each test."""
        import app as app_module
        app_module.jobs.clear()

    def test_cleanup_skips_active_jobs(self):
        """Jobs with status='processing' and completed_at=None must NOT be removed."""
        import app as app_module
        from app import _cleanup_stale_jobs

        app_module.jobs["active-1"] = _make_job(status="processing", completed_at=None)

        removed = _cleanup_stale_jobs(max_age_seconds=0)  # 0 = remove everything eligible

        assert "active-1" in app_module.jobs, "Active job should not have been removed"
        assert removed == 0

    def test_cleanup_removes_old_complete_jobs(self):
        """A complete job finished 2 hours ago must be removed when TTL is 1 hour."""
        import app as app_module
        from app import _cleanup_stale_jobs

        two_hours_ago = time.time() - 7200
        app_module.jobs["old-job"] = _make_job(status="complete", completed_at=two_hours_ago)

        removed = _cleanup_stale_jobs(max_age_seconds=3600)

        assert "old-job" not in app_module.jobs, "Old complete job should have been removed"
        assert removed == 1

    def test_cleanup_keeps_recent_complete_jobs(self):
        """A complete job finished 30 minutes ago must NOT be removed when TTL is 1 hour."""
        import app as app_module
        from app import _cleanup_stale_jobs

        thirty_min_ago = time.time() - 1800
        app_module.jobs["recent-job"] = _make_job(status="complete", completed_at=thirty_min_ago)

        removed = _cleanup_stale_jobs(max_age_seconds=3600)

        assert "recent-job" in app_module.jobs, "Recent complete job should NOT have been removed"
        assert removed == 0

    def test_cleanup_returns_count(self):
        """_cleanup_stale_jobs must return an integer count of removed jobs."""
        import app as app_module
        from app import _cleanup_stale_jobs

        old_ts = time.time() - 9000
        app_module.jobs["job-a"] = _make_job(status="complete", completed_at=old_ts)
        app_module.jobs["job-b"] = _make_job(status="error", completed_at=old_ts)
        app_module.jobs["job-c"] = _make_job(status="processing", completed_at=None)

        removed = _cleanup_stale_jobs(max_age_seconds=3600)

        assert isinstance(removed, int)
        assert removed == 2  # job-a and job-b removed; job-c skipped

    def test_cleanup_removes_temp_dir(self):
        """Cleanup must delete the job's job_dir from disk."""
        import app as app_module
        from app import _cleanup_stale_jobs

        # Create a real temporary directory
        job_dir = tempfile.mkdtemp(prefix="epub2mp3_test_")
        assert Path(job_dir).exists()

        old_ts = time.time() - 7200
        app_module.jobs["dir-job"] = _make_job(
            status="complete", completed_at=old_ts, job_dir=job_dir
        )

        _cleanup_stale_jobs(max_age_seconds=3600)

        assert not Path(job_dir).exists(), "job_dir should have been deleted from disk"

    def test_cleanup_handles_missing_dir_gracefully(self):
        """Cleanup must not raise when a job's job_dir no longer exists."""
        import app as app_module
        from app import _cleanup_stale_jobs

        nonexistent = "/tmp/epub2mp3_this_dir_does_not_exist_12345"
        old_ts = time.time() - 7200
        app_module.jobs["ghost-job"] = _make_job(
            status="complete", completed_at=old_ts, job_dir=nonexistent
        )

        # Should not raise any exception
        try:
            removed = _cleanup_stale_jobs(max_age_seconds=3600)
        except Exception as exc:
            pytest.fail(f"_cleanup_stale_jobs raised unexpectedly: {exc}")

        assert removed == 1
        assert "ghost-job" not in app_module.jobs


# ---------------------------------------------------------------------------
# TestCleanupOrphanedDirs
# ---------------------------------------------------------------------------

class TestCleanupOrphanedDirs:
    """Tests for _cleanup_orphaned_tmp_dirs()."""

    def test_cleanup_removes_old_epub2mp3_dirs(self):
        """Dirs named 'epub2mp3_*' with mtime older than max_age_seconds must be removed."""
        from app import _cleanup_orphaned_tmp_dirs

        tmp_root = tempfile.gettempdir()
        old_dir = tempfile.mkdtemp(prefix="epub2mp3_test_old_", dir=tmp_root)

        # Set mtime to 4 hours ago
        four_hours_ago = time.time() - 14400
        os.utime(old_dir, (four_hours_ago, four_hours_ago))

        try:
            removed = _cleanup_orphaned_tmp_dirs(max_age_seconds=7200)
            assert not Path(old_dir).exists(), "Old epub2mp3_ dir should have been removed"
            assert removed >= 1
        finally:
            # Safety cleanup in case the assertion failed
            shutil.rmtree(old_dir, ignore_errors=True)

    def test_cleanup_keeps_recent_dirs(self):
        """Dirs named 'epub2mp3_*' with a fresh mtime must NOT be removed."""
        from app import _cleanup_orphaned_tmp_dirs

        tmp_root = tempfile.gettempdir()
        recent_dir = tempfile.mkdtemp(prefix="epub2mp3_test_recent_", dir=tmp_root)
        # mtime is current (just created) — no utime call needed

        try:
            removed = _cleanup_orphaned_tmp_dirs(max_age_seconds=7200)
            assert Path(recent_dir).exists(), "Recent epub2mp3_ dir should NOT have been removed"
        finally:
            shutil.rmtree(recent_dir, ignore_errors=True)

    def test_cleanup_ignores_non_epub2mp3_dirs(self):
        """Dirs without the 'epub2mp3_' prefix must never be touched."""
        from app import _cleanup_orphaned_tmp_dirs

        tmp_root = tempfile.gettempdir()
        other_dir = tempfile.mkdtemp(prefix="other_app_dir_", dir=tmp_root)

        # Set mtime to a very old time to ensure it would be eligible if checked
        very_old = time.time() - 86400  # 24 hours ago
        os.utime(other_dir, (very_old, very_old))

        try:
            _cleanup_orphaned_tmp_dirs(max_age_seconds=7200)
            assert Path(other_dir).exists(), "Non-epub2mp3_ dir should NOT have been removed"
        finally:
            shutil.rmtree(other_dir, ignore_errors=True)

    def test_cleanup_returns_count(self):
        """_cleanup_orphaned_tmp_dirs must return an integer count of removed dirs."""
        from app import _cleanup_orphaned_tmp_dirs

        tmp_root = tempfile.gettempdir()
        old_ts = time.time() - 14400  # 4 hours ago

        dir1 = tempfile.mkdtemp(prefix="epub2mp3_test_count1_", dir=tmp_root)
        dir2 = tempfile.mkdtemp(prefix="epub2mp3_test_count2_", dir=tmp_root)
        os.utime(dir1, (old_ts, old_ts))
        os.utime(dir2, (old_ts, old_ts))

        try:
            removed = _cleanup_orphaned_tmp_dirs(max_age_seconds=7200)
            assert isinstance(removed, int)
            assert removed >= 2, f"Expected at least 2 dirs removed, got {removed}"
        finally:
            shutil.rmtree(dir1, ignore_errors=True)
            shutil.rmtree(dir2, ignore_errors=True)


# ---------------------------------------------------------------------------
# TestStatsEndpoint
# ---------------------------------------------------------------------------

class TestStatsEndpoint:
    """Tests for GET /api/stats."""

    def setup_method(self):
        import app as app_module
        app_module.jobs.clear()

    def teardown_method(self):
        import app as app_module
        app_module.jobs.clear()

    @pytest.fixture
    def client(self):
        from app import app
        return TestClient(app, raise_server_exceptions=True)

    def test_stats_endpoint_exists(self, client):
        """GET /api/stats must return HTTP 200."""
        response = client.get("/api/stats")
        assert response.status_code == 200

    def test_stats_response_structure(self, client):
        """Response must contain active_jobs, completed_jobs, total_disk_usage_mb."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "active_jobs" in data, "Response missing 'active_jobs'"
        assert "completed_jobs" in data, "Response missing 'completed_jobs'"
        assert "total_disk_usage_mb" in data, "Response missing 'total_disk_usage_mb'"

    def test_stats_active_jobs_count(self, client):
        """Stats must correctly count active vs completed jobs."""
        import app as app_module

        app_module.jobs["p1"] = _make_job(status="processing")
        app_module.jobs["p2"] = _make_job(status="processing")
        app_module.jobs["c1"] = _make_job(status="complete", completed_at=time.time())

        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()

        assert data["active_jobs"] == 2, f"Expected 2 active jobs, got {data['active_jobs']}"
        assert data["completed_jobs"] == 1, f"Expected 1 completed job, got {data['completed_jobs']}"

    def test_stats_disk_usage_type(self, client):
        """total_disk_usage_mb must be a float >= 0."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()

        disk_usage = data["total_disk_usage_mb"]
        assert isinstance(disk_usage, (int, float)), (
            f"total_disk_usage_mb should be numeric, got {type(disk_usage)}"
        )
        assert disk_usage >= 0, f"total_disk_usage_mb must be >= 0, got {disk_usage}"


# ---------------------------------------------------------------------------
# TestJobCompletedAt
# ---------------------------------------------------------------------------

class TestJobCompletedAt:
    """Tests for the completed_at field on job dicts."""

    def setup_method(self):
        import app as app_module
        app_module.jobs.clear()

    def teardown_method(self):
        import app as app_module
        app_module.jobs.clear()

    def test_job_initializes_with_completed_at_none(self):
        """A newly-created job dict must have completed_at=None."""
        import app as app_module

        # Simulate what start_conversion does when initializing a job
        job_id = "init-test"
        app_module.jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting...",
            "output_dir": None,
            "job_dir": None,
            "files": [],
            "completed_at": None,
            "details": {},
        }

        assert app_module.jobs[job_id]["completed_at"] is None, (
            "Newly created job should have completed_at=None"
        )

    def test_completed_at_set_on_success(self):
        """run_conversion must set completed_at when conversion succeeds."""
        import asyncio
        import app as app_module
        from app import run_conversion

        job_id = "success-test"
        job_dir = Path(tempfile.mkdtemp(prefix="epub2mp3_test_success_"))
        output_dir = job_dir / "output"
        output_dir.mkdir()

        app_module.jobs[job_id] = _make_job(status="processing")
        app_module.jobs[job_id]["output_dir"] = output_dir
        app_module.jobs[job_id]["job_dir"] = str(job_dir)

        # Patch convert_epub_to_mp3 to return a success list without doing real work
        with patch("app.convert_epub_to_mp3", return_value=["/fake/output.mp3"]):
            asyncio.get_event_loop().run_until_complete(
                run_conversion(
                    job_id=job_id,
                    epub_path="/fake/input.epub",
                    voice="Kore",
                    per_chapter=True,
                )
            )

        job = app_module.jobs[job_id]
        assert job["status"] == "complete"
        assert job["completed_at"] is not None, "completed_at should be set after success"
        assert isinstance(job["completed_at"], float), (
            "completed_at should be a float (Unix timestamp)"
        )
        assert job["completed_at"] <= time.time()

        # Cleanup
        shutil.rmtree(job_dir, ignore_errors=True)

    def test_completed_at_set_on_error(self):
        """run_conversion must set completed_at even when conversion raises an exception."""
        import asyncio
        import app as app_module
        from app import run_conversion

        job_id = "error-test"
        job_dir = Path(tempfile.mkdtemp(prefix="epub2mp3_test_error_"))
        output_dir = job_dir / "output"
        output_dir.mkdir()

        app_module.jobs[job_id] = _make_job(status="processing")
        app_module.jobs[job_id]["output_dir"] = output_dir
        app_module.jobs[job_id]["job_dir"] = str(job_dir)

        # Patch convert_epub_to_mp3 to raise an exception
        with patch("app.convert_epub_to_mp3", side_effect=RuntimeError("conversion failed")):
            asyncio.get_event_loop().run_until_complete(
                run_conversion(
                    job_id=job_id,
                    epub_path="/fake/input.epub",
                    voice="Kore",
                    per_chapter=True,
                )
            )

        job = app_module.jobs[job_id]
        assert job["status"] == "error"
        assert job["completed_at"] is not None, "completed_at should be set even on error"
        assert isinstance(job["completed_at"], float), (
            "completed_at should be a float (Unix timestamp)"
        )
        assert job["completed_at"] <= time.time()

        # Cleanup
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
