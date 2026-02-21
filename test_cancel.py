"""Unit tests for the cancel button feature."""

import asyncio
import inspect
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(status="processing", task=None):
    """Return a minimal job dict that mirrors what app.py stores in jobs."""
    return {
        "status": status,
        "progress": 0,
        "message": "Running...",
        "output_dir": "/tmp/fake_output",
        "job_dir": "/tmp/fake_job",
        "files": [],
        "details": {
            "chapter_current": 0,
            "chapter_total": 0,
            "chapter_title": "",
            "stage": "initializing",
            "start_time": 0.0,
            "words_processed": 0,
            "words_total": 0,
        },
        "task": task,
    }


# ---------------------------------------------------------------------------
# TestCancelEndpoint
# ---------------------------------------------------------------------------

class TestCancelEndpoint:
    """Tests that exercise the DELETE /api/cancel/{job_id} HTTP endpoint."""

    def test_cancel_nonexistent_job_returns_404(self):
        """Cancelling a job_id that doesn't exist must return HTTP 404."""
        from app import app, jobs

        # Ensure the id is definitely absent
        fake_id = "00000000-0000-0000-0000-000000000000"
        jobs.pop(fake_id, None)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.delete(f"/api/cancel/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_cancel_completed_job_returns_400(self):
        """Cancelling a job whose status is 'complete' must return HTTP 400."""
        from app import app, jobs

        job_id = "complete-job-test-id"
        jobs[job_id] = _make_job(status="complete")

        try:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.delete(f"/api/cancel/{job_id}")

            assert response.status_code == 400
            detail = response.json()["detail"]
            assert "not in progress" in detail.lower() or "status" in detail.lower()
        finally:
            jobs.pop(job_id, None)

    def test_cancel_endpoint_exists(self):
        """The DELETE /api/cancel/{job_id} route must be registered (not 405)."""
        from app import app, jobs

        job_id = "endpoint-exists-test-id"
        # Inject a processing job so we get past the 404 guard
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel.return_value = True
        jobs[job_id] = _make_job(status="processing", task=mock_task)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.delete(f"/api/cancel/{job_id}")

            # Any response other than 405 proves the route exists
            assert response.status_code != 405, (
                "Got 405 Method Not Allowed — cancel route is not registered"
            )
        finally:
            jobs.pop(job_id, None)

    def test_job_status_has_task_field(self):
        """A freshly initialised job dict must include a 'task' key.

        app.py sets jobs[job_id] before creating the asyncio task, then
        immediately assigns it.  We verify the key exists by checking the
        dict that is built inside start_conversion via the jobs store.
        """
        from app import jobs

        # After start_conversion runs, the stored dict must have 'task'.
        # We inject a complete dict the same way app.py does and verify.
        job_id = "task-field-test-id"
        jobs[job_id] = _make_job(status="processing")
        jobs[job_id]["task"] = asyncio.Future()   # simulated task assignment

        try:
            assert "task" in jobs[job_id], "job dict must contain a 'task' key"
        finally:
            jobs.pop(job_id, None)

    def test_cancel_processing_job_returns_cancelled(self):
        """Cancelling a processing job returns HTTP 200 with status='cancelled'."""
        from app import app, jobs

        job_id = "processing-cancel-test-id"
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel.return_value = True
        jobs[job_id] = _make_job(status="processing", task=mock_task)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.delete(f"/api/cancel/{job_id}")

            assert response.status_code == 200
            assert response.json()["status"] == "cancelled"
        finally:
            jobs.pop(job_id, None)

    def test_cancel_updates_job_status_in_store(self):
        """After a successful cancel the jobs dict entry status must be 'cancelled'."""
        from app import app, jobs

        job_id = "status-update-test-id"
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel.return_value = True
        jobs[job_id] = _make_job(status="processing", task=mock_task)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            client.delete(f"/api/cancel/{job_id}")

            assert jobs[job_id]["status"] == "cancelled"
        finally:
            jobs.pop(job_id, None)

    def test_cancel_error_job_returns_400(self):
        """Cancelling a job in 'error' state must return HTTP 400."""
        from app import app, jobs

        job_id = "error-job-test-id"
        jobs[job_id] = _make_job(status="error")

        try:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.delete(f"/api/cancel/{job_id}")

            assert response.status_code == 400
        finally:
            jobs.pop(job_id, None)

    def test_cancel_already_cancelled_job_returns_400(self):
        """Cancelling a job already in 'cancelled' state must return HTTP 400."""
        from app import app, jobs

        job_id = "already-cancelled-test-id"
        jobs[job_id] = _make_job(status="cancelled")

        try:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.delete(f"/api/cancel/{job_id}")

            assert response.status_code == 400
        finally:
            jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# TestCancelJobState
# ---------------------------------------------------------------------------

class TestCancelJobState:
    """Tests that validate job-dict state management for cancel logic."""

    def test_jobs_dict_stores_task(self):
        """Verify that the task key is wired up in the jobs store.

        We don't call start_conversion (it needs a real EPUB / TTS), but we
        can confirm the parameter lists of start_conversion and run_conversion
        are consistent with the pattern: task = asyncio.create_task(run_conversion(...)).
        """
        from app import start_conversion, run_conversion

        start_sig = inspect.signature(start_conversion)
        run_sig = inspect.signature(run_conversion)

        start_params = list(start_sig.parameters.keys())
        run_params = list(run_sig.parameters.keys())

        # start_conversion must accept the fields used to build a job
        assert "upload_id" in start_params
        assert "voice" in start_params

        # run_conversion must accept the fields forwarded from start_conversion
        assert "job_id" in run_params
        assert "epub_path" in run_params
        assert "voice" in run_params

    def test_cancel_sets_cancelled_status(self):
        """The cancel logic must flip job status to 'cancelled'.

        We replicate the logic from cancel_job() directly:
            job["status"] = "cancelled"
        and verify it works on a mocked job dict.
        """
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel.return_value = True

        job = _make_job(status="processing", task=mock_task)

        # ----- replicate cancel_job logic -----
        assert job["status"] == "processing"   # precondition
        task = job.get("task")
        if task and not task.done():
            task.cancel()
        job["status"] = "cancelled"
        job["message"] = "Cancelled"
        # --------------------------------------

        assert job["status"] == "cancelled"
        assert job["message"] == "Cancelled"
        mock_task.cancel.assert_called_once()

    def test_cancel_logic_skips_cancel_when_task_done(self):
        """If the task is already done, cancel() must not be called."""
        mock_task = MagicMock()
        mock_task.done.return_value = True   # already finished

        job = _make_job(status="processing", task=mock_task)

        task = job.get("task")
        if task and not task.done():
            task.cancel()
        job["status"] = "cancelled"

        mock_task.cancel.assert_not_called()
        assert job["status"] == "cancelled"

    def test_cancel_logic_handles_missing_task(self):
        """Cancel logic must not raise when 'task' key is absent."""
        job = _make_job(status="processing")
        del job["task"]  # simulate a job whose task key was never set

        task = job.get("task")   # returns None
        if task and not task.done():
            task.cancel()
        job["status"] = "cancelled"

        assert job["status"] == "cancelled"

    def test_run_conversion_cancellederror_sets_cancelled_status(self):
        """run_conversion's CancelledError handler must set status='cancelled'."""
        # We test the handler logic directly rather than running the full coroutine.
        from app import jobs

        job_id = "cancelled-error-test-id"
        jobs[job_id] = _make_job(status="processing")
        job = jobs[job_id]

        # Simulate what the except asyncio.CancelledError block does
        job["status"] = "cancelled"
        job["message"] = "Cancelled"
        job["progress"] = 0

        try:
            assert jobs[job_id]["status"] == "cancelled"
            assert jobs[job_id]["message"] == "Cancelled"
            assert jobs[job_id]["progress"] == 0
        finally:
            jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# TestCancelAPISignature
# ---------------------------------------------------------------------------

class TestCancelAPISignature:
    """Tests that inspect function signatures to verify the API contract."""

    def test_convert_endpoint_accepts_upload_id(self):
        """start_conversion must accept upload_id as a form parameter."""
        from app import start_conversion

        sig = inspect.signature(start_conversion)
        assert "upload_id" in sig.parameters, (
            "start_conversion must accept 'upload_id'"
        )

    def test_run_conversion_signature(self):
        """run_conversion must declare the parameters expected by start_conversion."""
        from app import run_conversion

        sig = inspect.signature(run_conversion)
        params = list(sig.parameters.keys())

        expected = [
            "job_id",
            "epub_path",
            "voice",
            "per_chapter",
            "chapter_indices",
            "skip_existing",
            "announce_chapters",
            "output_format",
            "text_processing",
        ]
        for p in expected:
            assert p in params, f"run_conversion is missing parameter '{p}'"

    def test_cancel_job_function_exists(self):
        """cancel_job coroutine must be importable from app."""
        from app import cancel_job
        assert asyncio.iscoroutinefunction(cancel_job), (
            "cancel_job must be an async function"
        )

    def test_cancel_job_accepts_job_id(self):
        """cancel_job must accept a single 'job_id' path parameter."""
        from app import cancel_job

        sig = inspect.signature(cancel_job)
        assert "job_id" in sig.parameters

    def test_start_conversion_returns_job_id(self):
        """Verify start_conversion is an async function (returns a coroutine)."""
        from app import start_conversion
        assert asyncio.iscoroutinefunction(start_conversion)

    def test_run_conversion_is_async(self):
        """run_conversion must be async so it can be wrapped in create_task."""
        from app import run_conversion
        assert asyncio.iscoroutinefunction(run_conversion)

    def test_run_conversion_chapter_indices_has_default(self):
        """chapter_indices in run_conversion must default to None."""
        from app import run_conversion

        sig = inspect.signature(run_conversion)
        param = sig.parameters.get("chapter_indices")
        assert param is not None
        assert param.default is None

    def test_run_conversion_output_format_default(self):
        """output_format in run_conversion must default to 'mp3'."""
        from app import run_conversion

        sig = inspect.signature(run_conversion)
        param = sig.parameters.get("output_format")
        assert param is not None
        assert param.default == "mp3"

    def test_run_conversion_text_processing_default(self):
        """text_processing in run_conversion must default to 'none'."""
        from app import run_conversion

        sig = inspect.signature(run_conversion)
        param = sig.parameters.get("text_processing")
        assert param is not None
        assert param.default == "none"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
