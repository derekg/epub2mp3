"""Tests for the persistent audiobook library endpoints.

Covers:
  - GET  /api/library         returns completed jobs with correct metadata
  - GET  /api/library/{id}/cover  returns 404 when no cover, 200 when present
  - DELETE /api/library/{id}  removes files and manifest entry
"""

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(tmp_path: Path, job_id: str, title: str = "Test Book",
              author: str = "Test Author", with_cover: bool = False,
              with_mp3: bool = True, settings: dict = None) -> dict:
    """Create a fake completed-job directory structure and return a manifest entry."""
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)

    files = []
    if with_mp3:
        mp3 = job_dir / "001_Chapter_One.mp3"
        mp3.write_bytes(b"\xff\xfb" + b"\x00" * 512)  # minimal fake MP3
        files.append(mp3.name)

    if with_cover:
        cover = job_dir / "cover.jpg"
        # Minimal JPEG header
        cover.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)

    entry = {
        "job_id": job_id,
        "status": "complete",
        "progress": 100,
        "message": "Conversion complete!",
        "files": files,
        "completed_at": time.time() - 3600,
        "output_dir": str(job_dir),
        "title": title,
        "author": author,
        "settings": settings or {"voice": "af_sky", "output_format": "mp3", "bitrate": 128},
        "has_cover": with_cover,
    }
    return entry


def _write_manifest(manifest_path: Path, entries: list[dict]) -> None:
    manifest_path.write_text(json.dumps(entries, indent=2))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Return a TestClient with OUTPUT_DIR and JOBS_MANIFEST redirected to tmp_path."""
    import app as app_module

    fake_output = tmp_path / "epub2mp3_output"
    fake_output.mkdir()
    fake_manifest = fake_output / "jobs.json"

    monkeypatch.setattr(app_module, "OUTPUT_DIR", fake_output)
    monkeypatch.setattr(app_module, "JOBS_MANIFEST", fake_manifest)
    # Clear in-memory jobs so tests start clean
    app_module.jobs.clear()

    with TestClient(app_module.app) as c:
        yield c, app_module, fake_output, fake_manifest


# ---------------------------------------------------------------------------
# GET /api/library
# ---------------------------------------------------------------------------

class TestGetLibrary:
    def test_empty_library(self, client):
        c, _, output_dir, manifest = client
        _write_manifest(manifest, [])
        resp = c.get("/api/library")
        assert resp.status_code == 200
        data = resp.json()
        assert "library" in data
        assert data["library"] == []

    def test_returns_completed_jobs(self, client):
        c, _, output_dir, manifest = client
        entry = _make_job(output_dir, "job-aaa", title="Great Book", author="Alice")
        _write_manifest(manifest, [entry])

        resp = c.get("/api/library")
        assert resp.status_code == 200
        lib = resp.json()["library"]
        assert len(lib) == 1
        item = lib[0]
        assert item["job_id"] == "job-aaa"
        assert item["title"] == "Great Book"
        assert item["author"] == "Alice"

    def test_metadata_fields_present(self, client):
        c, _, output_dir, manifest = client
        entry = _make_job(output_dir, "job-bbb", with_cover=True)
        _write_manifest(manifest, [entry])

        resp = c.get("/api/library")
        item = resp.json()["library"][0]
        assert "files" in item
        assert "file_sizes" in item
        assert "total_size_bytes" in item
        assert "has_cover" in item
        assert "cover_url" in item
        assert "settings" in item
        assert "completed_at" in item

    def test_has_cover_true_when_cover_exists(self, client):
        c, _, output_dir, manifest = client
        entry = _make_job(output_dir, "job-ccc", with_cover=True)
        _write_manifest(manifest, [entry])

        item = c.get("/api/library").json()["library"][0]
        assert item["has_cover"] is True
        assert item["cover_url"] == "/api/library/job-ccc/cover"

    def test_has_cover_false_when_no_cover(self, client):
        c, _, output_dir, manifest = client
        entry = _make_job(output_dir, "job-ddd", with_cover=False)
        _write_manifest(manifest, [entry])

        item = c.get("/api/library").json()["library"][0]
        assert item["has_cover"] is False
        assert item["cover_url"] is None

    def test_skips_jobs_with_missing_files(self, client):
        c, _, output_dir, manifest = client
        # Entry pointing to non-existent directory
        ghost = {
            "job_id": "ghost-job",
            "status": "complete",
            "files": ["audiobook.mp3"],
            "completed_at": time.time(),
            "output_dir": str(output_dir / "ghost-job"),
            "title": "Ghost",
            "author": "",
        }
        real = _make_job(output_dir, "real-job", title="Real Book")
        _write_manifest(manifest, [ghost, real])

        lib = c.get("/api/library").json()["library"]
        job_ids = [x["job_id"] for x in lib]
        assert "ghost-job" not in job_ids
        assert "real-job" in job_ids

    def test_skips_non_complete_statuses(self, client):
        c, _, output_dir, manifest = client
        entry = _make_job(output_dir, "job-error")
        entry["status"] = "error"
        _write_manifest(manifest, [entry])

        lib = c.get("/api/library").json()["library"]
        assert lib == []

    def test_sorted_newest_first(self, client):
        c, _, output_dir, manifest = client
        old = _make_job(output_dir, "job-old", title="Old Book")
        old["completed_at"] = time.time() - 7200
        new = _make_job(output_dir, "job-new", title="New Book")
        new["completed_at"] = time.time() - 60
        _write_manifest(manifest, [old, new])

        lib = c.get("/api/library").json()["library"]
        assert lib[0]["job_id"] == "job-new"
        assert lib[1]["job_id"] == "job-old"

    def test_file_sizes_populated(self, client):
        c, _, output_dir, manifest = client
        entry = _make_job(output_dir, "job-size")
        _write_manifest(manifest, [entry])

        item = c.get("/api/library").json()["library"][0]
        assert item["total_size_bytes"] > 0
        for fname in item["files"]:
            assert fname in item["file_sizes"]
            assert item["file_sizes"][fname] > 0

    def test_in_memory_job_included(self, client):
        """Jobs only in app.jobs (not yet in manifest) should appear in library."""
        c, app_module, output_dir, manifest = client
        _write_manifest(manifest, [])

        job_id = "mem-only-job"
        job_dir = output_dir / job_id
        job_dir.mkdir()
        mp3 = job_dir / "audiobook.mp3"
        mp3.write_bytes(b"\xff\xfb" + b"\x00" * 256)

        app_module.jobs[job_id] = {
            "status": "complete",
            "files": [mp3.name],
            "output_dir": job_dir,
            "completed_at": time.time(),
            "title": "In-Memory Book",
            "author": "Mem Author",
            "settings": {},
        }

        lib = c.get("/api/library").json()["library"]
        ids = [x["job_id"] for x in lib]
        assert job_id in ids


# ---------------------------------------------------------------------------
# GET /api/library/{job_id}/cover
# ---------------------------------------------------------------------------

class TestGetLibraryCover:
    def test_404_when_no_cover(self, client):
        c, _, output_dir, _ = client
        _make_job(output_dir, "job-nocover", with_cover=False)
        resp = c.get("/api/library/job-nocover/cover")
        assert resp.status_code == 404

    def test_404_for_nonexistent_job(self, client):
        c, _, _, _ = client
        resp = c.get("/api/library/does-not-exist/cover")
        assert resp.status_code == 404

    def test_200_with_cover(self, client):
        c, _, output_dir, _ = client
        _make_job(output_dir, "job-withcover", with_cover=True)
        resp = c.get("/api/library/job-withcover/cover")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/jpeg")
        # Response body should start with JPEG magic bytes
        assert resp.content[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# DELETE /api/library/{job_id}
# ---------------------------------------------------------------------------

class TestDeleteLibraryEntry:
    def test_delete_removes_directory(self, client):
        c, _, output_dir, manifest = client
        entry = _make_job(output_dir, "del-job")
        _write_manifest(manifest, [entry])

        job_dir = output_dir / "del-job"
        assert job_dir.exists()

        resp = c.delete("/api/library/del-job")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert not job_dir.exists()

    def test_delete_removes_from_manifest(self, client):
        c, _, output_dir, manifest = client
        entry_a = _make_job(output_dir, "keep-job", title="Keep Me")
        entry_b = _make_job(output_dir, "del-job2", title="Delete Me")
        _write_manifest(manifest, [entry_a, entry_b])

        c.delete("/api/library/del-job2")

        remaining = json.loads(manifest.read_text())
        ids = [e["job_id"] for e in remaining]
        assert "del-job2" not in ids
        assert "keep-job" in ids

    def test_delete_removes_from_memory(self, client):
        c, app_module, output_dir, manifest = client
        _make_job(output_dir, "mem-del-job")
        _write_manifest(manifest, [])

        app_module.jobs["mem-del-job"] = {
            "status": "complete",
            "files": [],
            "output_dir": output_dir / "mem-del-job",
            "completed_at": time.time(),
            "title": "Mem Job",
            "author": "",
        }

        c.delete("/api/library/mem-del-job")
        assert "mem-del-job" not in app_module.jobs

    def test_delete_nonexistent_returns_404(self, client):
        c, _, _, _ = client
        resp = c.delete("/api/library/totally-fake-id")
        assert resp.status_code == 404

    def test_delete_response_fields(self, client):
        c, _, output_dir, manifest = client
        entry = _make_job(output_dir, "resp-job")
        _write_manifest(manifest, [entry])

        resp = c.delete("/api/library/resp-job")
        data = resp.json()
        assert data["deleted"] is True
        assert data["job_id"] == "resp-job"
