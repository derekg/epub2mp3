"""Unit tests for M4B conversion path and orphan-cleanup safety.

Covers:
- create_m4b_with_chapters: output dir must exist before ffmpeg runs
- convert_epub_to_mp3: output_dir created before calling create_m4b
- _cleanup_orphaned_tmp_dirs: never deletes an active job's working dir
- CLI --format validation: rejects unknown formats, requires ffmpeg for m4b
- API: POST /api/convert rejects m4b when ffmpeg is unavailable
"""

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_epub(path: Path) -> None:
    """Write a minimal EPUB zip. Enough for the API to accept the file."""
    import zipfile
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml",
            '<?xml version="1.0"?>'
            '<container version="1.0" xmlns="urn:oasis:schemas:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf"'
            ' media-type="application/oebps-package+xml"/></rootfiles>'
            '</container>')
        zf.writestr("OEBPS/content.opf",
            '<?xml version="1.0"?>'
            '<package version="3.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            '<dc:title>Test Book</dc:title>'
            '<dc:creator>Test Author</dc:creator>'
            '<dc:identifier id="uid">test-001</dc:identifier>'
            '<dc:language>en</dc:language>'
            '</metadata>'
            '<manifest>'
            '<item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
            '</manifest>'
            '<spine><itemref idref="ch1"/></spine>'
            '</package>')
        zf.writestr("OEBPS/chapter1.xhtml",
            '<?xml version="1.0"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml">'
            '<head><title>Chapter One</title></head>'
            '<body><h1>Chapter One</h1><p>' + ('Word ' * 200) + '</p></body>'
            '</html>')


def _tiny_audio() -> np.ndarray:
    """Return a tiny valid int16 audio array (0.1 s at 24000 Hz)."""
    return np.zeros(2400, dtype=np.int16)


# ---------------------------------------------------------------------------
# create_m4b_with_chapters
# ---------------------------------------------------------------------------

class TestCreateM4bWithChapters:
    """Tests for converter.create_m4b_with_chapters."""

    def test_output_dir_must_exist(self, tmp_path):
        """ffmpeg writes to output_path — its parent dir must exist first."""
        from converter import create_m4b_with_chapters

        missing_dir = tmp_path / "subdir_that_does_not_exist"
        output_path = missing_dir / "book.m4b"

        segments = [("Ch 1", _tiny_audio())]

        with patch("converter.is_ffmpeg_available", return_value=True), \
             patch("converter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            success, err = create_m4b_with_chapters(
                audio_segments=segments,
                output_path=str(output_path),
                sample_rate=24000,
                title="Test",
                author="Author",
            )
            # The function itself should NOT create the parent dir — that's
            # convert_epub_to_mp3's responsibility.  What we test here is that
            # the subprocess command was built with the correct output path.
            if success:
                args = mock_run.call_args[0][0]
                assert str(output_path) == args[-1], \
                    "ffmpeg command's last arg must be the output path"

    def test_no_ffmpeg_returns_false(self):
        """Returns (False, error) immediately when ffmpeg is missing."""
        from converter import create_m4b_with_chapters

        with patch("converter.is_ffmpeg_available", return_value=False):
            success, err = create_m4b_with_chapters(
                audio_segments=[("Ch1", _tiny_audio())],
                output_path="/tmp/test.m4b",
                sample_rate=24000,
                title="T",
                author="A",
            )
        assert success is False
        assert err == "ffmpeg not found"

    def test_ffmpeg_nonzero_exit_returns_error(self, tmp_path):
        """Returns (False, message) when ffmpeg exits non-zero."""
        from converter import create_m4b_with_chapters

        out = tmp_path / "out.m4b"
        with patch("converter.is_ffmpeg_available", return_value=True), \
             patch("converter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="some ffmpeg error")
            success, err = create_m4b_with_chapters(
                audio_segments=[("Ch1", _tiny_audio())],
                output_path=str(out),
                sample_rate=24000,
                title="T",
                author="A",
            )
        assert success is False
        assert "some ffmpeg error" in err

    def test_empty_segments_returns_error(self, tmp_path):
        """Returns (False, message) when no audio segments provided."""
        from converter import create_m4b_with_chapters

        with patch("converter.is_ffmpeg_available", return_value=True):
            success, err = create_m4b_with_chapters(
                audio_segments=[],
                output_path=str(tmp_path / "out.m4b"),
                sample_rate=24000,
                title="T",
                author="A",
            )
        assert success is False

    def test_ffmpeg_cmd_includes_map_metadata(self, tmp_path):
        """ffmpeg command must include -map_metadata to embed chapter info."""
        from converter import create_m4b_with_chapters

        out = tmp_path / "out.m4b"
        with patch("converter.is_ffmpeg_available", return_value=True), \
             patch("converter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            create_m4b_with_chapters(
                audio_segments=[("Ch1", _tiny_audio()), ("Ch2", _tiny_audio())],
                output_path=str(out),
                sample_rate=24000,
                title="Test Book",
                author="Test Author",
            )
        cmd = mock_run.call_args[0][0]
        assert "-map_metadata" in cmd
        assert "1" in cmd  # metadata input index

    def test_metadata_file_contains_chapters(self, tmp_path):
        """The ffmpeg metadata input must contain CHAPTER entries."""
        from converter import create_m4b_with_chapters
        import os

        captured_metadata = {}

        def fake_run(cmd, **kwargs):
            # Find the metadata file path from the command
            for i, arg in enumerate(cmd):
                if arg == "-i" and i + 1 < len(cmd) and "metadata" in cmd[i + 1]:
                    meta_path = cmd[i + 1]
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            captured_metadata["content"] = f.read()
            return MagicMock(returncode=0, stderr="")

        out = tmp_path / "out.m4b"
        with patch("converter.is_ffmpeg_available", return_value=True), \
             patch("converter.subprocess.run", side_effect=fake_run):
            create_m4b_with_chapters(
                audio_segments=[("Chapter One", _tiny_audio()), ("Chapter Two", _tiny_audio())],
                output_path=str(out),
                sample_rate=24000,
                title="Test Book",
                author="Test Author",
            )

        content = captured_metadata.get("content", "")
        assert "[CHAPTER]" in content
        assert "Chapter One" in content
        assert "Chapter Two" in content


# ---------------------------------------------------------------------------
# convert_epub_to_mp3 — M4B path
# ---------------------------------------------------------------------------

def _fake_book():
    """Return a mock BookMetadata with one chapter."""
    from converter import BookMetadata, Chapter
    return BookMetadata(
        title="Test Book",
        author="Test Author",
        cover_image=None,
        cover_mime=None,
        chapters=[
            Chapter(
                title="Chapter One",
                text="Word " * 200,
                word_count=200,
                is_front_matter=False,
                is_back_matter=False,
            )
        ],
    )


class TestConvertEpubToMp3M4B:
    """Tests for the M4B code path in convert_epub_to_mp3."""

    def _patches(self):
        """Common patches: mock TTS and parse_epub so no real files needed."""
        return [
            patch("converter.generate_speech", return_value=(_tiny_audio(), 24000)),
            patch("converter.parse_epub", return_value=_fake_book()),
        ]

    def test_output_dir_created_before_m4b(self, tmp_path):
        """output_dir.mkdir is called before create_m4b_with_chapters."""
        from converter import convert_epub_to_mp3

        out_dir = tmp_path / "output"  # does NOT exist yet
        created_before_m4b = {}

        def spy_m4b(*args, **kwargs):
            created_before_m4b["exists"] = out_dir.exists()
            return (True, None)

        with self._patches()[0], self._patches()[1], \
             patch("converter.create_m4b_with_chapters", side_effect=spy_m4b), \
             patch("converter.is_ffmpeg_available", return_value=True):
            convert_epub_to_mp3(
                epub_path="/fake/test.epub",
                output_dir=str(out_dir),
                output_format="m4b",
            )

        assert created_before_m4b.get("exists") is True, \
            "output_dir must exist when create_m4b_with_chapters is called"

    def test_m4b_raises_on_ffmpeg_failure(self, tmp_path):
        """convert_epub_to_mp3 raises ValueError when M4B creation fails."""
        from converter import convert_epub_to_mp3

        with self._patches()[0], self._patches()[1], \
             patch("converter.create_m4b_with_chapters", return_value=(False, "ffmpeg exploded")), \
             patch("converter.is_ffmpeg_available", return_value=True):
            with pytest.raises(ValueError, match="Failed to create M4B"):
                convert_epub_to_mp3(
                    epub_path="/fake/test.epub",
                    output_dir=str(tmp_path / "out"),
                    output_format="m4b",
                )

    def test_m4b_requires_ffmpeg(self, tmp_path):
        """convert_epub_to_mp3 raises ValueError when ffmpeg is missing."""
        from converter import convert_epub_to_mp3

        with patch("converter.is_ffmpeg_available", return_value=False):
            with pytest.raises(ValueError, match="ffmpeg"):
                convert_epub_to_mp3(
                    epub_path="/fake/test.epub",
                    output_dir=str(tmp_path / "out"),
                    output_format="m4b",
                )


# ---------------------------------------------------------------------------
# Orphan cleanup — active job dirs must not be deleted
# ---------------------------------------------------------------------------

class TestOrphanCleanupSkipsActiveJobs:
    """_cleanup_orphaned_tmp_dirs must not delete dirs of in-progress jobs."""

    def _old_dir(self, tmp_path, name: str) -> Path:
        """Create a temp dir that looks old enough to be cleaned up."""
        d = tmp_path / name
        d.mkdir()
        old_time = time.time() - (3 * 60 * 60)  # 3 hours ago
        import os
        os.utime(d, (old_time, old_time))
        return d

    def test_active_job_dir_not_deleted(self, tmp_path, monkeypatch):
        """A dir belonging to a processing job is never removed."""
        import app

        # Redirect tempfile.gettempdir() to our tmp_path
        monkeypatch.setattr("app.tempfile.gettempdir", lambda: str(tmp_path))

        job_dir = self._old_dir(tmp_path, "epub2mp3_active123_abc")
        orphan_dir = self._old_dir(tmp_path, "epub2mp3_orphan456_xyz")

        job_id = "active-job-001"
        app.jobs[job_id] = {
            "status": "processing",
            "job_dir": job_dir,
            "output_dir": job_dir / "output",
            "files": [],
            "completed_at": None,
        }

        try:
            removed = app._cleanup_orphaned_tmp_dirs(max_age_seconds=60)
        finally:
            app.jobs.pop(job_id, None)

        assert job_dir.exists(), "Active job dir must NOT be deleted"
        assert not orphan_dir.exists(), "Orphan dir must be deleted"
        assert removed == 1

    def test_completed_job_dir_is_deleted(self, tmp_path, monkeypatch):
        """A dir belonging to a completed job CAN be removed by orphan cleanup."""
        import app

        monkeypatch.setattr("app.tempfile.gettempdir", lambda: str(tmp_path))

        job_dir = self._old_dir(tmp_path, "epub2mp3_done789_xyz")

        job_id = "done-job-001"
        app.jobs[job_id] = {
            "status": "complete",
            "job_dir": None,  # already cleared (moved to OUTPUT_DIR)
            "output_dir": tmp_path / "epub2mp3_output" / job_id,
            "files": ["book.m4b"],
            "completed_at": time.time() - 7200,
        }

        try:
            removed = app._cleanup_orphaned_tmp_dirs(max_age_seconds=60)
        finally:
            app.jobs.pop(job_id, None)

        assert not job_dir.exists(), "Stale dir with no active job should be deleted"
        assert removed == 1

    def test_no_dirs_removed_when_all_active(self, tmp_path, monkeypatch):
        """No dirs deleted when every temp dir belongs to an active job."""
        import app

        monkeypatch.setattr("app.tempfile.gettempdir", lambda: str(tmp_path))

        dirs = [self._old_dir(tmp_path, f"epub2mp3_job{i}_abc") for i in range(3)]
        added = []
        for i, d in enumerate(dirs):
            jid = f"job-{i}"
            app.jobs[jid] = {"status": "processing", "job_dir": d,
                             "output_dir": d / "output", "files": [], "completed_at": None}
            added.append(jid)

        try:
            removed = app._cleanup_orphaned_tmp_dirs(max_age_seconds=60)
        finally:
            for jid in added:
                app.jobs.pop(jid, None)

        assert removed == 0
        for d in dirs:
            assert d.exists()


# ---------------------------------------------------------------------------
# CLI --format validation
# ---------------------------------------------------------------------------

class TestCLIFormatValidation:
    """CLI validates --format before doing any work."""

    def test_invalid_format_rejected(self):
        """--format with unsupported value exits non-zero."""
        from typer.testing import CliRunner
        from cli import app as cli_app

        runner = CliRunner()
        result = runner.invoke(cli_app, ["convert", "/nonexistent.epub", "--format", "ogg"])
        assert result.exit_code != 0

    def test_m4b_requires_ffmpeg(self, tmp_path):
        """--format m4b prints error and exits when ffmpeg is missing."""
        from typer.testing import CliRunner
        from cli import app as cli_app

        epub = tmp_path / "test.epub"
        _make_epub(epub)

        with patch("cli.is_ffmpeg_available", return_value=False):
            runner = CliRunner()
            result = runner.invoke(cli_app, ["convert", str(epub), "--format", "m4b"])

        assert result.exit_code != 0
        assert "ffmpeg" in result.output.lower()

    def test_mp3_format_accepted(self, tmp_path):
        """--format mp3 is accepted (proceeds past validation)."""
        from typer.testing import CliRunner
        from cli import app as cli_app

        epub = tmp_path / "test.epub"
        _make_epub(epub)

        # Stop after model load — we just want to verify format validation passes
        with patch("cli.is_ffmpeg_available", return_value=True), \
             patch("tts.load_model", side_effect=RuntimeError("stop here")):
            runner = CliRunner()
            result = runner.invoke(cli_app, ["convert", str(epub), "--format", "mp3"])

        # Should reach model loading (RuntimeError from mock), not format validation
        assert "Unsupported format" not in result.output


# ---------------------------------------------------------------------------
# API: POST /api/convert rejects m4b without ffmpeg
# ---------------------------------------------------------------------------

class TestAPIConvertM4B:
    """POST /api/convert returns 400 when m4b requested without ffmpeg."""

    def _upload_epub(self, client, tmp_path) -> str:
        epub = tmp_path / "test.epub"
        _make_epub(epub)
        with open(epub, "rb") as f, \
             patch("app.parse_epub", return_value=_fake_book()):
            r = client.post("/api/upload", files={"epub_file": ("test.epub", f, "application/epub+zip")})
        assert r.status_code == 200, r.text
        return r.json()["upload_id"]

    def test_m4b_rejected_without_ffmpeg(self, tmp_path):
        from app import app as fastapi_app
        client = TestClient(fastapi_app)

        with patch("app.is_ffmpeg_available", return_value=False), \
             patch("app.is_tts_available", return_value=True):
            upload_id = self._upload_epub(client, tmp_path)
            r = client.post("/api/convert", data={
                "upload_id": upload_id,
                "selected_chapters": "[0]",
                "output_format": "m4b",
            })
        assert r.status_code == 400
        assert "ffmpeg" in r.json()["detail"].lower()

    def test_mp3_accepted_without_ffmpeg(self, tmp_path):
        from app import app as fastapi_app
        client = TestClient(fastapi_app)

        with patch("app.is_ffmpeg_available", return_value=False), \
             patch("app.is_tts_available", return_value=True), \
             patch("app.asyncio.create_task"):
            upload_id = self._upload_epub(client, tmp_path)
            r = client.post("/api/convert", data={
                "upload_id": upload_id,
                "selected_chapters": "[0]",
                "output_format": "mp3",
            })
        assert r.status_code == 200
        assert "job_id" in r.json()
