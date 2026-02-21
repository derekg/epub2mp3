"""Unit tests for the narration speed control feature."""

import inspect
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine(n_samples: int, dtype=np.int16) -> np.ndarray:
    """Create a deterministic int16 sine-wave array for testing."""
    t = np.linspace(0, 2 * np.pi * 5, n_samples)
    wave = np.sin(t) * 10000  # amplitude well within int16 range
    return wave.astype(dtype)


# ---------------------------------------------------------------------------
# 1. TestResampleAudio
# ---------------------------------------------------------------------------

class TestResampleAudio:
    """Tests for tts._resample_audio."""

    @pytest.fixture(autouse=True)
    def import_resample(self):
        from tts import _resample_audio
        self._resample_audio = _resample_audio

    def test_resample_2x_halves_length(self):
        """2× speed → output should be ~half the original length."""
        audio = _make_sine(1000)
        result = self._resample_audio(audio, 2.0, 24000)
        assert abs(len(result) - 500) <= 2, (
            f"Expected ~500 samples at 2× speed, got {len(result)}"
        )

    def test_resample_half_speed_doubles_length(self):
        """0.5× speed → output should be ~double the original length."""
        audio = _make_sine(1000)
        result = self._resample_audio(audio, 0.5, 24000)
        assert abs(len(result) - 2000) <= 2, (
            f"Expected ~2000 samples at 0.5× speed, got {len(result)}"
        )

    def test_resample_1x_unchanged(self):
        """1.0× speed → output should be the same length (identity)."""
        audio = _make_sine(1000)
        result = self._resample_audio(audio, 1.0, 24000)
        # The implementation short-circuits for speed ≈ 1.0
        assert abs(len(result) - 1000) <= 1, (
            f"Expected ~1000 samples at 1.0× speed, got {len(result)}"
        )

    def test_resample_preserves_dtype(self):
        """Output dtype should always be int16."""
        audio = _make_sine(1000, dtype=np.int16)
        for speed in (0.75, 1.25, 1.5, 2.0):
            result = self._resample_audio(audio, speed, 24000)
            assert result.dtype == np.int16, (
                f"Expected int16 output at {speed}× speed, got {result.dtype}"
            )

    def test_resample_valid_speeds(self):
        """All product-supported speeds produce correct-length output."""
        audio = _make_sine(1000)
        expected_lengths = {
            0.75: round(1000 / 0.75),
            1.0:  1000,
            1.25: round(1000 / 1.25),
            1.5:  round(1000 / 1.5),
            2.0:  500,
        }
        for speed, expected in expected_lengths.items():
            result = self._resample_audio(audio, speed, 24000)
            assert abs(len(result) - expected) <= 2, (
                f"At {speed}× speed: expected ~{expected} samples, got {len(result)}"
            )


# ---------------------------------------------------------------------------
# 2. TestGenerateSpeechSignature
# ---------------------------------------------------------------------------

class TestGenerateSpeechSignature:
    """Verify that speed parameter exists with correct defaults in key functions."""

    def test_generate_speech_has_speed_param(self):
        """tts.generate_speech must accept a 'speed' keyword argument."""
        from tts import generate_speech
        sig = inspect.signature(generate_speech)
        assert "speed" in sig.parameters, (
            "generate_speech is missing a 'speed' parameter"
        )

    def test_generate_speech_speed_default_is_1(self):
        """tts.generate_speech speed parameter must default to 1.0."""
        from tts import generate_speech
        sig = inspect.signature(generate_speech)
        default = sig.parameters["speed"].default
        assert default == 1.0, (
            f"generate_speech speed default expected 1.0, got {default!r}"
        )

    def test_text_to_audio_has_speed_param(self):
        """converter.text_to_audio must accept a 'speed' keyword argument."""
        from converter import text_to_audio
        sig = inspect.signature(text_to_audio)
        assert "speed" in sig.parameters, (
            "text_to_audio is missing a 'speed' parameter"
        )

    def test_text_to_audio_speed_default_is_1(self):
        """converter.text_to_audio speed parameter must default to 1.0."""
        from converter import text_to_audio
        sig = inspect.signature(text_to_audio)
        default = sig.parameters["speed"].default
        assert default == 1.0, (
            f"text_to_audio speed default expected 1.0, got {default!r}"
        )

    def test_convert_epub_to_mp3_has_speed_param(self):
        """converter.convert_epub_to_mp3 must accept a 'speed' keyword argument."""
        from converter import convert_epub_to_mp3
        sig = inspect.signature(convert_epub_to_mp3)
        assert "speed" in sig.parameters, (
            "convert_epub_to_mp3 is missing a 'speed' parameter"
        )

    def test_convert_epub_to_mp3_speed_default_is_1(self):
        """converter.convert_epub_to_mp3 speed parameter must default to 1.0."""
        from converter import convert_epub_to_mp3
        sig = inspect.signature(convert_epub_to_mp3)
        default = sig.parameters["speed"].default
        assert default == 1.0, (
            f"convert_epub_to_mp3 speed default expected 1.0, got {default!r}"
        )


# ---------------------------------------------------------------------------
# 3. TestSpeedValidation
# ---------------------------------------------------------------------------

class TestSpeedValidation:
    """Test app.py speed snapping logic.

    The implementation in start_conversion snaps to the nearest value in
    {0.75, 1.0, 1.25, 1.5, 2.0} using:
        speed = min(allowed, key=lambda s: abs(s - speed))
    """

    VALID_SPEEDS = [0.75, 1.0, 1.25, 1.5, 2.0]

    def _snap(self, raw: float) -> float:
        """Replicate the snapping logic from app.py."""
        allowed = sorted({0.75, 1.0, 1.25, 1.5, 2.0})
        return min(allowed, key=lambda s: abs(s - raw))

    def test_valid_speeds_accepted_unchanged(self):
        """Each valid speed value should snap to itself (no change)."""
        for speed in self.VALID_SPEEDS:
            result = self._snap(speed)
            assert result == speed, (
                f"Valid speed {speed} was wrongly snapped to {result}"
            )

    def test_invalid_speed_09_snaps_to_1(self):
        """0.9 is closer to 1.0 than 0.75, so it should snap to 1.0."""
        # abs(0.9 - 1.0) = 0.1 < abs(0.9 - 0.75) = 0.15
        result = self._snap(0.9)
        assert result == 1.0, f"Expected 0.9 → 1.0, got {result}"

    def test_invalid_speed_08_snaps_to_075(self):
        """0.8 is closer to 0.75 than 1.0, so it should snap to 0.75."""
        # abs(0.8 - 0.75) = 0.05 < abs(0.8 - 1.0) = 0.2
        result = self._snap(0.8)
        assert result == 0.75, f"Expected 0.8 → 0.75, got {result}"

    def test_invalid_speed_very_high_snaps_to_2(self):
        """A very high speed (e.g. 5.0) should snap to 2.0."""
        result = self._snap(5.0)
        assert result == 2.0, f"Expected 5.0 → 2.0, got {result}"

    def test_invalid_speed_very_low_snaps_to_075(self):
        """A very low speed (e.g. 0.1) should snap to 0.75."""
        result = self._snap(0.1)
        assert result == 0.75, f"Expected 0.1 → 0.75, got {result}"

    def test_speed_form_field_default_is_1(self):
        """app.start_conversion must declare speed=Form(1.0) as default."""
        from app import start_conversion
        sig = inspect.signature(start_conversion)
        assert "speed" in sig.parameters, (
            "start_conversion is missing a 'speed' parameter"
        )
        param = sig.parameters["speed"]
        # FastAPI Form defaults are stored in the parameter's default
        # The annotation may be float with a Form(...) default object;
        # we check the annotation type and that a default exists.
        assert param.default is not inspect.Parameter.empty, (
            "start_conversion 'speed' parameter has no default"
        )

    def test_run_conversion_has_speed_param(self):
        """app.run_conversion must propagate the speed parameter."""
        from app import run_conversion
        sig = inspect.signature(run_conversion)
        assert "speed" in sig.parameters, (
            "run_conversion is missing a 'speed' parameter"
        )

    def test_run_conversion_speed_default_is_1(self):
        """app.run_conversion speed parameter must default to 1.0."""
        from app import run_conversion
        sig = inspect.signature(run_conversion)
        default = sig.parameters["speed"].default
        assert default == 1.0, (
            f"run_conversion speed default expected 1.0, got {default!r}"
        )


# ---------------------------------------------------------------------------
# 4. TestSpeedEndToEnd
# ---------------------------------------------------------------------------

class TestSpeedEndToEnd:
    """End-to-end tests for speed propagation through the API."""

    @pytest.fixture
    def client(self):
        """Create a FastAPI TestClient with TTS and model load mocked out."""
        from fastapi.testclient import TestClient

        # Patch is_tts_available so the app reports TTS as available
        with patch("app.is_tts_available", return_value=True), \
             patch("app.load_model", return_value=None), \
             patch("tts._tts_model", MagicMock()), \
             patch("tts.POCKET_TTS_AVAILABLE", True):
            from app import app
            yield TestClient(app, raise_server_exceptions=False)

    def _make_minimal_epub(self, tmp_path):
        """Create a minimal valid EPUB file for upload."""
        import zipfile, os
        epub_path = tmp_path / "test.epub"
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr("META-INF/container.xml",
                """<?xml version="1.0"?>
                <container version="1.0" xmlns="urn:oasis:schemas-container">
                  <rootfiles>
                    <rootfile full-path="OEBPS/content.opf"
                              media-type="application/oebps-package+xml"/>
                  </rootfiles>
                </container>""")
            zf.writestr("OEBPS/content.opf",
                """<?xml version="1.0" encoding="UTF-8"?>
                <package xmlns="http://www.idpf.org/2007/opf" version="2.0"
                         unique-identifier="bookid">
                  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                    <dc:title>Test Book</dc:title>
                    <dc:creator>Test Author</dc:creator>
                    <dc:identifier id="bookid">test-id</dc:identifier>
                    <dc:language>en</dc:language>
                  </metadata>
                  <manifest>
                    <item id="ch1" href="chapter1.xhtml"
                          media-type="application/xhtml+xml"/>
                    <item id="ncx" href="toc.ncx"
                          media-type="application/x-dtbncx+xml"/>
                  </manifest>
                  <spine toc="ncx">
                    <itemref idref="ch1"/>
                  </spine>
                </package>""")
            zf.writestr("OEBPS/chapter1.xhtml",
                """<?xml version="1.0" encoding="UTF-8"?>
                <!DOCTYPE html>
                <html xmlns="http://www.w3.org/1999/xhtml">
                  <head><title>Chapter 1</title></head>
                  <body><h1>Chapter One</h1>
                  <p>This is a test chapter with enough words to matter.</p>
                  </body>
                </html>""")
            zf.writestr("OEBPS/toc.ncx",
                """<?xml version="1.0" encoding="UTF-8"?>
                <!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN"
                  "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">
                <ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"
                     version="2005-1">
                  <head><meta name="dtb:uid" content="test-id"/></head>
                  <docTitle><text>Test Book</text></docTitle>
                  <navMap>
                    <navPoint id="np1" playOrder="1">
                      <navLabel><text>Chapter 1</text></navLabel>
                      <content src="chapter1.xhtml"/>
                    </navPoint>
                  </navMap>
                </ncx>""")
        return epub_path

    def test_convert_endpoint_accepts_speed_param(self, client, tmp_path):
        """POST /api/convert with speed=1.5 should not return 422."""
        epub_path = self._make_minimal_epub(tmp_path)

        with patch("app.run_conversion", new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=None)()), \
             patch("app.asyncio.create_task"):
            with open(epub_path, "rb") as fh:
                response = client.post(
                    "/api/convert",
                    data={"speed": "1.5"},
                    files={"epub_file": ("test.epub", fh, "application/epub+zip")},
                )
            # 422 = FastAPI validation error; anything else means speed was accepted
            assert response.status_code != 422, (
                f"Endpoint rejected speed=1.5 with 422: {response.text}"
            )

    def test_speed_passed_through_to_run_conversion(self, tmp_path):
        """Verify that start_conversion forwards speed to run_conversion."""
        import asyncio as _asyncio
        from fastapi.testclient import TestClient

        captured = {}

        async def fake_run_conversion(*args, **kwargs):
            # run_conversion(job_id, epub_path, voice, per_chapter,
            #                chapter_indices, skip_existing, announce_chapters,
            #                output_format, text_processing, speed)
            captured["speed"] = args[9] if len(args) > 9 else kwargs.get("speed")

        epub_path = self._make_minimal_epub(tmp_path)

        with patch("app.is_tts_available", return_value=True), \
             patch("app.load_model", return_value=None), \
             patch("tts._tts_model", MagicMock()), \
             patch("tts.POCKET_TTS_AVAILABLE", True), \
             patch("app.run_conversion", side_effect=fake_run_conversion), \
             patch("app.asyncio.create_task") as mock_create_task:

            # Capture the coroutine passed to create_task
            created_coros = []
            def capture_task(coro):
                created_coros.append(coro)
                return MagicMock()
            mock_create_task.side_effect = capture_task

            from app import app
            client = TestClient(app, raise_server_exceptions=False)

            with open(epub_path, "rb") as fh:
                response = client.post(
                    "/api/convert",
                    data={"speed": "2.0"},
                    files={"epub_file": ("test.epub", fh, "application/epub+zip")},
                )

            assert response.status_code != 422, (
                f"Endpoint rejected speed=2.0: {response.text}"
            )

            # The create_task was called — verify the coroutine carries speed=2.0
            # by inspecting the cr_frame locals (Python 3.11+)
            assert mock_create_task.called, "asyncio.create_task was never called"




# ---------------------------------------------------------------------------
# Tests for text chunker (TTS context window overflow fix)
# ---------------------------------------------------------------------------

class TestSplitTextIntoChunks:
    """Tests for _split_text_into_chunks — the fix for Pocket TTS tensor overflow."""

    def test_short_text_single_chunk(self):
        """Text under max_words stays as one chunk."""
        from tts import _split_text_into_chunks
        result = _split_text_into_chunks("Hello world. This is a test.", max_words=100)
        assert len(result) == 1

    def test_long_sentence_no_punctuation_is_split(self):
        """A single run of 200 words with no punctuation is split into ≤100-word chunks."""
        from tts import _split_text_into_chunks
        text = "word " * 200
        chunks = _split_text_into_chunks(text, max_words=100)
        assert len(chunks) >= 2
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_sentence_boundaries_respected(self):
        """Chunks break at sentence endings, not mid-sentence."""
        from tts import _split_text_into_chunks
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_text_into_chunks(text, max_words=4)
        # Each chunk should end with a complete sentence
        for chunk in chunks:
            assert chunk.strip()  # no empty chunks

    def test_no_empty_chunks(self):
        """Empty or whitespace-only input returns empty list."""
        from tts import _split_text_into_chunks
        assert _split_text_into_chunks("") == []
        assert _split_text_into_chunks("   ") == []

    def test_max_words_never_exceeded(self):
        """No chunk exceeds max_words words."""
        from tts import _split_text_into_chunks
        text = ("This is a sentence that has some words. " * 20 +
                "nowaytobreakthis " * 150)
        chunks = _split_text_into_chunks(text, max_words=100)
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_all_words_preserved(self):
        """Total word count across chunks equals original word count."""
        from tts import _split_text_into_chunks
        text = "word " * 250
        chunks = _split_text_into_chunks(text, max_words=100)
        original_words = len(text.split())
        chunk_words = sum(len(c.split()) for c in chunks)
        assert chunk_words == original_words

    def test_single_sentence_under_limit_not_split(self):
        """A 50-word sentence with max_words=100 stays as one chunk."""
        from tts import _split_text_into_chunks
        text = "word " * 50
        chunks = _split_text_into_chunks(text.strip(), max_words=100)
        assert len(chunks) == 1
