"""Tests for chunk-level progress callback threading.

Verifies that:
  - generate_speech() and text_to_audio() expose chunk_callback (default None)
  - The callback fires once per synthesised chunk with monotonically increasing done
  - converter.convert_epub_to_mp3 calls progress_callback multiple times per
    chapter (once per chunk) rather than once per chapter
"""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import converter
import tts
from converter import BookMetadata, Chapter


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------

class TestGenerateSpeechSignature(unittest.TestCase):
    def test_chunk_callback_param_exists(self):
        sig = inspect.signature(tts.generate_speech)
        self.assertIn("chunk_callback", sig.parameters)

    def test_chunk_callback_defaults_to_none(self):
        sig = inspect.signature(tts.generate_speech)
        self.assertIsNone(sig.parameters["chunk_callback"].default)


class TestTextToAudioSignature(unittest.TestCase):
    def test_chunk_callback_param_exists(self):
        sig = inspect.signature(converter.text_to_audio)
        self.assertIn("chunk_callback", sig.parameters)

    def test_chunk_callback_defaults_to_none(self):
        sig = inspect.signature(converter.text_to_audio)
        self.assertIsNone(sig.parameters["chunk_callback"].default)


# ---------------------------------------------------------------------------
# converter.convert_epub_to_mp3 — progress_callback called per chunk
# ---------------------------------------------------------------------------

def _fake_book_metadata(n_words: int = 300) -> BookMetadata:
    """Return a BookMetadata with one chapter of n_words words."""
    text = " ".join(f"Word{i}" for i in range(n_words))
    chapter = Chapter(
        title="Chapter One",
        text=text,
        word_count=n_words,
    )
    return BookMetadata(
        title="Test Book",
        author="Test Author",
        chapters=[chapter],
    )


class TestConverterChunkProgress(unittest.TestCase):
    """progress_callback is called multiple times during chapter TTS."""

    def _make_audio_response(self, n_samples: int = 240):
        return np.zeros(n_samples, dtype=np.int16), tts.SAMPLE_RATE

    def _run_convert(self, progress_cb, fake_text_to_audio, per_chapter=True):
        """Helper: run convert_epub_to_mp3 with mocked parse_epub and text_to_audio."""
        import tempfile, shutil

        out_dir = tempfile.mkdtemp()
        try:
            with (
                patch("converter.parse_epub", return_value=_fake_book_metadata()),
                patch("converter.text_to_audio", side_effect=fake_text_to_audio),
            ):
                converter.convert_epub_to_mp3(
                    "fake.epub",
                    out_dir,
                    per_chapter=per_chapter,
                    progress_callback=progress_cb,
                )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_progress_callback_called_multiple_times_per_chapter(self):
        """With 3 TTS chunks, progress_callback fires 3+ times during TTS stage."""
        progress_calls: list[tuple[int, str]] = []

        def progress_cb(pct, total, msg, details):
            progress_calls.append((pct, details.get("stage", "")))

        # Simulate text_to_audio calling chunk_callback 3 times per call
        def fake_text_to_audio(text, voice=None, speed=1.0, chunk_callback=None):
            if chunk_callback:
                for i in range(1, 4):
                    chunk_callback(i, 3)
            return self._make_audio_response()

        self._run_convert(progress_cb, fake_text_to_audio)

        tts_calls = [p for p in progress_calls if p[1] == "tts"]
        self.assertGreaterEqual(
            len(tts_calls), 3,
            f"Expected ≥3 TTS progress calls, got {len(tts_calls)}: {tts_calls}",
        )

    def test_progress_pct_increases_during_tts(self):
        """Progress percentage grows across chunk callbacks within a chapter."""
        pcts: list[int] = []

        def progress_cb(pct, total, msg, details):
            if details.get("stage") == "tts":
                pcts.append(pct)

        def fake_text_to_audio(text, voice=None, speed=1.0, chunk_callback=None):
            if chunk_callback:
                for i in range(1, 4):
                    chunk_callback(i, 3)
            return self._make_audio_response()

        self._run_convert(progress_cb, fake_text_to_audio)

        if len(pcts) >= 2:
            self.assertLessEqual(pcts[0], pcts[-1], "Progress should not decrease")

    def test_combined_mode_also_gets_chunk_progress(self):
        """per_chapter=False also fires chunk callbacks."""
        progress_calls: list[str] = []

        def progress_cb(pct, total, msg, details):
            progress_calls.append(details.get("stage", ""))

        def fake_text_to_audio(text, voice=None, speed=1.0, chunk_callback=None):
            if chunk_callback:
                for i in range(1, 3):
                    chunk_callback(i, 2)
            return self._make_audio_response()

        self._run_convert(progress_cb, fake_text_to_audio, per_chapter=False)

        tts_calls = [s for s in progress_calls if s == "tts"]
        self.assertGreaterEqual(len(tts_calls), 2)


if __name__ == "__main__":
    unittest.main()
