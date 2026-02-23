"""Tests for Kokoro TTS engine module.

These tests run regardless of whether mlx-audio / kokoro-onnx are installed —
they mock the model calls so no GPU/download is required.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import kokoro_tts
import tts


# ---------------------------------------------------------------------------
# Engine availability / sentinel checks
# ---------------------------------------------------------------------------

class TestEngineAvailability(unittest.TestCase):
    def test_kokoro_engine_is_valid_sentinel(self):
        """KOKORO_ENGINE is None, 'mlx', or 'onnx'."""
        self.assertIn(tts.KOKORO_ENGINE, (None, "mlx", "onnx"))

    def test_is_tts_available_returns_bool(self):
        result = tts.is_tts_available()
        self.assertIsInstance(result, bool)

    def test_is_tts_available_true_when_kokoro_active(self):
        if tts.KOKORO_ENGINE:
            self.assertTrue(tts.is_tts_available())

    # Pocket TTS disabled — only Kokoro is used
    # def test_is_tts_available_true_when_pocket_tts_installed(self): ...


# ---------------------------------------------------------------------------
# Voice catalogue schema
# ---------------------------------------------------------------------------

class TestKokoroVoicesSchema(unittest.TestCase):
    def test_all_required_fields_present(self):
        for name, info in kokoro_tts.KOKORO_VOICES.items():
            with self.subTest(voice=name):
                self.assertIn("char", info)
                self.assertIn("gender", info)
                self.assertIn("lang", info)
                self.assertIn("id", info)
                self.assertIn("best_for", info)

    def test_default_voice_is_in_voices(self):
        self.assertIn(kokoro_tts.DEFAULT_KOKORO_VOICE, kokoro_tts.KOKORO_VOICES)

    def test_voice_ids_have_lang_prefix(self):
        """Voice IDs must start with 'a' (en-us) or 'b' (en-gb)."""
        for name, info in kokoro_tts.KOKORO_VOICES.items():
            with self.subTest(voice=name):
                self.assertIn(info["id"][0], ("a", "b"))

    def test_gender_values(self):
        for name, info in kokoro_tts.KOKORO_VOICES.items():
            with self.subTest(voice=name):
                self.assertIn(info["gender"], ("F", "M"))


class TestGetVoiceList(unittest.TestCase):
    def test_voice_list_has_required_fields(self):
        voices = tts.get_voice_list()
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 0)
        for v in voices:
            with self.subTest(voice=v.get("name")):
                self.assertIn("name", v)
                self.assertIn("characteristic", v)
                self.assertIn("gender", v)
                self.assertIn("best_for", v)
                self.assertIn("label", v)
                self.assertIn("engine", v)

    def test_voice_list_engine_field_matches_active_engine(self):
        voices = tts.get_voice_list()
        expected_engine = "kokoro"
        for v in voices:
            self.assertEqual(v["engine"], expected_engine)


# ---------------------------------------------------------------------------
# generate_speech_kokoro — chunk callback behaviour
# ---------------------------------------------------------------------------

class TestGenerateSpeechKokoro(unittest.TestCase):
    def _make_float_audio(self, n: int = 2400) -> np.ndarray:
        """Float32 audio at 24 kHz for 0.1 s."""
        return np.zeros(n, dtype=np.float32)

    # -- MLX engine --

    def test_mlx_callback_fires_per_chunk(self):
        chunks = ["First chunk text.", "Second chunk text.", "Third chunk text."]
        calls: list[tuple[int, int]] = []
        mock_model = MagicMock()
        mock_model.generate.return_value = self._make_float_audio()

        with (
            patch("kokoro_tts._kokoro_model", mock_model),
            patch("tts._split_text_into_chunks", return_value=chunks),
        ):
            audio, sr = kokoro_tts.generate_speech_kokoro(
                "placeholder", "heart", 1.0,
                lambda done, total: calls.append((done, total)),
                "mlx",
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual([c[0] for c in calls], [1, 2, 3])
        self.assertTrue(all(c[1] == 3 for c in calls))
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.int16)
        self.assertEqual(sr, kokoro_tts.SAMPLE_RATE)

    def test_mlx_no_callback_no_error(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = self._make_float_audio()

        with (
            patch("kokoro_tts._kokoro_model", mock_model),
            patch("tts._split_text_into_chunks", return_value=["one chunk"]),
        ):
            audio, sr = kokoro_tts.generate_speech_kokoro(
                "text", "heart", 1.0, None, "mlx",
            )

        self.assertIsInstance(audio, np.ndarray)

    def test_mlx_empty_text_returns_empty_array(self):
        mock_model = MagicMock()

        with (
            patch("kokoro_tts._kokoro_model", mock_model),
            patch("tts._split_text_into_chunks", return_value=[]),
        ):
            audio, sr = kokoro_tts.generate_speech_kokoro(
                "", "heart", 1.0, None, "mlx",
            )

        self.assertEqual(len(audio), 0)
        self.assertEqual(audio.dtype, np.int16)

    # -- ONNX engine --

    def test_onnx_callback_fires_per_chunk(self):
        chunks = ["Chunk A.", "Chunk B."]
        calls: list[tuple[int, int]] = []
        mock_model = MagicMock()
        mock_model.create.return_value = (self._make_float_audio(), 24000)

        with (
            patch("kokoro_tts._kokoro_model", mock_model),
            patch("tts._split_text_into_chunks", return_value=chunks),
        ):
            audio, sr = kokoro_tts.generate_speech_kokoro(
                "placeholder", "emma", 1.0,
                lambda done, total: calls.append((done, total)),
                "onnx",
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual([c[0] for c in calls], [1, 2])
        mock_model.create.assert_called()

    # -- Audio format --

    def test_output_is_int16(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = self._make_float_audio()

        with (
            patch("kokoro_tts._kokoro_model", mock_model),
            patch("tts._split_text_into_chunks", return_value=["text"]),
        ):
            audio, _ = kokoro_tts.generate_speech_kokoro(
                "text", "heart", 1.0, None, "mlx",
            )

        self.assertEqual(audio.dtype, np.int16)

    def test_silence_inserted_between_chunks(self):
        """Audio with 2 chunks must be longer than a single chunk."""
        mock_model = MagicMock()
        single_chunk_samples = 1200
        mock_model.generate.return_value = np.zeros(single_chunk_samples, dtype=np.float32)

        with (
            patch("kokoro_tts._kokoro_model", mock_model),
            patch("tts._split_text_into_chunks", return_value=["a", "b"]),
        ):
            audio, sr = kokoro_tts.generate_speech_kokoro(
                "a b", "heart", 1.0, None, "mlx",
            )

        # 2 audio chunks + 2 silence segments (100ms each = 2400 samples)
        expected_min = single_chunk_samples * 2 + int(sr * 0.1) * 2
        self.assertGreaterEqual(len(audio), expected_min)


# ---------------------------------------------------------------------------
# Voice routing
# ---------------------------------------------------------------------------

class TestVoiceRouting(unittest.TestCase):
    def test_voice_uses_kokoro_when_active(self):
        """A regular voice name routes to Kokoro when engine is active."""
        if not tts.KOKORO_ENGINE:
            self.skipTest("Kokoro not active")

        with patch("kokoro_tts.generate_speech_kokoro") as mock_kokoro:
            mock_kokoro.return_value = (np.zeros(100, dtype=np.int16), 24000)
            tts.generate_speech("text", voice="heart")

        mock_kokoro.assert_called_once()


# ---------------------------------------------------------------------------
# _get_kokoro_id helper
# ---------------------------------------------------------------------------

class TestGetKokoroId(unittest.TestCase):
    def test_known_voice_returns_correct_id(self):
        kokoro_id, lang_prefix, onnx_lang = kokoro_tts._get_kokoro_id("heart")
        self.assertEqual(kokoro_id, "af_heart")
        self.assertEqual(lang_prefix, "a")
        self.assertEqual(onnx_lang, "en-us")

    def test_british_voice_has_b_lang_prefix(self):
        _, lang_prefix, onnx_lang = kokoro_tts._get_kokoro_id("george")
        self.assertEqual(lang_prefix, "b")
        self.assertEqual(onnx_lang, "en-gb")

    def test_unknown_voice_falls_back_to_default(self):
        default_id, _, _ = kokoro_tts._get_kokoro_id(kokoro_tts.DEFAULT_KOKORO_VOICE)
        fallback_id, _, _ = kokoro_tts._get_kokoro_id("nonexistent_voice_xyz")
        self.assertEqual(fallback_id, default_id)


# ---------------------------------------------------------------------------
# _to_int16 conversion helper
# ---------------------------------------------------------------------------

class TestToInt16(unittest.TestCase):
    def test_float32_is_scaled_and_clamped(self):
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0], dtype=np.float32)
        result = kokoro_tts._to_int16(audio)
        self.assertEqual(result.dtype, np.int16)
        self.assertLessEqual(result.max(), 32767)
        self.assertGreaterEqual(result.min(), -32768)

    def test_int16_passthrough(self):
        audio = np.array([0, 100, -100], dtype=np.int16)
        result = kokoro_tts._to_int16(audio)
        np.testing.assert_array_equal(result, audio)

    def test_2d_array_is_squeezed(self):
        audio = np.zeros((1, 100), dtype=np.float32)
        result = kokoro_tts._to_int16(audio)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)


if __name__ == "__main__":
    unittest.main()
