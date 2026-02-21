"""Unit tests for the bitrate selection feature."""

import inspect
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _silence_1s(sample_rate: int = 24000) -> np.ndarray:
    """Return 1 second of silence as int16 PCM at the given sample rate."""
    return np.zeros(sample_rate, dtype=np.int16)


# ---------------------------------------------------------------------------
# 1. Signature tests
# ---------------------------------------------------------------------------

class TestConvertWavToMp3Signature:
    """Verify that both converter functions expose the expected bitrate param."""

    def test_convert_wav_to_mp3_has_bitrate_param(self):
        """convert_wav_to_mp3 must have a 'bitrate' parameter."""
        from converter import convert_wav_to_mp3
        sig = inspect.signature(convert_wav_to_mp3)
        assert "bitrate" in sig.parameters, (
            "convert_wav_to_mp3 is missing the 'bitrate' parameter"
        )

    def test_convert_wav_to_mp3_bitrate_default_192(self):
        """convert_wav_to_mp3's bitrate parameter must default to 192."""
        from converter import convert_wav_to_mp3
        sig = inspect.signature(convert_wav_to_mp3)
        param = sig.parameters["bitrate"]
        assert param.default == 192, (
            f"Expected default bitrate 192, got {param.default}"
        )

    def test_convert_epub_to_mp3_has_bitrate_param(self):
        """convert_epub_to_mp3 must have a 'bitrate' parameter."""
        from converter import convert_epub_to_mp3
        sig = inspect.signature(convert_epub_to_mp3)
        assert "bitrate" in sig.parameters, (
            "convert_epub_to_mp3 is missing the 'bitrate' parameter"
        )

    def test_convert_epub_to_mp3_bitrate_default_192(self):
        """convert_epub_to_mp3's bitrate parameter must default to 192."""
        from converter import convert_epub_to_mp3
        sig = inspect.signature(convert_epub_to_mp3)
        param = sig.parameters["bitrate"]
        assert param.default == 192, (
            f"Expected default bitrate 192, got {param.default}"
        )


# ---------------------------------------------------------------------------
# 2. Actual encoding tests
# ---------------------------------------------------------------------------

class TestConvertWavToMp3Bitrate:
    """Verify that the bitrate parameter actually affects MP3 encoding."""

    def test_64kbps_produces_smaller_file_than_192kbps(self):
        """A 64 kbps encode must yield a smaller file than 192 kbps."""
        from converter import convert_wav_to_mp3

        audio = _silence_1s()
        with tempfile.TemporaryDirectory() as tmpdir:
            path_64 = os.path.join(tmpdir, "out_64.mp3")
            path_192 = os.path.join(tmpdir, "out_192.mp3")

            convert_wav_to_mp3(audio, 24000, path_64, bitrate=64)
            convert_wav_to_mp3(audio, 24000, path_192, bitrate=192)

            size_64 = os.path.getsize(path_64)
            size_192 = os.path.getsize(path_192)

            assert size_64 < size_192, (
                f"64 kbps file ({size_64} bytes) should be smaller than "
                f"192 kbps file ({size_192} bytes)"
            )

    def test_192kbps_produces_larger_file_than_128kbps(self):
        """A 192 kbps encode must yield a larger file than 128 kbps."""
        from converter import convert_wav_to_mp3

        audio = _silence_1s()
        with tempfile.TemporaryDirectory() as tmpdir:
            path_128 = os.path.join(tmpdir, "out_128.mp3")
            path_192 = os.path.join(tmpdir, "out_192.mp3")

            convert_wav_to_mp3(audio, 24000, path_128, bitrate=128)
            convert_wav_to_mp3(audio, 24000, path_192, bitrate=192)

            size_128 = os.path.getsize(path_128)
            size_192 = os.path.getsize(path_192)

            assert size_192 > size_128, (
                f"192 kbps file ({size_192} bytes) should be larger than "
                f"128 kbps file ({size_128} bytes)"
            )

    def test_valid_bitrates_produce_mp3(self):
        """64, 128, and 192 kbps must each produce a non-empty output file."""
        from converter import convert_wav_to_mp3

        audio = _silence_1s()
        for bitrate in (64, 128, 192):
            with tempfile.TemporaryDirectory() as tmpdir:
                out = os.path.join(tmpdir, f"out_{bitrate}.mp3")
                convert_wav_to_mp3(audio, 24000, out, bitrate=bitrate)
                size = os.path.getsize(out)
                assert size > 0, (
                    f"Encoding at {bitrate} kbps produced an empty file"
                )

    def test_output_is_valid_mp3(self):
        """The output file must start with a recognised MP3 frame sync or ID3 header."""
        from converter import convert_wav_to_mp3

        audio = _silence_1s()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.mp3")
            convert_wav_to_mp3(audio, 24000, out, bitrate=192)

            with open(out, "rb") as f:
                header = f.read(3)

            # ID3 tag header
            is_id3 = header[:3] == b"ID3"
            # MPEG frame sync bytes (various layer/bitrate combos)
            is_mp3_sync = (
                len(header) >= 2
                and header[0] == 0xFF
                and header[1] in (0xFB, 0xF3, 0xFA, 0xF2, 0xE3, 0xE2)
            )

            assert is_id3 or is_mp3_sync, (
                f"Output does not look like a valid MP3 file. "
                f"First 3 bytes: {header!r}"
            )


# ---------------------------------------------------------------------------
# 3. API endpoint tests
# ---------------------------------------------------------------------------

class TestBitrateAPIEndpoint:
    """Verify that the FastAPI layer exposes and forwards the bitrate field."""

    def test_convert_endpoint_accepts_bitrate_field(self):
        """start_conversion must accept a 'bitrate' parameter."""
        from app import start_conversion
        sig = inspect.signature(start_conversion)
        assert "bitrate" in sig.parameters, (
            "start_conversion is missing the 'bitrate' parameter"
        )

    def test_bitrate_default_is_192(self):
        """start_conversion's bitrate parameter must default to 192."""
        from app import start_conversion
        sig = inspect.signature(start_conversion)
        param = sig.parameters["bitrate"]
        # FastAPI Form defaults are stored on the parameter annotation/default.
        # The default value may be a fastapi.Form(...) instance or a plain int.
        default = param.default
        # If it's a FastAPI FieldInfo, check its default attribute
        if hasattr(default, "default"):
            actual = default.default
        else:
            actual = default
        assert actual == 192, (
            f"Expected default bitrate 192 in start_conversion, got {actual}"
        )

    def test_run_conversion_has_bitrate_param(self):
        """run_conversion must accept a 'bitrate' parameter."""
        from app import run_conversion
        sig = inspect.signature(run_conversion)
        assert "bitrate" in sig.parameters, (
            "run_conversion is missing the 'bitrate' parameter"
        )


# ---------------------------------------------------------------------------
# 4. Validation / encoder interaction tests
# ---------------------------------------------------------------------------

class TestBitrateValidation:
    """Verify that valid bitrates are accepted and forwarded to lameenc."""

    def test_valid_bitrates_accepted_by_converter(self):
        """64, 128, 192 kbps should all be accepted without raising an exception."""
        from converter import convert_wav_to_mp3

        audio = _silence_1s()
        for bitrate in (64, 128, 192):
            with tempfile.TemporaryDirectory() as tmpdir:
                out = os.path.join(tmpdir, f"out_{bitrate}.mp3")
                # Should not raise
                convert_wav_to_mp3(audio, 24000, out, bitrate=bitrate)

    def test_bitrate_passed_to_encoder(self):
        """convert_wav_to_mp3 must call lameenc.Encoder.set_bit_rate with the supplied value."""
        from converter import convert_wav_to_mp3

        audio = _silence_1s()

        for expected_bitrate in (64, 128, 192):
            with tempfile.TemporaryDirectory() as tmpdir:
                out = os.path.join(tmpdir, "out.mp3")

                mock_encoder = MagicMock()
                # flush() must return bytes so the file write doesn't fail
                mock_encoder.encode.return_value = b"\xff\xfb\x90\x00" * 100
                mock_encoder.flush.return_value = b""

                with patch("lameenc.Encoder", return_value=mock_encoder):
                    convert_wav_to_mp3(audio, 24000, out, bitrate=expected_bitrate)

                mock_encoder.set_bit_rate.assert_called_once_with(expected_bitrate), (
                    f"set_bit_rate was not called with {expected_bitrate}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
