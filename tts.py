"""Text-to-Speech via Kokoro TTS (Kokoro-82M).

Engine selection at import time:
  1. mlx_audio available  → Kokoro MLX  (fastest on Apple Silicon)
  2. kokoro_onnx available → Kokoro ONNX (cross-platform)

Run setup_kokoro.py once to download model files before first use.

Install:
  Apple Silicon:  pip install mlx-audio
  Cross-platform: pip install kokoro-onnx huggingface_hub
"""

import numpy as np
from typing import Callable

try:
    from scipy.signal import resample as scipy_resample
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- Engine detection (happens once at import time) ---
# Test Kokoro MLX: mlx_audio must be installed AND its Kokoro pipeline must be
# importable (it has optional deps like misaki/phonemizer that may be missing).
try:
    import mlx_audio  # noqa: F401
    from mlx_audio.tts.models.kokoro import Model as _KokoroMLXModel  # noqa: F401
    KOKORO_ENGINE: str = "mlx"
except (ImportError, AttributeError):
    try:
        from kokoro_onnx import Kokoro as _KokoroONNX  # noqa: F401
        KOKORO_ENGINE: str = "onnx"
    except ImportError:
        raise RuntimeError(
            "No Kokoro TTS engine found.\n"
            "Install one of:\n"
            "  Apple Silicon:  pip install mlx-audio\n"
            "  Cross-platform: pip install kokoro-onnx huggingface_hub\n"
            "Then run: python setup_kokoro.py"
        ) from None

# Sample rate
SAMPLE_RATE = 24000

# Voice catalogue from Kokoro
from kokoro_tts import KOKORO_VOICES, DEFAULT_KOKORO_VOICE
VOICES = {name: dict(info) for name, info in KOKORO_VOICES.items()}
DEFAULT_VOICE = DEFAULT_KOKORO_VOICE

# Global model instance and cached voice states
_tts_model = None
_voice_states = {}  # Cache voice states to avoid re-downloading


def load_model():
    """Load the Kokoro TTS model. Downloads weights on first run."""
    import kokoro_tts  # noqa: PLC0415
    return kokoro_tts.load_kokoro_model(KOKORO_ENGINE)


def is_tts_available() -> bool:
    """Check if any TTS engine is available."""
    return bool(KOKORO_ENGINE)


def get_voice_list() -> list[dict]:
    """Return list of voices with metadata for UI."""
    voices = []
    for name, info in VOICES.items():
        voices.append({
            "name": name,
            "characteristic": info["char"],
            "gender": info["gender"],
            "best_for": info["best_for"],
            "label": f"{name} ({info['char']}, {info['gender']})",
            "engine": "kokoro",
        })
    return voices



def _resample_audio(audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """
    Resample audio to change playback speed without altering pitch.

    speed > 1.0 speeds up (shortens), speed < 1.0 slows down (lengthens).
    Uses scipy.signal.resample when available, falls back to numpy linear
    interpolation.
    """
    if abs(speed - 1.0) < 0.001:
        return audio  # No change needed

    original_len = len(audio)
    target_len = int(round(original_len / speed))

    if target_len == 0:
        return audio

    dtype = audio.dtype
    # Work in float64 for precision
    audio_f = audio.astype(np.float64)

    if SCIPY_AVAILABLE:
        resampled = scipy_resample(audio_f, target_len)
    else:
        # Linear interpolation fallback
        old_indices = np.linspace(0, original_len - 1, original_len)
        new_indices = np.linspace(0, original_len - 1, target_len)
        resampled = np.interp(new_indices, old_indices, audio_f)

    # Clip and convert back to original dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        resampled = np.clip(resampled, info.min, info.max)
    return resampled.astype(dtype)


# Max words per chunk passed to the TTS model.
# Kokoro works best with short, natural-sentence segments.  Keeping chunks
# at ≤50 words avoids silent truncation on long comma-free runs and gives
# better prosody than feeding entire paragraphs at once.
_MAX_WORDS_PER_CHUNK = 50


def _split_text_into_chunks(text: str, max_words: int = _MAX_WORDS_PER_CHUNK) -> list[str]:
    """
    Split text into chunks of at most max_words words.

    Priority of split points (best to worst):
      1. Sentence boundaries  (.  !  ?  ;)
      2. Comma clause boundaries within long sentences
      3. Hard word-boundary cut (last resort)

    This prevents the TTS model from silently truncating long runs that
    have no natural sentence breaks.
    """
    import re

    # --- pass 1: split on sentence-ending punctuation ---
    sentences = re.split(r'(?<=[.!?;])\s+', text.strip())

    # --- pass 2: split long sentences at comma boundaries ---
    clauses: list[str] = []
    for sentence in sentences:
        if len(sentence.split()) > max_words:
            # Break at commas; each part keeps its trailing comma so the
            # model knows it's mid-thought (better prosody than a hard cut).
            parts = re.split(r'(?<=,)\s+', sentence)
            clauses.extend(parts)
        else:
            clauses.append(sentence)

    # --- pass 3: accumulate clauses into ≤max_words chunks ---
    chunks: list[str] = []
    current_words: list[str] = []
    current_count = 0

    for clause in clauses:
        words = clause.split()
        if not words:
            continue

        # Hard word-boundary split for any clause still over the limit
        # (e.g. a comma-free run-on sentence longer than max_words).
        if len(words) > max_words:
            if current_words:
                chunks.append(' '.join(current_words))
                current_words = []
                current_count = 0
            for i in range(0, len(words), max_words):
                chunks.append(' '.join(words[i:i + max_words]))
            continue

        if current_count + len(words) > max_words:
            if current_words:
                chunks.append(' '.join(current_words))
            current_words = words[:]
            current_count = len(words)
        else:
            current_words.extend(words)
            current_count += len(words)

    if current_words:
        chunks.append(' '.join(current_words))

    return [c for c in chunks if c.strip()]


def _tensor_to_int16(audio_tensor) -> np.ndarray:
    """Convert a TTS audio tensor to int16 numpy array."""
    audio_np = audio_tensor.squeeze().numpy()
    if audio_np.dtype != np.int16:
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)
    return audio_np


def generate_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    progress_callback: Callable[[str], None] = None,
    speed: float = 1.0,
    chunk_callback: Callable[[int, int], None] = None,
) -> tuple[np.ndarray, int]:
    """
    Generate speech audio from text using Kokoro TTS.

    Args:
        text: Text to convert to speech.
        voice: Voice name (see VOICES / kokoro_tts.KOKORO_VOICES).
        progress_callback: Optional callback(message) for status updates.
        speed: Playback speed multiplier (0.75 – 2.0).
        chunk_callback: Optional callback(done, total) fired after each ~50-word
            chunk is synthesised.  Enables fine-grained progress reporting.

    Returns:
        Tuple of (audio numpy array int16, sample rate).
    """
    import kokoro_tts as _kt  # noqa: PLC0415
    return _kt.generate_speech_kokoro(text, voice, speed, chunk_callback, KOKORO_ENGINE)


def generate_preview(voice: str = DEFAULT_VOICE) -> tuple[np.ndarray, int]:
    """Generate a short preview for a voice."""
    preview_text = f"Hello, I'm {voice}. I'll be reading your book."
    return generate_speech(preview_text, voice)


if __name__ == "__main__":
    print(f"TTS available: {is_tts_available()}")
    print(f"Default voice: {DEFAULT_VOICE}")
    print(f"Available voices: {list(VOICES.keys())}")

    if is_tts_available():
        print("\n--- Testing TTS ---")
        audio, sr = generate_speech("This is a test of Kokoro text to speech.", DEFAULT_VOICE)
        print(f"Generated {len(audio)} samples at {sr}Hz ({len(audio)/sr:.2f} seconds)")
