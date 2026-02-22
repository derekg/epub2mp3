"""Text-to-Speech using Pocket TTS (local, no API limits).

Pocket TTS provides offline speech synthesis with voice cloning.
Uses predefined voice embeddings from HuggingFace.
"""

import numpy as np
from typing import Callable

try:
    from scipy.signal import resample as scipy_resample
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import pocket-tts
try:
    from pocket_tts import TTSModel
    from pocket_tts.utils.utils import PREDEFINED_VOICES
    POCKET_TTS_AVAILABLE = True
except ImportError:
    POCKET_TTS_AVAILABLE = False
    TTSModel = None
    PREDEFINED_VOICES = {}

# Sample rate for Pocket TTS
SAMPLE_RATE = 24000

# Available Pocket TTS voices (from predefined embeddings)
VOICES = {
    "alba": {"char": "Warm", "gender": "F", "best_for": "narration"},
    "marius": {"char": "Clear", "gender": "M", "best_for": "narration"},
    "javert": {"char": "Firm", "gender": "M", "best_for": "narration"},
    "jean": {"char": "Smooth", "gender": "M", "best_for": "narration"},
    "fantine": {"char": "Soft", "gender": "F", "best_for": "narration"},
    "cosette": {"char": "Youthful", "gender": "F", "best_for": "casual"},
    "eponine": {"char": "Bright", "gender": "F", "best_for": "casual"},
    "azelma": {"char": "Light", "gender": "F", "best_for": "casual"},
}

# Default voice
DEFAULT_VOICE = "alba"

# Global model instance and cached voice states
_tts_model = None
_voice_states = {}  # Cache voice states to avoid re-downloading


def load_model():
    """Load the Pocket TTS model. Call once at startup."""
    global _tts_model
    if not POCKET_TTS_AVAILABLE:
        raise RuntimeError("pocket-tts not installed. Run: pip install pocket-tts")

    print("Loading TTS model...")
    _tts_model = TTSModel.load_model()
    print("TTS model loaded!")
    return _tts_model


def get_model():
    """Get the loaded TTS model."""
    global _tts_model
    if _tts_model is None:
        load_model()
    return _tts_model


def _get_voice_state(voice: str) -> dict:
    """Get or create a voice state for the given voice name."""
    global _voice_states

    if voice in _voice_states:
        return _voice_states[voice]

    model = get_model()

    # Use predefined voice embedding (downloads from HuggingFace if needed)
    if voice in PREDEFINED_VOICES:
        print(f"Loading voice '{voice}' from HuggingFace...")
        state = model._cached_get_state_for_audio_prompt(voice, truncate=True)
        _voice_states[voice] = state
        return state
    else:
        # Fallback to default voice
        print(f"Unknown voice '{voice}', using {DEFAULT_VOICE}")
        return _get_voice_state(DEFAULT_VOICE)


def is_tts_available() -> bool:
    """Check if Pocket TTS is available."""
    return POCKET_TTS_AVAILABLE


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


# Max words per chunk passed to model.generate_audio().
# Pocket TTS internally sequences audio frames; long inputs without clear
# sentence boundaries can exceed its limit, causing silent truncation
# (the model emits a "Maximum generation length reached without EOS" warning
# and drops the tail of the chunk).  The model's own splitter targets ~27
# words/chunk.  Keeping our chunks at ≤50 words gives it at most ~2 of its
# own internal chunks per call, safely under any frame limit.
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
) -> tuple[np.ndarray, int]:
    """
    Generate speech audio from text using Pocket TTS.

    Args:
        text: Text to convert to speech
        voice: Voice name (default: alba)
        progress_callback: Optional callback for progress updates
        speed: Playback speed multiplier (0.75, 1.0, 1.25, 1.5, 2.0).
               Implemented via resampling since Pocket TTS has no native
               speed control.

    Returns:
        Tuple of (audio numpy array int16, sample rate)
    """
    model = get_model()

    # Validate voice
    if voice not in VOICES:
        if progress_callback:
            progress_callback(f"Unknown voice '{voice}', using {DEFAULT_VOICE}")
        voice = DEFAULT_VOICE

    if progress_callback:
        progress_callback(f"Generating speech with {voice}...")

    # Get voice state
    voice_state = _get_voice_state(voice)

    # Pre-chunk text to avoid Pocket TTS autoregressive sequence length overflow.
    # The model hard-limits at 1000 audio frames; long sentences without punctuation
    # produce a single oversized chunk that crashes generation.
    chunks = _split_text_into_chunks(text)

    # 100ms silence between chunks for natural pacing
    silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.int16)

    audio_parts = []
    for chunk in chunks:
        audio_tensor = model.generate_audio(voice_state, chunk)
        audio_parts.append(_tensor_to_int16(audio_tensor))
        if len(chunks) > 1:
            audio_parts.append(silence)

    if not audio_parts:
        return np.array([], dtype=np.int16), SAMPLE_RATE

    audio_np = np.concatenate(audio_parts)

    # Apply speed adjustment via resampling
    if abs(speed - 1.0) >= 0.001:
        audio_np = _resample_audio(audio_np, speed, SAMPLE_RATE)

    return audio_np, SAMPLE_RATE


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
        audio, sr = generate_speech("This is a test of Pocket text to speech.", "alba")
        print(f"Generated {len(audio)} samples at {sr}Hz ({len(audio)/sr:.2f} seconds)")
