"""Kokoro TTS engine backend (MLX and ONNX variants).

Kokoro-82M is a lightweight, high-quality TTS model with 49 voices.
- MLX variant: ~13-18x real-time on Apple Silicon via mlx-audio
- ONNX variant: cross-platform via kokoro-onnx

Install:
  Apple Silicon:  pip install mlx-audio
  Cross-platform: pip install kokoro-onnx huggingface_hub
"""

from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 24000

KOKORO_VOICES = {
    "heart":   {"char": "Warm",       "gender": "F", "lang": "en-us", "id": "af_heart",   "best_for": "narration"},
    "bella":   {"char": "Expressive", "gender": "F", "lang": "en-us", "id": "af_bella",   "best_for": "narration"},
    "sarah":   {"char": "Clear",      "gender": "F", "lang": "en-us", "id": "af_sarah",   "best_for": "narration"},
    "nova":    {"char": "Bright",     "gender": "F", "lang": "en-us", "id": "af_nova",    "best_for": "casual"},
    "adam":    {"char": "Confident",  "gender": "M", "lang": "en-us", "id": "am_adam",    "best_for": "narration"},
    "michael": {"char": "Deep",       "gender": "M", "lang": "en-us", "id": "am_michael", "best_for": "narration"},
    "emma":    {"char": "British F",  "gender": "F", "lang": "en-gb", "id": "bf_emma",    "best_for": "narration"},
    "george":  {"char": "British M",  "gender": "M", "lang": "en-gb", "id": "bm_george",  "best_for": "narration"},
}

DEFAULT_KOKORO_VOICE = "george"

# Global model instance (set by load_kokoro_model)
_kokoro_model = None


def _get_kokoro_id(voice: str) -> tuple[str, str, str]:
    """Return (kokoro_voice_id, mlx_lang_prefix, onnx_lang) for a voice name.

    mlx_lang_prefix: 'a' for en-us, 'b' for en-gb (used by mlx_audio).
    onnx_lang: 'en-us' or 'en-gb' (used by kokoro-onnx).
    """
    info = KOKORO_VOICES.get(voice, KOKORO_VOICES[DEFAULT_KOKORO_VOICE])
    kokoro_id = info["id"]
    # First char of voice ID encodes language: "af_heart" → "a", "bm_george" → "b"
    lang_prefix = kokoro_id[0]
    onnx_lang = info["lang"]   # "en-us" or "en-gb"
    return kokoro_id, lang_prefix, onnx_lang


def _to_int16(audio) -> np.ndarray:
    """Convert audio (mlx.array, torch tensor, or numpy) to int16 numpy array."""
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    audio = audio.squeeze()
    if audio.dtype != np.int16:
        if audio.dtype in (np.float32, np.float64):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        else:
            audio = audio.astype(np.int16)
    return audio


def load_kokoro_model(engine: str):
    """Load the Kokoro model for the given engine ('mlx' or 'onnx').

    Downloads model weights automatically on first run.
    MLX: fetches from mlx-community/Kokoro-82M-bf16
    ONNX: fetches kokoro-v1.0.onnx + voices-v1.0.bin into ~/.cache/kokoro/
    """
    global _kokoro_model

    if engine == "mlx":
        print("Loading Kokoro model (MLX)...")
        import mlx_audio.tts as _mlx_tts  # noqa: F401
        _kokoro_model = _mlx_tts.load_model("mlx-community/Kokoro-82M-bf16")
        print("Kokoro MLX model loaded!")

    elif engine == "onnx":
        print("Loading Kokoro model (ONNX)...")
        from huggingface_hub import hf_hub_download
        from kokoro_onnx import Kokoro

        cache_dir = Path.home() / ".cache" / "kokoro"
        cache_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M-ONNX",
            filename="kokoro-v1.0.onnx",
            local_dir=str(cache_dir),
        )
        voices_path = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M-ONNX",
            filename="voices-v1.0.bin",
            local_dir=str(cache_dir),
        )
        _kokoro_model = Kokoro(onnx_path, voices_path)
        print("Kokoro ONNX model loaded!")

    else:
        raise ValueError(f"Unknown Kokoro engine: {engine!r}")

    return _kokoro_model


def get_kokoro_model():
    """Return the loaded Kokoro model instance."""
    return _kokoro_model


def generate_speech_kokoro(
    text: str,
    voice: str,
    speed: float,
    chunk_callback: Callable[[int, int], None],
    engine: str,
) -> tuple[np.ndarray, int]:
    """Generate speech using Kokoro TTS.

    Args:
        text: Input text.
        voice: Kokoro voice name (key in KOKORO_VOICES).
        speed: Playback speed multiplier.
        chunk_callback: Called after each chunk as (done, total). May be None.
        engine: 'mlx' or 'onnx'.

    Returns:
        (audio_int16, sample_rate) tuple.
    """
    # Lazy import avoids circular dependency (tts imports kokoro_tts; kokoro_tts
    # uses _split_text_into_chunks from tts only inside function bodies).
    from tts import _split_text_into_chunks  # noqa: PLC0415

    model = get_kokoro_model()
    kokoro_id, lang_prefix, onnx_lang = _get_kokoro_id(voice)

    chunks = _split_text_into_chunks(text)
    if not chunks:
        return np.array([], dtype=np.int16), SAMPLE_RATE

    # 100ms silence between chunks for natural pacing (same as Pocket TTS)
    silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.int16)
    audio_parts = []
    chunks_total = len(chunks)

    for idx, chunk in enumerate(chunks):
        if engine == "mlx":
            # model.generate() is a generator yielding GenerationResult objects;
            # each result has an .audio float32 numpy array.
            parts = [np.array(r.audio) for r in model.generate(
                chunk, voice=kokoro_id, speed=speed, lang_code=lang_prefix
            )]
            raw = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        else:  # onnx
            # kokoro_onnx returns (numpy_float32_array, sample_rate)
            raw, _ = model.create(chunk, voice=kokoro_id, speed=speed, lang=onnx_lang)

        audio_np = _to_int16(raw)
        audio_parts.append(audio_np)

        if chunks_total > 1:
            audio_parts.append(silence)

        if chunk_callback:
            chunk_callback(idx + 1, chunks_total)

    return np.concatenate(audio_parts), SAMPLE_RATE
