"""Kokoro TTS engine backend (MLX and ONNX variants).

Kokoro-82M is a lightweight, high-quality TTS model with 49 voices.
- MLX variant: ~13-18x real-time on Apple Silicon via mlx-audio
- ONNX variant: cross-platform via kokoro-onnx

Install:
  Apple Silicon:  pip install mlx-audio
  Cross-platform: pip install kokoro-onnx
"""

import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 24000

# Source for ONNX model files.  The Hugging Face repo these used to come from
# (hexgrad/Kokoro-82M-ONNX) is no longer publicly available; the kokoro-onnx
# project's own GitHub release is the canonical source.
KOKORO_ONNX_RELEASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
KOKORO_ONNX_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_ONNX_VOICES_FILE = "voices-v1.0.bin"

KOKORO_VOICES = {
    "heart":   {"char": "Warm American",      "gender": "F", "lang": "en-us", "id": "af_heart",   "best_for": "narration"},
    "bella":   {"char": "Expressive American", "gender": "F", "lang": "en-us", "id": "af_bella",   "best_for": "narration"},
    "sarah":   {"char": "Clear American",      "gender": "F", "lang": "en-us", "id": "af_sarah",   "best_for": "narration"},
    "nova":    {"char": "Bright & Upbeat",     "gender": "F", "lang": "en-us", "id": "af_nova",    "best_for": "casual"},
    "adam":    {"char": "Confident American",  "gender": "M", "lang": "en-us", "id": "am_adam",    "best_for": "narration"},
    "michael": {"char": "Deep American",       "gender": "M", "lang": "en-us", "id": "am_michael", "best_for": "narration"},
    "emma":    {"char": "Warm British",        "gender": "F", "lang": "en-gb", "id": "bf_emma",    "best_for": "narration"},
    "george":  {"char": "Classic British",     "gender": "M", "lang": "en-gb", "id": "bm_george",  "best_for": "narration"},
}

DEFAULT_KOKORO_VOICE = "george"

# Global model instance (set by load_kokoro_model)
_kokoro_model = None

# Cached KokoroPipeline per language prefix (MLX only).  Calling the pipeline
# directly keeps voice packs cached across chunks; model.generate() resets
# them and flushes the Metal buffer cache on every call, which costs real
# time when invoked once per ~50-word chunk.
_mlx_pipelines: dict = {}


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


def _download_file(url: str, dest: Path) -> Path:
    """Download url to dest atomically, skipping if already present."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_name(dest.name + ".part")
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def ensure_onnx_files() -> tuple[Path, Path]:
    """Ensure the ONNX model and voices files exist in ~/.cache/kokoro/.

    Returns (model_path, voices_path).  Filenames match what earlier versions
    cached via huggingface_hub, so existing installs are reused as-is.
    """
    cache_dir = Path.home() / ".cache" / "kokoro"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = _download_file(
        f"{KOKORO_ONNX_RELEASE_URL}/{KOKORO_ONNX_MODEL_FILE}",
        cache_dir / KOKORO_ONNX_MODEL_FILE,
    )
    voices_path = _download_file(
        f"{KOKORO_ONNX_RELEASE_URL}/{KOKORO_ONNX_VOICES_FILE}",
        cache_dir / KOKORO_ONNX_VOICES_FILE,
    )
    return model_path, voices_path


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
        from kokoro_onnx import Kokoro

        model_path, voices_path = ensure_onnx_files()
        _kokoro_model = Kokoro(str(model_path), str(voices_path))
        print("Kokoro ONNX model loaded!")

    else:
        raise ValueError(f"Unknown Kokoro engine: {engine!r}")

    return _kokoro_model


def get_kokoro_model():
    """Return the loaded Kokoro model instance."""
    return _kokoro_model


def _get_mlx_pipeline(model, lang_prefix: str):
    """Return a cached MLX KokoroPipeline, or None to use model.generate().

    Uses the model's private pipeline accessor; if mlx-audio changes that
    API, synthesis falls back to the (slower) public generate() path.
    """
    pipe = _mlx_pipelines.get(lang_prefix)
    if pipe is None:
        try:
            pipe = model._get_pipeline(lang_prefix)
        except Exception:
            return None
        _mlx_pipelines[lang_prefix] = pipe
    return pipe


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

    mlx_pipeline = _get_mlx_pipeline(model, lang_prefix) if engine == "mlx" else None

    for idx, chunk in enumerate(chunks):
        if engine == "mlx":
            if mlx_pipeline is not None:
                # Fast path: call the pipeline directly.  model.generate()
                # resets the pipeline's voice cache and flushes the Metal
                # buffer cache on every call — per-chunk overhead we avoid by
                # keeping one pipeline warm for the whole chapter.
                parts = [
                    np.atleast_1d(np.array(r.audio).squeeze())
                    for r in mlx_pipeline(chunk, voice=kokoro_id, speed=speed)
                    if r.audio is not None
                ]
            else:
                # model.generate() is a generator yielding GenerationResult
                # objects; each result has an .audio float32 array.
                parts = [np.atleast_1d(np.array(r.audio).squeeze()) for r in model.generate(
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

    if engine == "mlx":
        # Trim the Metal buffer cache once per call (chapter) instead of the
        # per-segment flush model.generate() would do, so memory stays bounded
        # without paying reallocation cost on every chunk.
        try:
            import mlx.core as mx  # noqa: PLC0415
            mx.clear_cache()
        except Exception:
            pass

    return np.concatenate(audio_parts), SAMPLE_RATE
