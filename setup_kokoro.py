#!/usr/bin/env python3
"""Download and verify Kokoro TTS model files before first startup.

Run once after installing dependencies:
  python setup_kokoro.py

Engine priority (same as tts.py):
  1. mlx_audio   → Kokoro MLX  (Apple Silicon, fastest)
  2. kokoro_onnx → Kokoro ONNX (cross-platform)
"""

import sys
from pathlib import Path


def detect_engine() -> str | None:
    try:
        import mlx_audio  # noqa: F401
        from mlx_audio.tts.models.kokoro import Model as _  # noqa: F401
        return "mlx"
    except (ImportError, AttributeError):
        pass
    try:
        from kokoro_onnx import Kokoro as _  # noqa: F401
        return "onnx"
    except ImportError:
        pass
    return None


def download_mlx() -> bool:
    print("Engine: Kokoro MLX (Apple Silicon)")
    print("Downloading model from mlx-community/Kokoro-82M-bf16 ...")
    try:
        import mlx_audio.tts as _mlx_tts
        import numpy as np
        model = _mlx_tts.load_model("mlx-community/Kokoro-82M-bf16")
        print("  Model loaded successfully.")

        print("Smoke-testing synthesis (short phrase) ...")
        parts = [np.array(r.audio) for r in model.generate(
            "Hello.", voice="af_heart", speed=1.0, lang_code="a"
        )]
        audio_np = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        if audio_np.size == 0:
            print("  ERROR: synthesis returned empty audio.")
            return False
        print(f"  OK — {audio_np.size / model.sample_rate:.2f}s of audio generated.")
        return True
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False


def download_onnx() -> bool:
    print("Engine: Kokoro ONNX (cross-platform)")
    cache_dir = Path.home() / ".cache" / "kokoro"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    try:
        from huggingface_hub import hf_hub_download
        from kokoro_onnx import Kokoro

        print("Downloading kokoro-v1.0.onnx ...")
        onnx_path = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M-ONNX",
            filename="kokoro-v1.0.onnx",
            local_dir=str(cache_dir),
        )
        print(f"  Saved: {onnx_path}")

        print("Downloading voices-v1.0.bin ...")
        voices_path = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M-ONNX",
            filename="voices-v1.0.bin",
            local_dir=str(cache_dir),
        )
        print(f"  Saved: {voices_path}")

        print("Loading model ...")
        model = Kokoro(onnx_path, voices_path)

        print("Smoke-testing synthesis (short phrase) ...")
        audio, sr = model.create("Hello.", voice="af_heart", speed=1.0, lang="en-us")
        import numpy as np
        audio_np = np.array(audio).squeeze()
        if audio_np.size == 0:
            print("  ERROR: synthesis returned empty audio.")
            return False
        print(f"  OK — {audio_np.size / sr:.2f}s of audio at {sr}Hz.")
        return True
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False


def main():
    print("=" * 55)
    print("  Kokoro TTS setup")
    print("=" * 55)

    engine = detect_engine()
    if engine is None:
        print(
            "\nNo Kokoro engine found. Install one of:\n"
            "  Apple Silicon:  pip install mlx-audio\n"
            "  Cross-platform: pip install kokoro-onnx huggingface_hub\n"
        )
        sys.exit(1)

    print()
    ok = download_mlx() if engine == "mlx" else download_onnx()

    print()
    if ok:
        print("Setup complete. You can now start the server:")
        print("  python -m uvicorn app:app --host 0.0.0.0 --port 8000")
        sys.exit(0)
    else:
        print("Setup failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
