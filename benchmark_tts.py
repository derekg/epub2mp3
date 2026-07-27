#!/usr/bin/env python3
"""Benchmark the active Kokoro TTS engine.

Synthesizes a fixed ~140-word passage and reports wall-clock time, seconds
of audio produced, and the real-time factor.  Run it before and after a
dependency or code change to see whether synthesis actually got faster on
your hardware:

  python benchmark_tts.py [voice]
"""

import sys
import time

from tts import DEFAULT_VOICE, KOKORO_ENGINE, generate_speech, load_model

PASSAGE = (
    "The lighthouse keeper climbed the spiral stairs for the last time that "
    "evening, counting each worn step out of habit rather than need. Below "
    "him the harbour lights flickered on one by one, and the fishing boats "
    "turned for home ahead of the weather. He had watched this same scene "
    "for thirty years, yet it never looked quite the same twice. Some nights "
    "the sea lay flat as hammered tin; other nights it threw itself against "
    "the rocks as if it held a grudge. Tonight it was calm, and the beam "
    "swept out over the water in long, patient circles. He filled the "
    "logbook in his careful hand, noted the wind, the pressure, and the "
    "tide, then set the kettle on the small stove and waited for the first "
    "stars to come out over the headland."
)


def main() -> None:
    voice = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VOICE
    print(f"Engine: {KOKORO_ENGINE}   Voice: {voice}")

    load_model()

    # First call pays one-time costs (pipeline build, voice load) — warm up.
    generate_speech("Warming up the model.", voice)

    best = None
    for i in range(2):
        start = time.perf_counter()
        audio, sample_rate = generate_speech(PASSAGE, voice)
        elapsed = time.perf_counter() - start
        audio_secs = len(audio) / sample_rate
        print(
            f"pass {i + 1}: {elapsed:6.2f}s wall for {audio_secs:5.1f}s of audio"
            f"  ({audio_secs / elapsed:5.2f}x real-time)"
        )
        best = elapsed if best is None else min(best, elapsed)

    audio_secs = len(audio) / sample_rate
    print(f"best:   {best:6.2f}s wall  ({audio_secs / best:5.2f}x real-time)")


if __name__ == "__main__":
    main()
