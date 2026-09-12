"""Adaptive throttling based on whether the user is actively using the Mac.

TTS generation is CPU/GPU-heavy (MLX on the Metal queue). Running it flat
out while someone is using the machine for other things makes everything
else sluggish. This module answers "is anyone actually at the keyboard
right now?" using macOS's HID idle time, and the conversion pipeline uses
that to pace itself: full speed when the machine is idle, brief pauses
between TTS chunks when it isn't.

Env vars (all optional):
  EPUB2MP3_ADAPTIVE_THROTTLE       "0" to disable entirely (default: on)
  EPUB2MP3_IDLE_THRESHOLD_SECONDS  seconds of no input before we call the
                                    machine "idle" (default: 120)
  EPUB2MP3_ACTIVE_THROTTLE_SLEEP   seconds to pause between chunks while
                                    the machine is actively in use
                                    (default: 0.4)
"""

import os
import re
import subprocess
import time

ENABLED = os.environ.get("EPUB2MP3_ADAPTIVE_THROTTLE", "1") != "0"
IDLE_THRESHOLD_SECONDS = float(os.environ.get("EPUB2MP3_IDLE_THRESHOLD_SECONDS", 120))
ACTIVE_THROTTLE_SLEEP = float(os.environ.get("EPUB2MP3_ACTIVE_THROTTLE_SLEEP", 0.4))

# Avoid spawning `ioreg` on every single TTS chunk (they can complete in a
# couple seconds); a short cache keeps the overhead negligible.
_IDLE_CACHE_TTL_SECONDS = 2.0

_HID_IDLE_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')

_idle_cache_value = 0.0
_idle_cache_at = 0.0


def _parse_hid_idle_time(ioreg_output: str) -> float | None:
    """Extract HIDIdleTime (nanoseconds) from `ioreg -c IOHIDSystem` output
    and convert to seconds. Returns None if the field wasn't found."""
    match = _HID_IDLE_RE.search(ioreg_output)
    if not match:
        return None
    return int(match.group(1)) / 1_000_000_000


def get_idle_seconds() -> float:
    """Seconds since the last keyboard/mouse/trackpad event.

    Returns 0.0 (i.e. "assume active") on non-macOS systems or if the
    underlying `ioreg` call fails, so throttling fails safe rather than
    silently running unthrottled.
    """
    global _idle_cache_value, _idle_cache_at
    now = time.monotonic()
    if now - _idle_cache_at < _IDLE_CACHE_TTL_SECONDS:
        return _idle_cache_value

    idle = None
    try:
        result = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True, text=True, timeout=2,
        )
        idle = _parse_hid_idle_time(result.stdout)
    except (OSError, subprocess.SubprocessError):
        idle = None

    _idle_cache_value = idle if idle is not None else 0.0
    _idle_cache_at = now
    return _idle_cache_value


def is_system_active(idle_seconds: float, threshold: float = IDLE_THRESHOLD_SECONDS) -> bool:
    """True if someone appears to be actively using the machine right now."""
    return idle_seconds < threshold


def throttle_pause() -> bool:
    """Briefly yield the CPU/GPU if the machine looks actively in use.

    Meant to be called between TTS chunks (every ~50 words / few seconds).
    Returns True if it paused.
    """
    if not ENABLED:
        return False
    if is_system_active(get_idle_seconds()):
        time.sleep(ACTIVE_THROTTLE_SLEEP)
        return True
    return False


def get_status() -> dict:
    """Snapshot of current throttle state, for surfacing in the UI/API."""
    idle = get_idle_seconds()
    active = is_system_active(idle)
    return {
        "adaptive_throttle_enabled": ENABLED,
        "idle_seconds": round(idle, 1),
        "idle_threshold_seconds": IDLE_THRESHOLD_SECONDS,
        "system_active": active,
        "throttled": ENABLED and active,
    }
