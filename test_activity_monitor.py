"""Unit tests for activity_monitor: idle-time parsing and TTS throttling."""

from unittest.mock import MagicMock, patch

import activity_monitor


SAMPLE_IOREG_OUTPUT = '''
+-o Root  <class IORegistryEntry, id 0x100000100, retain 9>
  +-o IOHIDSystem  <class IOHIDSystem, id 0x100000308, retain 9>
    | | |   "HIDIdleTime" = 54908555708
    | | |   "OtherField" = 12345
'''


class TestParseHidIdleTime:
    def test_parses_nanoseconds_to_seconds(self):
        seconds = activity_monitor._parse_hid_idle_time(SAMPLE_IOREG_OUTPUT)
        assert seconds == 54908555708 / 1_000_000_000

    def test_returns_none_when_field_missing(self):
        assert activity_monitor._parse_hid_idle_time("no such field here") is None

    def test_returns_none_on_empty_string(self):
        assert activity_monitor._parse_hid_idle_time("") is None


class TestIsSystemActive:
    def test_recently_used_is_active(self):
        assert activity_monitor.is_system_active(5.0, threshold=120) is True

    def test_long_idle_is_not_active(self):
        assert activity_monitor.is_system_active(300.0, threshold=120) is False

    def test_boundary_is_not_active(self):
        """idle_seconds exactly at the threshold counts as idle, not active."""
        assert activity_monitor.is_system_active(120.0, threshold=120) is False


class TestGetIdleSeconds:
    def setup_method(self):
        # Force a cache miss on every test regardless of run order/timing.
        activity_monitor._idle_cache_at = 0.0

    def test_reads_from_ioreg(self):
        fake_result = MagicMock(stdout=SAMPLE_IOREG_OUTPUT)
        with patch("activity_monitor.subprocess.run", return_value=fake_result) as mock_run:
            seconds = activity_monitor.get_idle_seconds()
        assert seconds == 54908555708 / 1_000_000_000
        mock_run.assert_called_once()

    def test_caches_within_ttl(self):
        fake_result = MagicMock(stdout=SAMPLE_IOREG_OUTPUT)
        with patch("activity_monitor.subprocess.run", return_value=fake_result) as mock_run:
            activity_monitor.get_idle_seconds()
            activity_monitor.get_idle_seconds()
            activity_monitor.get_idle_seconds()
        assert mock_run.call_count == 1, "repeated calls within the TTL must not re-invoke ioreg"

    def test_defaults_to_zero_on_subprocess_failure(self):
        with patch("activity_monitor.subprocess.run", side_effect=OSError("no ioreg")):
            seconds = activity_monitor.get_idle_seconds()
        assert seconds == 0.0

    def teardown_method(self):
        activity_monitor._idle_cache_at = 0.0
        activity_monitor._idle_cache_value = 0.0


class TestThrottlePause:
    def test_sleeps_when_active(self):
        with patch("activity_monitor.ENABLED", True), \
             patch("activity_monitor.ACTIVE_THROTTLE_SLEEP", 0.4), \
             patch("activity_monitor.get_idle_seconds", return_value=1.0), \
             patch("activity_monitor.IDLE_THRESHOLD_SECONDS", 120), \
             patch("activity_monitor.time.sleep") as mock_sleep:
            paused = activity_monitor.throttle_pause()
        assert paused is True
        mock_sleep.assert_called_once_with(0.4)

    def test_no_sleep_when_idle(self):
        with patch("activity_monitor.ENABLED", True), \
             patch("activity_monitor.get_idle_seconds", return_value=999.0), \
             patch("activity_monitor.IDLE_THRESHOLD_SECONDS", 120), \
             patch("activity_monitor.time.sleep") as mock_sleep:
            paused = activity_monitor.throttle_pause()
        assert paused is False
        mock_sleep.assert_not_called()

    def test_no_sleep_when_disabled(self):
        with patch("activity_monitor.ENABLED", False), \
             patch("activity_monitor.get_idle_seconds", return_value=1.0), \
             patch("activity_monitor.time.sleep") as mock_sleep:
            paused = activity_monitor.throttle_pause()
        assert paused is False
        mock_sleep.assert_not_called()


class TestGetStatus:
    def test_reports_throttled_when_active_and_enabled(self):
        with patch("activity_monitor.ENABLED", True), \
             patch("activity_monitor.get_idle_seconds", return_value=1.0), \
             patch("activity_monitor.IDLE_THRESHOLD_SECONDS", 120):
            status = activity_monitor.get_status()
        assert status["throttled"] is True
        assert status["system_active"] is True

    def test_not_throttled_when_idle(self):
        with patch("activity_monitor.ENABLED", True), \
             patch("activity_monitor.get_idle_seconds", return_value=999.0), \
             patch("activity_monitor.IDLE_THRESHOLD_SECONDS", 120):
            status = activity_monitor.get_status()
        assert status["throttled"] is False
        assert status["system_active"] is False

    def test_not_throttled_when_disabled_even_if_active(self):
        with patch("activity_monitor.ENABLED", False), \
             patch("activity_monitor.get_idle_seconds", return_value=1.0), \
             patch("activity_monitor.IDLE_THRESHOLD_SECONDS", 120):
            status = activity_monitor.get_status()
        assert status["throttled"] is False
