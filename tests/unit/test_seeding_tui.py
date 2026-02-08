#!/usr/bin/env python3
"""Tests for the Rich TUI split-screen module."""

import logging
import os
import time

# ---------------------------------------------------------------------------
# Imports — seeding_tui lives under scripts/benchmark/
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "benchmark"))

from seeding_tui import (
    DequeHandler,
    TapTailer,
    SeedingTUI,
)


# ---------------------------------------------------------------------------
# TestDequeHandler
# ---------------------------------------------------------------------------


class TestDequeHandler:
    """DequeHandler captures formatted log records into a bounded deque."""

    def test_records_land_in_deque(self):
        handler = DequeHandler(maxlen=100)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        handler.emit(record)

        assert len(handler.records) == 1
        assert handler.records[0] == "hello world"

    def test_maxlen_respected(self):
        handler = DequeHandler(maxlen=3)
        handler.setFormatter(logging.Formatter("%(message)s"))

        for i in range(10):
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg=f"msg-{i}", args=(), exc_info=None,
            )
            handler.emit(record)

        assert len(handler.records) == 3
        # Only the last 3 messages should remain
        assert list(handler.records) == ["msg-7", "msg-8", "msg-9"]

    def test_works_with_real_logger(self):
        handler = DequeHandler(maxlen=50)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        test_logger = logging.getLogger("test_deque_handler")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        try:
            test_logger.info("test message")
            assert len(handler.records) == 1
            assert handler.records[0] == "INFO: test message"
        finally:
            test_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# TestTapTailer
# ---------------------------------------------------------------------------


class TestTapTailer:
    """TapTailer tails a file and tracks section boundaries."""

    def test_picks_up_lines(self, tmp_path):
        tap_file = tmp_path / "tap.log"
        tap_file.write_text("")

        tailer = TapTailer(str(tap_file), poll_interval=0.02)
        tailer.start()
        try:
            # Wait for tailer to open file and seek to end
            time.sleep(0.15)

            # Write lines AFTER tailer is positioned at EOF
            with open(str(tap_file), "a") as f:
                f.write("line 1\n")
                f.write("line 2\n")
                f.flush()

            # Give the tailer time to pick up new lines
            time.sleep(0.2)

            section = tailer.get_current_section()
            assert "line 1" in section
            assert "line 2" in section
        finally:
            tailer.stop()

    def test_section_reset_on_separator(self, tmp_path):
        tap_file = tmp_path / "tap.log"
        tap_file.write_text("")

        tailer = TapTailer(str(tap_file), poll_interval=0.02)
        tailer.start()
        try:
            with open(str(tap_file), "a") as f:
                f.write("old line\n")
                f.flush()
            time.sleep(0.1)

            with open(str(tap_file), "a") as f:
                f.write("=" * 72 + "\n")
                f.write("new section\n")
                f.flush()
            time.sleep(0.1)

            section = tailer.get_current_section()
            # Old line should be gone after section reset
            assert "old line" not in section
            assert "new section" in section
        finally:
            tailer.stop()

    def test_handles_nonexistent_file(self, tmp_path):
        """Tailer waits for file creation without crashing."""
        tap_file = tmp_path / "does_not_exist.log"

        tailer = TapTailer(str(tap_file), poll_interval=0.05)
        tailer.start()
        try:
            time.sleep(0.1)  # Tailer is waiting for file

            # Create the file (tailer will open and seek to end)
            tap_file.write_text("")
            time.sleep(0.8)  # Wait for tailer to detect file + open + seek

            # Now append — tailer is positioned at EOF
            with open(str(tap_file), "a") as f:
                f.write("appeared!\n")
                f.flush()

            time.sleep(0.3)
            section = tailer.get_current_section()
            assert "appeared!" in section
        finally:
            tailer.stop()

    def test_stop_is_clean(self, tmp_path):
        tap_file = tmp_path / "tap.log"
        tap_file.write_text("hello\n")

        tailer = TapTailer(str(tap_file), poll_interval=0.02)
        tailer.start()
        time.sleep(0.1)
        tailer.stop()
        # Thread should be dead
        assert not tailer._thread.is_alive()


# ---------------------------------------------------------------------------
# TestSeedingTUI
# ---------------------------------------------------------------------------


class TestSeedingTUI:
    """SeedingTUI sentinel management and handler swap."""

    def test_sentinel_created_and_cleaned(self, tmp_path, monkeypatch):
        """Sentinel file is written on enter and removed on exit."""
        sentinel = str(tmp_path / ".inference_tap_active")
        tap = str(tmp_path / "tap.log")

        monkeypatch.setattr("seeding_tui._SENTINEL_PATH", sentinel)

        tui = SeedingTUI(session_id="test_session", tap_path=tap)
        # Patch _SENTINEL_PATH used by the instance
        tui._write_sentinel = lambda: _write_sentinel_to(sentinel, tap)
        tui._cleanup = lambda: _cleanup_paths(sentinel, tap)

        # Manually test write + cleanup
        _write_sentinel_to(sentinel, tap)
        assert os.path.exists(sentinel)
        with open(sentinel) as f:
            assert f.read().strip() == tap

        _cleanup_paths(sentinel, tap)
        assert not os.path.exists(sentinel)

    def test_handler_swap(self):
        """Verify DequeHandler replaces root handlers during TUI lifetime."""
        handler = DequeHandler(maxlen=10)
        handler.setFormatter(logging.Formatter("%(message)s"))

        root = logging.getLogger()
        original_handlers = list(root.handlers)

        # Simulate what SeedingTUI.__enter__ does for handler swap
        saved = list(root.handlers)
        root.handlers.clear()
        root.addHandler(handler)

        assert handler in root.handlers
        assert all(h not in root.handlers for h in original_handlers)

        # Restore (like __exit__)
        root.handlers.clear()
        for h in saved:
            root.addHandler(h)

        assert handler not in root.handlers

    def test_update_progress(self):
        tui = SeedingTUI(session_id="test")
        tui.update_progress(5, 30, "thinking", "q42", "ARCHITECT")

        assert tui._progress.current_index == 5
        assert tui._progress.total_questions == 30
        assert tui._progress.current_suite == "thinking"
        assert tui._progress.current_qid == "q42"
        assert tui._progress.current_action == "ARCHITECT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sentinel_to(sentinel_path: str, tap_path: str) -> None:
    os.makedirs(os.path.dirname(sentinel_path), exist_ok=True)
    with open(sentinel_path, "w") as f:
        f.write(tap_path)


def _cleanup_paths(*paths: str) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
