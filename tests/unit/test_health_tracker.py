#!/usr/bin/env python3
"""Tests for BackendHealthTracker circuit breaker.

Covers: state transitions, cooldown timing, exponential backoff,
thread safety, and status snapshots.
"""

import threading
import time
from unittest.mock import patch

import pytest

from src.api.health_tracker import (
    BackendCircuit,
    BackendHealthTracker,
    DEFAULT_COOLDOWN_S,
    DEFAULT_FAILURE_THRESHOLD,
    MAX_COOLDOWN_S,
)


URL_A = "http://localhost:8080"
URL_B = "http://localhost:8081"


class TestBackendCircuitDefaults:
    """Test BackendCircuit dataclass defaults."""

    def test_defaults(self):
        c = BackendCircuit()
        assert c.state == "closed"
        assert c.failure_count == 0
        assert c.cooldown_s == DEFAULT_COOLDOWN_S
        assert c.failure_threshold == DEFAULT_FAILURE_THRESHOLD


class TestInitialState:
    """Test tracker behavior with no recorded events."""

    def test_unknown_backend_is_available(self):
        tracker = BackendHealthTracker()
        assert tracker.is_available(URL_A) is True

    def test_empty_status(self):
        tracker = BackendHealthTracker()
        assert tracker.get_status() == {}

    def test_reset_clears_circuits(self):
        tracker = BackendHealthTracker()
        tracker.record_success(URL_A)
        assert len(tracker.get_status()) == 1
        tracker.reset()
        assert tracker.get_status() == {}


class TestClosedCircuit:
    """Test behavior in the closed (healthy) state."""

    def test_single_failure_stays_closed(self):
        tracker = BackendHealthTracker(failure_threshold=3)
        tracker.record_failure(URL_A)
        assert tracker.is_available(URL_A) is True
        status = tracker.get_status()
        assert status[URL_A]["state"] == "closed"
        assert status[URL_A]["failure_count"] == 1

    def test_two_failures_stays_closed(self):
        tracker = BackendHealthTracker(failure_threshold=3)
        tracker.record_failure(URL_A)
        tracker.record_failure(URL_A)
        assert tracker.is_available(URL_A) is True
        assert tracker.get_status()[URL_A]["state"] == "closed"

    def test_success_resets_failure_count(self):
        tracker = BackendHealthTracker(failure_threshold=3)
        tracker.record_failure(URL_A)
        tracker.record_failure(URL_A)
        tracker.record_success(URL_A)
        assert tracker.get_status()[URL_A]["failure_count"] == 0
        assert tracker.get_status()[URL_A]["state"] == "closed"


class TestCircuitOpening:
    """Test transition from closed → open."""

    def test_threshold_opens_circuit(self):
        tracker = BackendHealthTracker(failure_threshold=3)
        for _ in range(3):
            tracker.record_failure(URL_A)
        assert tracker.is_available(URL_A) is False
        assert tracker.get_status()[URL_A]["state"] == "open"

    def test_open_circuit_fast_fails(self):
        tracker = BackendHealthTracker(failure_threshold=2)
        tracker.record_failure(URL_A)
        tracker.record_failure(URL_A)
        # Circuit is open — fast-fail
        assert tracker.is_available(URL_A) is False


class TestCooldownAndHalfOpen:
    """Test open → half-open → closed transitions."""

    def test_cooldown_transitions_to_half_open(self):
        tracker = BackendHealthTracker(failure_threshold=2, cooldown_s=0.1)
        tracker.record_failure(URL_A)
        tracker.record_failure(URL_A)
        assert tracker.is_available(URL_A) is False

        # Wait for cooldown
        time.sleep(0.15)
        assert tracker.is_available(URL_A) is True
        assert tracker.get_status()[URL_A]["state"] == "half-open"

    def test_half_open_success_closes(self):
        tracker = BackendHealthTracker(failure_threshold=2, cooldown_s=0.1)
        tracker.record_failure(URL_A)
        tracker.record_failure(URL_A)

        time.sleep(0.15)
        tracker.is_available(URL_A)  # Transition to half-open
        tracker.record_success(URL_A)

        assert tracker.get_status()[URL_A]["state"] == "closed"
        assert tracker.get_status()[URL_A]["failure_count"] == 0

    def test_half_open_failure_reopens_with_doubled_cooldown(self):
        tracker = BackendHealthTracker(failure_threshold=2, cooldown_s=0.1)
        tracker.record_failure(URL_A)
        tracker.record_failure(URL_A)

        time.sleep(0.15)
        tracker.is_available(URL_A)  # Transition to half-open
        tracker.record_failure(URL_A)

        status = tracker.get_status()[URL_A]
        assert status["state"] == "open"
        assert status["cooldown_s"] == pytest.approx(0.2, abs=0.01)

    def test_cooldown_caps_at_max(self):
        tracker = BackendHealthTracker(failure_threshold=1, cooldown_s=200.0)
        tracker.record_failure(URL_A)
        # Force half-open by setting last_failure in past
        with tracker._lock:
            tracker._circuits[URL_A].last_failure = time.monotonic() - 300
        tracker.is_available(URL_A)  # half-open
        tracker.record_failure(URL_A)  # back to open, cooldown doubled

        status = tracker.get_status()[URL_A]
        assert status["cooldown_s"] <= MAX_COOLDOWN_S

    def test_success_resets_cooldown_to_default(self):
        tracker = BackendHealthTracker(failure_threshold=2, cooldown_s=0.1)
        tracker.record_failure(URL_A)
        tracker.record_failure(URL_A)

        time.sleep(0.15)
        tracker.is_available(URL_A)  # half-open
        tracker.record_failure(URL_A)  # doubled cooldown

        # Now recover
        with tracker._lock:
            tracker._circuits[URL_A].last_failure = time.monotonic() - 1.0
        tracker.is_available(URL_A)  # half-open again
        tracker.record_success(URL_A)

        assert tracker.get_status()[URL_A]["cooldown_s"] == pytest.approx(0.1, abs=0.01)


class TestMultipleBackends:
    """Test independent tracking of multiple backends."""

    def test_independent_circuits(self):
        tracker = BackendHealthTracker(failure_threshold=2)
        tracker.record_failure(URL_A)
        tracker.record_failure(URL_A)
        tracker.record_success(URL_B)

        assert tracker.is_available(URL_A) is False
        assert tracker.is_available(URL_B) is True

    def test_status_shows_all_backends(self):
        tracker = BackendHealthTracker()
        tracker.record_success(URL_A)
        tracker.record_failure(URL_B)

        status = tracker.get_status()
        assert URL_A in status
        assert URL_B in status


class TestThreadSafety:
    """Test concurrent access doesn't corrupt state."""

    def test_concurrent_record_calls(self):
        tracker = BackendHealthTracker(failure_threshold=100)
        errors = []

        def record_many(url: str, n: int):
            try:
                for _ in range(n):
                    tracker.record_failure(url)
                    tracker.record_success(url)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_many, args=(URL_A, 100)),
            threading.Thread(target=record_many, args=(URL_A, 100)),
            threading.Thread(target=record_many, args=(URL_B, 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Both URLs should be in status
        status = tracker.get_status()
        assert URL_A in status
        assert URL_B in status


class TestGetStatusSnapshot:
    """Test get_status returns complete snapshot."""

    def test_status_contains_expected_fields(self):
        tracker = BackendHealthTracker()
        tracker.record_failure(URL_A)

        status = tracker.get_status()
        entry = status[URL_A]
        assert "state" in entry
        assert "failure_count" in entry
        assert "cooldown_s" in entry
        assert "last_failure" in entry
        assert "last_success" in entry

    def test_status_is_snapshot_not_reference(self):
        tracker = BackendHealthTracker()
        tracker.record_failure(URL_A)
        status1 = tracker.get_status()
        tracker.record_failure(URL_A)
        status2 = tracker.get_status()

        # status1 should not be affected by later changes
        assert status1[URL_A]["failure_count"] == 1
        assert status2[URL_A]["failure_count"] == 2
