"""Tests for AdmissionController per-backend concurrency limiter."""

import threading

from src.api.admission import AdmissionController


class TestAdmissionController:
    """Tests for thread-safe admission control."""

    def test_unknown_backend_always_admitted(self):
        ctrl = AdmissionController({"http://localhost:8080": 1})
        assert ctrl.try_acquire("http://unknown:9999") is True

    def test_acquire_within_limit(self):
        ctrl = AdmissionController({"http://localhost:8080": 2})
        assert ctrl.try_acquire("http://localhost:8080") is True
        assert ctrl.try_acquire("http://localhost:8080") is True

    def test_acquire_beyond_limit_rejected(self):
        ctrl = AdmissionController({"http://localhost:8083": 1})
        assert ctrl.try_acquire("http://localhost:8083") is True
        # Second acquire should be rejected (limit=1)
        assert ctrl.try_acquire("http://localhost:8083") is False

    def test_release_allows_new_acquire(self):
        ctrl = AdmissionController({"http://localhost:8083": 1})
        assert ctrl.try_acquire("http://localhost:8083") is True
        assert ctrl.try_acquire("http://localhost:8083") is False
        ctrl.release("http://localhost:8083")
        assert ctrl.try_acquire("http://localhost:8083") is True

    def test_status_shows_in_flight(self):
        ctrl = AdmissionController({"http://localhost:8084": 2})
        ctrl.try_acquire("http://localhost:8084")
        status = ctrl.get_status()
        assert status["http://localhost:8084"]["limit"] == 2
        assert status["http://localhost:8084"]["in_flight"] == 1
        assert status["http://localhost:8084"]["available"] == 1

    def test_from_defaults_creates_controller(self):
        ctrl = AdmissionController.from_defaults()
        # Architect backends should have limit=1
        status = ctrl.get_status()
        assert status["http://localhost:8083"]["limit"] == 1
        assert status["http://localhost:8084"]["limit"] == 1
        # Worker backend limit follows concurrent sweep defaults (serialized).
        assert status["http://localhost:8082"]["limit"] == 1

    def test_thread_safety(self):
        """Concurrent acquire/release should not corrupt state."""
        ctrl = AdmissionController({"http://localhost:8080": 2})
        errors = []

        def worker():
            try:
                for _ in range(100):
                    if ctrl.try_acquire("http://localhost:8080"):
                        ctrl.release("http://localhost:8080")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All slots should be available after all threads complete
        status = ctrl.get_status()
        assert status["http://localhost:8080"]["in_flight"] == 0
