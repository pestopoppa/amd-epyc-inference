"""Backend health tracking with circuit breaker pattern.

Tracks per-backend health state to prevent cascading failures when
llama-server instances become unhealthy (KV cache pressure, OOM, etc.).

Circuit states:
    closed ──(N failures)──> open ──(cooldown)──> half-open ──(success)──> closed
                                                       │
                                                  (failure)
                                                       └──> open (double cooldown)

Usage:
    tracker = BackendHealthTracker()

    # Before dispatch
    if not tracker.is_available("http://localhost:8080"):
        raise RuntimeError("Backend circuit open")

    # After dispatch
    if result.success:
        tracker.record_success("http://localhost:8080")
    else:
        tracker.record_failure("http://localhost:8080")

    # Health endpoint
    status = tracker.get_status()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Circuit breaker constants
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_S = 30.0
MAX_COOLDOWN_S = 300.0


@dataclass
class BackendCircuit:
    """Health state for a single backend.

    Attributes:
        state: Circuit state — "closed" (healthy), "open" (failing), "half-open" (probing).
        failure_count: Consecutive failures since last success.
        last_failure: Timestamp of most recent failure.
        last_success: Timestamp of most recent success.
        cooldown_s: Current cooldown before half-open probe (doubles on repeated failure).
        failure_threshold: Failures required to open the circuit.
    """

    state: str = "closed"
    failure_count: int = 0
    last_failure: float = 0.0
    last_success: float = 0.0
    cooldown_s: float = DEFAULT_COOLDOWN_S
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD


class BackendHealthTracker:
    """Thread-safe circuit breaker for backend health tracking.

    Manages per-backend circuit state to fast-fail when backends are down,
    preventing cascading timeouts (300s waits on dead backends).

    All public methods are thread-safe via internal lock.
    """

    def __init__(
        self,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
    ) -> None:
        """Initialize the health tracker.

        Args:
            failure_threshold: Consecutive failures before opening circuit.
            cooldown_s: Base cooldown in seconds before probing (doubles on repeated failure).
        """
        self._circuits: dict[str, BackendCircuit] = {}
        self._lock = threading.Lock()
        self._default_threshold = failure_threshold
        self._default_cooldown = cooldown_s

    def _get_circuit(self, backend_url: str) -> BackendCircuit:
        """Get or create circuit for a backend. Must be called with lock held."""
        if backend_url not in self._circuits:
            self._circuits[backend_url] = BackendCircuit(
                failure_threshold=self._default_threshold,
                cooldown_s=self._default_cooldown,
            )
        return self._circuits[backend_url]

    def is_available(self, backend_url: str) -> bool:
        """Check if a backend can accept requests.

        Fast path: unknown backends and closed circuits return True immediately.
        Open circuits check if cooldown has elapsed → transition to half-open.

        Args:
            backend_url: The backend URL to check.

        Returns:
            True if the backend should be tried, False if circuit is open.
        """
        with self._lock:
            if backend_url not in self._circuits:
                return True

            circuit = self._circuits[backend_url]

            if circuit.state == "closed":
                return True

            if circuit.state == "half-open":
                # Allow one probe request through
                return True

            # state == "open"
            elapsed = time.monotonic() - circuit.last_failure
            if elapsed >= circuit.cooldown_s:
                # Cooldown elapsed — transition to half-open for probe
                circuit.state = "half-open"
                logger.info(
                    f"Circuit half-open for {backend_url} "
                    f"(cooldown {circuit.cooldown_s:.0f}s elapsed)"
                )
                return True

            return False

    def record_success(self, backend_url: str) -> None:
        """Record a successful request to a backend.

        Resets failure counter. Closes circuit if half-open.

        Args:
            backend_url: The backend URL that succeeded.
        """
        with self._lock:
            circuit = self._get_circuit(backend_url)

            if circuit.state == "half-open":
                logger.info(f"Circuit closed for {backend_url} (probe succeeded)")

            circuit.state = "closed"
            circuit.failure_count = 0
            circuit.last_success = time.monotonic()
            # Reset cooldown to default on success
            circuit.cooldown_s = self._default_cooldown

    def record_failure(self, backend_url: str) -> None:
        """Record a failed request to a backend.

        Increments failure counter. Opens circuit if threshold reached.
        Doubles cooldown if already open/half-open.

        Args:
            backend_url: The backend URL that failed.
        """
        with self._lock:
            circuit = self._get_circuit(backend_url)
            circuit.failure_count += 1
            circuit.last_failure = time.monotonic()

            if circuit.state == "half-open":
                # Probe failed — reopen with doubled cooldown
                circuit.state = "open"
                circuit.cooldown_s = min(circuit.cooldown_s * 2, MAX_COOLDOWN_S)
                logger.warning(
                    f"Circuit reopened for {backend_url} "
                    f"(probe failed, cooldown now {circuit.cooldown_s:.0f}s)"
                )
            elif circuit.state == "closed" and circuit.failure_count >= circuit.failure_threshold:
                circuit.state = "open"
                logger.warning(
                    f"Circuit opened for {backend_url} "
                    f"({circuit.failure_count} consecutive failures)"
                )

    def get_status(self) -> dict[str, dict]:
        """Get a snapshot of all backend circuit states.

        Returns:
            Dict mapping backend URL to circuit state info.
        """
        with self._lock:
            return {
                url: {
                    "state": c.state,
                    "failure_count": c.failure_count,
                    "cooldown_s": c.cooldown_s,
                    "last_failure": c.last_failure,
                    "last_success": c.last_success,
                }
                for url, c in self._circuits.items()
            }

    def reset(self) -> None:
        """Reset all circuits. Used in tests."""
        with self._lock:
            self._circuits.clear()
