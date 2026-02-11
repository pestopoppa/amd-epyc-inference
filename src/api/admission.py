"""Per-backend admission control using asyncio semaphores.

Prevents request pile-up on slow backends (especially architects) by limiting
concurrent in-flight requests per backend URL. When a backend's queue is full,
requests are rejected immediately with a clear error rather than blocking
until timeout.

Usage:
    controller = AdmissionController.from_defaults()
    if not controller.try_acquire(backend_url):
        raise RuntimeError("Backend queue full")
    try:
        result = do_inference(backend_url, ...)
    finally:
        controller.release(backend_url)
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# Default concurrency limits per backend URL.
# Architects (serial, huge models) = 1; workers = 4; vision = 2; embedders = 4.
DEFAULT_LIMITS: dict[str, int] = {
    "http://localhost:8080": 2,   # frontdoor / coder_primary (30B MoE)
    "http://localhost:8081": 2,   # coder_escalation / worker_summarize (32B)
    "http://localhost:8082": 4,   # worker_explore (7B, 2 slots)
    "http://localhost:8083": 1,   # architect_general (235B) — SERIAL
    "http://localhost:8084": 1,   # architect_coding (480B) — SERIAL
    "http://localhost:8085": 1,   # ingest_long_context (80B SSM) — SERIAL
    "http://localhost:8086": 2,   # worker_vision (7B VL)
    "http://localhost:8087": 1,   # vision_escalation (30B VL MoE)
    "http://localhost:8102": 4,   # worker_fast (1.5B, 4 slots)
}

# Embedding servers (8090-8095) are not gated — they're lightweight.


class AdmissionController:
    """Thread-safe per-backend concurrency limiter.

    Uses threading.Semaphore (not asyncio.Semaphore) because inference calls
    happen in thread pool via asyncio.to_thread / inference_lock (fcntl).
    """

    def __init__(self, limits: dict[str, int] | None = None):
        self._limits = limits or DEFAULT_LIMITS
        self._semaphores: dict[str, threading.Semaphore] = {
            url: threading.Semaphore(n) for url, n in self._limits.items()
        }
        self._lock = threading.Lock()

    @classmethod
    def from_defaults(cls) -> AdmissionController:
        return cls(DEFAULT_LIMITS)

    def try_acquire(self, backend_url: str) -> bool:
        """Non-blocking acquire. Returns True if admitted, False if queue full."""
        sem = self._semaphores.get(backend_url)
        if sem is None:
            # Unknown backend — no limit applied
            return True
        acquired = sem.acquire(blocking=False)
        if not acquired:
            logger.warning(
                "Admission rejected for %s (limit=%d)",
                backend_url,
                self._limits.get(backend_url, 0),
            )
        return acquired

    def release(self, backend_url: str) -> None:
        """Release a slot for the backend."""
        sem = self._semaphores.get(backend_url)
        if sem is not None:
            sem.release()

    def get_status(self) -> dict[str, dict[str, int]]:
        """Return current admission status per backend."""
        status = {}
        for url, sem in self._semaphores.items():
            limit = self._limits.get(url, 0)
            # Semaphore._value is the current counter (available slots)
            available = sem._value  # type: ignore[attr-defined]
            status[url] = {
                "limit": limit,
                "available": available,
                "in_flight": limit - available,
            }
        return status
