"""In-memory token-bucket rate limiter middleware.

Uses per-IP token buckets with configurable sustained rate (RPM)
and burst capacity. No external dependencies.

Usage:
    from src.api.rate_limit import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware, rpm=60, burst=10)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class TokenBucket:
    """Single token bucket for one client."""

    __slots__ = ("capacity", "tokens", "refill_rate", "last_refill")

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting middleware using token buckets.

    Args:
        app: The ASGI application.
        rpm: Sustained requests per minute per IP.
        burst: Maximum burst above sustained rate.
    """

    def __init__(self, app: Any, rpm: int = 60, burst: int = 10):
        super().__init__(app)
        self.rpm = rpm
        self.burst = burst
        self.refill_rate = rpm / 60.0  # tokens per second
        self.capacity = rpm + burst
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.capacity, self.refill_rate)
        )
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 300.0  # prune stale buckets every 5 min

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _maybe_cleanup(self) -> None:
        """Periodically remove stale buckets to prevent memory growth."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_threshold = now - 600.0  # 10 min idle
        stale_keys = [
            k for k, v in self._buckets.items()
            if v.last_refill < stale_threshold
        ]
        for k in stale_keys:
            del self._buckets[k]

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/healthz", "/ready"):
            return await call_next(request)

        self._maybe_cleanup()

        client_ip = self._get_client_ip(request)
        bucket = self._buckets[client_ip]

        if not bucket.consume():
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(int(60 / max(self.rpm, 1)))},
            )

        return await call_next(request)
