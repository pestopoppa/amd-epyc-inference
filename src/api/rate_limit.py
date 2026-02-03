"""In-memory token-bucket rate limiter middleware.

Uses per-IP token buckets with configurable sustained rate (RPM)
and burst capacity. No external dependencies.

Implemented as a pure ASGI middleware (not BaseHTTPMiddleware) to
avoid the known Starlette body-consumption deadlock on POST requests.

Usage:
    from src.api.rate_limit import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware, rpm=60, burst=10)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

from starlette.types import ASGIApp, Receive, Scope, Send


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


_SKIP_PATHS = frozenset({"/health", "/healthz", "/ready"})

_RATE_LIMIT_BODY = json.dumps({"detail": "Rate limit exceeded. Try again later."}).encode()


class RateLimitMiddleware:
    """Pure-ASGI per-IP rate limiting middleware using token buckets.

    Args:
        app: The ASGI application.
        rpm: Sustained requests per minute per IP.
        burst: Maximum burst above sustained rate.
        trust_proxy: Whether to trust X-Forwarded-For header.
    """

    def __init__(
        self,
        app: ASGIApp,
        rpm: int = 60,
        burst: int = 10,
        trust_proxy: bool = True,
    ):
        self.app = app
        self.rpm = rpm
        self.burst = burst
        self.refill_rate = rpm / 60.0  # tokens per second
        self.capacity = rpm + burst
        self._trust_proxy = trust_proxy
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.capacity, self.refill_rate)
        )
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 300.0  # prune stale buckets every 5 min
        self._retry_after = str(int(60 / max(rpm, 1))).encode()

    def _get_client_ip(self, scope: Scope) -> str:
        """Extract client IP from ASGI scope.

        When trust_proxy=True (default), the first IP from x-forwarded-for
        is used.  This requires a reverse proxy that strips untrusted
        x-forwarded-for headers; otherwise clients can spoof their IP to
        bypass rate limits.  Set trust_proxy=False for direct exposure.
        """
        if self._trust_proxy:
            headers = dict(scope.get("headers", []))
            forwarded = headers.get(b"x-forwarded-for")
            if forwarded:
                return forwarded.decode().split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"

    def _maybe_cleanup(self) -> None:
        """Periodically remove stale buckets to prevent memory growth."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_threshold = now - 600.0  # 10 min idle
        stale_keys = [k for k, v in self._buckets.items() if v.last_refill < stale_threshold]
        for k in stale_keys:
            del self._buckets[k]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Skip rate limiting for health checks
        if path in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        self._maybe_cleanup()

        client_ip = self._get_client_ip(scope)
        bucket = self._buckets[client_ip]

        if not bucket.consume():
            # Send 429 response directly
            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"retry-after", self._retry_after],
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": _RATE_LIMIT_BODY,
                }
            )
            return

        await self.app(scope, receive, send)
