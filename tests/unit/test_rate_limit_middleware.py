"""Unit tests for src.api.rate_limit."""

from __future__ import annotations

import time
from unittest.mock import Mock

from src.api.rate_limit import RateLimitMiddleware


class TestRateLimitMiddlewareConfig:
    """Validate cleanup-related tunables are wired correctly."""

    def test_cleanup_threshold_uses_constructor_overrides(self) -> None:
        middleware = RateLimitMiddleware(
            app=Mock(),
            cleanup_interval_seconds=1.25,
            stale_bucket_ttl_seconds=2.5,
        )
        assert middleware._cleanup_interval == 1.25
        assert middleware._stale_bucket_ttl_seconds == 2.5

    def test_cleanup_prunes_stale_buckets_using_ttl(self) -> None:
        middleware = RateLimitMiddleware(
            app=Mock(),
            cleanup_interval_seconds=0.0,
            stale_bucket_ttl_seconds=0.0,
        )
        bucket = middleware._buckets["127.0.0.1"]
        bucket.last_refill = time.monotonic() - 10.0

        middleware._maybe_cleanup()

        assert "127.0.0.1" not in middleware._buckets
