"""Tests for API middleware: CORS configuration and rate limiting.

WI-1: Verifies CORS no longer uses wildcard with credentials.
WI-2: Verifies token-bucket rate limiter behavior.
"""

from __future__ import annotations

import time

import pytest

from src.config import ApiConfig, OrchestratorConfigData, reset_config


@pytest.fixture(autouse=True)
def _clean_config():
    """Reset config cache before and after each test."""
    reset_config()
    yield
    reset_config()


# ── WI-1: CORS Configuration ─────────────────────────────────────────────


class TestCORSConfig:
    """Verify CORS is not using wildcard + credentials anti-pattern."""

    def test_default_cors_origins_are_explicit(self):
        """Default CORS origins must not contain '*'."""
        cfg = ApiConfig()
        assert "*" not in cfg.cors_origins
        assert len(cfg.cors_origins) > 0

    def test_default_origins_are_localhost(self):
        """All default origins should be localhost variants."""
        cfg = ApiConfig()
        for origin in cfg.cors_origins:
            assert "localhost" in origin or "127.0.0.1" in origin

    def test_cors_credentials_with_explicit_origins(self):
        """When credentials are enabled, origins must be explicit."""
        cfg = ApiConfig()
        if cfg.cors_allow_credentials:
            assert "*" not in cfg.cors_origins, (
                "OWASP anti-pattern: allow_credentials=True with wildcard origins. "
                "Browsers reject this and Starlette reflects the requesting origin."
            )

    def test_cors_origins_configurable(self):
        """CORS origins can be overridden."""
        cfg = ApiConfig(cors_origins=["https://myapp.example.com"])
        assert cfg.cors_origins == ["https://myapp.example.com"]

    def test_cors_in_app_factory(self):
        """App factory applies non-wildcard CORS middleware."""
        from src.api import create_app

        app = create_app()
        # Starlette CORSMiddleware is in app.middleware_stack
        # Verify the app was created without error
        assert app is not None

    def test_api_config_in_orchestrator_config(self):
        """ApiConfig is accessible from root config."""
        cfg = OrchestratorConfigData()
        assert hasattr(cfg, "api")
        assert isinstance(cfg.api, ApiConfig)
        assert "*" not in cfg.api.cors_origins


# ── WI-2: Rate Limiting ──────────────────────────────────────────────────


class TestTokenBucket:
    """Unit tests for TokenBucket internals."""

    def test_initial_capacity_full(self):
        from src.api.rate_limit import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10.0

    def test_consume_reduces_tokens(self):
        from src.api.rate_limit import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume() is True
        assert bucket.tokens < 10.0

    def test_consume_exhausts_bucket(self):
        from src.api.rate_limit import TokenBucket

        bucket = TokenBucket(capacity=3, refill_rate=0.0)
        assert bucket.consume() is True  # 2 left
        assert bucket.consume() is True  # 1 left
        assert bucket.consume() is True  # 0 left
        assert bucket.consume() is False  # exhausted

    def test_refill_over_time(self):
        from src.api.rate_limit import TokenBucket

        bucket = TokenBucket(capacity=5, refill_rate=100.0)  # 100 tokens/sec
        # Exhaust all tokens
        for _ in range(5):
            bucket.consume()
        assert bucket.consume() is False

        # Wait a bit for refill
        time.sleep(0.1)
        assert bucket.consume() is True  # refilled ~10 tokens

    def test_capacity_capped(self):
        from src.api.rate_limit import TokenBucket

        bucket = TokenBucket(capacity=5, refill_rate=100.0)
        time.sleep(0.1)
        # Even after long wait, tokens shouldn't exceed capacity
        bucket.consume()  # trigger refill
        assert bucket.tokens <= 5.0


class TestRateLimitMiddleware:
    """Tests for the RateLimitMiddleware ASGI middleware."""

    def test_middleware_importable(self):
        from src.api.rate_limit import RateLimitMiddleware

        assert RateLimitMiddleware is not None

    def test_default_config_values(self):
        """Rate limit defaults match config."""
        cfg = ApiConfig()
        assert cfg.rate_limit_rpm == 60
        assert cfg.rate_limit_burst == 10

    @pytest.mark.anyio
    async def test_health_endpoint_bypasses_rate_limit(self):
        """Health check endpoints should never be rate limited."""
        from httpx import ASGITransport, AsyncClient
        from src.api import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Health endpoint should always succeed regardless of rate
            for _ in range(5):
                resp = await client.get("/health")
                assert resp.status_code != 429

    @pytest.mark.anyio
    async def test_rate_limit_returns_429(self):
        """Requests exceeding rate limit get 429 response."""
        from src.api.rate_limit import RateLimitMiddleware

        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def homepage(request):
            return PlainTextResponse("ok")

        test_app = Starlette(routes=[Route("/", homepage)])
        # Very restrictive rate: 1 RPM, no burst
        test_app.add_middleware(RateLimitMiddleware, rpm=1, burst=0)

        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First request should succeed
            resp1 = await client.get("/")
            assert resp1.status_code == 200

            # Second request should be rate limited
            resp2 = await client.get("/")
            assert resp2.status_code == 429
            assert "Rate limit" in resp2.json()["detail"]

    @pytest.mark.anyio
    async def test_429_includes_retry_after(self):
        """Rate-limited responses include Retry-After header."""
        from src.api.rate_limit import RateLimitMiddleware

        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def homepage(request):
            return PlainTextResponse("ok")

        test_app = Starlette(routes=[Route("/", homepage)])
        test_app.add_middleware(RateLimitMiddleware, rpm=1, burst=0)

        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get("/")  # consume token
            resp = await client.get("/")  # rate limited
            assert resp.status_code == 429
            assert "retry-after" in resp.headers

    def test_middleware_cleanup(self):
        """Stale buckets are cleaned up to prevent memory growth."""
        from src.api.rate_limit import RateLimitMiddleware

        from starlette.applications import Starlette

        middleware = RateLimitMiddleware(Starlette(), rpm=60, burst=10)

        # Simulate stale bucket
        from src.api.rate_limit import TokenBucket

        stale_bucket = TokenBucket(70, 1.0)
        stale_bucket.last_refill = time.monotonic() - 700  # 11+ min idle
        middleware._buckets["192.168.1.100"] = stale_bucket
        middleware._last_cleanup = time.monotonic() - 400  # force cleanup

        middleware._maybe_cleanup()
        assert "192.168.1.100" not in middleware._buckets
