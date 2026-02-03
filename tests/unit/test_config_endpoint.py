"""Tests for the runtime config endpoint (src/api/routes/config.py)."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import dep_features
from src.api.routes.config import router
from src.features import Features


def _make_app():
    """Create a minimal FastAPI app with the config router."""
    app = FastAPI()
    app.include_router(router)
    return app


class _LocalhostMiddleware:
    """ASGI middleware that overrides client host to 127.0.0.1 for testing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope["client"] = ("127.0.0.1", 0)
        await self.app(scope, receive, send)


class TestUpdateConfig:
    """Test POST /config endpoint."""

    def test_localhost_allowed(self):
        """Config changes from localhost should succeed."""
        app = _make_app()
        test_features = Features(memrl=False, tools=True)
        app.dependency_overrides[dep_features] = lambda: test_features
        # Wrap with middleware to fake localhost
        app = _LocalhostMiddleware(app)

        with patch("src.api.routes.config.set_features"):
            client = TestClient(app)
            response = client.post("/config", json={"memrl": True})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "features" in data

    def test_localhost_updates_features(self):
        """Config should update feature flags and return new state."""
        app = _make_app()
        test_features = Features(memrl=False, tools=True)
        app.dependency_overrides[dep_features] = lambda: test_features
        app = _LocalhostMiddleware(app)

        with patch("src.api.routes.config.set_features") as mock_set:
            client = TestClient(app)
            response = client.post("/config", json={"memrl": True})

        assert response.status_code == 200
        data = response.json()
        assert data["features"]["memrl"] is True
        assert data["features"]["tools"] is True
        mock_set.assert_called_once()

    def test_remote_blocked(self):
        """Config changes from non-localhost should be rejected."""
        app = _make_app()
        test_features = Features()
        app.dependency_overrides[dep_features] = lambda: test_features

        # TestClient uses 'testclient' as host — should be blocked
        client = TestClient(app)
        response = client.post("/config", json={"memrl": True})

        assert response.status_code == 403
        assert "localhost" in response.json()["detail"]

    def test_unknown_keys_ignored(self):
        """Unknown feature keys should be silently ignored."""
        app = _make_app()
        test_features = Features(memrl=False, tools=True)
        app.dependency_overrides[dep_features] = lambda: test_features
        app = _LocalhostMiddleware(app)

        with patch("src.api.routes.config.set_features"):
            client = TestClient(app)
            response = client.post(
                "/config",
                json={"memrl": True, "nonexistent_flag": True},
            )

        assert response.status_code == 200
        # Unknown key should not appear in features
        data = response.json()
        assert "nonexistent_flag" not in data["features"]
        # Known key should be applied
        assert data["features"]["memrl"] is True

    def test_empty_body(self):
        """Empty JSON body should return current features unchanged."""
        app = _make_app()
        test_features = Features(memrl=False, tools=True)
        app.dependency_overrides[dep_features] = lambda: test_features
        app = _LocalhostMiddleware(app)

        with patch("src.api.routes.config.set_features"):
            client = TestClient(app)
            response = client.post("/config", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["features"]["memrl"] is False
        assert data["features"]["tools"] is True

    def test_ipv6_localhost_allowed(self):
        """Config changes from IPv6 localhost (::1) should succeed."""

        class IPv6LocalhostMiddleware:
            def __init__(self, inner_app):
                self.app = inner_app

            async def __call__(self, scope, receive, send):
                if scope["type"] == "http":
                    scope["client"] = ("::1", 0)
                await self.app(scope, receive, send)

        app = _make_app()
        test_features = Features()
        app.dependency_overrides[dep_features] = lambda: test_features
        app = IPv6LocalhostMiddleware(app)

        with patch("src.api.routes.config.set_features"):
            client = TestClient(app)
            response = client.post("/config", json={"memrl": True})

        assert response.status_code == 200
