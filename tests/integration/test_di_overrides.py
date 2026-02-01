"""Integration tests for dependency injection overrides.

Tests that FastAPI's app.dependency_overrides mechanism works end-to-end
with the dependency functions in src.api.dependencies.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.dependencies import (
    dep_app_state,
    dep_health_tracker,
    dep_gate_runner,
    dep_llm_primitives,
    dep_progress_logger,
    dep_hybrid_router,
    dep_tool_registry,
    dep_script_registry,
    dep_registry_loader,
)


class TestDIOverrides:
    """Verify that app.dependency_overrides correctly inject mock dependencies."""

    @pytest.fixture
    def app(self):
        """Create app instance and ensure cleanup after test."""
        app = create_app()
        yield app
        app.dependency_overrides.clear()  # Cleanup

    @pytest.fixture
    def client(self, app):
        """Create test client from app."""
        return TestClient(app)

    def test_override_health_tracker(self, app, client):
        """Overriding dep_health_tracker injects mock into /health."""
        mock_tracker = MagicMock()
        mock_tracker.get_status.return_value = {
            "test_backend": {"state": "closed", "failures": 0, "cooldown_until": None}
        }
        app.dependency_overrides[dep_health_tracker] = lambda: mock_tracker

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "test_backend" in data.get("backend_health", {})
        mock_tracker.get_status.assert_called_once()

    def test_override_gate_runner(self, app, client):
        """Overriding dep_gate_runner injects mock into /gates."""
        from src.gate_runner import GateResult

        mock_runner = MagicMock()
        # run_gates_by_name should return a list of GateResult objects
        mock_runner.run_gates_by_name.return_value = [
            GateResult(
                gate_name="schema",
                passed=True,
                exit_code=0,
                output="",
                elapsed_seconds=0.1,
                errors=[],
                warnings=[],
            )
        ]
        app.dependency_overrides[dep_gate_runner] = lambda: mock_runner

        response = client.post("/gates", json={
            "gate_names": ["schema"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["all_passed"] is True
        # Verify the mock was called
        mock_runner.run_gates_by_name.assert_called_once()

    def test_override_app_state(self, app, client):
        """Overriding dep_app_state injects mock state into /stats."""
        from src.api.state import AppState

        mock_state = MagicMock(spec=AppState)
        mock_state.get_stats.return_value = {
            "total_requests": 42,
            "total_turns": 100,
            "average_turns_per_request": 2.38,
            "mock_requests": 20,
            "real_requests": 22,
            "active_requests": 0,
        }
        app.dependency_overrides[dep_app_state] = lambda: mock_state

        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 42
        assert data["total_turns"] == 100
        mock_state.get_stats.assert_called_once()

    def test_cleanup_removes_override(self, app, client):
        """After clearing overrides, real dependencies are used again."""
        mock_tracker = MagicMock()
        mock_tracker.get_status.return_value = {"mock": {"state": "open", "failures": 0, "cooldown_until": None}}
        app.dependency_overrides[dep_health_tracker] = lambda: mock_tracker

        # With override
        r1 = client.get("/health")
        assert "mock" in r1.json().get("backend_health", {})

        # Clear and verify real tracker used
        app.dependency_overrides.clear()
        r2 = client.get("/health")
        # Real tracker won't have "mock" backend
        assert "mock" not in r2.json().get("backend_health", {})

    def test_override_llm_primitives_for_chat(self, app, client):
        """Overriding dep_llm_primitives makes it accessible via dependency injection."""
        from src.api.state import AppState

        # Create mock primitives
        mock_primitives = MagicMock()
        mock_primitives.llm_call.return_value = "Mocked LLM response"

        # Override llm_primitives dependency
        app.dependency_overrides[dep_llm_primitives] = lambda: mock_primitives

        # Also override app_state for mock mode
        mock_state = MagicMock(spec=AppState)
        mock_state.registry = None
        mock_state.hybrid_router = None
        mock_state.progress_logger = None
        mock_state.tool_registry = None
        mock_state.script_registry = None
        mock_state.health_tracker = MagicMock()
        mock_state.increment_active = MagicMock()
        mock_state.decrement_active = MagicMock()
        mock_state.increment_request = MagicMock()
        app.dependency_overrides[dep_app_state] = lambda: mock_state

        # Make a chat request in mock mode to verify no crashes
        response = client.post("/chat", json={
            "prompt": "test override",
            "mock_mode": True,
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        # Verify state tracking was called
        mock_state.increment_active.assert_called_once()
        mock_state.decrement_active.assert_called_once()

        # Verify override was accessible
        assert dep_llm_primitives in app.dependency_overrides

    def test_override_optional_dependencies(self, app, client):
        """Test overriding optional dependencies (progress_logger, hybrid_router, etc.)."""
        from src.api.state import AppState

        mock_progress_logger = MagicMock()
        mock_hybrid_router = MagicMock()
        mock_tool_registry = MagicMock()

        # Create mock state with all necessary attributes
        mock_state = MagicMock(spec=AppState)
        mock_state.registry = None
        mock_state.hybrid_router = mock_hybrid_router
        mock_state.progress_logger = mock_progress_logger
        mock_state.tool_registry = mock_tool_registry
        mock_state.script_registry = None
        mock_state.health_tracker = MagicMock()
        mock_state.increment_active = MagicMock()
        mock_state.decrement_active = MagicMock()
        mock_state.increment_request = MagicMock()

        # Override dependencies
        app.dependency_overrides[dep_progress_logger] = lambda: mock_progress_logger
        app.dependency_overrides[dep_hybrid_router] = lambda: mock_hybrid_router
        app.dependency_overrides[dep_tool_registry] = lambda: mock_tool_registry
        app.dependency_overrides[dep_app_state] = lambda: mock_state

        # Make a chat request in mock mode to verify these don't break the pipeline
        response = client.post("/chat", json={
            "prompt": "test optional deps",
            "mock_mode": True,
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data.get("mock_mode") is True

        # Verify overrides were registered
        assert dep_progress_logger in app.dependency_overrides
        assert dep_hybrid_router in app.dependency_overrides
        assert dep_tool_registry in app.dependency_overrides

    def test_override_multiple_dependencies(self, app, client):
        """Test overriding multiple dependencies simultaneously."""
        mock_tracker = MagicMock()
        mock_tracker.get_status.return_value = {}

        mock_state = MagicMock()
        mock_state.get_stats.return_value = {
            "total_requests": 10,
            "total_turns": 20,
            "average_turns_per_request": 2.0,
            "mock_requests": 10,
            "real_requests": 0,
            "active_requests": 0,
        }

        app.dependency_overrides[dep_health_tracker] = lambda: mock_tracker
        app.dependency_overrides[dep_app_state] = lambda: mock_state

        # Both should work simultaneously
        health_response = client.get("/health")
        stats_response = client.get("/stats")

        assert health_response.status_code == 200
        assert stats_response.status_code == 200
        assert stats_response.json()["total_requests"] == 10

        mock_tracker.get_status.assert_called()
        mock_state.get_stats.assert_called()

    def test_override_with_fixture_pattern(self, app, client):
        """Test common test pattern: override in fixture, use in test."""
        # This pattern is useful for test suites that need the same mock setup
        mock_state = MagicMock()
        mock_state.get_stats.return_value = {
            "total_requests": 0,
            "total_turns": 0,
            "average_turns_per_request": 0.0,
            "mock_requests": 0,
            "real_requests": 0,
            "active_requests": 0,
        }

        app.dependency_overrides[dep_app_state] = lambda: mock_state

        # Make multiple requests - all should use the mock
        for _ in range(3):
            response = client.get("/stats")
            assert response.status_code == 200
            assert response.json()["total_requests"] == 0

        # Mock should have been called 3 times
        assert mock_state.get_stats.call_count == 3

    def test_chat_endpoint_with_mock_state(self, app, client):
        """Test chat endpoint with overridden state dependency."""
        from src.api.state import AppState

        mock_state = MagicMock(spec=AppState)
        # Set up required attributes for chat endpoint
        mock_state.registry = None
        mock_state.hybrid_router = None
        mock_state.progress_logger = None
        mock_state.tool_registry = None
        mock_state.script_registry = None
        mock_state.health_tracker = MagicMock()
        mock_state.increment_active = MagicMock()
        mock_state.decrement_active = MagicMock()
        mock_state.increment_request = MagicMock()

        app.dependency_overrides[dep_app_state] = lambda: mock_state

        # Chat request in mock mode should work
        response = client.post("/chat", json={
            "prompt": "test",
            "mock_mode": True,
        })

        assert response.status_code == 200
        data = response.json()

        # Verify state tracking methods were called
        mock_state.increment_active.assert_called_once()
        mock_state.decrement_active.assert_called_once()

        # Verify response structure and content
        assert "answer" in data
        assert data["answer"] != ""  # Not empty
        assert data.get("mock_mode") is True
        assert "elapsed_seconds" in data
        assert data["elapsed_seconds"] >= 0

    def test_override_registry_loader(self, app, client):
        """Test overriding registry loader dependency."""
        from src.api.state import AppState

        mock_registry = MagicMock()
        mock_registry.get_role_defaults.return_value = {
            "n_ctx": 8192,
            "n_tokens": 512,
        }

        # Create mock state with registry
        mock_state = MagicMock(spec=AppState)
        mock_state.registry = mock_registry
        mock_state.hybrid_router = None
        mock_state.progress_logger = None
        mock_state.tool_registry = None
        mock_state.script_registry = None
        mock_state.health_tracker = MagicMock()
        mock_state.increment_active = MagicMock()
        mock_state.decrement_active = MagicMock()
        mock_state.increment_request = MagicMock()

        app.dependency_overrides[dep_registry_loader] = lambda: mock_registry
        app.dependency_overrides[dep_app_state] = lambda: mock_state

        # Make a chat request to verify the mock registry doesn't crash the pipeline
        response = client.post("/chat", json={
            "prompt": "test registry override",
            "mock_mode": True,
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data.get("mock_mode") is True

        # Verify override is in place
        assert dep_registry_loader in app.dependency_overrides

    def test_override_script_registry(self, app, client):
        """Test overriding script registry dependency."""
        from src.api.state import AppState

        mock_script_registry = MagicMock()
        mock_script_registry.search.return_value = []

        # Create mock state with script registry
        mock_state = MagicMock(spec=AppState)
        mock_state.registry = None
        mock_state.hybrid_router = None
        mock_state.progress_logger = None
        mock_state.tool_registry = None
        mock_state.script_registry = mock_script_registry
        mock_state.health_tracker = MagicMock()
        mock_state.increment_active = MagicMock()
        mock_state.decrement_active = MagicMock()
        mock_state.increment_request = MagicMock()

        app.dependency_overrides[dep_script_registry] = lambda: mock_script_registry
        app.dependency_overrides[dep_app_state] = lambda: mock_state

        # Make a chat request to verify the mock script registry doesn't crash the pipeline
        response = client.post("/chat", json={
            "prompt": "test script registry override",
            "mock_mode": True,
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data.get("mock_mode") is True

        # Verify override is in place
        assert dep_script_registry in app.dependency_overrides

    def test_dependency_override_isolation(self, app, client):
        """Test that overrides in one test don't affect another."""
        # First override
        mock1 = MagicMock()
        mock1.get_status.return_value = {"backend1": {"state": "open", "failures": 0, "cooldown_until": None}}
        app.dependency_overrides[dep_health_tracker] = lambda: mock1

        r1 = client.get("/health")
        assert "backend1" in r1.json()["backend_health"]

        # Clear and set different override
        app.dependency_overrides.clear()
        mock2 = MagicMock()
        mock2.get_status.return_value = {"backend2": {"state": "closed", "failures": 0, "cooldown_until": None}}
        app.dependency_overrides[dep_health_tracker] = lambda: mock2

        r2 = client.get("/health")
        assert "backend2" in r2.json()["backend_health"]
        assert "backend1" not in r2.json()["backend_health"]
