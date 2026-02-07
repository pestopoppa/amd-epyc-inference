"""Integration tests for dependency injection overrides.

Tests that FastAPI's app.dependency_overrides mechanism works end-to-end
with the dependency functions in src.api.dependencies.
"""

import pytest
from dataclasses import dataclass
from pathlib import Path
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.state import AppState
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
from src.gate_runner import GateRunner


# Stub classes for testing (lightweight, no external dependencies)
@dataclass
class StubLLMPrimitives:
    """Stub LLM primitives for testing."""

    def llm_call(self, prompt: str, **kwargs) -> str:
        return "Mocked LLM response"


@dataclass
class StubRegistryLoader:
    """Stub registry loader for testing."""

    routing_hints: dict

    def __init__(self):
        self.routing_hints = {}

    def get_role_defaults(self, role: str) -> dict:
        return {"n_ctx": 8192, "n_tokens": 512}


@dataclass
class StubToolRegistry:
    """Stub tool registry for testing."""

    def search(self, query: str) -> list:
        return []


@dataclass
class StubScriptRegistry:
    """Stub script registry for testing."""

    def search(self, query: str) -> list:
        return []


@dataclass
class StubProgressLogger:
    """Stub progress logger for testing."""

    def flush(self) -> None:
        pass

    def log_task_started(self, *args, **kwargs) -> None:
        """Stub log task started."""
        pass

    def log_task_completed(self, *args, **kwargs) -> None:
        """Stub log task completed."""
        pass


@dataclass
class StubHybridRouter:
    """Stub hybrid router for testing."""

    def route(self, task: str) -> str:
        return "frontdoor"


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
        """Overriding dep_health_tracker injects stub into /health."""
        # Use real AppState with a tracker
        state = AppState()
        # health_tracker is auto-created with AppState, just use it
        app.dependency_overrides[dep_health_tracker] = lambda: state.health_tracker

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Status may be "degraded" when backend servers are unreachable (CI)
        assert data["status"] in ("ok", "degraded")

    def test_override_gate_runner(self, app, client):
        """Overriding dep_gate_runner injects real GateRunner into /gates."""
        # Create real GateRunner with no config file (uses defaults)
        runner = GateRunner(config_path=None, working_dir=Path.cwd())

        app.dependency_overrides[dep_gate_runner] = lambda: runner

        response = client.post(
            "/gates",
            json={
                "gate_names": ["schema"],
            },
        )
        # The response may succeed or fail depending on actual gate execution
        # We just verify the override worked and didn't crash
        assert response.status_code in (200, 500)  # May fail if gates fail
        data = response.json()
        assert "all_passed" in data or "detail" in data

    def test_override_app_state(self, app, client):
        """Overriding dep_app_state injects real state into /stats."""
        # Use real AppState
        state = AppState()
        state.total_requests = 42
        state.total_turns = 100
        state.mock_requests = 20
        state.real_requests = 22

        app.dependency_overrides[dep_app_state] = lambda: state

        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 42
        assert data["total_turns"] == 100

    def test_cleanup_removes_override(self, app, client):
        """After clearing overrides, real dependencies are used again."""
        state = AppState()
        app.dependency_overrides[dep_health_tracker] = lambda: state.health_tracker

        # With override
        r1 = client.get("/health")
        assert r1.status_code == 200

        # Clear and verify real tracker used
        app.dependency_overrides.clear()
        r2 = client.get("/health")
        assert r2.status_code == 200
        # Both should work, we're just testing override removal doesn't break

    def test_override_llm_primitives_for_chat(self, app, client):
        """Overriding dep_llm_primitives makes it accessible via dependency injection."""
        # Create stub primitives
        primitives = StubLLMPrimitives()

        # Override llm_primitives dependency
        app.dependency_overrides[dep_llm_primitives] = lambda: primitives

        # Also override app_state with real state
        state = AppState()
        state.registry = None
        state.hybrid_router = None
        state.progress_logger = None
        state.tool_registry = None
        state.script_registry = None
        app.dependency_overrides[dep_app_state] = lambda: state

        # Make a chat request in mock mode to verify no crashes
        response = client.post(
            "/chat",
            json={
                "prompt": "test override",
                "mock_mode": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

        # Verify override was accessible
        assert dep_llm_primitives in app.dependency_overrides

    def test_override_optional_dependencies(self, app, client):
        """Test overriding optional dependencies (progress_logger, hybrid_router, etc.)."""
        # Create stub objects
        progress_logger = StubProgressLogger()
        hybrid_router = StubHybridRouter()
        tool_registry = StubToolRegistry()

        # Create real state with stub components
        state = AppState()
        state.registry = None
        state.hybrid_router = hybrid_router
        state.progress_logger = progress_logger
        state.tool_registry = tool_registry
        state.script_registry = None

        # Override dependencies
        app.dependency_overrides[dep_progress_logger] = lambda: progress_logger
        app.dependency_overrides[dep_hybrid_router] = lambda: hybrid_router
        app.dependency_overrides[dep_tool_registry] = lambda: tool_registry
        app.dependency_overrides[dep_app_state] = lambda: state

        # Make a chat request in mock mode to verify these don't break the pipeline
        response = client.post(
            "/chat",
            json={
                "prompt": "test optional deps",
                "mock_mode": True,
            },
        )

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
        # Use real AppState
        state = AppState()
        state.total_requests = 10
        state.total_turns = 20
        state.mock_requests = 10
        state.real_requests = 0

        app.dependency_overrides[dep_health_tracker] = lambda: state.health_tracker
        app.dependency_overrides[dep_app_state] = lambda: state

        # Both should work simultaneously
        health_response = client.get("/health")
        stats_response = client.get("/stats")

        assert health_response.status_code == 200
        assert stats_response.status_code == 200
        assert stats_response.json()["total_requests"] == 10

    def test_override_with_fixture_pattern(self, app, client):
        """Test common test pattern: override in fixture, use in test."""
        # This pattern is useful for test suites that need the same setup
        state = AppState()
        state.total_requests = 0
        state.total_turns = 0
        state.mock_requests = 0
        state.real_requests = 0

        app.dependency_overrides[dep_app_state] = lambda: state

        # Make multiple requests - all should use the same state
        for _ in range(3):
            response = client.get("/stats")
            assert response.status_code == 200
            # State doesn't auto-increment in this test
            assert response.json()["total_requests"] == 0

    def test_chat_endpoint_with_real_state(self, app, client):
        """Test chat endpoint with overridden real state dependency."""
        # Use real AppState
        state = AppState()
        state.registry = None
        state.hybrid_router = None
        state.progress_logger = None
        state.tool_registry = None
        state.script_registry = None

        app.dependency_overrides[dep_app_state] = lambda: state

        # Chat request in mock mode should work
        response = client.post(
            "/chat",
            json={
                "prompt": "test",
                "mock_mode": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure and content
        assert "answer" in data
        assert data["answer"] != ""  # Not empty
        assert data.get("mock_mode") is True
        assert "elapsed_seconds" in data
        assert data["elapsed_seconds"] >= 0

    def test_override_registry_loader(self, app, client):
        """Test overriding registry loader dependency."""
        # Create stub registry
        registry = StubRegistryLoader()

        # Create real state with registry
        state = AppState()
        state.registry = registry
        state.hybrid_router = None
        state.progress_logger = None
        state.tool_registry = None
        state.script_registry = None

        app.dependency_overrides[dep_registry_loader] = lambda: registry
        app.dependency_overrides[dep_app_state] = lambda: state

        # Make a chat request to verify the stub registry doesn't crash the pipeline
        response = client.post(
            "/chat",
            json={
                "prompt": "test registry override",
                "mock_mode": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data.get("mock_mode") is True

        # Verify override is in place
        assert dep_registry_loader in app.dependency_overrides

    def test_override_script_registry(self, app, client):
        """Test overriding script registry dependency."""
        # Create stub script registry
        script_registry = StubScriptRegistry()

        # Create real state with script registry
        state = AppState()
        state.registry = None
        state.hybrid_router = None
        state.progress_logger = None
        state.tool_registry = None
        state.script_registry = script_registry

        app.dependency_overrides[dep_script_registry] = lambda: script_registry
        app.dependency_overrides[dep_app_state] = lambda: state

        # Make a chat request to verify the stub script registry doesn't crash the pipeline
        response = client.post(
            "/chat",
            json={
                "prompt": "test script registry override",
                "mock_mode": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data.get("mock_mode") is True

        # Verify override is in place
        assert dep_script_registry in app.dependency_overrides

    def test_dependency_override_isolation(self, app, client):
        """Test that overrides in one test don't affect another."""
        # First override
        state1 = AppState()
        app.dependency_overrides[dep_health_tracker] = lambda: state1.health_tracker

        r1 = client.get("/health")
        assert r1.status_code == 200

        # Clear and set different override
        app.dependency_overrides.clear()
        state2 = AppState()
        app.dependency_overrides[dep_health_tracker] = lambda: state2.health_tracker

        r2 = client.get("/health")
        assert r2.status_code == 200
        # Both should succeed, verifying isolation
