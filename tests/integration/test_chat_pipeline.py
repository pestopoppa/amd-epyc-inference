"""Integration tests for the chat pipeline.

Tests the chat pipeline end-to-end using FastAPI TestClient with mock LLM responses.
Tests routing logic, mode selection, and endpoint behavior.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.routes.chat_routing import _classify_and_route, _should_use_direct_mode, _select_mode
from src.roles import Role


class TestClassifyAndRoute:
    """Test _classify_and_route() keyword-based routing."""

    def test_image_routes_to_vision(self):
        """Images should route to vision worker."""
        role, strategy = _classify_and_route("describe this image", has_image=True)
        assert role == "worker_vision"
        assert strategy == "classified"

    def test_code_prompt_routes_to_coder_when_specialist_routing_enabled(self):
        """Code keywords should route to coder when specialist_routing is enabled."""
        from src.features import set_features, Features

        # Enable specialist routing
        features = Features(specialist_routing=True)
        set_features(features)

        try:
            role, strategy = _classify_and_route("implement a binary search function")
            assert role == str(Role.CODER_PRIMARY)
            assert strategy == "classified"
        finally:
            # Reset features
            from src.features import reset_features
            reset_features()

    def test_architecture_prompt_routes_to_architect_when_specialist_routing_enabled(self):
        """Architecture keywords should route to architect when specialist_routing is enabled."""
        from src.features import set_features, Features

        # Enable specialist routing
        features = Features(specialist_routing=True)
        set_features(features)

        try:
            role, strategy = _classify_and_route("design a microservice architecture")
            assert role == str(Role.ARCHITECT_GENERAL)
            assert strategy == "classified"
        finally:
            # Reset features
            from src.features import reset_features
            reset_features()

    def test_default_routes_to_frontdoor(self):
        """Default routing should go to frontdoor."""
        role, strategy = _classify_and_route("hello, how are you?")
        assert role == str(Role.FRONTDOOR)
        assert strategy == "rules"

    def test_specialist_routing_disabled_by_default(self):
        """With specialist_routing disabled, code prompts should still route to frontdoor."""
        from src.features import set_features, Features

        # Ensure specialist routing is disabled
        features = Features(specialist_routing=False)
        set_features(features)

        try:
            role, strategy = _classify_and_route("implement a function")
            assert role == str(Role.FRONTDOOR)
            assert strategy == "rules"
        finally:
            # Reset features
            from src.features import reset_features
            reset_features()


class TestShouldUseDirectMode:
    """Test _should_use_direct_mode() heuristic."""

    def test_simple_question_uses_direct(self):
        """Simple questions should use direct mode."""
        assert _should_use_direct_mode("What is 2+2?") is True

    def test_file_operation_uses_repl(self):
        """File operations should use REPL mode."""
        assert _should_use_direct_mode("read the file foo.py") is False

    def test_large_context_uses_repl(self):
        """Large context should use REPL mode."""
        large_context = "x" * 30000
        assert _should_use_direct_mode("summarize this", large_context) is False

    def test_code_execution_uses_repl(self):
        """Code execution requests should use REPL mode."""
        assert _should_use_direct_mode("execute this code") is False

    def test_small_context_can_use_direct(self):
        """Small context with simple question can use direct mode."""
        small_context = "The capital of France is Paris."
        assert _should_use_direct_mode("What is the capital?", small_context) is True


class TestSelectMode:
    """Test _select_mode() with mock state."""

    def test_no_hybrid_router_uses_heuristic(self):
        """Without hybrid_router, should use heuristic fallback."""
        state = MagicMock()
        state.hybrid_router = None

        mode = _select_mode("hello world", "", state)
        assert mode in ("direct", "react", "repl")

    def test_react_mode_when_keywords_match(self):
        """Should return react mode when keywords match."""
        from src.features import set_features, Features

        # Enable react mode
        features = Features(react_mode=True)
        set_features(features)

        try:
            state = MagicMock()
            state.hybrid_router = None

            mode = _select_mode("search for quantum computing papers", "", state)
            assert mode == "react"
        finally:
            # Reset features
            from src.features import reset_features
            reset_features()

    def test_hybrid_router_used_when_available(self):
        """When hybrid_router is available, it should be consulted."""
        state = MagicMock()
        state.hybrid_router = MagicMock()
        state.hybrid_router.route_with_mode = MagicMock(
            return_value=(["frontdoor"], "learned", "direct")
        )

        mode = _select_mode("test query", "", state)
        assert mode == "direct"
        state.hybrid_router.route_with_mode.assert_called_once()


class TestChatEndpoint:
    """Test /api/chat endpoint with mock mode."""

    @pytest.fixture
    def client(self):
        """Create test client with fresh app instance."""
        app = create_app()
        return TestClient(app)

    def test_mock_mode_returns_200(self, client):
        """Mock mode should return 200 with valid response."""
        response = client.post("/chat", json={
            "prompt": "Hello world",
            "mock_mode": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["mock_mode"] is True

    def test_empty_prompt_returns_422(self, client):
        """Empty prompt should return 422 validation error."""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_mock_response_contains_expected_fields(self, client):
        """Mock response should contain all expected fields."""
        response = client.post("/chat", json={
            "prompt": "test query",
            "mock_mode": True,
        })
        data = response.json()
        assert response.status_code == 200
        assert "answer" in data
        assert "turns" in data
        assert "elapsed_seconds" in data
        assert "mock_mode" in data
        assert data["mock_mode"] is True

    def test_routing_decision_in_response(self, client):
        """Response should include routing decision."""
        response = client.post("/chat", json={
            "prompt": "test query",
            "mock_mode": True,
        })
        data = response.json()
        assert "routed_to" in data
        assert "routing_strategy" in data

    def test_force_role_parameter(self, client):
        """force_role parameter should be accepted without errors."""
        response = client.post("/chat", json={
            "prompt": "test query",
            "mock_mode": True,
            "force_role": "coder_primary",
        })
        data = response.json()
        assert response.status_code == 200
        # Mock mode returns empty routed_to, but accepts the parameter
        assert "routed_to" in data

    def test_force_mode_parameter(self, client):
        """force_mode parameter should be accepted without errors."""
        response = client.post("/chat", json={
            "prompt": "test query",
            "mock_mode": True,
            "force_mode": "direct",
        })
        data = response.json()
        assert response.status_code == 200
        assert data.get("mode") == "mock"


class TestRewardEndpoint:
    """Test /api/chat/reward endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with fresh app instance."""
        app = create_app()
        return TestClient(app)

    def test_reward_endpoint_with_valid_data(self, client):
        """Reward endpoint should accept valid reward data."""
        response = client.post("/chat/reward", json={
            "task_description": "test task",
            "action": "frontdoor:direct",
            "reward": 0.8,
        })
        # Should return 200 even without MemRL initialized (graceful degradation)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_reward_endpoint_with_context(self, client):
        """Reward endpoint should accept optional context."""
        response = client.post("/chat/reward", json={
            "task_description": "test task",
            "action": "frontdoor:direct",
            "reward": 0.5,
            "context": {"suite": "thinking", "tier": 1},
        })
        assert response.status_code == 200

    def test_reward_endpoint_validation(self, client):
        """Reward endpoint should validate reward range."""
        # Test out of range reward
        response = client.post("/chat/reward", json={
            "task_description": "test task",
            "action": "frontdoor:direct",
            "reward": 2.0,  # Out of range
        })
        assert response.status_code == 422


class TestStreamEndpoint:
    """Test /api/chat/stream SSE endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with fresh app instance."""
        app = create_app()
        return TestClient(app)

    def test_stream_mock_mode(self, client):
        """Stream endpoint should work in mock mode."""
        response = client.post("/chat/stream", json={
            "prompt": "Hello",
            "mock_mode": True,
        })
        assert response.status_code == 200
        # SSE responses have text/event-stream content type
        assert "text/event-stream" in response.headers.get("content-type", "").lower()

    def test_stream_returns_sse_format(self, client):
        """Stream endpoint should return SSE formatted data."""
        response = client.post("/chat/stream", json={
            "prompt": "Hello",
            "mock_mode": True,
        })
        assert response.status_code == 200

        # Check that we get some content back
        content = response.text
        assert len(content) > 0

    def test_stream_with_thinking_budget(self, client):
        """Stream endpoint should accept thinking_budget parameter."""
        response = client.post("/chat/stream", json={
            "prompt": "Hello",
            "mock_mode": True,
            "thinking_budget": 1000,
        })
        assert response.status_code == 200

    def test_stream_with_permission_mode(self, client):
        """Stream endpoint should accept permission_mode parameter."""
        response = client.post("/chat/stream", json={
            "prompt": "Hello",
            "mock_mode": True,
            "permission_mode": "plan",
        })
        assert response.status_code == 200
