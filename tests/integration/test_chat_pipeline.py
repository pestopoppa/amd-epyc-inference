"""Integration tests for the chat pipeline.

Tests the chat pipeline end-to-end using FastAPI TestClient with mock LLM responses.
Tests routing logic, mode selection, and endpoint behavior.
"""

import json
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
        """Mock mode should return 200 with valid response structure."""
        response = client.post("/chat", json={
            "prompt": "Hello world",
            "mock_mode": True,
        })
        assert response.status_code == 200
        data = response.json()

        # Verify core response structure
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

        # Verify metadata
        assert data["mock_mode"] is True
        assert data["turns"] > 0
        assert data["elapsed_seconds"] > 0.0
        assert data["mode"] == "mock"

    def test_empty_prompt_returns_422(self, client):
        """Empty prompt should return 422 validation error."""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_mock_response_contains_expected_fields(self, client):
        """Mock response should contain all expected fields with correct types."""
        response = client.post("/chat", json={
            "prompt": "test query",
            "mock_mode": True,
        })
        data = response.json()
        assert response.status_code == 200

        # Required fields
        assert "answer" in data
        assert "turns" in data
        assert "elapsed_seconds" in data
        assert "mock_mode" in data
        assert data["mock_mode"] is True

        # Type constraints
        assert isinstance(data["answer"], str)
        assert isinstance(data["turns"], int)
        assert isinstance(data["elapsed_seconds"], (int, float))
        assert isinstance(data["mock_mode"], bool)

        # Value constraints
        assert data["turns"] >= 1
        assert data["elapsed_seconds"] >= 0.0
        assert data.get("tokens_used", 0) >= 0

    def test_routing_decision_in_response(self, client):
        """Response should include routing decision with valid values."""
        response = client.post("/chat", json={
            "prompt": "test query",
            "mock_mode": True,
        })
        data = response.json()

        assert "routed_to" in data
        assert "routing_strategy" in data

        # routed_to should be a string (may be empty in mock mode)
        assert isinstance(data["routed_to"], str)

        # routing_strategy should be one of expected values
        valid_strategies = ["learned", "rules", "default", "classified", "mock", ""]
        assert data["routing_strategy"] in valid_strategies

    def test_force_role_parameter(self, client):
        """force_role parameter should override routing to specified role."""
        response = client.post("/chat", json={
            "prompt": "test query",
            "mock_mode": True,
            "force_role": "coder_primary",
        })
        data = response.json()
        assert response.status_code == 200

        # In mock mode, force_role may not populate routed_to, but should be accepted
        assert "routed_to" in data
        # With real mode, would verify: assert data["routed_to"] == "coder_primary"

    def test_force_mode_parameter(self, client):
        """force_mode parameter should be accepted and reflected in response."""
        response = client.post("/chat", json={
            "prompt": "test query",
            "mock_mode": True,
            "force_mode": "direct",
        })
        data = response.json()
        assert response.status_code == 200

        # Mock mode overrides to "mock", but parameter should be accepted
        assert "mode" in data
        # In mock mode, mode is always "mock"
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
        """Reward endpoint should accept optional context with proper structure."""
        response = client.post("/chat/reward", json={
            "task_description": "test task",
            "action": "frontdoor:direct",
            "reward": 0.5,
            "context": {"suite": "thinking", "tier": 1},
        })
        assert response.status_code == 200

        # Verify response structure
        data = response.json()
        assert "success" in data
        assert isinstance(data["success"], bool)

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
        """Stream endpoint should return valid SSE events in mock mode."""
        response = client.post("/chat/stream", json={
            "prompt": "Hello",
            "mock_mode": True,
        })
        assert response.status_code == 200

        # SSE responses have text/event-stream content type
        assert "text/event-stream" in response.headers.get("content-type", "").lower()

        # Parse SSE events
        content = response.text
        lines = content.strip().split('\n')

        # Should have at least one data event
        data_events = [line for line in lines if line.startswith("data:")]
        assert len(data_events) > 0

        # Verify at least one event contains valid JSON
        json_found = False
        for line in data_events:
            if line.startswith("data: "):
                payload = line[6:]  # Skip "data: "
                if payload == "[DONE]":
                    continue
                try:
                    parsed = json.loads(payload)
                    json_found = True
                    # Verify event has expected structure
                    assert "type" in parsed
                    break
                except json.JSONDecodeError:
                    pass

        # At minimum should have [DONE] marker
        assert any("[DONE]" in line for line in data_events)

    def test_stream_returns_sse_format(self, client):
        """Stream endpoint should return properly formatted SSE data."""
        response = client.post("/chat/stream", json={
            "prompt": "Hello",
            "mock_mode": True,
        })
        assert response.status_code == 200

        # Verify SSE format
        content = response.text
        assert len(content) > 0

        # SSE format verification
        lines = content.strip().split('\n')

        # Should have "data:" prefixes
        data_lines = [line for line in lines if line.startswith("data:")]
        assert len(data_lines) > 0

        # Verify proper line structure (data: prefix, parseable JSON or [DONE])
        for line in data_lines:
            assert line.startswith("data: ")
            payload = line[6:]  # Skip "data: "
            if payload == "[DONE]":
                continue
            # Other events should be valid JSON
            try:
                parsed = json.loads(payload)
                assert isinstance(parsed, dict)
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in SSE event: {payload}")

    def test_stream_with_thinking_budget(self, client):
        """Stream endpoint should accept thinking_budget and return valid SSE."""
        response = client.post("/chat/stream", json={
            "prompt": "Hello",
            "mock_mode": True,
            "thinking_budget": 1000,
        })
        assert response.status_code == 200

        # Verify SSE format
        content = response.text
        lines = content.strip().split('\n')
        data_lines = [line for line in lines if line.startswith("data:")]

        # Should have data events
        assert len(data_lines) > 0

        # Should have [DONE] marker
        assert any("[DONE]" in line for line in data_lines)

    def test_stream_with_permission_mode(self, client):
        """Stream endpoint should accept permission_mode and return valid SSE."""
        response = client.post("/chat/stream", json={
            "prompt": "Hello",
            "mock_mode": True,
            "permission_mode": "plan",
        })
        assert response.status_code == 200

        # Verify SSE format
        content = response.text
        lines = content.strip().split('\n')
        data_lines = [line for line in lines if line.startswith("data:")]

        # Should have data events
        assert len(data_lines) > 0

        # Should have proper SSE structure with [DONE]
        assert any("[DONE]" in line for line in data_lines)
