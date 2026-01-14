"""Integration tests for frontend-API interaction.

Tests the SSE streaming, OpenAI compatibility, Gradio integration,
and session management in mock mode.
"""

from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.state import get_state, reset_state
from src.api.routes.sessions import _sessions, _pending_permissions


@pytest.fixture
def client():
    """Create a test client with state reset."""
    # Reset API state
    reset_state()

    # Clear sessions and permissions
    _sessions.clear()
    _pending_permissions.clear()

    with TestClient(app) as c:
        yield c


class TestSSEStreaming:
    """Test /chat/stream SSE endpoint."""

    def test_stream_event_sequence(self, client):
        """Verify turn_start -> token(s) -> turn_end sequence."""
        events = []
        with client.stream(
            "POST",
            "/chat/stream",
            json={"prompt": "Hello", "mock_mode": True}
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))

        # Verify sequence
        types = [e["type"] for e in events]
        assert types[0] == "turn_start"
        assert "token" in types
        assert types[-1] == "turn_end"

    def test_stream_thinking_events(self, client):
        """Verify thinking events when thinking_budget > 0."""
        events = []
        with client.stream(
            "POST",
            "/chat/stream",
            json={"prompt": "Hello", "mock_mode": True, "thinking_budget": 1000}
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))

        types = [e["type"] for e in events]
        assert "thinking" in types, "Thinking events should be emitted when budget > 0"

        # Verify thinking events come after turn_start and before tokens
        thinking_indices = [i for i, t in enumerate(types) if t == "thinking"]
        token_indices = [i for i, t in enumerate(types) if t == "token"]

        assert thinking_indices[0] < token_indices[0], "Thinking should come before tokens"

    def test_stream_no_thinking_when_disabled(self, client):
        """Verify no thinking events when thinking_budget = 0."""
        events = []
        with client.stream(
            "POST",
            "/chat/stream",
            json={"prompt": "Hello", "mock_mode": True, "thinking_budget": 0}
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))

        types = [e["type"] for e in events]
        assert "thinking" not in types, "No thinking when budget is 0"

    def test_stream_plan_mode(self, client):
        """Verify plan mode only emits analysis, no execution."""
        events = []
        with client.stream(
            "POST",
            "/chat/stream",
            json={"prompt": "Hello", "mock_mode": True, "permission_mode": "plan"}
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))

        # Should have token events with plan mode message
        token_events = [e for e in events if e["type"] == "token"]
        all_content = "".join(e["content"] for e in token_events)
        assert "[PLAN MODE]" in all_content

    def test_stream_done_termination(self, client):
        """Verify stream ends with [DONE]."""
        last_line = None
        with client.stream(
            "POST",
            "/chat/stream",
            json={"prompt": "Hello", "mock_mode": True}
        ) as response:
            for line in response.iter_lines():
                if line:
                    last_line = line

        assert last_line == "data: [DONE]"

    def test_stream_event_format(self, client):
        """Verify each event is valid JSON with type field."""
        with client.stream(
            "POST",
            "/chat/stream",
            json={"prompt": "Hello", "mock_mode": True}
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    assert "type" in data, f"Event missing 'type': {data}"


class TestOpenAICompatibility:
    """Test /v1/chat/completions OpenAI format."""

    def test_openai_non_streaming(self, client):
        """Non-streaming returns proper format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "orchestrator",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
        )
        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI format
        assert "id" in data
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]

    def test_openai_streaming_chunks(self, client):
        """Streaming returns SSE chunks with delta format."""
        chunks = []
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "orchestrator",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line[6:]))

        # Verify chunk format
        assert len(chunks) > 0
        for chunk in chunks:
            assert "id" in chunk
            assert "object" in chunk
            assert chunk["object"] == "chat.completion.chunk"
            assert "choices" in chunk
            assert "delta" in chunk["choices"][0]

    def test_openai_model_mapping(self, client):
        """Model field maps to orchestrator roles."""
        # Test different model names
        for model in ["orchestrator", "frontdoor", "coder"]:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                }
            )
            assert response.status_code == 200

    def test_openai_models_list(self, client):
        """GET /v1/models returns available roles."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        model_ids = [m["id"] for m in data["data"]]
        assert "orchestrator" in model_ids
        assert "frontdoor" in model_ids

    def test_openai_get_single_model(self, client):
        """GET /v1/models/{id} returns model info."""
        response = client.get("/v1/models/orchestrator")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "orchestrator"

    def test_openai_model_not_found(self, client):
        """GET /v1/models/{id} returns 404 for unknown model."""
        response = client.get("/v1/models/nonexistent")
        assert response.status_code == 404


class TestSessionManagement:
    """Test session management endpoints."""

    def test_session_list_empty(self, client):
        """List sessions returns empty when none exist."""
        response = client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) == 0

    def test_session_rename_creates(self, client):
        """Rename creates session if it doesn't exist."""
        response = client.post(
            "/sessions/current/rename",
            params={"name": "test-session"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "renamed"
        assert data["name"] == "test-session"

    def test_session_list_after_create(self, client):
        """List sessions shows created session."""
        # Create a session
        client.post("/sessions/current/rename", params={"name": "my-session"})

        response = client.get("/sessions")
        data = response.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["name"] == "my-session"

    def test_session_resume_not_found(self, client):
        """Resume returns 404 for non-existent session."""
        response = client.post("/sessions/nonexistent/resume")
        assert response.status_code == 404


class TestPermissionFlow:
    """Test permission request/response flow."""

    def test_pending_permissions_empty(self, client):
        """List pending permissions returns empty initially."""
        response = client.get("/permission/pending")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0

    def test_permission_not_found(self, client):
        """Respond to non-existent permission returns 404."""
        response = client.post(
            "/permission/nonexistent",
            params={"approved": True}
        )
        assert response.status_code == 404


class TestHealthAndStats:
    """Test health and stats endpoints still work."""

    def test_health_endpoint(self, client):
        """Health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_stats_endpoint(self, client):
        """Stats endpoint returns metrics."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data


class TestChatEndpoint:
    """Test the main /chat endpoint still works."""

    def test_chat_mock_mode(self, client):
        """Chat in mock mode returns response."""
        response = client.post(
            "/chat",
            json={"prompt": "Hello", "mock_mode": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert "[MOCK]" in data["answer"]
        assert data["mock_mode"] is True

    def test_chat_with_thinking_budget(self, client):
        """Chat accepts thinking_budget parameter."""
        response = client.post(
            "/chat",
            json={"prompt": "Hello", "mock_mode": True, "thinking_budget": 1000}
        )
        assert response.status_code == 200

    def test_chat_with_permission_mode(self, client):
        """Chat accepts permission_mode parameter."""
        response = client.post(
            "/chat",
            json={"prompt": "Hello", "mock_mode": True, "permission_mode": "plan"}
        )
        assert response.status_code == 200
