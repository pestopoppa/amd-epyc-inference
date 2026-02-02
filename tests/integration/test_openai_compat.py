"""Integration tests for OpenAI-compatible endpoints.

Tests cover:
- /v1/models endpoint
- /v1/chat/completions non-streaming
- /v1/chat/completions streaming
- Request validation
- Response format compliance
- Error handling
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import create_app


@pytest.fixture
def client():
    """Create test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestModelsEndpoint:
    """Test the /v1/models endpoint."""

    def test_list_models(self, client):
        """Test GET /v1/models returns available models."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        # Check OpenAI-compatible format
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0

        # Check first model has correct fields
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model

    def test_get_specific_model(self, client):
        """Test GET /v1/models/{model_id}."""
        response = client.get("/v1/models/orchestrator")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == "orchestrator"
        assert data["object"] == "model"

    def test_get_nonexistent_model(self, client):
        """Test GET /v1/models/{model_id} with invalid ID."""
        response = client.get("/v1/models/nonexistent_model")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestChatCompletionsNonStreaming:
    """Test non-streaming chat completions."""

    def test_chat_completions_basic(self, client):
        """Test basic chat completion request."""
        request = {
            "model": "orchestrator",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.0,
            "max_tokens": 100,
        }

        response = client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        # Check OpenAI-compatible response structure
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert data["model"] == "orchestrator"
        assert "choices" in data
        assert len(data["choices"]) > 0

        # Check choice structure
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert choice["finish_reason"] == "stop"

        # Check usage
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    def test_chat_completions_with_system_message(self, client):
        """Test chat completion with system message."""
        request = {
            "model": "frontdoor",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
        }

        response = client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completions_with_conversation_history(self, client):
        """Test chat completion with multi-turn conversation."""
        request = {
            "model": "orchestrator",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
        }

        response = client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "content" in data["choices"][0]["message"]

    def test_chat_completions_missing_messages(self, client):
        """Test validation error for missing messages."""
        request = {
            "model": "orchestrator",
            # Missing messages field
        }

        response = client.post("/v1/chat/completions", json=request)

        assert response.status_code == 422  # Validation error

    def test_chat_completions_empty_messages(self, client):
        """Test error handling for empty messages array."""
        request = {
            "model": "orchestrator",
            "messages": [],
        }

        response = client.post("/v1/chat/completions", json=request)

        # Should return 400 for no user message
        assert response.status_code == 400
        assert "No user message" in response.json()["detail"]

    def test_chat_completions_no_user_message(self, client):
        """Test error when no user message provided."""
        request = {
            "model": "orchestrator",
            "messages": [
                {"role": "system", "content": "You are helpful."}
            ],
        }

        response = client.post("/v1/chat/completions", json=request)

        assert response.status_code == 400
        assert "No user message" in response.json()["detail"]

    def test_chat_completions_different_models(self, client):
        """Test chat completions with different model names."""
        for model_name in ["orchestrator", "frontdoor", "coder"]:
            request = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Test"}],
            }

            response = client.post("/v1/chat/completions", json=request)

            assert response.status_code == 200
            data = response.json()
            assert data["model"] == model_name

    def test_chat_completions_with_x_orchestrator_role(self, client):
        """Test forcing specific role via x_orchestrator_role."""
        request = {
            "model": "orchestrator",
            "messages": [{"role": "user", "content": "Test"}],
            "x_orchestrator_role": "coder",
        }

        response = client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200

    def test_chat_completions_with_x_show_routing(self, client):
        """Test routing metadata via x_show_routing."""
        request = {
            "model": "orchestrator",
            "messages": [{"role": "user", "content": "Test"}],
            "x_show_routing": True,
        }

        response = client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        # Should include orchestrator metadata
        assert "x_orchestrator_metadata" in data
        assert "role" in data["x_orchestrator_metadata"]
        assert "elapsed_seconds" in data["x_orchestrator_metadata"]


class TestChatCompletionsStreaming:
    """Test streaming chat completions."""

    def test_chat_completions_streaming_basic(self, client):
        """Test streaming chat completion."""
        request = {
            "model": "orchestrator",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": True,
        }

        with client.stream("POST", "/v1/chat/completions", json=request) as response:

            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            # Parse streaming response
            chunks = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break
                        chunk = json.loads(data_str)
                        chunks.append(chunk)

            # Should have received multiple chunks
            assert len(chunks) > 0

            # Check first chunk has role delta
            first_chunk = chunks[0]
            assert first_chunk["object"] == "chat.completion.chunk"
            assert "choices" in first_chunk
            assert "delta" in first_chunk["choices"][0]

            # Check last chunk has finish_reason
            last_chunk = chunks[-1]
            assert last_chunk["choices"][0]["finish_reason"] == "stop"

    def test_chat_completions_streaming_format(self, client):
        """Test streaming response follows OpenAI SSE format."""
        request = {
            "model": "orchestrator",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        with client.stream("POST", "/v1/chat/completions", json=request) as response:
            # Each chunk should be valid JSON after "data: " prefix
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str != "[DONE]":
                            # Should be valid JSON
                            chunk = json.loads(data_str)
                            assert "id" in chunk
                            assert "object" in chunk
                            assert chunk["object"] == "chat.completion.chunk"

    def test_chat_completions_streaming_with_metadata(self, client):
        """Test streaming with routing metadata."""
        request = {
            "model": "orchestrator",
            "messages": [{"role": "user", "content": "Test"}],
            "stream": True,
            "x_show_routing": True,
        }

        with client.stream("POST", "/v1/chat/completions", json=request) as response:
            assert response.status_code == 200

            # Final chunk should have metadata
            chunks = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        chunks.append(json.loads(data_str))

            if chunks:
                # Check for metadata in chunks
                has_metadata = any("x_orchestrator_metadata" in c for c in chunks)
                # Metadata may be in final chunk
                assert has_metadata or "x_role" in chunks[-1]


class TestRequestValidation:
    """Test request validation."""

    def test_invalid_temperature(self, client):
        """Test validation of temperature parameter."""
        request = {
            "model": "orchestrator",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 3.0,  # Out of range
        }

        response = client.post("/v1/chat/completions", json=request)

        # Should return 422 validation error
        assert response.status_code == 422

    def test_invalid_max_tokens(self, client):
        """Test validation of max_tokens parameter."""
        request = {
            "model": "orchestrator",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": -1,  # Invalid
        }

        response = client.post("/v1/chat/completions", json=request)

        assert response.status_code == 422

    def test_invalid_message_role(self, client):
        """Test message with invalid role."""
        request = {
            "model": "orchestrator",
            "messages": [{"role": "invalid_role", "content": "Test"}],
        }

        response = client.post("/v1/chat/completions", json=request)

        # May still process but extract no user messages
        # Depends on implementation - at minimum should not crash


class TestErrorHandling:
    """Test error handling."""

    def test_backend_error_handling(self, client):
        """Test graceful error handling when backend fails."""
        # Use an edge case that might trigger an error
        request = {
            "model": "orchestrator",
            "messages": [{"role": "user", "content": ""}],  # Empty content
        }

        response = client.post("/v1/chat/completions", json=request)

        # Should not crash - either success or graceful error
        assert response.status_code in [200, 400, 503]

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import concurrent.futures

        def make_request():
            request = {
                "model": "orchestrator",
                "messages": [{"role": "user", "content": "Test"}],
            }
            return client.post("/v1/chat/completions", json=request)

        # Make 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            results = [f.result() for f in futures]

        # All should succeed
        for response in results:
            assert response.status_code == 200
