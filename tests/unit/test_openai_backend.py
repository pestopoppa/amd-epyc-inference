"""Unit tests for OpenAIBackend.

Tests are mocked by default. Use pytest -m integration with OPENAI_API_KEY
set to run live tests.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.backends.openai import OpenAIBackend
from src.backends.protocol import InferenceRequest
from src.config import ExternalAPIConfig


@pytest.fixture
def mock_config() -> ExternalAPIConfig:
    """Create a mock API config."""
    return ExternalAPIConfig(
        api_key="test-api-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        timeout=30,
        max_retries=3,
    )


@pytest.fixture
def mock_response() -> dict:
    """Create a mock API response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1699000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 9,
            "total_tokens": 21,
        },
    }


class TestOpenAIBackendInit:
    """Tests for OpenAIBackend initialization."""

    def test_init_with_valid_config(self, mock_config: ExternalAPIConfig) -> None:
        """Test initialization with valid config."""
        with patch("httpx.Client"):
            backend = OpenAIBackend(mock_config)
            assert backend.config == mock_config
            assert backend.stats.total_requests == 0

    def test_init_without_api_key_raises(self) -> None:
        """Test that missing API key raises ValueError."""
        config = ExternalAPIConfig(api_key="", base_url="https://api.openai.com/v1")
        with pytest.raises(ValueError, match="API key is required"):
            OpenAIBackend(config)


class TestOpenAIBackendInfer:
    """Tests for OpenAIBackend.infer()."""

    def test_infer_success(
        self, mock_config: ExternalAPIConfig, mock_response: dict
    ) -> None:
        """Test successful inference."""
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = mock_response
            mock_http_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_http_response

            backend = OpenAIBackend(mock_config)
            request = InferenceRequest(prompt="Hello!", max_tokens=100)
            result = backend.infer(request)

            assert result.success is True
            assert result.text == "Hello! How can I help you today?"
            assert result.tokens_generated == 9
            assert result.prompt_tokens == 12
            assert result.elapsed_seconds > 0
            assert backend.stats.total_requests == 1

    def test_infer_http_error(self, mock_config: ExternalAPIConfig) -> None:
        """Test HTTP error handling."""
        import httpx

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            error_response = MagicMock()
            error_response.status_code = 429
            error_response.text = "Rate limit exceeded"

            mock_client.post.side_effect = httpx.HTTPStatusError(
                message="Rate limit",
                request=MagicMock(),
                response=error_response,
            )

            backend = OpenAIBackend(mock_config)
            request = InferenceRequest(prompt="Hello!")
            result = backend.infer(request)

            assert result.success is False
            assert "429" in result.error
            assert backend.stats.errors == 1

    def test_infer_network_error(self, mock_config: ExternalAPIConfig) -> None:
        """Test network error handling."""
        import httpx

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            mock_client.post.side_effect = httpx.RequestError(
                message="Connection refused",
                request=MagicMock(),
            )

            backend = OpenAIBackend(mock_config)
            request = InferenceRequest(prompt="Hello!")
            result = backend.infer(request)

            assert result.success is False
            assert "Request error" in result.error
            assert backend.stats.errors == 1


class TestOpenAIBackendPayload:
    """Tests for payload building."""

    def test_build_payload_basic(self, mock_config: ExternalAPIConfig) -> None:
        """Test basic payload building."""
        with patch("httpx.Client"):
            backend = OpenAIBackend(mock_config)
            request = InferenceRequest(
                prompt="Test prompt",
                max_tokens=100,
                temperature=0.7,
            )
            payload = backend._build_payload(request)

            assert payload["model"] == "gpt-4o"
            assert payload["max_tokens"] == 100
            assert payload["temperature"] == 0.7
            assert len(payload["messages"]) == 1
            assert payload["messages"][0]["content"] == "Test prompt"
            assert payload["messages"][0]["role"] == "user"

    def test_build_payload_with_system(self, mock_config: ExternalAPIConfig) -> None:
        """Test payload with system prompt."""
        with patch("httpx.Client"):
            backend = OpenAIBackend(mock_config)
            request = InferenceRequest(
                prompt="Hello",
                max_tokens=50,
                extra={"system": "You are a helpful assistant."},
            )
            payload = backend._build_payload(request)

            assert len(payload["messages"]) == 2
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][0]["content"] == "You are a helpful assistant."
            assert payload["messages"][1]["role"] == "user"

    def test_build_payload_with_stop_sequences(
        self, mock_config: ExternalAPIConfig
    ) -> None:
        """Test payload with stop sequences."""
        with patch("httpx.Client"):
            backend = OpenAIBackend(mock_config)
            request = InferenceRequest(
                prompt="Hello",
                max_tokens=50,
                stop_sequences=["END", "STOP"],
            )
            payload = backend._build_payload(request)

            assert payload["stop"] == ["END", "STOP"]

    def test_build_payload_with_custom_messages(
        self, mock_config: ExternalAPIConfig
    ) -> None:
        """Test payload with pre-built messages."""
        with patch("httpx.Client"):
            backend = OpenAIBackend(mock_config)
            messages = [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ]
            request = InferenceRequest(
                prompt="",  # ignored when messages provided
                max_tokens=50,
                extra={"messages": messages},
            )
            payload = backend._build_payload(request)

            assert payload["messages"] == messages


class TestOpenAIBackendHealthCheck:
    """Tests for health check."""

    def test_health_check_success(self, mock_config: ExternalAPIConfig) -> None:
        """Test successful health check."""
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response

            backend = OpenAIBackend(mock_config)
            assert backend.health_check() is True

    def test_health_check_failure(self, mock_config: ExternalAPIConfig) -> None:
        """Test failed health check."""
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            mock_client.get.side_effect = Exception("Connection failed")

            backend = OpenAIBackend(mock_config)
            assert backend.health_check() is False


class TestOpenAIBackendStats:
    """Tests for statistics tracking."""

    def test_stats_accumulate(
        self, mock_config: ExternalAPIConfig, mock_response: dict
    ) -> None:
        """Test that stats accumulate across requests."""
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = mock_response
            mock_http_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_http_response

            backend = OpenAIBackend(mock_config)

            # Make multiple requests
            for _ in range(3):
                request = InferenceRequest(prompt="Hello!", max_tokens=100)
                backend.infer(request)

            stats = backend.get_stats()
            assert stats.total_requests == 3
            assert stats.total_tokens_generated == 27  # 9 * 3
            assert stats.total_prompt_tokens == 36  # 12 * 3


class TestOpenAIBackendContextManager:
    """Tests for context manager protocol."""

    def test_context_manager(self, mock_config: ExternalAPIConfig) -> None:
        """Test context manager usage."""
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            with OpenAIBackend(mock_config) as backend:
                assert backend is not None

            mock_client.close.assert_called_once()


class TestOpenAIBackendCompatibility:
    """Tests for OpenAI-compatible endpoints."""

    def test_custom_base_url(self) -> None:
        """Test with custom base URL (e.g., local server)."""
        config = ExternalAPIConfig(
            api_key="local-key",
            base_url="http://localhost:8080/v1",
            default_model="local-model",
        )

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            backend = OpenAIBackend(config)
            assert backend.config.base_url == "http://localhost:8080/v1"

            # Verify client was configured with correct base URL
            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:8080/v1"


# Integration tests (require API key)
@pytest.mark.integration
class TestOpenAIBackendIntegration:
    """Integration tests requiring actual API key."""

    @pytest.fixture
    def live_config(self) -> ExternalAPIConfig:
        """Create config from environment."""
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        return ExternalAPIConfig(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            default_model="gpt-4o-mini",  # Use cheaper model for tests
            timeout=60,
        )

    def test_live_inference(self, live_config: ExternalAPIConfig) -> None:
        """Test live inference against real API."""
        backend = OpenAIBackend(live_config)
        request = InferenceRequest(
            prompt="Say 'Hello, World!' and nothing else.",
            max_tokens=20,
            temperature=0,
        )
        result = backend.infer(request)

        assert result.success is True
        assert "Hello" in result.text
        assert result.tokens_generated > 0

    def test_live_health_check(self, live_config: ExternalAPIConfig) -> None:
        """Test live health check."""
        backend = OpenAIBackend(live_config)
        assert backend.health_check() is True
