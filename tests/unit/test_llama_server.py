"""Unit tests for llama_server backend."""

from unittest.mock import Mock, patch

import httpx
import pytest

from src.backends.llama_server import (
    CacheStats,
    LlamaServerBackend,
    LlamaServerError,
    ServerConfig,
)
from src.model_server import InferenceRequest
from src.registry_loader import (
    AccelerationConfig,
    MemoryConfig,
    ModelConfig,
    PerformanceMetrics,
    RoleConfig,
)


@pytest.fixture
def role_config():
    """Create a minimal role config for testing."""
    return RoleConfig(
        name="test_role",
        tier="C",
        description="Test role",
        model=ModelConfig(
            name="test-model",
            path="test-model.gguf",
            quant="Q4_K_M",
            size_gb=1.0,
        ),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=10.0),
        memory=MemoryConfig(residency="hot"),
    )


@pytest.fixture
def server_config():
    """Create a server config for testing."""
    return ServerConfig(
        base_url="http://localhost:8080",
        timeout=120,
        num_slots=4,
        connect_timeout=5,
    )


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values from config module."""
        with patch("src.backends.llama_server._server_cfg") as mock_cfg:
            mock_cfg.return_value = Mock(
                default_url="http://test:8080",
                timeout=300,
                num_slots=8,
                connect_timeout=10,
                retry_count=3,
                retry_backoff=0.5,
            )
            config = ServerConfig()

            assert config.base_url == "http://test:8080"
            assert config.timeout == 300
            assert config.num_slots == 8

    def test_config_creation_with_overrides(self):
        """Test creating config with explicit values."""
        config = ServerConfig(
            base_url="http://custom:9090",
            timeout=60,
            num_slots=2,
        )

        assert config.base_url == "http://custom:9090"
        assert config.timeout == 60
        assert config.num_slots == 2


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        stats = CacheStats(total_requests=100, cache_hits=75, cache_misses=25)
        assert stats.hit_rate == 75.0

    def test_hit_rate_zero_requests(self):
        """Test hit rate with zero requests (division by zero edge case)."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_token_savings_rate(self):
        """Test token savings rate calculation."""
        stats = CacheStats(
            total_prompt_tokens=1000,
            cached_prompt_tokens=600,
        )
        assert stats.token_savings_rate == 60.0

    def test_token_savings_zero_tokens(self):
        """Test token savings with zero tokens (division by zero edge case)."""
        stats = CacheStats()
        assert stats.token_savings_rate == 0.0


class TestLlamaServerBackend:
    """Tests for LlamaServerBackend."""

    def test_initialization_with_config(self, server_config):
        """Test backend initialization with ServerConfig."""
        backend = LlamaServerBackend(config=server_config)

        assert backend.config.base_url == "http://localhost:8080"
        assert backend.config.timeout == 120
        assert isinstance(backend.cache_stats, CacheStats)
        assert backend._healthy is False

    def test_initialization_with_base_url(self):
        """Test backend initialization with just base_url."""
        backend = LlamaServerBackend(base_url="http://custom:9090")

        assert backend.config.base_url == "http://custom:9090"
        assert isinstance(backend.client, httpx.Client)

    def test_build_payload_minimal(self, role_config):
        """Test building payload with minimal request."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", n_tokens=128)

        payload = backend._build_payload(role_config, request)

        assert payload["prompt"] == "Hello"
        assert payload["n_predict"] == 128
        assert payload["cache_prompt"] is True  # Default
        assert payload["temperature"] == 0.0
        assert "top_k" in payload
        assert "top_p" in payload

    def test_build_payload_with_stop_sequences(self, role_config):
        """Test building payload with stop sequences."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(
            role="test",
            prompt="Hello",
            stop_sequences=["END", "STOP"],
        )

        payload = backend._build_payload(role_config, request)

        assert payload["stop"] == ["END", "STOP"]

    def test_build_payload_cache_prompt_override(self, role_config):
        """Test cache_prompt can be overridden per-request."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(
            role="test",
            prompt="Hello",
            cache_prompt=False,  # Override default
        )

        payload = backend._build_payload(role_config, request)

        assert payload["cache_prompt"] is False

    def test_build_payload_temperature_from_role(self, role_config):
        """Test temperature from role config overrides request."""
        role_config.acceleration.temperature = 0.7
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", temperature=0.2)

        payload = backend._build_payload(role_config, request)

        assert payload["temperature"] == 0.7  # Role wins

    def test_infer_success(self, role_config):
        """Test successful inference with mocked HTTP response."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", n_tokens=64)

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "Hello world",
            "tokens_predicted": 5,
            "tokens_evaluated": 10,
            "tokens_cached": 3,
            "timings": {
                "prompt_ms": 100.0,
                "predicted_ms": 50.0,
                "predicted_per_second": 33.0,
            },
        }

        with patch.object(backend.client, "post", return_value=mock_response):
            result = backend.infer(role_config, request)

        assert result.success is True
        assert result.output == "Hello world"
        assert result.tokens_generated == 5
        assert result.generation_speed == 33.0
        assert result.prompt_eval_ms == 100.0
        assert result.generation_ms == 50.0
        assert result.predicted_per_second == 33.0
        assert backend.cache_stats.cache_hits == 1  # cached_tokens > 0

    def test_infer_timeout(self, role_config):
        """Test inference timeout handling."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", timeout=10)

        with patch.object(backend.client, "post", side_effect=httpx.TimeoutException("Timeout")):
            result = backend.infer(role_config, request)

        assert result.success is False
        assert "timed out" in result.error_message.lower()
        assert result.tokens_generated == 0

    def test_infer_http_error(self, role_config):
        """Test inference with HTTP request error."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello")

        # Simulate network error
        with patch.object(
            backend.client, "post", side_effect=httpx.RequestError("Connection failed")
        ):
            result = backend.infer(role_config, request)

        assert result.success is False
        assert "Server request failed" in result.error_message

    def test_health_check_success(self):
        """Test health check with healthy server."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(backend.client, "get", return_value=mock_response):
            healthy = backend.health_check(0)

        assert healthy is True
        assert backend._healthy is True

    def test_health_check_failure(self):
        """Test health check with unreachable server."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        with patch.object(
            backend.client, "get", side_effect=httpx.RequestError("Connection failed")
        ):
            healthy = backend.health_check(0)

        assert healthy is False
        assert backend._healthy is False

    def test_health_check_rate_limiting(self):
        """Test health check is rate limited (< 1s between checks)."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        backend._healthy = True
        backend._last_health_check = 999999999.0  # Recent

        with patch("time.time", return_value=999999999.5):  # 0.5s later
            with patch.object(backend.client, "get") as mock_get:
                healthy = backend.health_check(0)

        # Should not make HTTP call (rate limited)
        mock_get.assert_not_called()
        assert healthy is True

    def test_get_slots(self):
        """Test fetching slot information."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 0, "state": "idle", "n_past": 100, "n_cache": 50},
            {"id": 1, "state": "processing", "n_past": 200, "n_cache": 0},
        ]

        with patch.object(backend.client, "get", return_value=mock_response):
            slots = backend.get_slots()

        assert len(slots) == 2
        assert slots[0].slot_id == 0
        assert slots[0].state == "idle"
        assert slots[0].prompt_tokens == 100
        assert slots[0].cache_tokens == 50
        assert slots[1].slot_id == 1
        assert slots[1].state == "processing"

    def test_save_slot_success(self):
        """Test saving slot state."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(backend.client, "post", return_value=mock_response):
            success = backend.save_slot(0, "/tmp/slot_0.cache")

        assert success is True

    def test_save_slot_failure(self):
        """Test saving slot with HTTP error."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        with patch.object(backend.client, "post", side_effect=httpx.RequestError("Network error")):
            success = backend.save_slot(0, "/tmp/slot_0.cache")

        assert success is False

    def test_load_raises_error_on_unhealthy_server(self, role_config):
        """Test load() raises error if server not reachable."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        with patch.object(backend, "health_check", return_value=False):
            with pytest.raises(LlamaServerError, match="Cannot reach"):
                backend.load(role_config)

    def test_unload_always_succeeds(self):
        """Test unload() is a no-op that returns True."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        assert backend.unload(12345) is True


class TestEarlyStopTiming:
    """Tests for timing telemetry when early-stop breaks the SSE stream."""

    def test_early_stop_produces_timing(self, role_config, server_config):
        """When on_chunk raises StopIteration, timings should still be set."""
        backend = LlamaServerBackend(config=server_config)

        # Simulate SSE stream with 5 chunks before early-stop
        sse_lines = [
            'data: {"content": "Hello"}',
            'data: {"content": " world"}',
            'data: {"content": "!"}',
            'data: {"content": " FINAL"}',
            'data: {"content": "(42)"}',
            # stop event would follow but early-stop breaks before it
        ]

        call_count = 0

        def on_chunk(content):
            nonlocal call_count
            call_count += 1
            if call_count >= 4:
                raise StopIteration

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(sse_lines)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with patch.object(backend.client, "stream", return_value=FakeResponse()):
            request = InferenceRequest(
                role="test_role",
                prompt="test",
                n_tokens=100,
                timeout=30,
            )
            result = backend.infer_stream_text(role_config, request, on_chunk=on_chunk)

        # Before the fix: generation_ms == 0 because timings dict was empty
        # After the fix: generation_ms > 0 computed from wall clock
        assert result.tokens_generated == 4  # chunks before stop
        assert result.generation_ms > 0, "Early-stop should still produce timing"
        assert result.predicted_per_second > 0, "Early-stop should still produce TPS"
