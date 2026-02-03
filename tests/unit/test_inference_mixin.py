"""Unit tests for inference mixin in LLMPrimitives.

Tests InferenceMixin methods through LLMPrimitives integration:
- _real_call() with CachingBackend and ModelServer paths
- _call_caching_backend() with circuit breaker
- _real_batch() with ThreadPoolExecutor
- Error handling (connection errors, timeouts)
"""

from unittest.mock import Mock, patch

import pytest

from src.llm_primitives import LLMPrimitives
from src.model_server import InferenceRequest, InferenceResult


@pytest.fixture
def mock_backend():
    """Create a mock backend that implements the caching interface."""
    backend = Mock()
    backend.infer = Mock()
    return backend


@pytest.fixture
def mock_model_server():
    """Create a mock ModelServer."""
    server = Mock()
    server.infer = Mock()
    return server


@pytest.fixture
def mock_health_tracker():
    """Create a mock BackendHealthTracker."""
    tracker = Mock()
    tracker.is_available = Mock(return_value=True)
    tracker.record_success = Mock()
    tracker.record_failure = Mock()
    return tracker


class TestInferenceMixinRealCall:
    """Tests for _real_call() method."""

    def test_real_call_with_caching_backend(self, mock_backend, mock_health_tracker):
        """Test _real_call uses CachingBackend when available."""
        # Create LLMPrimitives with backend
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"test_role": "http://localhost:8080"},
            health_tracker=mock_health_tracker,
        )
        prims._backends["test_role"] = mock_backend

        # Mock successful inference
        mock_backend.infer.return_value = InferenceResult(
            role="test_role",
            output="Backend response",
            tokens_generated=10,
            generation_speed=25.0,
            elapsed_time=0.4,
            success=True,
            prompt_eval_ms=100.0,
            generation_ms=300.0,
            predicted_per_second=25.0,
            http_overhead_ms=10.0,
        )

        result = prims._real_call("Test prompt", "test_role", n_tokens=128)

        assert result == "Backend response"
        mock_backend.infer.assert_called_once()
        assert prims.total_tokens_generated == 10
        assert prims.total_prompt_eval_ms == 100.0
        assert prims.total_generation_ms == 300.0
        assert prims.total_http_overhead_ms == 10.0

    def test_real_call_fallback_to_model_server(self, mock_model_server):
        """Test _real_call falls back to ModelServer when no backend."""
        prims = LLMPrimitives(
            mock_mode=False,
            model_server=mock_model_server,
        )

        # Mock successful inference
        mock_model_server.infer.return_value = InferenceResult(
            role="test_role",
            output="Server response",
            tokens_generated=15,
            generation_speed=30.0,
            elapsed_time=0.5,
            success=True,
        )

        result = prims._real_call("Test prompt", "test_role", n_tokens=256)

        assert result == "Server response"
        mock_model_server.infer.assert_called_once()
        # Check request structure
        call_args = mock_model_server.infer.call_args
        assert call_args[0][0] == "test_role"  # First positional arg is role
        request = call_args[0][1]
        assert isinstance(request, InferenceRequest)
        assert request.n_tokens == 256

    def test_real_call_no_backend_raises_error(self):
        """Test _real_call raises error when no backend configured."""
        prims = LLMPrimitives(mock_mode=False)

        with pytest.raises(RuntimeError, match="No backend configured"):
            prims._real_call("Test prompt", "unknown_role")


class TestCallCachingBackend:
    """Tests for _call_caching_backend() method."""

    def test_call_caching_backend_success(self, mock_backend, mock_health_tracker):
        """Test successful call to caching backend."""
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"coder": "http://localhost:8081"},
            health_tracker=mock_health_tracker,
        )

        mock_backend.infer.return_value = InferenceResult(
            role="coder",
            output="def hello(): pass",
            tokens_generated=20,
            generation_speed=40.0,
            elapsed_time=0.5,
            success=True,
            predicted_per_second=40.0,
        )

        result = prims._call_caching_backend(
            mock_backend, "Write hello function", "coder", n_tokens=128
        )

        assert result == "def hello(): pass"
        assert prims.total_tokens_generated == 20
        mock_health_tracker.record_success.assert_called_once_with("http://localhost:8081")

    def test_call_caching_backend_circuit_breaker_open(self, mock_backend, mock_health_tracker):
        """Test circuit breaker prevents call to unhealthy backend."""
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"coder": "http://localhost:8081"},
            health_tracker=mock_health_tracker,
        )

        # Simulate circuit open
        mock_health_tracker.is_available.return_value = False

        with pytest.raises(RuntimeError, match="circuit open"):
            prims._call_caching_backend(mock_backend, "Test prompt", "coder", n_tokens=128)

        # Should not call backend
        mock_backend.infer.assert_not_called()

    def test_call_caching_backend_inference_failure(self, mock_backend, mock_health_tracker):
        """Test backend records failure when inference fails."""
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"coder": "http://localhost:8081"},
            health_tracker=mock_health_tracker,
        )

        mock_backend.infer.return_value = InferenceResult(
            role="coder",
            output="",
            tokens_generated=0,
            generation_speed=0.0,
            elapsed_time=1.0,
            success=False,
            error_message="Model crashed",
        )

        with pytest.raises(RuntimeError, match="Inference failed"):
            prims._call_caching_backend(mock_backend, "Test prompt", "coder", n_tokens=128)

        mock_health_tracker.record_failure.assert_called_once_with("http://localhost:8081")

    def test_call_caching_backend_with_stop_sequences(self, mock_backend):
        """Test backend call with stop sequences."""
        prims = LLMPrimitives(mock_mode=False)

        mock_backend.infer.return_value = InferenceResult(
            role="worker",
            output="Output",
            tokens_generated=5,
            generation_speed=20.0,
            elapsed_time=0.25,
            success=True,
        )

        result = prims._call_caching_backend(
            mock_backend,
            "Test",
            "worker",
            n_tokens=64,
            stop_sequences=["END", "STOP"],
        )

        assert result == "Output"
        # Check that stop_sequences were passed
        call_args = mock_backend.infer.call_args
        request = call_args[0][1]
        assert request.stop_sequences == ["END", "STOP"]


class TestRealBatch:
    """Tests for _real_batch() method."""

    def test_real_batch_parallel_execution(self, mock_backend):
        """Test _real_batch executes calls in parallel."""
        prims = LLMPrimitives(mock_mode=False)
        prims._backends["worker"] = mock_backend

        # Mock responses
        def mock_infer(role_config, request):
            # Return different responses based on prompt
            prompt = request.prompt
            return InferenceResult(
                role="worker",
                output=f"Response to: {prompt[:20]}",
                tokens_generated=10,
                generation_speed=25.0,
                elapsed_time=0.1,
                success=True,
            )

        mock_backend.infer.side_effect = mock_infer

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = prims._real_batch(prompts, "worker")

        assert len(results) == 3
        assert "Prompt 1" in results[0]
        assert "Prompt 2" in results[1]
        assert "Prompt 3" in results[2]
        assert mock_backend.infer.call_count == 3

    def test_real_batch_handles_errors(self, mock_backend):
        """Test _real_batch handles individual call errors."""
        prims = LLMPrimitives(mock_mode=False)
        prims._backends["worker"] = mock_backend

        # Second call raises error
        call_count = [0]

        def mock_infer(role_config, request):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated failure")
            return InferenceResult(
                role="worker",
                output="Success",
                tokens_generated=5,
                generation_speed=20.0,
                elapsed_time=0.1,
                success=True,
            )

        mock_backend.infer.side_effect = mock_infer

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = prims._real_batch(prompts, "worker")

        assert len(results) == 3
        assert results[0] == "Success"
        assert "[ERROR:" in results[1]  # Error formatted
        assert results[2] == "Success"

    def test_real_batch_no_backend_raises_error(self):
        """Test _real_batch raises error when no backend available."""
        prims = LLMPrimitives(mock_mode=False)

        with pytest.raises(RuntimeError, match="No backend configured"):
            prims._real_batch(["Prompt 1", "Prompt 2"], "unknown_role")

    def test_real_batch_with_model_server(self, mock_model_server):
        """Test _real_batch uses ModelServer when no backends."""
        prims = LLMPrimitives(mock_mode=False, model_server=mock_model_server)

        # Mock _real_call to verify it's used
        with patch.object(prims, "_real_call", return_value="Mocked response") as mock_call:
            prompts = ["P1", "P2"]
            results = prims._real_batch(prompts, "worker")

        assert len(results) == 2
        assert all(r == "Mocked response" for r in results)
        assert mock_call.call_count == 2


class TestWorkerPoolBatch:
    """Tests for _worker_pool_batch() method."""

    def test_worker_pool_batch_routes_correctly(self):
        """Test worker pool batch routes to correct task type."""
        mock_pool = Mock()

        # mock_pool.batch returns a coroutine when called by production code
        async def mock_batch_async(prompts, task_type):
            return [f"Pool response: {p}" for p in prompts]

        mock_pool.batch = mock_batch_async

        prims = LLMPrimitives(
            mock_mode=False,
            worker_pool=mock_pool,
            use_worker_pool=True,
        )

        prompts = ["Explore this", "Analyze that"]

        # Patch get_event_loop to return a non-running loop so we always
        # hit the asyncio.run() branch (event loop state varies in full suite).
        mock_loop = Mock()
        mock_loop.is_running.return_value = False

        with (
            patch("asyncio.get_event_loop", return_value=mock_loop),
            patch("asyncio.run") as mock_run,
        ):
            mock_run.return_value = ["Result 1", "Result 2"]
            results = prims._worker_pool_batch(prompts, "worker_explore")

        assert len(results) == 2
        mock_run.assert_called_once()

    def test_worker_pool_batch_fallback_on_error(self, mock_model_server):
        """Test worker pool batch falls back on error."""
        mock_pool = Mock()
        mock_pool.batch.side_effect = RuntimeError("Pool unavailable")

        prims = LLMPrimitives(
            mock_mode=False,
            worker_pool=mock_pool,
            use_worker_pool=True,
            model_server=mock_model_server,
        )

        # Mock fallback
        with patch.object(
            prims, "_fallback_batch", return_value=["Fallback 1", "Fallback 2"]
        ) as mock_fallback:
            results = prims._worker_pool_batch(["P1", "P2"], "worker_code")

        assert results == ["Fallback 1", "Fallback 2"]
        mock_fallback.assert_called_once_with(["P1", "P2"], "worker_code")
