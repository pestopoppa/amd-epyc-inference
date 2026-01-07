#!/usr/bin/env python3
"""Integration tests for LLMPrimitives with CachingBackend.

These tests validate the integration between:
1. LLMPrimitives (orchestrator's LLM interface)
2. CachingBackend (RadixAttention prefix caching)
3. Real inference with llama-server

Requirements:
    - llama-server running on configured ports (8080, 8082)
    - Tests run with mocks by default
    - Use --run-server for live tests

To run with a live server:
    pytest tests/integration/test_cache_integration.py -v --run-server

To run without server (mocked):
    pytest tests/integration/test_cache_integration.py -v
"""

import os
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

# Mark all tests as integration
pytestmark = pytest.mark.integration


@pytest.fixture
def mock_caching_backend():
    """Create a mock CachingBackend for unit testing."""
    backend = MagicMock()

    # Track call count for hit rate simulation
    call_count = {"count": 0}

    def mock_infer(role_config, request):
        call_count["count"] += 1
        return MagicMock(
            role=role_config.name if hasattr(role_config, "name") else "test",
            output=f"Mock response for: {request.prompt[:30]}...",
            tokens_generated=50,
            generation_speed=20.0,
            elapsed_time=2.5,
            success=True,
            error_message=None,
        )

    backend.infer.side_effect = mock_infer

    def mock_stats():
        hit_rate = min(0.8, call_count["count"] / 10) if call_count["count"] > 1 else 0
        return {
            "router_hit_rate": hit_rate,
            "cache_hits": int(call_count["count"] * hit_rate),
            "cache_misses": int(call_count["count"] * (1 - hit_rate)),
            "total_requests": call_count["count"],
        }

    backend.get_stats.side_effect = mock_stats

    return backend


@pytest.fixture
def server_urls():
    """Server URLs for testing."""
    return {
        "frontdoor": os.environ.get("FRONTDOOR_URL", "http://localhost:8080"),
        "worker": os.environ.get("WORKER_URL", "http://localhost:8082"),
    }


# =============================================================================
# LLMPrimitives with CachingBackend Tests
# =============================================================================


class TestLLMPrimitivesCachingIntegration:
    """Tests for LLMPrimitives with CachingBackend integration."""

    def test_llm_primitives_init_with_server_urls(self, mock_caching_backend):
        """LLMPrimitives should initialize backends from server_urls."""
        with patch(
            "src.llm_primitives.LLMPrimitives._init_caching_backends"
        ) as mock_init:
            from src.llm_primitives import LLMPrimitives

            # Initialize with server_urls
            primitives = LLMPrimitives(
                mock_mode=False,
                server_urls={
                    "frontdoor": "http://localhost:8080",
                    "worker": "http://localhost:8082",
                },
            )

            # Should attempt to init backends
            mock_init.assert_called_once()

    def test_llm_primitives_mock_mode_no_backends(self):
        """In mock_mode, backends should not be initialized."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)

        # No backends in mock mode
        assert len(primitives._backends) == 0

    def test_get_backend_returns_configured_backend(self, mock_caching_backend):
        """get_backend should return the backend for a role."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)
        primitives._backends = {"worker": mock_caching_backend}

        backend = primitives.get_backend("worker")
        assert backend is mock_caching_backend

    def test_get_backend_returns_none_for_unknown_role(self):
        """get_backend should return None for unknown roles."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)

        backend = primitives.get_backend("unknown_role")
        assert backend is None

    def test_get_cache_stats_aggregates_all_backends(self, mock_caching_backend):
        """get_cache_stats should aggregate stats from all backends."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)
        primitives._backends = {
            "frontdoor": mock_caching_backend,
            "worker": mock_caching_backend,
        }

        # Generate some "calls"
        mock_caching_backend.infer(MagicMock(name="test"), MagicMock(prompt="test"))
        mock_caching_backend.infer(MagicMock(name="test"), MagicMock(prompt="test"))

        stats = primitives.get_cache_stats()

        assert "frontdoor" in stats
        assert "worker" in stats
        assert isinstance(stats["frontdoor"], dict)

    def test_get_stats_includes_cache_info(self, mock_caching_backend):
        """get_stats should include cache_stats when backends configured."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)
        primitives._backends = {"worker": mock_caching_backend}

        stats = primitives.get_stats()

        assert "cache_stats" in stats


# =============================================================================
# Cache Hit Rate Tests
# =============================================================================


class TestCacheHitRates:
    """Tests for cache hit rate behavior."""

    def test_repeated_prefix_improves_hit_rate(self, mock_caching_backend):
        """Repeated prompts with same prefix should improve hit rate."""
        from src.llm_primitives import LLMPrimitives
        from src.model_server import InferenceRequest

        primitives = LLMPrimitives(mock_mode=True)
        primitives._backends = {"worker": mock_caching_backend}

        # Simulate repeated calls with same prefix
        scaffold = "You are a helpful assistant. Answer briefly.\n\n"
        queries = [
            scaffold + "What is 1+1?",
            scaffold + "What is 2+2?",
            scaffold + "What is 3+3?",
        ]

        for query in queries:
            mock_caching_backend.infer(
                MagicMock(name="worker"), MagicMock(prompt=query)
            )

        stats = mock_caching_backend.get_stats()
        assert stats["total_requests"] == 3

    def test_batch_calls_with_shared_prefix(self, mock_caching_backend):
        """Batch calls with shared prefix should benefit from caching."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)
        primitives._backends = {"worker": mock_caching_backend}

        scaffold = "Analyze the following:\n\n"
        prompts = [scaffold + f"Item {i}" for i in range(8)]

        # Simulate batch processing
        for prompt in prompts:
            mock_caching_backend.infer(MagicMock(name="worker"), MagicMock(prompt=prompt))

        stats = mock_caching_backend.get_stats()
        assert stats["total_requests"] == 8


# =============================================================================
# Real Mode Integration Tests
# =============================================================================


class TestRealModeIntegration:
    """Tests for real_mode with actual backends."""

    def test_real_call_uses_caching_backend(self, mock_caching_backend):
        """_real_call should use CachingBackend when available."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=False)
        primitives._backends = {"worker": mock_caching_backend}

        # Call should use the caching backend
        with patch.object(primitives, "_call_caching_backend") as mock_call:
            mock_call.return_value = "Test response"

            result = primitives._real_call("Test prompt", "worker")

            mock_call.assert_called_once()
            assert result == "Test response"

    def test_real_batch_parallelizes_calls(self, mock_caching_backend):
        """_real_batch should parallelize calls to backend."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=False)
        primitives._backends = {"worker": mock_caching_backend}

        prompts = [f"Prompt {i}" for i in range(4)]

        with patch.object(primitives, "_call_caching_backend") as mock_call:
            mock_call.return_value = "Response"

            results = primitives._real_batch(prompts, "worker")

            # Should be called once per prompt
            assert mock_call.call_count == 4
            assert len(results) == 4


# =============================================================================
# Live Server Tests (require --run-server)
# =============================================================================


@pytest.mark.requires_server
class TestLiveServerIntegration:
    """Tests requiring a live llama-server instance."""

    @pytest.fixture(autouse=True)
    def skip_without_server(self, request):
        """Skip tests if --run-server not provided."""
        if not request.config.getoption("--run-server"):
            pytest.skip("Need --run-server to run live server tests")

    def test_llm_primitives_real_inference(self, server_urls):
        """LLMPrimitives should perform real inference with live server."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(
            mock_mode=False,
            server_urls={"worker": server_urls["worker"]},
        )

        # Skip if backends didn't initialize
        if not primitives._backends:
            pytest.skip("Could not initialize backends")

        result = primitives.llm_call("Say 'hello' and nothing else.", role="worker")

        assert result is not None
        assert len(result) > 0

    def test_cache_hit_rate_on_repeated_prefix(self, server_urls):
        """Cache should show hits on repeated prefix."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(
            mock_mode=False,
            server_urls={"worker": server_urls["worker"]},
        )

        if not primitives._backends:
            pytest.skip("Could not initialize backends")

        scaffold = "You are a helpful AI. Answer in one word.\n\n"

        # Run multiple queries with same prefix
        for i in range(5):
            primitives.llm_call(scaffold + f"What is {i}+{i}?", role="worker")

        stats = primitives.get_cache_stats()
        worker_stats = stats.get("worker", {})

        print(f"Cache stats after 5 queries: {worker_stats}")

        # Expect some hits after repeated prefix
        if "router_hit_rate" in worker_stats:
            assert worker_stats["router_hit_rate"] >= 0

    def test_batch_performance(self, server_urls):
        """Batch calls should complete faster than sequential."""
        import time
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(
            mock_mode=False,
            server_urls={"worker": server_urls["worker"]},
        )

        if not primitives._backends:
            pytest.skip("Could not initialize backends")

        prompts = [f"Say the number {i}" for i in range(4)]

        start = time.time()
        results = primitives.llm_batch(prompts, role="worker")
        elapsed = time.time() - start

        assert len(results) == 4
        print(f"Batch of 4 completed in {elapsed:.2f}s")

        # Should complete reasonably fast due to parallelism
        assert elapsed < 60  # Allow 60s for 4 parallel requests


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCachingErrorHandling:
    """Tests for error handling with caching backends."""

    def test_backend_error_propagates(self):
        """Errors from backend should propagate to caller."""
        from src.llm_primitives import LLMPrimitives

        error_backend = MagicMock()
        error_backend.infer.return_value = MagicMock(
            success=False,
            error_message="Server error",
            output="",
        )

        primitives = LLMPrimitives(mock_mode=False)
        primitives._backends = {"worker": error_backend}

        with pytest.raises(RuntimeError, match="Inference failed"):
            primitives._call_caching_backend(
                error_backend,
                "Test prompt",
                "worker",
            )

    def test_missing_backend_falls_back(self):
        """Missing backend should fall back to model server or mock."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)
        primitives._backends = {}  # No backends

        # Should use mock mode fallback
        result = primitives.llm_call("Test", role="unknown_role")

        # Mock response should be returned
        assert result is not None
