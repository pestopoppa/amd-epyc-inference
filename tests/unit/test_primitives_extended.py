"""Extended unit tests for LLMPrimitives.

Focuses on uncovered paths in primitives.py:
- Recursion depth limit enforcement
- Output truncation
- llm_batch() in mock mode
- Token estimation methods
- Call log entries
- get_stats() structure
"""

from unittest.mock import patch

import pytest

from src.llm_primitives import LLMPrimitives
from src.llm_primitives.config import LLMPrimitivesConfig


class TestRecursionDepthLimit:
    """Tests for recursion depth enforcement."""

    def test_recursion_depth_limit_enforced(self):
        """Test that max_recursion_depth is enforced."""
        config = LLMPrimitivesConfig(max_recursion_depth=2)

        # Create a mock that triggers recursive calls
        recursion_count = [0]

        def mock_response_factory():
            recursion_count[0] += 1
            # Return a value that will be interpreted by the test logic
            return f"recursion_{recursion_count[0]}"

        prims = LLMPrimitives(mock_mode=True, config=config)

        # Simulate recursion by having llm_call call itself
        # We'll do this by patching the implementation to recurse
        original_impl = prims._llm_call_impl

        def recursive_impl(
            prompt,
            context_slice="",
            role="worker",
            n_tokens=None,
            skip_suffix=False,
            stop_sequences=None,
            persona=None,
            json_schema=None,
            grammar=None,
        ):
            # Call llm_call again (which will check recursion depth)
            if prims._recursion_depth < 5:  # Try to exceed limit of 2
                return prims.llm_call("Nested call", role=role)
            return "Done"

        with pytest.raises(RecursionError, match="Maximum recursion depth"):
            prims._llm_call_impl = recursive_impl
            try:
                prims.llm_call("Start", role="worker")
            finally:
                prims._llm_call_impl = original_impl

    def test_recursion_depth_tracking(self):
        """Test that recursion depth is tracked correctly."""
        config = LLMPrimitivesConfig(max_recursion_depth=5)
        prims = LLMPrimitives(mock_mode=True, config=config)

        # Single call - depth should be 1
        prims.llm_call("test", role="worker")
        # After call completes, depth should be back to 0
        assert prims._recursion_depth == 0
        # Max reached should be 1
        assert prims._max_recursion_depth_reached == 1

    def test_recursion_depth_resets_after_error(self):
        """Test recursion depth resets even after error."""
        LLMPrimitivesConfig(max_recursion_depth=3)
        prims = LLMPrimitives(mock_mode=False)  # No backend - will error

        try:
            prims.llm_call("test", role="worker")
        except RuntimeError:
            pass  # Expected

        # Depth should be reset
        assert prims._recursion_depth == 0


class TestOutputTruncation:
    """Tests for output capping."""

    def test_output_truncation_in_llm_call(self):
        """Test that long outputs are truncated."""
        config = LLMPrimitivesConfig(output_cap=100)
        long_response = "x" * 200
        prims = LLMPrimitives(
            mock_mode=True,
            config=config,
            mock_responses={"test": long_response},  # Key must match prompt
        )

        result = prims.llm_call("test", role="worker")

        assert len(result) <= 100 + 50  # Cap + truncation message
        assert "truncated at 100 chars" in result

    def test_output_not_truncated_when_under_cap(self):
        """Test that short outputs are not truncated."""
        config = LLMPrimitivesConfig(output_cap=1000)
        short_response = "Short response"
        prims = LLMPrimitives(
            mock_mode=True,
            config=config,
            mock_responses={"test": short_response},  # Key must match prompt
        )

        result = prims.llm_call("test", role="worker")

        assert result == short_response
        assert "truncated" not in result

    def test_output_truncation_in_batch(self):
        """Test that batch outputs are truncated individually."""
        config = LLMPrimitivesConfig(output_cap=50)
        prims = LLMPrimitives(mock_mode=True, config=config)

        # Mock batch will generate long responses
        long_text = "y" * 100

        with patch.object(prims, "_mock_batch", return_value=[long_text, long_text]):
            results = prims.llm_batch(["p1", "p2"], role="worker")

        assert len(results) == 2
        for result in results:
            assert len(result) <= 50 + 50  # Cap + message
            assert "truncated at 50 chars" in result


class TestLLMBatchMockMode:
    """Tests for llm_batch() in mock mode."""

    def test_llm_batch_mock_mode(self):
        """Test llm_batch with multiple prompts in mock mode."""
        prims = LLMPrimitives(mock_mode=True)

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = prims.llm_batch(prompts, role="worker")

        assert len(results) == 3
        # All should be mock responses (default format)
        assert all("[MOCK]" in r or "Batch" in r for r in results)

    def test_llm_batch_with_persona(self):
        """Test llm_batch applies persona to all prompts."""
        prims = LLMPrimitives(mock_mode=True)

        with patch.object(prims, "_apply_persona_prefix") as mock_persona:
            mock_persona.side_effect = lambda p, persona: f"[{persona}] {p}"

            prompts = ["P1", "P2"]
            prims.llm_batch(prompts, role="worker", persona="security")

        # Should be called for each prompt
        assert mock_persona.call_count == 2
        assert all(call[0][1] == "security" for call in mock_persona.call_args_list)

    def test_llm_batch_updates_stats(self):
        """Test llm_batch updates batch call counter."""
        prims = LLMPrimitives(mock_mode=True)

        initial_count = prims.total_batch_calls
        prims.llm_batch(["P1", "P2", "P3"], role="worker")

        assert prims.total_batch_calls == initial_count + 1


class TestTokenEstimation:
    """Tests for token estimation methods."""

    def test_estimate_prompt_tokens(self):
        """Test prompt token estimation."""
        prims = LLMPrimitives(mock_mode=True)

        # Rough estimate: 1 token per 4 chars
        prompt = "This is a test prompt with some words"
        estimated = prims._estimate_prompt_tokens(prompt)

        # Should be roughly len(prompt) / 4
        assert estimated > 0
        assert estimated == len(prompt) // 4

    def test_estimate_completion_tokens(self):
        """Test completion token estimation."""
        prims = LLMPrimitives(mock_mode=True)

        completion = "This is the model's response text"
        estimated = prims._estimate_completion_tokens(completion)

        assert estimated > 0
        assert estimated == len(completion) // 4

    def test_token_estimation_empty_string(self):
        """Test token estimation with empty strings."""
        prims = LLMPrimitives(mock_mode=True)

        assert prims._estimate_prompt_tokens("") == 0
        assert prims._estimate_completion_tokens("") == 0


class TestCallLog:
    """Tests for call log tracking."""

    def test_call_log_entries_created(self):
        """Test that call log entries are created."""
        prims = LLMPrimitives(mock_mode=True)

        prims.llm_call("Test prompt", role="worker")

        assert len(prims.call_log) == 1
        entry = prims.call_log[0]
        assert entry.call_type == "call"
        assert entry.role == "worker"
        assert entry.prompt == "Test prompt"
        assert entry.result is not None

    def test_call_log_batch_entries(self):
        """Test batch calls create batch log entries."""
        prims = LLMPrimitives(mock_mode=True)

        prims.llm_batch(["P1", "P2"], role="coder")

        assert len(prims.call_log) == 1
        entry = prims.call_log[0]
        assert entry.call_type == "batch"
        assert entry.role == "coder"
        assert entry.prompts is not None
        assert len(entry.prompts) == 2

    def test_call_log_truncates_context(self):
        """Test call log truncates long context slices."""
        prims = LLMPrimitives(mock_mode=True)

        long_context = "x" * 1000
        prims.llm_call("Test", context_slice=long_context, role="worker")

        entry = prims.call_log[0]
        # Context should be truncated to 500 chars
        assert len(entry.context_slice) <= 500

    def test_call_log_records_persona(self):
        """Test call log records persona parameter."""
        prims = LLMPrimitives(mock_mode=True)

        prims.llm_call("Test", role="worker", persona="security_auditor")

        entry = prims.call_log[0]
        assert entry.persona == "security_auditor"

    def test_call_log_records_elapsed_time(self):
        """Test call log records elapsed time."""
        prims = LLMPrimitives(mock_mode=True)

        prims.llm_call("Test", role="worker")

        entry = prims.call_log[0]
        assert entry.elapsed_seconds > 0  # Should have some time

    def test_call_log_records_errors(self):
        """Test call log records errors."""
        prims = LLMPrimitives(mock_mode=False)  # No backend - will error

        result = prims.llm_call("Test", role="worker")

        entry = prims.call_log[0]
        assert entry.error is not None
        assert "ERROR" in result


class TestGetStats:
    """Tests for get_stats() method."""

    def test_get_stats_structure(self):
        """Test get_stats returns expected structure."""
        prims = LLMPrimitives(mock_mode=True)

        # Make some calls
        prims.llm_call("Test 1", role="worker")
        prims.llm_call("Test 2", role="worker")
        prims.llm_batch(["B1", "B2"], role="coder")

        stats = prims.get_stats()

        assert "total_calls" in stats
        assert "total_batch_calls" in stats
        assert "total_tokens_generated" in stats
        assert stats["total_calls"] == 2
        assert stats["total_batch_calls"] == 1

    def test_get_stats_timing_data(self):
        """Test get_stats includes timing data."""
        prims = LLMPrimitives(mock_mode=True)

        prims.llm_call("Test", role="worker")

        stats = prims.get_stats()

        # Should have timing fields (even if zero in mock mode)
        assert "total_prompt_eval_ms" in stats
        assert "total_generation_ms" in stats

    def test_get_stats_recursion_depth(self):
        """Test get_stats includes max recursion depth."""
        config = LLMPrimitivesConfig(max_recursion_depth=5)
        prims = LLMPrimitives(mock_mode=True, config=config)

        prims.llm_call("Test", role="worker")

        stats = prims.get_stats()

        assert "max_recursion_depth_reached" in stats
        assert stats["max_recursion_depth_reached"] == 1


class TestQueryCostTracking:
    """Tests for per-query cost tracking integration."""

    def test_call_updates_current_query_cost(self):
        """Test that calls update current query cost when set."""
        prims = LLMPrimitives(mock_mode=True)

        # Set current query
        from src.llm_primitives.types import QueryCost

        query = QueryCost(query_id="test_query")
        prims._current_query = query

        prims.llm_call("Test prompt", role="worker")

        # Query should have accumulated costs
        assert query.prompt_tokens > 0
        assert query.completion_tokens > 0
        assert query.calls_made == 1

    def test_batch_updates_current_query_cost(self):
        """Test that batch calls update current query cost."""
        prims = LLMPrimitives(mock_mode=True)

        from src.llm_primitives.types import QueryCost

        query = QueryCost(query_id="batch_query")
        prims._current_query = query

        prims.llm_batch(["P1", "P2", "P3"], role="worker")

        assert query.batch_calls_made == 1
        assert query.total_tokens > 0

    def test_no_query_tracking_when_none(self):
        """Test no error when current query is None."""
        prims = LLMPrimitives(mock_mode=True)
        prims._current_query = None

        # Should not raise
        prims.llm_call("Test", role="worker")
        prims.llm_batch(["P1"], role="worker")
