#!/usr/bin/env python3
"""Unit tests for LLM primitives."""

from src.llm_primitives import (
    LLMPrimitives,
    LLMPrimitivesConfig,
)


class TestMockMode:
    """Test mock mode functionality."""

    def test_mock_mode_enabled_by_default(self):
        """Test that mock mode is enabled by default."""
        primitives = LLMPrimitives()
        assert primitives.mock_mode is True

    def test_mock_call_returns_mock_response(self):
        """Test that llm_call returns mock response in mock mode."""
        primitives = LLMPrimitives(mock_mode=True)
        result = primitives.llm_call("Test prompt")

        assert "[MOCK]" in result
        assert "worker" in result  # Default role

    def test_mock_call_includes_role(self):
        """Test that mock response includes the role."""
        primitives = LLMPrimitives(mock_mode=True)
        result = primitives.llm_call("Test prompt", role="coder")

        assert "coder" in result

    def test_mock_batch_returns_list(self):
        """Test that llm_batch returns list of mock responses."""
        primitives = LLMPrimitives(mock_mode=True)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = primitives.llm_batch(prompts)

        assert len(results) == 3
        assert all("[MOCK]" in r for r in results)
        assert "Batch[0]" in results[0]
        assert "Batch[1]" in results[1]
        assert "Batch[2]" in results[2]

    def test_custom_mock_responses(self):
        """Test that custom mock responses are used."""
        mock_responses = {
            "summarize": "This is a summary.",
            "translate": "Esta es una traduccion.",
        }
        primitives = LLMPrimitives(mock_mode=True, mock_responses=mock_responses)

        result1 = primitives.llm_call("Please summarize this text")
        result2 = primitives.llm_call("Please translate this")

        assert result1 == "This is a summary."
        assert result2 == "Esta es una traduccion."


class TestCallLogging:
    """Test call logging functionality."""

    def test_call_log_records_calls(self):
        """Test that calls are logged."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Test prompt")

        assert len(primitives.call_log) == 1
        assert primitives.call_log[0].call_type == "call"
        assert primitives.call_log[0].prompt == "Test prompt"

    def test_call_log_records_batch_calls(self):
        """Test that batch calls are logged."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_batch(["Prompt 1", "Prompt 2"])

        assert len(primitives.call_log) == 1
        assert primitives.call_log[0].call_type == "batch"
        assert len(primitives.call_log[0].prompts) == 2

    def test_call_log_includes_role(self):
        """Test that role is logged."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Test", role="architect")

        assert primitives.call_log[0].role == "architect"

    def test_call_log_includes_elapsed_time(self):
        """Test that elapsed time is recorded."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Test")

        assert primitives.call_log[0].elapsed_seconds >= 0

    def test_get_recent_calls(self):
        """Test get_recent_calls method."""
        primitives = LLMPrimitives(mock_mode=True)
        for i in range(5):
            primitives.llm_call(f"Prompt {i}")

        recent = primitives.get_recent_calls(3)
        assert len(recent) == 3

    def test_clear_log(self):
        """Test clear_log method."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Test")
        primitives.clear_log()

        assert len(primitives.call_log) == 0


class TestStats:
    """Test statistics tracking."""

    def test_total_calls_incremented(self):
        """Test that total_calls is incremented."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Test 1")
        primitives.llm_call("Test 2")

        assert primitives.total_calls == 2

    def test_total_batch_calls_incremented(self):
        """Test that total_batch_calls is incremented."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_batch(["A", "B"])
        primitives.llm_batch(["C", "D", "E"])

        assert primitives.total_batch_calls == 2

    def test_get_stats(self):
        """Test get_stats method."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Test")
        primitives.llm_batch(["A", "B"])

        stats = primitives.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_batch_calls"] == 1
        assert stats["mock_mode"] is True

    def test_reset_stats(self):
        """Test reset_stats method."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Test")
        primitives.reset_stats()

        assert primitives.total_calls == 0
        assert len(primitives.call_log) == 0


class TestContextSlice:
    """Test context slice handling."""

    def test_context_slice_appended_to_prompt(self):
        """Test that context_slice is appended to prompt."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Summarize:", context_slice="Some long text here")

        # Check that the call was logged with context_slice
        assert primitives.call_log[0].context_slice is not None

    def test_empty_context_slice(self):
        """Test that empty context_slice doesn't affect prompt."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Summarize:", context_slice="")

        assert primitives.call_log[0].context_slice is None

    def test_context_slice_truncated_in_log(self):
        """Test that long context_slice is truncated in log."""
        primitives = LLMPrimitives(mock_mode=True)
        long_context = "x" * 1000
        primitives.llm_call("Test", context_slice=long_context)

        # Log should have truncated context
        assert len(primitives.call_log[0].context_slice) <= 500


class TestOutputCapping:
    """Test output capping functionality."""

    def test_output_capped_at_limit(self):
        """Test that output is capped at configured limit."""
        config = LLMPrimitivesConfig(output_cap=100)
        mock_responses = {"test": "x" * 200}
        primitives = LLMPrimitives(
            mock_mode=True,
            config=config,
            mock_responses=mock_responses,
        )

        result = primitives.llm_call("test")
        assert len(result) <= 150  # Allow for truncation message
        assert "truncated" in result

    def test_short_output_not_capped(self):
        """Test that short output is not capped."""
        config = LLMPrimitivesConfig(output_cap=1000)
        mock_responses = {"test": "short response"}
        primitives = LLMPrimitives(
            mock_mode=True,
            config=config,
            mock_responses=mock_responses,
        )

        result = primitives.llm_call("test")
        assert "truncated" not in result

    def test_batch_output_capped(self):
        """Test that batch outputs are individually capped."""
        config = LLMPrimitivesConfig(output_cap=50)
        primitives = LLMPrimitives(mock_mode=True, config=config)

        # Mock responses will be longer than 50 chars
        results = primitives.llm_batch(["A", "B"])

        for result in results:
            assert len(result) <= 100  # Allow for truncation message


class TestConfig:
    """Test configuration options."""

    def test_default_config(self):
        """Test default configuration values."""
        primitives = LLMPrimitives(mock_mode=True)

        assert primitives.config.output_cap == 8192
        assert primitives.config.batch_parallelism == 4
        assert primitives.config.call_timeout == 600

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMPrimitivesConfig(
            output_cap=4096,
            batch_parallelism=8,
            call_timeout=60,
        )
        primitives = LLMPrimitives(mock_mode=True, config=config)

        assert primitives.config.output_cap == 4096
        assert primitives.config.batch_parallelism == 8
        assert primitives.config.call_timeout == 60

    def test_custom_mock_prefix(self):
        """Test custom mock response prefix."""
        config = LLMPrimitivesConfig(mock_response_prefix="[TEST]")
        primitives = LLMPrimitives(mock_mode=True, config=config)

        result = primitives.llm_call("Test")
        assert "[TEST]" in result


class TestRealModeWithoutServer:
    """Test real mode behavior when server is not configured."""

    def test_real_call_without_server_returns_error(self):
        """Test that real call without server returns error."""
        primitives = LLMPrimitives(mock_mode=False, model_server=None)
        result = primitives.llm_call("Test")

        assert "[ERROR:" in result

    def test_real_batch_without_server_returns_errors(self):
        """Test that real batch without server returns errors."""
        primitives = LLMPrimitives(mock_mode=False, model_server=None)
        results = primitives.llm_batch(["A", "B"])

        assert all("[ERROR:" in r for r in results)


class TestRoles:
    """Test role handling."""

    def test_default_role_is_worker(self):
        """Test that default role is 'worker'."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Test")

        assert primitives.call_log[0].role == "worker"

    def test_custom_role(self):
        """Test custom role."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives.llm_call("Test", role="ingest")

        assert primitives.call_log[0].role == "ingest"

    def test_batch_uses_same_role(self):
        """Test that batch uses the same role for all prompts."""
        primitives = LLMPrimitives(mock_mode=True)
        results = primitives.llm_batch(["A", "B"], role="coder")

        assert primitives.call_log[0].role == "coder"
        assert all("coder" in r for r in results)


class TestIntegrationWithREPL:
    """Test integration with REPLEnvironment."""

    def test_primitives_work_in_repl(self):
        """Test that primitives can be used from REPL."""
        from src.repl_environment import REPLEnvironment

        primitives = LLMPrimitives(mock_mode=True)
        repl = REPLEnvironment(
            context="Test context",
            llm_primitives=primitives,
        )

        result = repl.execute("print(llm_call('Summarize', context[:10]))")

        assert "[MOCK]" in result.output
        assert primitives.total_calls == 1

    def test_batch_primitives_work_in_repl(self):
        """Test that batch primitives can be used from REPL."""
        from src.repl_environment import REPLEnvironment

        primitives = LLMPrimitives(mock_mode=True)
        repl = REPLEnvironment(
            context="Test context",
            llm_primitives=primitives,
        )

        result = repl.execute("results = llm_batch(['A', 'B']); print(len(results))")

        assert "2" in result.output
        assert primitives.total_batch_calls == 1
