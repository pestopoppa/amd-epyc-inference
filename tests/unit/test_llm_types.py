#!/usr/bin/env python3
"""Unit tests for LLM primitive types (CallLogEntry, LLMResult, QueryCost)."""

from src.llm_primitives.types import CallLogEntry, LLMResult, QueryCost


class TestCallLogEntry:
    """Test CallLogEntry dataclass."""

    def test_required_fields(self):
        entry = CallLogEntry(timestamp=1000.0, call_type="call")
        assert entry.timestamp == 1000.0
        assert entry.call_type == "call"

    def test_defaults(self):
        entry = CallLogEntry(timestamp=0.0, call_type="batch")
        assert entry.prompt is None
        assert entry.prompts is None
        assert entry.context_slice is None
        assert entry.role == "worker"
        assert entry.persona is None
        assert entry.result is None
        assert entry.elapsed_seconds == 0.0
        assert entry.error is None

    def test_all_fields_set(self):
        entry = CallLogEntry(
            timestamp=1234.5,
            call_type="call",
            prompt="hello",
            prompts=["a", "b"],
            context_slice="ctx",
            role="coder",
            persona="expert",
            result="output",
            elapsed_seconds=1.5,
            error="timeout",
        )
        assert entry.prompt == "hello"
        assert entry.prompts == ["a", "b"]
        assert entry.role == "coder"
        assert entry.error == "timeout"

    def test_result_can_be_list(self):
        entry = CallLogEntry(timestamp=0.0, call_type="batch", result=["a", "b"])
        assert entry.result == ["a", "b"]


class TestLLMResult:
    """Test LLMResult dataclass."""

    def test_minimal(self):
        result = LLMResult(text="Hello world")
        assert result.text == "Hello world"

    def test_defaults(self):
        result = LLMResult(text="")
        assert result.aborted is False
        assert result.abort_reason == ""
        assert result.tokens_generated == 0
        assert result.tokens_saved == 0
        assert result.failure_probability == 0.0
        assert result.elapsed_seconds == 0.0

    def test_aborted_result(self):
        result = LLMResult(
            text="partial output",
            aborted=True,
            abort_reason="high failure probability",
            tokens_generated=50,
            tokens_saved=200,
            failure_probability=0.85,
            elapsed_seconds=2.3,
        )
        assert result.aborted is True
        assert result.tokens_saved == 200
        assert result.failure_probability == 0.85


class TestQueryCost:
    """Test QueryCost dataclass and computed properties."""

    def test_defaults(self):
        cost = QueryCost(query_id="q1")
        assert cost.prompt_tokens == 0
        assert cost.completion_tokens == 0
        assert cost.total_tokens == 0
        assert cost.calls_made == 0
        assert cost.batch_calls_made == 0
        assert cost.elapsed_seconds == 0.0
        assert cost.prompt_rate == 0.0
        assert cost.completion_rate == 0.0

    def test_estimated_cost_zero_rates(self):
        cost = QueryCost(query_id="q1", prompt_tokens=1000, completion_tokens=500)
        assert cost.estimated_cost == 0.0

    def test_estimated_cost_calculation(self):
        cost = QueryCost(
            query_id="q2",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            prompt_rate=3.0,
            completion_rate=15.0,
        )
        # 1M prompt tokens at $3/M = $3, 1M completion at $15/M = $15
        assert cost.estimated_cost == 18.0

    def test_estimated_cost_fractional(self):
        cost = QueryCost(
            query_id="q3",
            prompt_tokens=500,
            completion_tokens=200,
            prompt_rate=2.0,
            completion_rate=10.0,
        )
        expected = (500 / 1_000_000) * 2.0 + (200 / 1_000_000) * 10.0
        assert abs(cost.estimated_cost - expected) < 1e-12

    def test_estimated_cost_zero_tokens(self):
        cost = QueryCost(query_id="q4", prompt_rate=5.0, completion_rate=15.0)
        assert cost.estimated_cost == 0.0

    def test_str_formatting(self):
        cost = QueryCost(
            query_id="test-q",
            total_tokens=1500,
            calls_made=3,
            batch_calls_made=1,
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_rate=2.0,
            completion_rate=10.0,
        )
        s = str(cost)
        assert "test-q" in s
        assert "tokens=1500" in s
        assert "calls=3" in s
        assert "batch=1" in s
        assert "cost=$" in s

    def test_str_zero_cost(self):
        cost = QueryCost(query_id="z")
        s = str(cost)
        assert "cost=$0.000000" in s
