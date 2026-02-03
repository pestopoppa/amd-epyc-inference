#!/usr/bin/env python3
"""Tests for the RestrictedPython executor module."""

import pytest
from unittest.mock import MagicMock

# Import the module
from src.restricted_executor import (
    is_available,
    RestrictedExecutor,
    RestrictedSecurityError,
    RestrictedTimeout,
    FinalSignal,
    ExecutionResult,
    _restricted_getattr,
    _restricted_getitem,
    _restricted_write,
)


class TestIsAvailable:
    """Tests for is_available function."""

    def test_is_available_returns_bool(self):
        """is_available should return a boolean."""
        result = is_available()
        assert isinstance(result, bool)


class TestExceptions:
    """Tests for custom exception classes."""

    def test_final_signal_stores_answer(self):
        """FinalSignal should store the answer."""
        signal = FinalSignal("The answer is 42")
        assert signal.answer == "The answer is 42"
        assert str(signal) == "The answer is 42"

    def test_restricted_security_error_is_exception(self):
        """RestrictedSecurityError should be an Exception."""
        error = RestrictedSecurityError("test")
        assert isinstance(error, Exception)

    def test_restricted_timeout_is_exception(self):
        """RestrictedTimeout should be an Exception."""
        error = RestrictedTimeout("timeout")
        assert isinstance(error, Exception)


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_execution_result_default_values(self):
        """ExecutionResult should have sensible defaults."""
        result = ExecutionResult(output="test", is_final=False)
        assert result.output == "test"
        assert result.is_final is False
        assert result.final_answer is None
        assert result.error is None
        assert result.elapsed_seconds == 0.0

    def test_execution_result_with_error(self):
        """ExecutionResult should support error field."""
        result = ExecutionResult(
            output="",
            is_final=False,
            error="SyntaxError: invalid syntax",
            elapsed_seconds=0.5,
        )
        assert result.error == "SyntaxError: invalid syntax"
        assert result.elapsed_seconds == 0.5


@pytest.mark.skipif(not is_available(), reason="RestrictedPython not installed")
class TestRestrictedGuards:
    """Tests for the guard functions."""

    def test_restricted_getattr_blocks_dunder(self):
        """_restricted_getattr should block access to dunder attributes."""
        obj = {"key": "value"}
        with pytest.raises(RestrictedSecurityError, match="private/dunder"):
            _restricted_getattr(obj, "__class__")

    def test_restricted_getattr_blocks_private(self):
        """_restricted_getattr should block access to private attributes."""
        obj = MagicMock()
        obj._private = "secret"
        with pytest.raises(RestrictedSecurityError, match="private/dunder"):
            _restricted_getattr(obj, "_private")

    def test_restricted_getattr_allows_public(self):
        """_restricted_getattr should allow public attributes."""
        obj = MagicMock()
        obj.public = "value"
        result = _restricted_getattr(obj, "public")
        assert result == "value"

    def test_restricted_getitem_blocks_dunder_keys(self):
        """_restricted_getitem should block keys starting with underscore."""
        obj = {"_private": "secret", "public": "value"}
        with pytest.raises(RestrictedSecurityError, match="key '_private' is not allowed"):
            _restricted_getitem(obj, "_private")

    def test_restricted_getitem_allows_public_keys(self):
        """_restricted_getitem should allow public keys."""
        obj = {"public": "value"}
        result = _restricted_getitem(obj, "public")
        assert result == "value"

    def test_restricted_getitem_allows_numeric_keys(self):
        """_restricted_getitem should allow numeric keys."""
        obj = [1, 2, 3]
        result = _restricted_getitem(obj, 1)
        assert result == 2

    def test_restricted_write_allows_basic_types(self):
        """_restricted_write should allow basic Python types."""
        assert _restricted_write("string") == "string"
        assert _restricted_write(42) == 42
        assert _restricted_write(3.14) == 3.14
        assert _restricted_write(True) is True
        assert _restricted_write(None) is None
        assert _restricted_write([1, 2, 3]) == [1, 2, 3]
        assert _restricted_write({"key": "val"}) == {"key": "val"}

    def test_restricted_write_allows_print_collector(self):
        """_restricted_write should allow objects with _call_print."""
        obj = MagicMock()
        obj._call_print = MagicMock()
        result = _restricted_write(obj)
        assert result is obj


@pytest.mark.skipif(not is_available(), reason="RestrictedPython not installed")
class TestRestrictedExecutor:
    """Tests for RestrictedExecutor class."""

    def test_init_without_restrictedpython_raises(self, monkeypatch):
        """Initialization should fail gracefully if RestrictedPython not available."""
        # Temporarily patch the availability check
        monkeypatch.setattr("src.restricted_executor.RESTRICTED_PYTHON_AVAILABLE", False)

        with pytest.raises(ImportError, match="RestrictedPython is not installed"):
            RestrictedExecutor(context="test")

    def test_init_with_default_params(self):
        """Executor should initialize with default parameters."""
        executor = RestrictedExecutor(context="Hello, world!")
        assert executor.context == "Hello, world!"
        assert executor.artifacts == {}
        assert executor.timeout_seconds == 120
        assert executor.output_cap == 8192
        assert executor.max_grep_results == 100
        assert executor.llm_primitives is None

    def test_init_with_custom_params(self):
        """Executor should accept custom parameters."""
        artifacts = {"key": "value"}
        executor = RestrictedExecutor(
            context="test",
            artifacts=artifacts,
            timeout_seconds=60,
            output_cap=4096,
            max_grep_results=50,
        )
        assert executor.artifacts is artifacts
        assert executor.timeout_seconds == 60
        assert executor.output_cap == 4096
        assert executor.max_grep_results == 50

    def test_peek_returns_substring(self):
        """_peek should return first n characters of context."""
        executor = RestrictedExecutor(context="Hello, world!")
        result = executor._peek(5)
        assert result == "Hello"
        assert executor._exploration_calls == 1

    def test_peek_with_large_n(self):
        """_peek should handle n larger than context."""
        executor = RestrictedExecutor(context="Short")
        result = executor._peek(1000)
        assert result == "Short"

    def test_grep_finds_matches(self):
        """_grep should find matching lines."""
        context = "Line 1: apple\nLine 2: banana\nLine 3: apple pie"
        executor = RestrictedExecutor(context=context)
        matches = executor._grep("apple")
        assert len(matches) == 2
        assert "Line 1: apple" in matches
        assert "Line 3: apple pie" in matches
        assert executor._exploration_calls == 1

    def test_grep_case_insensitive(self):
        """_grep should be case insensitive."""
        context = "Apple\nBANANA\napple"
        executor = RestrictedExecutor(context=context)
        matches = executor._grep("APPLE")
        assert len(matches) == 2

    def test_grep_with_invalid_regex(self):
        """_grep should handle invalid regex gracefully."""
        executor = RestrictedExecutor(context="test")
        matches = executor._grep("[invalid")
        assert len(matches) == 1
        assert "REGEX ERROR" in matches[0]

    def test_grep_truncates_results(self):
        """_grep should truncate results at max_grep_results."""
        lines = [f"Line {i}: match" for i in range(200)]
        context = "\n".join(lines)
        executor = RestrictedExecutor(context=context, max_grep_results=50)
        matches = executor._grep("match")
        assert len(matches) == 51  # 50 matches + 1 truncation message
        assert "truncated at 50" in matches[-1]

    def test_final_raises_signal(self):
        """_final should raise FinalSignal with the answer."""
        executor = RestrictedExecutor(context="test")
        with pytest.raises(FinalSignal) as exc_info:
            executor._final("The answer is 42")
        assert exc_info.value.answer == "The answer is 42"

    def test_llm_call_without_primitives_raises(self):
        """_llm_call should raise when llm_primitives is None."""
        executor = RestrictedExecutor(context="test", llm_primitives=None)
        with pytest.raises(RuntimeError, match="llm_call not available"):
            executor._llm_call("prompt")

    def test_llm_call_with_primitives(self):
        """_llm_call should delegate to llm_primitives."""
        mock_primitives = MagicMock()
        mock_primitives.llm_call.return_value = "response"
        executor = RestrictedExecutor(context="test", llm_primitives=mock_primitives)

        result = executor._llm_call("prompt", temperature=0.5)

        assert result == "response"
        mock_primitives.llm_call.assert_called_once_with("prompt", temperature=0.5)
        assert executor._exploration_calls == 1

    def test_llm_batch_without_primitives_raises(self):
        """_llm_batch should raise when llm_primitives is None."""
        executor = RestrictedExecutor(context="test", llm_primitives=None)
        with pytest.raises(RuntimeError, match="llm_batch not available"):
            executor._llm_batch(["prompt1", "prompt2"])

    def test_llm_batch_with_primitives(self):
        """_llm_batch should delegate to llm_primitives."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["response1", "response2"]
        executor = RestrictedExecutor(context="test", llm_primitives=mock_primitives)

        result = executor._llm_batch(["prompt1", "prompt2"])

        assert result == ["response1", "response2"]
        mock_primitives.llm_batch.assert_called_once_with(["prompt1", "prompt2"])
        assert executor._exploration_calls == 1

    def test_build_globals_includes_builtins(self):
        """_build_globals should include safe builtins."""
        executor = RestrictedExecutor(context="test")
        globals_dict = executor._build_globals()

        assert "__builtins__" in globals_dict
        assert "context" in globals_dict
        assert "artifacts" in globals_dict
        assert "peek" in globals_dict
        assert "grep" in globals_dict
        assert "FINAL" in globals_dict

    def test_build_globals_includes_llm_primitives(self):
        """_build_globals should include llm functions when available."""
        mock_primitives = MagicMock()
        executor = RestrictedExecutor(context="test", llm_primitives=mock_primitives)
        globals_dict = executor._build_globals()

        assert "llm_call" in globals_dict
        assert "llm_batch" in globals_dict

    def test_build_globals_excludes_llm_when_none(self):
        """_build_globals should not include llm functions when None."""
        executor = RestrictedExecutor(context="test", llm_primitives=None)
        globals_dict = executor._build_globals()

        assert "llm_call" not in globals_dict
        assert "llm_batch" not in globals_dict

    def test_execute_simple_code(self):
        """execute should run simple Python code."""
        executor = RestrictedExecutor(context="Hello, world!")
        result = executor.execute("x = 1 + 1")

        assert result.output == ""
        assert result.is_final is False
        assert result.error is None
        assert result.elapsed_seconds > 0

    def test_execute_with_print(self):
        """execute should capture print output."""
        executor = RestrictedExecutor(context="test")
        result = executor.execute("print('Hello from code')")

        assert "Hello from code" in result.output
        assert result.is_final is False
        assert result.error is None

    def test_execute_with_peek(self):
        """execute should allow calling peek function."""
        executor = RestrictedExecutor(context="Hello, world!")
        result = executor.execute("print(peek(5))")

        assert "Hello" in result.output
        assert result.error is None

    def test_execute_with_grep(self):
        """execute should allow calling grep function."""
        executor = RestrictedExecutor(context="apple\nbanana\napple pie")
        result = executor.execute("matches = grep('apple')\nprint(len(matches))")

        assert "2" in result.output
        assert result.error is None

    def test_execute_with_final(self):
        """execute should handle FINAL signal."""
        executor = RestrictedExecutor(context="test")
        result = executor.execute("FINAL('The answer is 42')")

        assert result.is_final is True
        assert result.final_answer == "The answer is 42"
        assert result.error is None

    def test_execute_with_syntax_error(self):
        """execute should handle syntax errors gracefully."""
        executor = RestrictedExecutor(context="test")
        result = executor.execute("if True print('no colon')")

        assert result.error is not None
        assert "SyntaxError" in result.error
        assert result.is_final is False

    def test_execute_with_runtime_error(self):
        """execute should handle runtime errors gracefully."""
        executor = RestrictedExecutor(context="test")
        result = executor.execute("x = 1 / 0")

        assert result.error is not None
        assert "ZeroDivisionError" in result.error
        assert result.is_final is False

    def test_execute_blocks_dunder_access(self):
        """execute should block access to dunder attributes."""
        executor = RestrictedExecutor(context="test")
        result = executor.execute("x = context.__class__")

        assert result.error is not None
        assert "SecurityError" in result.error

    def test_execute_with_output_cap(self):
        """execute should truncate output at output_cap."""
        executor = RestrictedExecutor(context="test", output_cap=100)
        code = "for i in range(1000):\n    print('X' * 100)"
        result = executor.execute(code)

        assert len(result.output) <= 200  # Cap + truncation message
        assert "truncated" in result.output

    def test_execute_with_artifacts(self):
        """execute should allow access to artifacts dict."""
        artifacts = {"key": "value", "count": 42}
        executor = RestrictedExecutor(context="test", artifacts=artifacts)
        result = executor.execute("print(artifacts['key'])\nprint(artifacts['count'])")

        assert "value" in result.output
        assert "42" in result.output
        assert result.error is None

    def test_get_state_shows_context_summary(self):
        """get_state should provide a summary of context and artifacts."""
        artifacts = {"key1": "value1", "key2": "a" * 150}
        executor = RestrictedExecutor(context="Hello, world!", artifacts=artifacts)
        state = executor.get_state()

        assert "13 chars" in state
        assert "key1" in state
        assert "key2" in state
        assert "value1" in state
        assert "..." in state  # Truncation of long value

    def test_get_state_empty_artifacts(self):
        """get_state should handle empty artifacts."""
        executor = RestrictedExecutor(context="test")
        state = executor.get_state()

        assert "artifacts: {}" in state

    def test_execute_captures_stderr(self):
        """execute should capture stderr output."""
        executor = RestrictedExecutor(context="test")
        code = "import sys\nsys.stderr.write('Error message\\n')"
        result = executor.execute(code)

        assert "[STDERR]" in result.output
        assert "Error message" in result.output

    def test_execute_with_final_preserves_print(self):
        """execute should capture print output before FINAL."""
        executor = RestrictedExecutor(context="test")
        code = "print('Before final')\nFINAL('answer')"
        result = executor.execute(code)

        assert result.is_final is True
        assert result.final_answer == "answer"
        assert "Before final" in result.output
