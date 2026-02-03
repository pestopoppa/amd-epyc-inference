#!/usr/bin/env python3
"""Unit tests for src/tools/base.py."""

import signal
import time

import pytest

from src.tools.base import (
    ToolResult,
    ToolTimeout,
    format_error,
    safe_execute,
    truncate_output,
    with_timeout,
)


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ToolResult(success=True, data="output", elapsed_ms=123.45)
        assert result.success is True
        assert result.data == "output"
        assert result.error is None
        assert result.elapsed_ms == 123.45
        assert result.truncated is False

    def test_failure_result(self):
        """Test creating a failure result."""
        result = ToolResult(success=False, error="Failed", elapsed_ms=50.0)
        assert result.success is False
        assert result.data is None
        assert result.error == "Failed"
        assert result.elapsed_ms == 50.0

    def test_default_values(self):
        """Test ToolResult default field values."""
        result = ToolResult(success=True)
        assert result.data is None
        assert result.error is None
        assert result.elapsed_ms == 0.0
        assert result.truncated is False

    def test_truncated_result(self):
        """Test result with truncation flag."""
        result = ToolResult(success=True, data="text", truncated=True)
        assert result.truncated is True


class TestTruncateOutput:
    """Test truncate_output() function."""

    def test_text_below_max_length(self):
        """Test text shorter than max_length is not truncated."""
        text = "Short text"
        result, was_truncated = truncate_output(text, max_length=100)
        assert result == text
        assert was_truncated is False

    def test_text_at_max_length(self):
        """Test text exactly at max_length is not truncated."""
        text = "x" * 100
        result, was_truncated = truncate_output(text, max_length=100)
        assert result == text
        assert was_truncated is False

    def test_text_above_max_length(self):
        """Test text longer than max_length is truncated."""
        text = "x" * 200
        result, was_truncated = truncate_output(text, max_length=100)
        assert len(result) > 100  # Includes truncation message
        assert result.startswith("x" * 100)
        assert "truncated at 100 chars" in result
        assert "total was 200" in result
        assert was_truncated is True

    def test_default_max_length(self):
        """Test default max_length of 8192."""
        text = "x" * 10000
        result, was_truncated = truncate_output(text)
        assert was_truncated is True
        assert "truncated at 8192 chars" in result


class TestSafeExecute:
    """Test safe_execute() function."""

    def test_successful_execution(self):
        """Test safe_execute with successful function."""

        def success_func(x):
            return x * 2

        result = safe_execute(success_func, 21, timeout_seconds=5)
        assert result.success is True
        assert result.data == 42
        assert result.error is None
        assert result.elapsed_ms > 0

    def test_function_raises_exception(self):
        """Test safe_execute with function that raises exception."""

        def failing_func():
            raise ValueError("Test error")

        result = safe_execute(failing_func, timeout_seconds=5)
        assert result.success is False
        assert result.data is None
        assert "ValueError: Test error" in result.error
        assert result.elapsed_ms > 0

    def test_string_output_truncation(self):
        """Test safe_execute truncates long string output."""

        def long_output():
            return "x" * 10000

        result = safe_execute(long_output, timeout_seconds=5, max_output=100)
        assert result.success is True
        assert result.truncated is True
        assert "truncated at 100 chars" in result.data

    def test_non_string_output_not_truncated(self):
        """Test safe_execute doesn't truncate non-string output."""

        def dict_output():
            return {"key": "x" * 10000}

        result = safe_execute(dict_output, timeout_seconds=5, max_output=100)
        assert result.success is True
        assert result.truncated is False
        assert isinstance(result.data, dict)

    def test_timing_measurement(self):
        """Test safe_execute measures elapsed time."""

        def slow_func():
            time.sleep(0.01)
            return "done"

        result = safe_execute(slow_func, timeout_seconds=5)
        assert result.success is True
        # Should be at least 10ms
        assert result.elapsed_ms >= 10

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM not available")
    def test_timeout_handling(self):
        """Test safe_execute handles timeouts (Unix only)."""

        def infinite_loop():
            while True:
                pass

        result = safe_execute(infinite_loop, timeout_seconds=1)
        assert result.success is False
        assert "timed out" in result.error.lower()


class TestWithTimeout:
    """Test with_timeout() decorator."""

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM not available")
    def test_timeout_decorator_success(self):
        """Test with_timeout allows successful completion."""

        @with_timeout(5)
        def fast_func():
            return "success"

        result = fast_func()
        assert result == "success"

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM not available")
    def test_timeout_decorator_timeout(self):
        """Test with_timeout raises ToolTimeout."""

        @with_timeout(1)
        def slow_func():
            time.sleep(2)
            return "never reached"

        with pytest.raises(ToolTimeout) as exc_info:
            slow_func()
        assert "timed out after 1s" in str(exc_info.value)

    @pytest.mark.skipif(hasattr(signal, "SIGALRM"), reason="Test for non-Unix systems")
    def test_timeout_decorator_no_sigalrm(self):
        """Test with_timeout works without SIGALRM (Windows)."""

        @with_timeout(1)
        def func():
            return "no timeout on Windows"

        # Should work without timing out (no SIGALRM available)
        result = func()
        assert result == "no timeout on Windows"


class TestFormatError:
    """Test format_error() function."""

    def test_format_error_without_traceback(self):
        """Test format_error without traceback."""
        error = ValueError("Test error")
        formatted = format_error(error, include_traceback=False)
        assert formatted == "ValueError: Test error"

    def test_format_error_with_traceback(self):
        """Test format_error with traceback."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            formatted = format_error(e, include_traceback=True)
            assert "ValueError: Test error" in formatted
            assert "Traceback" in formatted

    def test_format_error_custom_exception(self):
        """Test format_error with custom exception."""

        class CustomError(Exception):
            pass

        error = CustomError("custom message")
        formatted = format_error(error, include_traceback=False)
        assert formatted == "CustomError: custom message"
