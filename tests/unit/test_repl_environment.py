#!/usr/bin/env python3
"""Unit tests for the REPL environment."""

import pytest

from src.repl_environment import (
    REPLConfig,
    REPLEnvironment,
    REPLSecurityError,
    ExecutionResult,
)


class TestREPLBasicExecution:
    """Test basic code execution in the REPL."""

    def test_simple_print(self):
        """Test that print() output is captured."""
        repl = REPLEnvironment(context="test context")
        result = repl.execute('print("hello")')

        assert result.output.strip() == "hello"
        assert not result.is_final
        assert result.error is None

    def test_context_available(self):
        """Test that context variable is accessible."""
        repl = REPLEnvironment(context="This is the full context")
        result = repl.execute("print(len(context))")

        assert "24" in result.output  # len("This is the full context")
        assert not result.is_final

    def test_context_content(self):
        """Test that context contains the expected content."""
        repl = REPLEnvironment(context="Hello World")
        result = repl.execute("print(context)")

        assert "Hello World" in result.output

    def test_artifacts_available(self):
        """Test that artifacts dict is accessible."""
        repl = REPLEnvironment(context="test", artifacts={"key": "value"})
        result = repl.execute("print(artifacts['key'])")

        assert "value" in result.output

    def test_artifacts_can_be_modified(self):
        """Test that artifacts can be written to."""
        repl = REPLEnvironment(context="test")
        repl.execute("artifacts['result'] = 42")
        result = repl.execute("print(artifacts['result'])")

        assert "42" in result.output
        assert repl.artifacts["result"] == 42

    def test_multiline_code(self):
        """Test execution of multiline code."""
        repl = REPLEnvironment(context="test")
        code = """
x = 5
y = 10
print(x + y)
"""
        result = repl.execute(code)

        assert "15" in result.output

    def test_syntax_error_captured(self):
        """Test that syntax errors are captured."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(")

        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error_captured(self):
        """Test that runtime errors are captured."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("x = 1 / 0")

        assert result.error is not None
        assert "ZeroDivisionError" in result.error


class TestPeekFunction:
    """Test the peek() function."""

    def test_peek_default(self):
        """Test peek with default n=500."""
        context = "A" * 1000
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(len(peek()))")

        assert "500" in result.output

    def test_peek_custom_n(self):
        """Test peek with custom n."""
        context = "Hello World"
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(peek(5))")

        assert "Hello" in result.output

    def test_peek_larger_than_context(self):
        """Test peek when n > len(context)."""
        context = "Short"
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(peek(100))")

        assert "Short" in result.output


class TestGrepFunction:
    """Test the grep() function."""

    def test_grep_finds_matches(self):
        """Test that grep finds matching lines."""
        context = """Line 1: foo
Line 2: bar
Line 3: foo bar
Line 4: baz"""
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(grep('foo'))")

        assert "Line 1" in result.output
        assert "Line 3" in result.output
        assert "Line 2" not in result.output

    def test_grep_case_insensitive(self):
        """Test that grep is case insensitive."""
        context = "FOO\nfoo\nFoO"
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(len(grep('foo')))")

        assert "3" in result.output

    def test_grep_regex_pattern(self):
        """Test grep with regex pattern."""
        context = "test123\ntest456\nhello"
        repl = REPLEnvironment(context=context)
        result = repl.execute("print(grep(r'test\\d+'))")

        assert "test123" in result.output
        assert "test456" in result.output
        assert "hello" not in result.output

    def test_grep_invalid_regex(self):
        """Test grep with invalid regex returns error message."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(grep('[invalid'))")

        assert "REGEX ERROR" in result.output

    def test_grep_truncation(self):
        """Test that grep results are truncated."""
        # Create context with many matching lines
        context = "\n".join([f"match line {i}" for i in range(200)])
        config = REPLConfig(max_grep_results=50)
        repl = REPLEnvironment(context=context, config=config)
        result = repl.execute("print(grep('match'))")

        assert "truncated" in result.output


class TestFinalFunction:
    """Test the FINAL() and FINAL_VAR() functions."""

    def test_final_signals_completion(self):
        """Test that FINAL() signals completion."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("FINAL('The answer is 42')")

        assert result.is_final
        assert result.final_answer == "The answer is 42"

    def test_final_with_computation(self):
        """Test FINAL with computed value."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("FINAL(str(2 + 2))")

        assert result.is_final
        assert result.final_answer == "4"

    def test_final_var_returns_artifact(self):
        """Test that FINAL_VAR returns artifact content."""
        repl = REPLEnvironment(context="test")
        repl.execute("artifacts['result'] = 'computed value'")
        result = repl.execute("FINAL_VAR('result')")

        assert result.is_final
        assert result.final_answer == "computed value"

    def test_final_var_missing_key(self):
        """Test FINAL_VAR with missing key raises error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("FINAL_VAR('nonexistent')")

        assert not result.is_final
        assert result.error is not None
        assert "KeyError" in result.error

    def test_output_before_final_captured(self):
        """Test that output before FINAL is captured."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
print("before final")
FINAL("done")
""")

        assert result.is_final
        assert "before final" in result.output
        assert result.final_answer == "done"


class TestSecuritySandbox:
    """Test security restrictions in the sandbox."""

    def test_import_os_blocked(self):
        """Test that import os is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("import os")

        assert result.error is not None
        assert "not allowed" in result.error.lower() or "Dangerous" in result.error

    def test_import_subprocess_blocked(self):
        """Test that import subprocess is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("import subprocess")

        assert result.error is not None

    def test_from_os_import_blocked(self):
        """Test that from os import is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("from os import system")

        assert result.error is not None

    def test_dunder_import_blocked(self):
        """Test that __import__ is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("__import__('os')")

        assert result.error is not None

    def test_eval_blocked(self):
        """Test that eval() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("eval('1+1')")

        assert result.error is not None

    def test_exec_blocked(self):
        """Test that exec() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("exec('x=1')")

        assert result.error is not None

    def test_open_blocked(self):
        """Test that open() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("open('/etc/passwd')")

        assert result.error is not None

    def test_getattr_blocked(self):
        """Test that getattr() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("getattr(object, '__class__')")

        assert result.error is not None

    def test_dunder_class_blocked(self):
        """Test that __class__ access is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("''.__class__")

        assert result.error is not None

    def test_dunder_subclasses_blocked(self):
        """Test that __subclasses__ is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("object.__subclasses__()")

        assert result.error is not None

    def test_globals_blocked(self):
        """Test that globals() is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("globals()")

        assert result.error is not None

    def test_builtins_blocked(self):
        """Test that __builtins__ access is blocked."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(__builtins__)")

        assert result.error is not None


class TestOutputCapping:
    """Test output capping functionality."""

    def test_output_capped(self):
        """Test that output is capped at limit."""
        config = REPLConfig(output_cap=100)
        repl = REPLEnvironment(context="test", config=config)
        result = repl.execute("print('x' * 200)")

        assert len(result.output) <= 150  # Allow for truncation message
        assert "truncated" in result.output

    def test_normal_output_not_capped(self):
        """Test that normal output is not capped."""
        config = REPLConfig(output_cap=1000)
        repl = REPLEnvironment(context="test", config=config)
        result = repl.execute("print('hello')")

        assert result.output.strip() == "hello"
        assert "truncated" not in result.output


class TestREPLState:
    """Test REPL state management."""

    def test_get_state(self):
        """Test get_state() returns summary."""
        repl = REPLEnvironment(context="Hello World")
        repl.execute("artifacts['key'] = 'value'")
        state = repl.get_state()

        assert "context" in state
        assert "11 chars" in state  # len("Hello World")
        assert "key" in state

    def test_reset_clears_artifacts(self):
        """Test that reset() clears artifacts."""
        repl = REPLEnvironment(context="test")
        repl.execute("artifacts['key'] = 'value'")
        repl.reset()

        assert len(repl.artifacts) == 0

    def test_reset_keeps_context(self):
        """Test that reset() keeps context."""
        repl = REPLEnvironment(context="important context")
        repl.reset()

        assert repl.context == "important context"

    def test_state_persists_across_executions(self):
        """Test that state persists across execute() calls."""
        repl = REPLEnvironment(context="test")
        repl.execute("x = 42")
        result = repl.execute("print(x)")

        assert "42" in result.output


class TestAllowedBuiltins:
    """Test that allowed builtins work correctly."""

    def test_len_works(self):
        """Test that len() works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(len([1,2,3]))")
        assert "3" in result.output

    def test_range_works(self):
        """Test that range() works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(list(range(3)))")
        assert "[0, 1, 2]" in result.output

    def test_sorted_works(self):
        """Test that sorted() works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(sorted([3,1,2]))")
        assert "[1, 2, 3]" in result.output

    def test_dict_comprehension_works(self):
        """Test that dict comprehension works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print({i: i*2 for i in range(3)})")
        assert "0: 0" in result.output

    def test_list_comprehension_works(self):
        """Test that list comprehension works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print([x*2 for x in range(3)])")
        assert "[0, 2, 4]" in result.output

    def test_try_except_works(self):
        """Test that try/except works."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
try:
    x = 1/0
except ZeroDivisionError:
    print("caught")
""")
        assert "caught" in result.output
        assert result.error is None


class TestElapsedTime:
    """Test elapsed time tracking."""

    def test_elapsed_time_recorded(self):
        """Test that elapsed time is recorded."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("x = sum(range(1000))")

        assert result.elapsed_seconds >= 0


class TestConfig:
    """Test REPLConfig options."""

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        config = REPLConfig(timeout_seconds=60)
        repl = REPLEnvironment(context="test", config=config)

        assert repl.config.timeout_seconds == 60

    def test_custom_output_cap(self):
        """Test custom output cap configuration."""
        config = REPLConfig(output_cap=500)
        repl = REPLEnvironment(context="test", config=config)

        assert repl.config.output_cap == 500

    def test_custom_max_grep_results(self):
        """Test custom max grep results configuration."""
        config = REPLConfig(max_grep_results=10)
        repl = REPLEnvironment(context="test", config=config)

        assert repl.config.max_grep_results == 10
