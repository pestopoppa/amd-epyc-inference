"""Unit tests for parallel read-only tool dispatch.

Tests AST extraction, lock safety, result ordering, and fallback behavior.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from src.repl_environment.parallel_dispatch import (
    _ParallelCall,
    _eval_ast_arg,
    _extract_parallel_calls,
    execute_parallel_calls,
)


# ---------------------------------------------------------------------------
# _eval_ast_arg tests
# ---------------------------------------------------------------------------

class TestEvalAstArg:
    def test_string_literal(self):
        import ast
        node = ast.parse('"hello"', mode="eval").body
        assert _eval_ast_arg(node, {}) == "hello"

    def test_int_literal(self):
        import ast
        node = ast.parse("42", mode="eval").body
        assert _eval_ast_arg(node, {}) == 42

    def test_name_in_globals(self):
        import ast
        node = ast.parse("x", mode="eval").body
        assert _eval_ast_arg(node, {"x": 99}) == 99

    def test_name_not_in_globals(self):
        import ast
        node = ast.parse("x", mode="eval").body
        assert _eval_ast_arg(node, {}) is None

    def test_list_literal(self):
        import ast
        node = ast.parse("[1, 2, 3]", mode="eval").body
        assert _eval_ast_arg(node, {}) == [1, 2, 3]

    def test_negative_number(self):
        import ast
        node = ast.parse("-5", mode="eval").body
        assert _eval_ast_arg(node, {}) == -5

    def test_complex_expression_returns_none(self):
        import ast
        node = ast.parse("f(x)", mode="eval").body
        assert _eval_ast_arg(node, {}) is None


# ---------------------------------------------------------------------------
# _extract_parallel_calls tests
# ---------------------------------------------------------------------------

READ_ONLY_TOOLS = {"peek", "grep", "list_dir", "file_info", "recall"}


def _make_globals(**tools):
    """Create a globals dict with callable mock tools."""
    return {name: MagicMock(return_value=f"{name}_result") for name in tools}


class TestExtractParallelCalls:
    def test_two_independent_calls(self):
        code = 'a = peek(500)\nb = grep("pattern")'
        globs = _make_globals(peek=True, grep=True)
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is not None
        assert len(calls) == 2
        assert calls[0].func_name == "peek"
        assert calls[0].target_var == "a"
        assert calls[1].func_name == "grep"
        assert calls[1].target_var == "b"

    def test_bare_calls(self):
        code = 'peek(500)\ngrep("pattern")'
        globs = _make_globals(peek=True, grep=True)
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is not None
        assert len(calls) == 2
        assert calls[0].target_var is None
        assert calls[1].target_var is None

    def test_single_call_returns_none(self):
        code = 'a = peek(500)'
        globs = _make_globals(peek=True)
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is None  # Need >1 call for parallel

    def test_non_read_only_tool_returns_none(self):
        code = 'a = peek(500)\nb = escalate("help")'
        globs = _make_globals(peek=True, escalate=True)
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is None

    def test_dependency_detected(self):
        code = 'a = peek(500)\nb = grep(a)'
        globs = _make_globals(peek=True, grep=True)
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is None  # b depends on a

    def test_non_call_statement_returns_none(self):
        code = 'a = peek(500)\nprint("hello")\nb = grep("x")'
        globs = _make_globals(peek=True, grep=True)
        globs["print"] = print
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is None  # print is not in read_only_tools

    def test_syntax_error_returns_none(self):
        code = 'a = peek(500\nb = grep("x")'
        globs = _make_globals(peek=True, grep=True)
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is None

    def test_kwargs_extraction(self):
        code = 'a = peek(500)\nb = grep("pattern", file_path="/tmp/test")'
        globs = _make_globals(peek=True, grep=True)
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is not None
        assert calls[1].kwargs == {"file_path": "/tmp/test"}

    def test_three_calls(self):
        code = 'a = peek(500)\nb = grep("x")\nc = list_dir(".")'
        globs = _make_globals(peek=True, grep=True, list_dir=True)
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is not None
        assert len(calls) == 3

    def test_function_not_in_globals_returns_none(self):
        code = 'a = peek(500)\nb = grep("x")'
        globs = {"peek": MagicMock()}  # grep not in globals
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is None

    def test_method_call_returns_none(self):
        code = 'a = self.peek(500)\nb = self.grep("x")'
        globs = _make_globals(peek=True, grep=True)
        calls = _extract_parallel_calls(code, globs, READ_ONLY_TOOLS)
        assert calls is None  # self.method is not a simple Name


# ---------------------------------------------------------------------------
# execute_parallel_calls tests
# ---------------------------------------------------------------------------

class TestExecuteParallelCalls:
    def test_results_in_order(self):
        def slow_peek(*args, **kwargs):
            time.sleep(0.05)
            return "peek_result"

        def slow_grep(*args, **kwargs):
            time.sleep(0.05)
            return "grep_result"

        calls = [
            _ParallelCall("peek", [500], {}, "a", 0),
            _ParallelCall("grep", ["pattern"], {}, "b", 1),
        ]
        globs = {"peek": slow_peek, "grep": slow_grep}
        lock = threading.Lock()

        results = execute_parallel_calls(calls, globs, lock)
        assert results["a"] == "peek_result"
        assert results["b"] == "grep_result"

    def test_parallel_is_faster_than_sequential(self):
        """3 calls sleeping 0.1s each should complete in ~0.1s not ~0.3s."""
        def slow_tool(*args, **kwargs):
            time.sleep(0.1)
            return "done"

        calls = [
            _ParallelCall("t1", [], {}, "a", 0),
            _ParallelCall("t2", [], {}, "b", 1),
            _ParallelCall("t3", [], {}, "c", 2),
        ]
        globs = {"t1": slow_tool, "t2": slow_tool, "t3": slow_tool}
        lock = threading.Lock()

        start = time.perf_counter()
        results = execute_parallel_calls(calls, globs, lock)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.25  # Should be ~0.1s, definitely <0.25s
        assert len(results) == 3

    def test_error_handling(self):
        def failing_tool(*args, **kwargs):
            raise ValueError("test error")

        calls = [
            _ParallelCall("bad", [], {}, "a", 0),
            _ParallelCall("good", [], {}, "b", 1),
        ]
        globs = {"bad": failing_tool, "good": lambda: "ok"}
        lock = threading.Lock()

        results = execute_parallel_calls(calls, globs, lock)
        assert "[ERROR:" in results["a"]
        assert results["b"] == "ok"

    def test_bare_calls_use_index_keys(self):
        calls = [
            _ParallelCall("t1", [], {}, None, 0),
            _ParallelCall("t2", [], {}, None, 1),
        ]
        globs = {"t1": lambda: "r1", "t2": lambda: "r2"}
        lock = threading.Lock()

        results = execute_parallel_calls(calls, globs, lock)
        assert results["_result_0"] == "r1"
        assert results["_result_1"] == "r2"
