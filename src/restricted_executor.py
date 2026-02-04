"""RestrictedPython-based code execution for the REPL environment.

This module provides a more battle-tested sandbox using RestrictedPython,
which is a well-established library for safe Python code execution in
environments like Zope/Plone.

Usage:
    from src.restricted_executor import RestrictedExecutor, is_available

    if is_available():
        executor = RestrictedExecutor(context="...", artifacts={})
        result = executor.execute("print(peek(100))")

Features:
    - compile_restricted for safer bytecode generation
    - Built-in guards against attribute/getitem exploits
    - PrintCollector for stdout capture
    - Configurable safe builtins

This module is optional and requires: pip install RestrictedPython>=7.0
"""

from __future__ import annotations

import io
import signal
from contextlib import redirect_stderr
from dataclasses import dataclass
from typing import Any

# Try to import RestrictedPython
try:
    from RestrictedPython import (
        compile_restricted,
        safe_builtins,
        utility_builtins,
    )
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence,
        guarded_unpack_sequence,
    )
    from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem
    from RestrictedPython.PrintCollector import PrintCollector

    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    # Provide stubs for type checking
    compile_restricted = None
    safe_builtins = {}
    utility_builtins = {}

from src.config import _registry_timeout

# Default execution timeout from registry
_RESTRICTED_TIMEOUT = int(_registry_timeout("repl", "restricted_executor", 120))


def is_available() -> bool:
    """Check if RestrictedPython is available.

    Returns:
        True if RestrictedPython is installed.
    """
    return RESTRICTED_PYTHON_AVAILABLE


class RestrictedSecurityError(Exception):
    """Security violation in restricted execution."""

    pass


class RestrictedTimeout(Exception):
    """Execution timed out."""

    pass


class FinalSignal(Exception):
    """Signal that FINAL() was called with the answer."""

    def __init__(self, answer: str):
        self.answer = answer
        super().__init__(answer)


@dataclass
class ExecutionResult:
    """Result of executing code in the restricted environment."""

    output: str
    is_final: bool
    final_answer: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


def _restricted_getattr(obj: Any, name: str) -> Any:
    """Guarded getattr that blocks dangerous attribute access.

    Args:
        obj: Object to get attribute from.
        name: Attribute name.

    Returns:
        Attribute value.

    Raises:
        RestrictedSecurityError: If attribute access is forbidden.
    """
    # Block dunder attributes
    if name.startswith("_"):
        raise RestrictedSecurityError(f"Access to private/dunder attribute '{name}' is not allowed")

    # Use RestrictedPython's default guard
    return default_guarded_getattr(obj, name)


def _restricted_getitem(obj: Any, key: Any) -> Any:
    """Guarded getitem that blocks dangerous subscript access.

    Args:
        obj: Object to get item from.
        key: Item key.

    Returns:
        Item value.

    Raises:
        RestrictedSecurityError: If key access is forbidden.
    """
    # Block string keys that are dunder names
    if isinstance(key, str) and key.startswith("_"):
        raise RestrictedSecurityError(f"Access to key '{key}' is not allowed")

    # Use RestrictedPython's default guard
    return default_guarded_getitem(obj, key)


def _restricted_write(obj: Any) -> Any:
    """Guard for write operations.

    Args:
        obj: Object to check for write.

    Returns:
        The object if safe.
    """
    # Basic types are always safe to write
    if isinstance(obj, (str, bytes, int, float, bool, type(None), list, dict, set, tuple)):
        return obj
    # For other types, check if they're a PrintCollector
    if hasattr(obj, "_call_print"):
        return obj
    return obj


class RestrictedExecutor:
    """Execute Python code in a RestrictedPython sandbox.

    This provides stronger security guarantees than custom AST validation
    by using RestrictedPython's compile_restricted which modifies the
    bytecode to use guarded operations.

    Example:
        executor = RestrictedExecutor(context="Hello world")
        result = executor.execute("print(peek(5))")
        print(result.output)  # "Hello"
    """

    def __init__(
        self,
        context: str,
        artifacts: dict[str, Any] | None = None,
        timeout_seconds: int = _RESTRICTED_TIMEOUT,
        output_cap: int = 8192,
        max_grep_results: int = 100,
        llm_primitives: Any | None = None,
    ):
        """Initialize the restricted executor.

        Args:
            context: The full input context.
            artifacts: Pre-existing artifacts dict.
            timeout_seconds: Execution timeout (from registry).
            output_cap: Maximum output characters.
            max_grep_results: Maximum grep results.
            llm_primitives: Optional LLMPrimitives for llm_call.
        """
        if not RESTRICTED_PYTHON_AVAILABLE:
            raise ImportError(
                "RestrictedPython is not installed. Install with: pip install RestrictedPython>=7.0"
            )

        self.context = context
        self.artifacts = artifacts if artifacts is not None else {}
        self.timeout_seconds = timeout_seconds
        self.output_cap = output_cap
        self.max_grep_results = max_grep_results
        self.llm_primitives = llm_primitives

        self._final_answer: str | None = None
        self._exploration_calls = 0

    def _peek(self, n: int = 500) -> str:
        """Return first n characters of context."""
        self._exploration_calls += 1
        return self.context[:n]

    def _grep(self, pattern: str) -> list[str]:
        """Search context with regex."""
        import re

        self._exploration_calls += 1
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [f"[REGEX ERROR: {e}]"]

        matches = []
        for line in self.context.split("\n"):
            if regex.search(line):
                matches.append(line)
                if len(matches) >= self.max_grep_results:
                    matches.append(f"[... truncated at {self.max_grep_results} results]")
                    break

        return matches

    def _final(self, answer: str) -> None:
        """Signal completion with final answer."""
        raise FinalSignal(str(answer))

    def _llm_call(self, *args, **kwargs) -> str:
        """Wrapper for llm_call."""
        if self.llm_primitives is None:
            raise RuntimeError("llm_call not available (no LLM primitives)")
        self._exploration_calls += 1
        return self.llm_primitives.llm_call(*args, **kwargs)

    def _llm_batch(self, *args, **kwargs) -> list[str]:
        """Wrapper for llm_batch."""
        if self.llm_primitives is None:
            raise RuntimeError("llm_batch not available (no LLM primitives)")
        self._exploration_calls += 1
        return self.llm_primitives.llm_batch(*args, **kwargs)

    def _build_globals(self) -> dict[str, Any]:
        """Build the restricted globals dict."""
        # Start with safe builtins
        restricted_builtins = dict(safe_builtins)
        restricted_builtins.update(utility_builtins)

        # Add guards
        restricted_builtins["_getattr_"] = _restricted_getattr
        restricted_builtins["_getitem_"] = _restricted_getitem
        restricted_builtins["_write_"] = _restricted_write
        restricted_builtins["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        restricted_builtins["_unpack_sequence_"] = guarded_unpack_sequence

        # PrintCollector for stdout capture
        restricted_builtins["_print_"] = PrintCollector

        # Build globals
        globals_dict = {
            "__builtins__": restricted_builtins,
            # Context variables
            "context": self.context,
            "artifacts": self.artifacts,
            # Built-in functions
            "peek": self._peek,
            "grep": self._grep,
            "FINAL": self._final,
        }

        # Add LLM primitives if available
        if self.llm_primitives is not None:
            globals_dict["llm_call"] = self._llm_call
            globals_dict["llm_batch"] = self._llm_batch

        return globals_dict

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the restricted environment.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with output, is_final flag, and optional error.
        """
        import time

        start_time = time.perf_counter()

        # Compile with RestrictedPython
        try:
            compiled = compile_restricted(
                code,
                filename="<repl>",
                mode="exec",
            )
        except SyntaxError as e:
            return ExecutionResult(
                output="",
                is_final=False,
                error=f"SyntaxError: {e}",
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Check for compilation errors
        if compiled.errors:
            return ExecutionResult(
                output="",
                is_final=False,
                error=f"RestrictedPython error: {compiled.errors[0]}",
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Build execution environment
        globals_dict = self._build_globals()
        locals_dict: dict[str, Any] = {}

        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise RestrictedTimeout(f"Execution timed out after {self.timeout_seconds}s")

        stderr_capture = io.StringIO()

        try:
            # Set timeout (Unix only)
            old_handler = None
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_seconds)

            try:
                with redirect_stderr(stderr_capture):
                    exec(compiled.code, globals_dict, locals_dict)

                # Collect printed output
                output = ""
                if "_print" in locals_dict:
                    printed = locals_dict["_print"]
                    if hasattr(printed, "__call__"):
                        output = printed()
                    else:
                        output = str(printed)
                elif "_print" in globals_dict:
                    printed = globals_dict.get("_print")
                    if printed and hasattr(printed, "__call__"):
                        output = printed()

                if stderr_capture.getvalue():
                    output += "\n[STDERR]\n" + stderr_capture.getvalue()

                # Cap output
                if len(output) > self.output_cap:
                    output = (
                        output[: self.output_cap] + f"\n[... truncated at {self.output_cap} chars]"
                    )

                return ExecutionResult(
                    output=output,
                    is_final=False,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except FinalSignal as e:
                output = ""
                if "_print" in locals_dict:
                    printed = locals_dict["_print"]
                    if hasattr(printed, "__call__"):
                        output = printed()
                return ExecutionResult(
                    output=output,
                    is_final=True,
                    final_answer=e.answer,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except RestrictedSecurityError as e:
                return ExecutionResult(
                    output="",
                    is_final=False,
                    error=f"SecurityError: {e}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except RestrictedTimeout as e:
                return ExecutionResult(
                    output="",
                    is_final=False,
                    error=str(e),
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except Exception as e:
                return ExecutionResult(
                    output="",
                    is_final=False,
                    error=f"{type(e).__name__}: {e}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

        finally:
            # Clear timeout
            if hasattr(signal, "SIGALRM") and old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def get_state(self) -> str:
        """Get a summary of current state."""
        state_lines = [
            f"context: str ({len(self.context)} chars)",
            f"artifacts: {list(self.artifacts.keys()) if self.artifacts else '{}'}",
        ]

        for key, value in self.artifacts.items():
            preview = str(value)[:100]
            if len(str(value)) > 100:
                preview += "..."
            state_lines.append(f"  artifacts['{key}']: {preview}")

        return "\n".join(state_lines)
