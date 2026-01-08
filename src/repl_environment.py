#!/usr/bin/env python3
"""Sandboxed Python REPL environment for RLM-style orchestration.

This module provides a restricted Python execution environment where:
- Large context is stored as a variable, not sent to LLM
- Built-in functions (peek, grep, FINAL) enable context manipulation
- Dangerous operations (file I/O, subprocess, imports) are blocked
- Execution is time-limited and output-capped

Usage:
    from src.repl_environment import REPLEnvironment

    repl = REPLEnvironment(context="Large document here...")
    output, is_final = repl.execute("print(peek(100))")
    output, is_final = repl.execute("FINAL('Done!')")
"""

from __future__ import annotations

import io
import re
import signal
import sys
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Callable


class REPLError(Exception):
    """Error during REPL execution."""

    pass


class REPLTimeout(REPLError):
    """Execution timed out."""

    pass


class REPLSecurityError(REPLError):
    """Attempted to execute dangerous code."""

    pass


class FinalSignal(Exception):
    """Signal that FINAL() was called with the answer."""

    def __init__(self, answer: str):
        self.answer = answer
        super().__init__(answer)


@dataclass
class REPLConfig:
    """Configuration for the REPL environment."""

    timeout_seconds: int = 120
    output_cap: int = 8192
    max_grep_results: int = 100
    allowed_builtins: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                # Safe builtins
                "abs",
                "all",
                "any",
                "ascii",
                "bin",
                "bool",
                "bytes",
                "chr",
                "dict",
                "divmod",
                "enumerate",
                "filter",
                "float",
                "format",
                "frozenset",
                "hash",
                "hex",
                "int",
                "isinstance",
                "issubclass",
                "iter",
                "len",
                "list",
                "map",
                "max",
                "min",
                "next",
                "oct",
                "ord",
                "pow",
                "print",
                "range",
                "repr",
                "reversed",
                "round",
                "set",
                "slice",
                "sorted",
                "str",
                "sum",
                "tuple",
                "type",
                "zip",
                # Exceptions (needed for try/except)
                "Exception",
                "ValueError",
                "TypeError",
                "KeyError",
                "IndexError",
                "AttributeError",
                "RuntimeError",
                "StopIteration",
                "ZeroDivisionError",
                "NameError",
                "FileNotFoundError",
                "IOError",
                "OSError",
                # Constants
                "True",
                "False",
                "None",
            }
        )
    )


@dataclass
class ExecutionResult:
    """Result of executing code in the REPL."""

    output: str
    is_final: bool
    final_answer: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


class REPLEnvironment:
    """Sandboxed Python REPL with context-as-variable pattern.

    The REPL provides:
    - `context`: The full input as a string (never sent to LLM)
    - `artifacts`: Dict for storing intermediate results
    - `peek(n)`: Return first n characters of context
    - `grep(pattern)`: Regex search in context
    - `FINAL(answer)`: Signal completion with final answer
    - `FINAL_VAR(var_name)`: Signal completion, return variable contents

    When tool_registry is provided:
    - `TOOL(name, **kwargs)`: Invoke a registered tool
    - `list_tools()`: List available tools for current role

    When script_registry is provided:
    - `SCRIPT(id, **kwargs)`: Invoke a prepared script by ID
    - `find_scripts(query)`: Find scripts matching a description
    """

    def __init__(
        self,
        context: str,
        artifacts: dict[str, Any] | None = None,
        config: REPLConfig | None = None,
        llm_primitives: Any | None = None,  # LLMPrimitives instance
        tool_registry: Any | None = None,  # ToolRegistry instance
        script_registry: Any | None = None,  # ScriptRegistry instance
        role: str | None = None,  # Role for permission checking
    ):
        """Initialize the REPL environment.

        Args:
            context: The full input context (stored as variable, not in LLM prompt).
            artifacts: Optional dict of pre-existing artifacts from previous turns.
            config: Optional configuration for timeouts, output caps, etc.
            llm_primitives: Optional LLMPrimitives instance for llm_call/llm_batch.
            tool_registry: Optional ToolRegistry for TOOL() invocations.
            script_registry: Optional ScriptRegistry for SCRIPT() invocations.
            role: Role name for permission checking (e.g., "frontdoor", "coder_primary").
        """
        self.context = context
        self.artifacts = artifacts if artifacts is not None else {}
        self.config = config if config is not None else REPLConfig()
        self.llm_primitives = llm_primitives
        self.tool_registry = tool_registry
        self.script_registry = script_registry
        self.role = role or "worker_general"  # Default to restricted role

        # Execution state
        self._final_answer: str | None = None
        self._execution_count = 0

        # Build restricted globals
        self._globals = self._build_globals()

    def _build_globals(self) -> dict[str, Any]:
        """Build the restricted globals dict for exec()."""
        import builtins

        # Build safe builtins from the builtins module
        safe_builtins = {}
        for name in self.config.allowed_builtins:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        globals_dict = {
            "__builtins__": safe_builtins,
            # Context variables
            "context": self.context,
            "artifacts": self.artifacts,
            # Built-in functions
            "peek": self._peek,
            "grep": self._grep,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
        }

        # Add LLM primitives if available
        if self.llm_primitives is not None:
            globals_dict["llm_call"] = self.llm_primitives.llm_call
            globals_dict["llm_batch"] = self.llm_primitives.llm_batch

        # Add tool registry functions if available
        if self.tool_registry is not None:
            globals_dict["TOOL"] = self._invoke_tool
            globals_dict["list_tools"] = self._list_tools

        # Add script registry functions if available
        if self.script_registry is not None:
            globals_dict["SCRIPT"] = self._invoke_script
            globals_dict["find_scripts"] = self._find_scripts

        return globals_dict

    def _peek(self, n: int = 500) -> str:
        """Return first n characters of context.

        Args:
            n: Number of characters to return (default 500).

        Returns:
            First n characters of the context.
        """
        return self.context[:n]

    def _grep(self, pattern: str) -> list[str]:
        """Search context with regex and return matching lines.

        Args:
            pattern: Regular expression pattern to search for.

        Returns:
            List of lines containing matches (capped at max_grep_results).
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [f"[REGEX ERROR: {e}]"]

        matches = []
        for line in self.context.split("\n"):
            if regex.search(line):
                matches.append(line)
                if len(matches) >= self.config.max_grep_results:
                    matches.append(f"[... truncated at {self.config.max_grep_results} results]")
                    break

        return matches

    def _final(self, answer: str) -> None:
        """Signal completion with final answer.

        Args:
            answer: The final answer to return.

        Raises:
            FinalSignal: Always raised to terminate execution.
        """
        raise FinalSignal(str(answer))

    def _final_var(self, var_name: str) -> None:
        """Signal completion, returning contents of a variable.

        Args:
            var_name: Name of variable in artifacts dict to return.

        Raises:
            FinalSignal: Always raised to terminate execution.
            KeyError: If variable not found in artifacts.
        """
        if var_name not in self.artifacts:
            raise KeyError(f"Variable '{var_name}' not found in artifacts")
        raise FinalSignal(str(self.artifacts[var_name]))

    def _invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """Invoke a registered tool.

        Args:
            tool_name: Name of the tool to invoke.
            **kwargs: Tool arguments.

        Returns:
            Tool result.

        Raises:
            ValueError: If tool doesn't exist.
            PermissionError: If role cannot use this tool.
        """
        if self.tool_registry is None:
            raise RuntimeError("No tool registry configured")

        return self.tool_registry.invoke(tool_name, self.role, **kwargs)

    def _list_tools(self) -> list[dict[str, Any]]:
        """List available tools for the current role.

        Returns:
            List of tool info dicts.
        """
        if self.tool_registry is None:
            return []

        return self.tool_registry.list_tools(role=self.role)

    def _invoke_script(self, script_id: str, **kwargs) -> Any:
        """Invoke a prepared script by ID.

        Args:
            script_id: Script identifier.
            **kwargs: Script arguments.

        Returns:
            Script result.

        Raises:
            ValueError: If script doesn't exist.
        """
        if self.script_registry is None:
            raise RuntimeError("No script registry configured")

        # Pass sandbox globals for code execution
        return self.script_registry.invoke(
            script_id,
            sandbox_globals=self._globals,
            **kwargs,
        )

    def _find_scripts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find scripts matching a natural language query.

        Args:
            query: Search query (e.g., "fetch python documentation").
            limit: Maximum results to return.

        Returns:
            List of matching script info dicts.
        """
        if self.script_registry is None:
            return []

        matches = self.script_registry.find_scripts(query, limit=limit)
        return [
            {
                "id": m.script.id,
                "description": m.script.description,
                "score": round(m.score, 2),
                "matched_on": m.matched_on,
            }
            for m in matches
        ]

    def _validate_code(self, code: str) -> None:
        """Validate code for dangerous patterns before execution.

        Args:
            code: Python code to validate.

        Raises:
            REPLSecurityError: If dangerous patterns detected.
        """
        # Patterns that indicate dangerous operations
        dangerous_patterns = [
            (r"\bimport\s+os\b", "import os"),
            (r"\bimport\s+sys\b", "import sys"),
            (r"\bimport\s+subprocess\b", "import subprocess"),
            (r"\bimport\s+socket\b", "import socket"),
            (r"\bimport\s+shutil\b", "import shutil"),
            (r"\bfrom\s+os\s+import\b", "from os import"),
            (r"\bfrom\s+sys\s+import\b", "from sys import"),
            (r"\bfrom\s+subprocess\s+import\b", "from subprocess import"),
            (r"\b__import__\s*\(", "__import__()"),
            (r"\beval\s*\(", "eval()"),
            (r"\bexec\s*\(", "exec()"),
            (r"\bcompile\s*\(", "compile()"),
            (r"\bopen\s*\(", "open()"),
            (r"\bgetattr\s*\(", "getattr()"),
            (r"\bsetattr\s*\(", "setattr()"),
            (r"\bdelattr\s*\(", "delattr()"),
            (r"\bglobals\s*\(", "globals()"),
            (r"\blocals\s*\(", "locals()"),
            (r"\bvars\s*\(", "vars()"),
            (r"\b__class__\b", "__class__"),
            (r"\b__bases__\b", "__bases__"),
            (r"\b__subclasses__\b", "__subclasses__"),
            (r"\b__mro__\b", "__mro__"),
            (r"\b__dict__\b", "__dict__"),
            (r"\b__globals__\b", "__globals__"),
            (r"\b__code__\b", "__code__"),
            (r"\b__builtins__\b", "__builtins__"),
        ]

        for pattern, description in dangerous_patterns:
            if re.search(pattern, code):
                raise REPLSecurityError(f"Dangerous operation not allowed: {description}")

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the sandboxed environment.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with output, is_final flag, and optional error.
        """
        import time

        self._execution_count += 1
        start_time = time.perf_counter()

        # Validate code for dangerous patterns
        try:
            self._validate_code(code)
        except REPLSecurityError as e:
            return ExecutionResult(
                output="",
                is_final=False,
                error=str(e),
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise REPLTimeout(f"Execution timed out after {self.config.timeout_seconds}s")

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Set timeout (Unix only)
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.config.timeout_seconds)

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, self._globals)

                output = stdout_capture.getvalue()
                if stderr_capture.getvalue():
                    output += "\n[STDERR]\n" + stderr_capture.getvalue()

                # Cap output
                if len(output) > self.config.output_cap:
                    output = output[: self.config.output_cap] + f"\n[... truncated at {self.config.output_cap} chars]"

                return ExecutionResult(
                    output=output,
                    is_final=False,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except FinalSignal as e:
                output = stdout_capture.getvalue()
                return ExecutionResult(
                    output=output,
                    is_final=True,
                    final_answer=e.answer,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except SyntaxError as e:
                return ExecutionResult(
                    output="",
                    is_final=False,
                    error=f"SyntaxError: {e}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except Exception as e:
                return ExecutionResult(
                    output=stdout_capture.getvalue(),
                    is_final=False,
                    error=f"{type(e).__name__}: {e}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

        finally:
            # Clear timeout
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def get_state(self) -> str:
        """Get a summary of current REPL state for the Root LM.

        Returns:
            String describing available variables and artifacts.
        """
        state_lines = [
            f"context: str ({len(self.context)} chars)",
            f"artifacts: {list(self.artifacts.keys()) if self.artifacts else '{}'}",
        ]

        # Show artifact previews
        for key, value in self.artifacts.items():
            preview = str(value)[:100]
            if len(str(value)) > 100:
                preview += "..."
            state_lines.append(f"  artifacts['{key}']: {preview}")

        return "\n".join(state_lines)

    def reset(self) -> None:
        """Reset the REPL state (clear artifacts, keep context)."""
        self.artifacts.clear()
        self._final_answer = None
        self._execution_count = 0
        self._globals = self._build_globals()
