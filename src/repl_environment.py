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

import ast
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


class ASTSecurityVisitor(ast.NodeVisitor):
    """AST visitor that checks for dangerous code patterns.

    This is more robust than regex because it analyzes the parsed syntax tree,
    making it immune to string concatenation tricks like:
        getattr(__builtins__, '__im' + 'port__')('os')
    """

    # Forbidden module imports
    FORBIDDEN_MODULES = frozenset({
        "os", "sys", "subprocess", "socket", "shutil", "pathlib",
        "tempfile", "multiprocessing", "threading", "ctypes", "pickle",
        "importlib", "builtins", "code", "codeop", "runpy", "pkgutil",
    })

    # Forbidden built-in function calls
    FORBIDDEN_CALLS = frozenset({
        "__import__", "eval", "exec", "compile", "open",
        "getattr", "setattr", "delattr", "hasattr",
        "globals", "locals", "vars", "dir",
        "input", "breakpoint", "memoryview",
    })

    # Forbidden attribute accesses (dunder attributes for escaping sandbox)
    FORBIDDEN_ATTRS = frozenset({
        "__class__", "__bases__", "__subclasses__", "__mro__",
        "__dict__", "__globals__", "__locals__", "__code__",
        "__builtins__", "__closure__", "__func__", "__self__",
        "__module__", "__qualname__", "__annotations__",
        "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
    })

    def __init__(self):
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Check regular imports: import os"""
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from imports: from os import path"""
        if node.module:
            module = node.module.split(".")[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"from {node.module} import ...")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for forbidden functions."""
        # Check direct calls: eval(...)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_CALLS:
                self.violations.append(f"{node.func.id}()")

        # Check attribute calls: obj.__class__()
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.FORBIDDEN_ATTRS:
                self.violations.append(f".{node.func.attr}()")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access for forbidden dunder attributes."""
        if node.attr in self.FORBIDDEN_ATTRS:
            self.violations.append(f".{node.attr}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Check subscript access for string-based dunder bypass attempts.

        Catches patterns like: obj['__class__'] or obj["__globals__"]
        """
        if isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                if node.slice.value in self.FORBIDDEN_ATTRS:
                    self.violations.append(f"['{node.slice.value}']")
        self.generic_visit(node)


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
class ExplorationEvent:
    """A single exploration event for logging."""

    function: str  # peek, grep, llm_call, llm_batch
    args: dict[str, Any]  # Arguments passed
    result_size: int  # Size of result (chars or items)
    timestamp: float  # Time of call
    token_estimate: int = 0  # Estimated tokens used


@dataclass
class ExplorationLog:
    """Log of exploration events for a REPL session."""

    events: list[ExplorationEvent] = field(default_factory=list)
    total_exploration_tokens: int = 0

    def add_event(
        self,
        function: str,
        args: dict[str, Any],
        result: Any,
    ) -> None:
        """Add an exploration event to the log."""
        import time

        # Estimate result size
        if isinstance(result, str):
            result_size = len(result)
        elif isinstance(result, list):
            result_size = len(result)
        else:
            result_size = 0

        # Rough token estimate (4 chars per token)
        token_estimate = result_size // 4

        event = ExplorationEvent(
            function=function,
            args=args,
            result_size=result_size,
            timestamp=time.time(),
            token_estimate=token_estimate,
        )
        self.events.append(event)
        self.total_exploration_tokens += token_estimate

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get a summary of the exploration strategy used."""
        function_counts: dict[str, int] = {}
        for event in self.events:
            function_counts[event.function] = function_counts.get(event.function, 0) + 1

        return {
            "total_events": len(self.events),
            "function_counts": function_counts,
            "total_tokens": self.total_exploration_tokens,
            "strategy_type": self._classify_strategy(function_counts),
        }

    def _classify_strategy(self, counts: dict[str, int]) -> str:
        """Classify the exploration strategy based on function usage."""
        if not counts:
            return "none"
        if counts.get("llm_call", 0) > 0 or counts.get("llm_batch", 0) > 0:
            return "delegated"  # Used sub-LLM calls
        if counts.get("grep", 0) > counts.get("peek", 0):
            return "search"  # Primarily used grep
        if counts.get("peek", 0) > 0:
            return "scan"  # Primarily used peek
        return "mixed"


@dataclass
class REPLConfig:
    """Configuration for the REPL environment."""

    timeout_seconds: int = 120
    output_cap: int = 8192
    max_grep_results: int = 100
    # Forced exploration validation (prevent premature FINAL)
    # Default False for backwards compatibility - enable for production use
    require_exploration_before_final: bool = False
    min_exploration_calls: int = 1  # Minimum peek/grep/llm_call before FINAL
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

        # Exploration tracking (for forced exploration validation)
        self._exploration_calls = 0  # Count of peek/grep/llm_call calls
        self._exploration_log = ExplorationLog()  # Detailed exploration log

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

        # Add LLM primitives if available (wrapped for exploration tracking)
        if self.llm_primitives is not None:
            globals_dict["llm_call"] = self._tracked_llm_call
            globals_dict["llm_batch"] = self._tracked_llm_batch

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
        self._exploration_calls += 1
        result = self.context[:n]
        self._exploration_log.add_event("peek", {"n": n}, result)
        return result

    def _grep(self, pattern: str) -> list[str]:
        """Search context with regex and return matching lines.

        Args:
            pattern: Regular expression pattern to search for.

        Returns:
            List of lines containing matches (capped at max_grep_results).
        """
        self._exploration_calls += 1
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

        self._exploration_log.add_event("grep", {"pattern": pattern}, matches)
        return matches

    def _tracked_llm_call(self, *args, **kwargs) -> str:
        """Wrapper for llm_call that tracks exploration.

        Args:
            *args: Positional arguments for llm_call.
            **kwargs: Keyword arguments for llm_call.

        Returns:
            llm_call result.
        """
        self._exploration_calls += 1
        result = self.llm_primitives.llm_call(*args, **kwargs)
        self._exploration_log.add_event("llm_call", {"args": args, "kwargs": kwargs}, result)
        return result

    def _tracked_llm_batch(self, *args, **kwargs) -> list[str]:
        """Wrapper for llm_batch that tracks exploration.

        Args:
            *args: Positional arguments for llm_batch.
            **kwargs: Keyword arguments for llm_batch.

        Returns:
            llm_batch result.
        """
        self._exploration_calls += 1
        result = self.llm_primitives.llm_batch(*args, **kwargs)
        self._exploration_log.add_event("llm_batch", {"args": args, "kwargs": kwargs}, result)
        return result

    def _final(self, answer: str) -> None:
        """Signal completion with final answer.

        Args:
            answer: The final answer to return.

        Raises:
            FinalSignal: Raised to terminate execution (after validation).
            ValueError: If exploration requirement not met.
        """
        # Check forced exploration validation
        if self.config.require_exploration_before_final:
            if self._exploration_calls < self.config.min_exploration_calls:
                raise ValueError(
                    f"Premature FINAL: Must call at least {self.config.min_exploration_calls} "
                    f"exploration function(s) (peek, grep, llm_call, llm_batch) before FINAL(). "
                    f"Current exploration calls: {self._exploration_calls}. "
                    "Use peek() or grep() to examine the context first."
                )
        raise FinalSignal(str(answer))

    def _final_var(self, var_name: str) -> None:
        """Signal completion, returning contents of a variable.

        Args:
            var_name: Name of variable in artifacts dict to return.

        Raises:
            FinalSignal: Raised to terminate execution (after validation).
            KeyError: If variable not found in artifacts.
            ValueError: If exploration requirement not met.
        """
        # Check forced exploration validation
        if self.config.require_exploration_before_final:
            if self._exploration_calls < self.config.min_exploration_calls:
                raise ValueError(
                    f"Premature FINAL_VAR: Must call at least {self.config.min_exploration_calls} "
                    f"exploration function(s) (peek, grep, llm_call, llm_batch) before FINAL_VAR(). "
                    f"Current exploration calls: {self._exploration_calls}. "
                    "Use peek() or grep() to examine the context first."
                )
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
        """Validate code for dangerous patterns using AST analysis.

        Uses AST-based validation which is more robust than regex because it
        analyzes the parsed syntax tree, making it immune to string tricks like:
            getattr(__builtins__, '__im' + 'port__')('os')

        Args:
            code: Python code to validate.

        Raises:
            REPLSecurityError: If dangerous patterns detected or syntax is invalid.
        """
        # First, try to parse the code to get an AST
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            # Let the actual execution handle syntax errors for better messages
            return

        # Run AST-based security analysis
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        if visitor.violations:
            # Report first violation (avoids info disclosure about all checks)
            raise REPLSecurityError(
                f"Dangerous operation not allowed: {visitor.violations[0]}"
            )

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

    def get_exploration_log(self) -> ExplorationLog:
        """Get the detailed exploration log.

        Returns:
            ExplorationLog containing all exploration events.
        """
        return self._exploration_log

    def get_exploration_strategy(self) -> dict[str, Any]:
        """Get a summary of the exploration strategy used.

        Returns:
            Dictionary with strategy summary including event counts and type.
        """
        return self._exploration_log.get_strategy_summary()

    def suggest_exploration(
        self,
        task_description: str,
        retriever: Any | None = None,
    ) -> list[str]:
        """Suggest exploration strategies based on similar past tasks.

        Uses episodic memory (if available) to suggest effective exploration
        strategies based on successful past tasks.

        Args:
            task_description: Description of the current task.
            retriever: TwoPhaseRetriever from orchestration.repl_memory (optional).

        Returns:
            List of suggested exploration function calls as strings.
        """
        suggestions = []

        # Default suggestions based on context characteristics
        context_len = len(self.context)

        if context_len < 500:
            suggestions.append("peek(500)  # Context is short, read it all")
        elif context_len < 2000:
            suggestions.append("peek(1000)  # Scan the beginning")
        else:
            suggestions.append("peek(500)  # Preview context")
            suggestions.append("grep('keyword')  # Search for specific patterns")

        # If retriever available, query for similar successful tasks
        if retriever is not None:
            try:
                # Try to get suggestions from episodic memory
                task_ir = {"objective": task_description, "task_type": "exploration"}
                results = retriever.retrieve_for_routing(task_ir)

                if results:
                    for r in results[:3]:
                        if hasattr(r.memory, "metadata") and r.memory.metadata:
                            metadata = r.memory.metadata
                            strategy = metadata.get("exploration_strategy", {})
                            if strategy.get("function_counts"):
                                counts = strategy["function_counts"]
                                if counts.get("grep", 0) > 0:
                                    suggestions.insert(0, "# Similar task used grep effectively")
                                if counts.get("llm_call", 0) > 0:
                                    suggestions.insert(0, "# Similar task delegated to sub-LLM")
            except Exception:
                pass  # Silently ignore retrieval errors

        return suggestions

    def reset(self) -> None:
        """Reset the REPL state (clear artifacts, keep context)."""
        self.artifacts.clear()
        self._final_answer = None
        self._execution_count = 0
        self._exploration_calls = 0
        self._exploration_log = ExplorationLog()  # Reset exploration log
        self._globals = self._build_globals()


def create_repl_environment(
    context: str,
    artifacts: dict[str, Any] | None = None,
    config: REPLConfig | None = None,
    llm_primitives: Any | None = None,
    tool_registry: Any | None = None,
    script_registry: Any | None = None,
    role: str | None = None,
    use_restricted_python: bool | None = None,
) -> REPLEnvironment:
    """Factory function to create REPL environment with optional RestrictedPython.

    When use_restricted_python is True (or None with feature flag enabled),
    creates a REPLEnvironment that delegates to RestrictedExecutor for execution.

    Args:
        context: The full input context.
        artifacts: Optional pre-existing artifacts.
        config: Optional REPL configuration.
        llm_primitives: Optional LLMPrimitives for llm_call/llm_batch.
        tool_registry: Optional ToolRegistry for TOOL() invocations.
        script_registry: Optional ScriptRegistry for SCRIPT() invocations.
        role: Role name for permission checking.
        use_restricted_python: Override feature flag (None = use flag).

    Returns:
        REPLEnvironment instance (may use RestrictedExecutor internally).
    """
    # Check feature flag if not explicitly set
    if use_restricted_python is None:
        from src.features import features
        use_restricted_python = features().restricted_python

    # Try to use RestrictedPython if requested
    if use_restricted_python:
        try:
            from src.restricted_executor import is_available, RestrictedExecutor

            if is_available():
                # Create a wrapped environment that uses RestrictedExecutor
                return _RestrictedREPLEnvironment(
                    context=context,
                    artifacts=artifacts,
                    config=config,
                    llm_primitives=llm_primitives,
                    tool_registry=tool_registry,
                    script_registry=script_registry,
                    role=role,
                )
        except ImportError:
            pass  # Fall back to standard REPL

    # Standard REPL environment
    return REPLEnvironment(
        context=context,
        artifacts=artifacts,
        config=config,
        llm_primitives=llm_primitives,
        tool_registry=tool_registry,
        script_registry=script_registry,
        role=role,
    )


class _RestrictedREPLEnvironment(REPLEnvironment):
    """REPL environment that uses RestrictedPython for execution.

    This is a wrapper around the standard REPLEnvironment that delegates
    execution to RestrictedExecutor for stronger security guarantees.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with RestrictedExecutor."""
        super().__init__(*args, **kwargs)

        # Import here to avoid issues if not available
        from src.restricted_executor import RestrictedExecutor

        # Create restricted executor with same config
        self._restricted_executor = RestrictedExecutor(
            context=self.context,
            artifacts=self.artifacts,
            timeout_seconds=self.config.timeout_seconds,
            output_cap=self.config.output_cap,
            max_grep_results=self.config.max_grep_results,
            llm_primitives=self.llm_primitives,
        )

    def execute(self, code: str) -> ExecutionResult:
        """Execute using RestrictedPython.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with output, is_final flag, and optional error.
        """
        # Use the restricted executor
        from src.restricted_executor import ExecutionResult as RestrictedResult

        result = self._restricted_executor.execute(code)

        # Convert to our ExecutionResult type
        return ExecutionResult(
            output=result.output,
            is_final=result.is_final,
            final_answer=result.final_answer,
            error=result.error,
            elapsed_seconds=result.elapsed_seconds,
        )
