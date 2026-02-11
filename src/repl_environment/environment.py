"""Core REPLEnvironment class, factory function, and restricted variant.

This module assembles all mixins into the final REPLEnvironment class and
provides the create_repl_environment() factory function.
"""

from __future__ import annotations

import ast
import io
import signal
import uuid
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, TYPE_CHECKING

from src.repl_environment.types import (
    ExplorationLog,
    ExecutionResult,
    FinalSignal,
    REPLConfig,
    REPLSecurityError,
    REPLTimeout,
)
from src.repl_environment.security import ASTSecurityVisitor
from src.repl_environment.unicode_sanitizer import sanitize_code_unicode
from src.repl_environment.file_tools import _FileToolsMixin
from src.repl_environment.document_tools import _DocumentToolsMixin
from src.repl_environment.routing import _RoutingMixin
from src.repl_environment.code_search import _CodeSearchMixin
from src.repl_environment.procedure_tools import _ProcedureToolsMixin
from src.repl_environment.context import _ContextMixin
from src.repl_environment.state import _StateMixin
from src.research_context import ResearchContext

if TYPE_CHECKING:
    from orchestration.repl_memory.progress_logger import ProgressLogger


def _get_allowed_file_paths() -> list[str]:
    """Get allowed file path prefixes from config with fallback."""
    try:
        from src.config import get_config

        cfg = get_config()
        llm_root = str(cfg.paths.llm_root)
        # Ensure trailing slash for prefix matching
        return [
            llm_root if llm_root.endswith("/") else f"{llm_root}/",
            "/tmp/",
        ]
    except Exception:
        return ["/mnt/raid0/llm/", "/tmp/"]


class REPLEnvironment(
    _FileToolsMixin,
    _DocumentToolsMixin,
    _RoutingMixin,
    _CodeSearchMixin,
    _ProcedureToolsMixin,
    _ContextMixin,
    _StateMixin,
):
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

    # Allowed file paths for file operations (security)
    ALLOWED_FILE_PATHS = _get_allowed_file_paths()

    def __init__(
        self,
        context: str,
        artifacts: dict[str, Any] | None = None,
        config: REPLConfig | None = None,
        llm_primitives: Any | None = None,  # LLMPrimitives instance
        tool_registry: Any | None = None,  # ToolRegistry instance
        script_registry: Any | None = None,  # ScriptRegistry instance
        role: str | None = None,  # Role for permission checking
        progress_logger: ProgressLogger | None = None,  # For exploration logging
        task_id: str | None = None,  # Task ID for progress logging
        # MemRL components for routing-aware REPL
        retriever: Any | None = None,  # TwoPhaseRetriever for recall/route_advice
        hybrid_router: Any | None = None,  # HybridRouter for route_advice
        # Structured mode for React-style execution
        structured_mode: bool = False,
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
            progress_logger: Optional ProgressLogger for logging exploration events.
            task_id: Optional task ID for associating exploration logs with tasks.
            retriever: Optional TwoPhaseRetriever for recall/route_advice.
            hybrid_router: Optional HybridRouter for route_advice.
            structured_mode: If True, enforce one-tool-per-turn React-style execution.
        """
        self.context = context
        self.artifacts = artifacts if artifacts is not None else {}
        self.config = config if config is not None else REPLConfig()
        self.llm_primitives = llm_primitives
        self.progress_logger = progress_logger
        self.task_id = task_id or str(uuid.uuid4())
        self.tool_registry = tool_registry
        self.script_registry = script_registry
        self.role = role or "worker_general"  # Default to restricted role

        # MemRL components for routing-aware functions (recall, route_advice)
        self._retriever = retriever
        self._hybrid_router = hybrid_router

        # Structured mode: React-style one-tool-per-turn execution
        # Can be set via config or constructor parameter (constructor takes precedence)
        self._structured_mode = structured_mode or self.config.structured_mode

        # Execution state
        self._final_answer: str | None = None
        self._execution_count = 0
        self._tool_invocations = 0  # Count of TOOL()/CALL() invocations

        # Exploration tracking (for forced exploration validation)
        self._exploration_calls = 0  # Count of peek/grep/llm_call calls
        self._exploration_log = ExplorationLog()  # Detailed exploration log

        # Grep hits buffer for two-stage summarization pipeline
        self._grep_hits_buffer: list[dict[str, Any]] = []

        # Key findings buffer for session persistence
        self._findings_buffer: list[dict[str, Any]] = []

        # Research context tracker for tool result lineage
        self._research_context = ResearchContext()
        self._last_research_node: str | None = None

        # Build restricted globals
        self._globals = self._build_globals()

        # Verify mixin contracts are satisfied (runtime validation)
        # This ensures all required attributes exist before mixin methods are called
        required_attrs = [
            # Core state
            "config",
            "context",
            "artifacts",
            "role",
            "task_id",
            # Counters
            "_exploration_calls",
            "_execution_count",
            "_tool_invocations",
            # Logs and buffers
            "_exploration_log",
            "_grep_hits_buffer",
            "_findings_buffer",
            # Research context
            "_research_context",
            "_last_research_node",
            # State
            "_final_answer",
            "_globals",
            # Optional dependencies
            "llm_primitives",
            "tool_registry",
            "script_registry",
            "progress_logger",
            "_retriever",
            "_hybrid_router",
            # Methods
            "_validate_file_path",
            "_build_globals",
        ]
        for attr in required_attrs:
            assert hasattr(self, attr), (
                f"Mixin contract violated: missing required attribute '{attr}'"
            )

    # Modules safe for import inside the REPL sandbox.  These are
    # pure-computation or data-structure libraries — no filesystem,
    # network, or process-spawning capabilities.
    SAFE_IMPORT_MODULES: frozenset[str] = frozenset({
        # Standard library — math & data
        "math", "cmath", "decimal", "fractions", "statistics",
        "random", "numbers",
        # Standard library — data structures & iteration
        "collections", "itertools", "functools", "operator",
        "heapq", "bisect", "array", "queue",
        # Standard library — string & text
        "re", "string", "textwrap", "unicodedata",
        # Standard library — time (read-only)
        "time", "datetime", "calendar",
        # Standard library — misc safe
        "copy", "enum", "dataclasses", "typing",
        "json", "csv", "io", "struct", "base64",
        "hashlib", "hmac", "secrets",
        "pprint", "reprlib",
        # Scientific computing (if installed)
        "numpy", "scipy", "sympy",
    })

    def _build_globals(self) -> dict[str, Any]:
        """Build the restricted globals dict for exec()."""
        import builtins

        # Build safe builtins from the builtins module
        safe_builtins = {}
        for name in self.config.allowed_builtins:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        # Safe import: allows whitelisted modules only
        _safe_modules = self.SAFE_IMPORT_MODULES

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            top = name.split(".")[0]
            if top not in _safe_modules:
                raise ImportError(
                    f"Module '{name}' is not allowed in the REPL sandbox. "
                    f"Allowed: {', '.join(sorted(_safe_modules))}"
                )
            return builtins.__import__(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = _safe_import

        # Pre-load commonly used safe modules so _strip_import_lines()
        # doesn't cause NameErrors when it removes import statements.
        import json as _json
        import math as _math
        import re as _re
        import collections as _collections
        import itertools as _itertools
        import functools as _functools
        import statistics as _statistics
        import datetime as _datetime
        import fractions as _fractions
        import decimal as _decimal
        import copy as _copy

        # Optional scientific modules (may not be installed)
        try:
            import numpy as _numpy
        except ImportError:
            _numpy = None
        try:
            import scipy as _scipy
        except ImportError:
            _scipy = None

        globals_dict = {
            "__builtins__": safe_builtins,
            # Context variables
            "context": self.context,
            "artifacts": self.artifacts,
            # Safe modules (pre-loaded to match _strip_import_lines behavior)
            "json": _json,
            "math": _math,
            "re": _re,
            "collections": _collections,
            "itertools": _itertools,
            "functools": _functools,
            "statistics": _statistics,
            "datetime": _datetime,
            "fractions": _fractions,
            "decimal": _decimal,
            "copy": _copy,
            **({"numpy": _numpy} if _numpy else {}),
            **({"scipy": _scipy} if _scipy else {}),
            # Built-in functions
            "peek": self._peek,
            "grep": self._grep,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            # Document processing tools
            "ocr_document": self._ocr_document,
            "analyze_figure": self._analyze_figure,
            "extract_figure": self._extract_figure,
            # File system tools
            "list_dir": self._list_dir,
            "file_info": self._file_info,
            # Archive tools
            "archive_open": self._archive_open,
            "archive_extract": self._archive_extract,
            "archive_file": self._archive_file,
            "archive_search": self._archive_search,
            # Web tools
            "web_fetch": self._web_fetch,
            # Memory tools
            "recall": self._recall,
            # Code & document retrieval (NextPLAID multi-vector search)
            "code_search": self._code_search,
            "doc_search": self._doc_search,
            # Session persistence tools
            "mark_finding": self._mark_finding,
            "list_findings": self._list_findings,
            # Routing & self-assessment tools
            "escalate": self._escalate,
            "my_role": self._my_role,
            "route_advice": self._route_advice,
            "delegate": self._delegate,
            # Shell tools (sandboxed)
            "run_shell": self._run_shell,
            "run_python_code": self._run_python_code,
            # Self-management procedure tools
            "run_procedure": self._run_procedure,
            "list_procedures": self._list_procedures,
            "get_procedure_status": self._get_procedure_status,
            "checkpoint_create": self._checkpoint_create,
            "checkpoint_restore": self._checkpoint_restore,
            "registry_lookup": self._registry_lookup,
            "registry_update": self._registry_update,
            "benchmark_run": self._benchmark_run,
            "benchmark_compare": self._benchmark_compare,
            "gate_run": self._gate_run,
            "log_append": self._log_append,
            "file_write_safe": self._file_write_safe,
            # Long context exploration tools
            "chunk_context": self._chunk_context,
            "summarize_chunks": self._summarize_chunks,
            "context_len": self._context_len,
        }

        # Add LLM primitives if available (wrapped for exploration tracking)
        if self.llm_primitives is not None:
            globals_dict["llm_call"] = self._tracked_llm_call
            globals_dict["llm_batch"] = self._tracked_llm_batch

        # Add tool registry functions if available
        if self.tool_registry is not None:
            globals_dict["TOOL"] = self._invoke_tool
            globals_dict["CALL"] = self._call_tool
            globals_dict["list_tools"] = self._list_tools

        # Add script registry functions if available
        if self.script_registry is not None:
            globals_dict["SCRIPT"] = self._invoke_script
            globals_dict["find_scripts"] = self._find_scripts

        return globals_dict

    def _validate_file_path(self, path: str) -> tuple[bool, str | None]:
        """Validate that a file path is allowed.

        Args:
            path: Path to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        import os

        resolved = os.path.realpath(path)
        for allowed in self.ALLOWED_FILE_PATHS:
            if resolved.startswith(allowed):
                return True, None
        return False, f"Path not in allowed locations: {self.ALLOWED_FILE_PATHS}"

    def _validate_code(self, code: str) -> None:
        """Validate code for dangerous patterns using AST analysis.

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
            violation = visitor.violations[0]
            # Provide actionable hints for common cases
            hints = ""
            if "input()" in violation:
                hints = (
                    " Hint: input() is not available in the REPL. "
                    "For competitive programming solutions that read from stdin, "
                    "wrap your solution in a string and use "
                    'CALL("run_python_code", code=your_code_string, stdin_data=test_input) '
                    "to test it, then FINAL(your_code_string) to submit."
                )
            elif "exec()" in violation or "eval()" in violation:
                hints = (
                    " Hint: use run_python_code(code_string, stdin_data) "
                    "to test code, or write code directly (it is executed "
                    "automatically). Do NOT use exec()/eval()."
                )
            elif "open()" in violation:
                hints = (
                    " Hint: use peek(path) to read files or "
                    "file_write_safe(path, content) to write."
                )
            elif "import " in violation:
                hints = (
                    " Hint: for competitive programming solutions that need sys/os, "
                    "wrap your solution in a string and use "
                    'CALL("run_python_code", code=your_code_string, stdin_data=test_input) '
                    "to test it, then FINAL(your_code_string) to submit."
                )
            raise REPLSecurityError(
                f"Dangerous operation not allowed: {violation}.{hints}"
            )

    def _execute_structured(self, code: str, start_time: float) -> ExecutionResult:
        """Execute in structured (React-style) mode: one tool call per turn.

        This mode enforces the Thought/Action/Observation pattern:
        - Parses code for a single tool function call
        - If FINAL() found, treats as completion
        - If multiple tools, returns error requesting one at a time
        - Otherwise executes single tool and returns observation

        Args:
            code: Python code to execute (should be single tool call).
            start_time: Performance timer start for elapsed_seconds.

        Returns:
            ExecutionResult with observation output or final answer.
        """
        import re
        import time

        code = sanitize_code_unicode(code)

        # Validate code for dangerous patterns first
        try:
            self._validate_code(code)
        except REPLSecurityError as e:
            return ExecutionResult(
                output="",
                is_final=False,
                error=str(e),
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Check for FINAL() call - treat as completion.
        # Try triple-quoted strings first (models wrap multi-line code in
        # FINAL('''...''') or FINAL(\"\"\"...\"\"\") — auto_wrap_final
        # also generates this form), then fall back to single-quoted.
        final_match = (
            re.search(r"""\bFINAL\s*\(\s*'{3}(.+?)'{3}\s*\)""", code, re.DOTALL)
            or re.search(r'''\bFINAL\s*\(\s*"{3}(.+?)"{3}\s*\)''', code, re.DOTALL)
            or re.search(r'\bFINAL\s*\(\s*["\'](.+?)["\']\s*\)', code, re.DOTALL)
        )
        final_var_match = re.search(r'\bFINAL_VAR\s*\(\s*["\'](\w+)["\']\s*\)', code)

        if final_match:
            answer = final_match.group(1)
            return ExecutionResult(
                output="",
                is_final=True,
                final_answer=answer,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        if final_var_match:
            var_name = final_var_match.group(1)
            answer = str(self._globals.get(var_name, f"[Variable '{var_name}' not found]"))
            return ExecutionResult(
                output="",
                is_final=True,
                final_answer=answer,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # List of known tool functions in the REPL environment
        tool_functions = {
            "peek",
            "grep",
            "llm_call",
            "llm_batch",
            "TOOL",
            "CALL",
            "list_tools",
            "SCRIPT",
            "find_scripts",
            "list_dir",
            "file_info",
            "ocr_document",
            "analyze_figure",
            "extract_figure",
            "archive_open",
            "archive_extract",
            "archive_file",
            "archive_search",
            "web_fetch",
            "recall",
            "mark_finding",
            "list_findings",
            "escalate",
            "my_role",
            "route_advice",
            "delegate",
            "run_shell",
            "run_python_code",
            "run_procedure",
            "list_procedures",
            "get_procedure_status",
            "checkpoint_create",
            "checkpoint_restore",
            "registry_lookup",
            "registry_update",
            "benchmark_run",
            "benchmark_compare",
            "gate_run",
            "log_append",
            "file_write_safe",
            "chunk_context",
            "summarize_chunks",
            "context_len",
        }

        # Count tool function calls in the code
        tool_calls = []
        for func in tool_functions:
            # Match function calls: func_name( with word boundary
            pattern = rf"\b{func}\s*\("
            matches = list(re.finditer(pattern, code))
            tool_calls.extend([(func, m.start()) for m in matches])

        # Sort by position to get order of calls
        tool_calls.sort(key=lambda x: x[1])

        if len(tool_calls) > 1:
            # Multiple tool calls - request one at a time (React style)
            tool_names = [t[0] for t in tool_calls]
            return ExecutionResult(
                output="",
                is_final=False,
                error=f"Structured mode: Only one tool call per turn. "
                f"Found {len(tool_calls)} calls: {', '.join(tool_names)}. "
                f"Execute one tool, observe the result, then call the next.",
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Execute the code (single tool or simple expression)
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_handler = None
        _alarm_set = False
        try:
            if hasattr(signal, "SIGALRM"):
                try:

                    def timeout_handler(signum, frame):
                        raise REPLTimeout(
                            f"Execution timed out after {self.config.timeout_seconds}s"
                        )

                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.config.timeout_seconds)
                    _alarm_set = True
                except ValueError:
                    pass

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, self._globals)

                output = stdout_capture.getvalue()
                if stderr_capture.getvalue():
                    output += "\n[STDERR]\n" + stderr_capture.getvalue()

                # Format as observation for React-style loop
                if output:
                    observation = f"Observation: {output.strip()}"
                else:
                    observation = "Observation: [No output]"

                # Cap output
                if len(observation) > self.config.output_cap:
                    observation = (
                        observation[: self.config.output_cap]
                        + f"\n[... truncated at {self.config.output_cap} chars]"
                    )

                return ExecutionResult(
                    output=observation,
                    is_final=False,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except FinalSignal as e:
                return ExecutionResult(
                    output=stdout_capture.getvalue(),
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
            if _alarm_set:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

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

        # Structured mode: React-style one-tool-per-turn execution
        if self._structured_mode:
            return self._execute_structured(code, start_time)

        # Sanitize Unicode characters that models copy from question text
        code = sanitize_code_unicode(code)

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

        old_handler = None
        _alarm_set = False
        try:
            # Set timeout (Unix only, main thread only)
            if hasattr(signal, "SIGALRM"):
                try:
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.config.timeout_seconds)
                    _alarm_set = True
                except ValueError:
                    pass  # Not main thread — skip signal-based timeout

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, self._globals)

                output = stdout_capture.getvalue()
                if stderr_capture.getvalue():
                    output += "\n[STDERR]\n" + stderr_capture.getvalue()

                # Cap output
                if len(output) > self.config.output_cap:
                    output = (
                        output[: self.config.output_cap]
                        + f"\n[... truncated at {self.config.output_cap} chars]"
                    )

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
            if _alarm_set:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


def create_repl_environment(
    context: str,
    artifacts: dict[str, Any] | None = None,
    config: REPLConfig | None = None,
    llm_primitives: Any | None = None,
    tool_registry: Any | None = None,
    script_registry: Any | None = None,
    role: str | None = None,
    use_restricted_python: bool | None = None,
    progress_logger: ProgressLogger | None = None,
    task_id: str | None = None,
    structured_mode: bool = False,
) -> REPLEnvironment:
    """Factory function to create REPL environment with optional RestrictedPython.

    Args:
        context: The full input context.
        artifacts: Optional pre-existing artifacts.
        config: Optional REPL configuration.
        llm_primitives: Optional LLMPrimitives for llm_call/llm_batch.
        tool_registry: Optional ToolRegistry for TOOL() invocations.
        script_registry: Optional ScriptRegistry for SCRIPT() invocations.
        role: Role name for permission checking.
        use_restricted_python: Override feature flag (None = use flag).
        progress_logger: Optional ProgressLogger for exploration logging.
        task_id: Optional task ID for associating exploration logs.
        structured_mode: If True, enforce one-tool-per-turn React-style execution.

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
            from src.restricted_executor import is_available, RestrictedExecutor  # noqa: F401

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
                    progress_logger=progress_logger,
                    task_id=task_id,
                    structured_mode=structured_mode,
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
        progress_logger=progress_logger,
        task_id=task_id,
        structured_mode=structured_mode,
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

        result = self._restricted_executor.execute(code)

        # Convert to our ExecutionResult type
        return ExecutionResult(
            output=result.output,
            is_final=result.is_final,
            final_answer=result.final_answer,
            error=result.error,
            elapsed_seconds=result.elapsed_seconds,
        )
