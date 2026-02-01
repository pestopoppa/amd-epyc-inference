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
from src.repl_environment.file_tools import _FileToolsMixin
from src.repl_environment.document_tools import _DocumentToolsMixin
from src.repl_environment.routing import _RoutingMixin
from src.repl_environment.procedure_tools import _ProcedureToolsMixin
from src.repl_environment.context import _ContextMixin
from src.repl_environment.state import _StateMixin

if TYPE_CHECKING:
    from orchestration.repl_memory.progress_logger import ProgressLogger
    from orchestration.repl_memory.retriever import TwoPhaseRetriever


class REPLEnvironment(
    _FileToolsMixin,
    _DocumentToolsMixin,
    _RoutingMixin,
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
    ALLOWED_FILE_PATHS = [
        "/mnt/raid0/llm/",
        "/tmp/",
    ]

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

        # Import json for REPL use (needed for document processing)
        import json as _json

        globals_dict = {
            "__builtins__": safe_builtins,
            # Context variables
            "context": self.context,
            "artifacts": self.artifacts,
            # Safe modules
            "json": _json,
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
                    progress_logger=progress_logger,
                    task_id=task_id,
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
