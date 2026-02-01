"""Sandboxed Python REPL with context-as-variable pattern.

Package providing the REPLEnvironment class and supporting types.
Split into focused modules for maintainability:
- types: Data types, exceptions, constants
- security: AST-based code validation
- file_tools: File I/O, grep, shell, archive tools
- document_tools: OCR, figure analysis
- routing: Delegation, escalation, episodic memory
- procedure_tools: Self-management procedures, benchmarks, gates
- context: Context chunking, FINAL signals, LLM wrappers, tool dispatch
- state: State inspection, checkpoint/restore, exploration tracking
- environment: REPLEnvironment class, factory function
"""

from src.repl_environment.types import (
    TOOL_OUTPUT_START,
    TOOL_OUTPUT_END,
    wrap_tool_output,
    REPLError,
    REPLTimeout,
    REPLSecurityError,
    FinalSignal,
    ExplorationEvent,
    ExplorationLog,
    REPLConfig,
    ExecutionResult,
)
from src.repl_environment.security import ASTSecurityVisitor
from src.repl_environment.environment import (
    REPLEnvironment,
    create_repl_environment,
    _RestrictedREPLEnvironment,
)

__all__ = [
    # Types and constants
    "TOOL_OUTPUT_START",
    "TOOL_OUTPUT_END",
    "wrap_tool_output",
    "REPLError",
    "REPLTimeout",
    "REPLSecurityError",
    "FinalSignal",
    "ExplorationEvent",
    "ExplorationLog",
    "REPLConfig",
    "ExecutionResult",
    # Security
    "ASTSecurityVisitor",
    # Core classes
    "REPLEnvironment",
    "create_repl_environment",
    "_RestrictedREPLEnvironment",
]
