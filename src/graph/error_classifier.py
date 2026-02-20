"""Error classification helpers for orchestration graph execution."""

from __future__ import annotations

from src.escalation import ErrorCategory


def classify_error(error_message: str) -> ErrorCategory:
    """Classify an error message into an ErrorCategory."""
    lower = error_message.lower()

    if any(kw in lower for kw in ("timeout", "timed out", "deadline")):
        return ErrorCategory.TIMEOUT
    if any(kw in lower for kw in ("json", "schema", "validation", "jsonschema")):
        return ErrorCategory.SCHEMA
    if any(kw in lower for kw in ("format", "style", "lint", "ruff", "markdown")):
        return ErrorCategory.FORMAT
    if any(
        kw in lower
        for kw in ("abort", "generation aborted", "early_abort", "early abort")
    ):
        return ErrorCategory.EARLY_ABORT
    if any(
        kw in lower
        for kw in ("backend", "connection", "unreachable", "infrastructure", "502", "503")
    ):
        return ErrorCategory.INFRASTRUCTURE
    if any(
        kw in lower
        for kw in ("syntax", "type error", "typeerror", "nameerror", "import error", "test fail")
    ):
        return ErrorCategory.CODE
    if any(kw in lower for kw in ("wrong", "incorrect", "assertion", "logic")):
        return ErrorCategory.LOGIC

    return ErrorCategory.UNKNOWN

