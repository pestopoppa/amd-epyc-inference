#!/usr/bin/env python3
"""Base classes and utilities for tool implementations.

Provides common functionality for all tools:
- Timeout handling
- Output truncation
- Error formatting
- Result wrapping
"""

from __future__ import annotations

import functools
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from src.config import _registry_timeout

logger = logging.getLogger(__name__)

# Default tool timeout from registry
_TOOL_TIMEOUT = int(_registry_timeout("tools", "base_default", 60))

T = TypeVar("T")


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    data: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0
    truncated: bool = False


class ToolTimeout(Exception):
    """Tool execution timed out."""

    pass


def with_timeout(seconds: int) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add timeout to a function.

    Args:
        seconds: Maximum execution time in seconds.

    Returns:
        Decorated function that raises ToolTimeout if time exceeded.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            def timeout_handler(signum: int, frame: Any) -> None:
                raise ToolTimeout(f"Execution timed out after {seconds}s")

            # Only set alarm on Unix
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)

            try:
                return func(*args, **kwargs)
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


def truncate_output(text: str, max_length: int = 8192) -> tuple[str, bool]:
    """Truncate text if it exceeds max length.

    Args:
        text: Text to potentially truncate.
        max_length: Maximum allowed length.

    Returns:
        Tuple of (truncated_text, was_truncated).
    """
    if len(text) <= max_length:
        return text, False

    truncated = text[:max_length]
    truncated += f"\n\n[... truncated at {max_length} chars, total was {len(text)}]"
    return truncated, True


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    timeout_seconds: int = _TOOL_TIMEOUT,
    max_output: int = 8192,
    **kwargs: Any,
) -> ToolResult:
    """Safely execute a function with timeout and output handling.

    Args:
        func: Function to execute.
        *args: Positional arguments.
        timeout_seconds: Timeout in seconds (from registry).
        max_output: Maximum output size.
        **kwargs: Keyword arguments.

    Returns:
        ToolResult with success status and data or error.
    """
    start = time.perf_counter()

    try:
        # Apply timeout
        timed_func = with_timeout(timeout_seconds)(func)
        result = timed_func(*args, **kwargs)

        elapsed = (time.perf_counter() - start) * 1000

        # Handle string output truncation
        truncated = False
        if isinstance(result, str):
            result, truncated = truncate_output(result, max_output)

        return ToolResult(
            success=True,
            data=result,
            elapsed_ms=elapsed,
            truncated=truncated,
        )

    except ToolTimeout as e:
        elapsed = (time.perf_counter() - start) * 1000
        return ToolResult(
            success=False,
            error=str(e),
            elapsed_ms=elapsed,
        )

    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception(f"Tool execution failed: {e}")
        return ToolResult(
            success=False,
            error=f"{type(e).__name__}: {e}",
            elapsed_ms=elapsed,
        )


def format_error(error: Exception, include_traceback: bool = False) -> str:
    """Format an exception for tool output.

    Args:
        error: Exception to format.
        include_traceback: Whether to include full traceback.

    Returns:
        Formatted error string.
    """
    if include_traceback:
        import traceback

        return traceback.format_exc()

    return f"{type(error).__name__}: {error}"
