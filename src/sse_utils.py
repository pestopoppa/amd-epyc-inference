"""SSE (Server-Sent Events) utilities for the orchestration API.

This module provides SSE streaming helpers that can use either:
1. sse-starlette library (preferred, feature flag controlled)
2. Manual SSE formatting with StreamingResponse (fallback)

Usage:
    from src.sse_utils import create_sse_response, sse_event

    async def generate():
        yield sse_event("token", {"content": "Hello"})
        yield sse_event("done", {})

    return create_sse_response(generate())
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

# Try to import sse-starlette
try:
    from sse_starlette.sse import EventSourceResponse, ServerSentEvent

    SSE_STARLETTE_AVAILABLE = True
except ImportError:
    SSE_STARLETTE_AVAILABLE = False
    EventSourceResponse = None
    ServerSentEvent = None

# Import FastAPI's StreamingResponse as fallback
from starlette.responses import StreamingResponse


def is_sse_starlette_available() -> bool:
    """Check if sse-starlette library is available.

    Returns:
        True if sse-starlette is installed.
    """
    return SSE_STARLETTE_AVAILABLE


def sse_event(event_type: str, data: dict[str, Any] | str) -> dict[str, Any]:
    """Create an SSE event dictionary.

    This helper creates a standardized event format that works with both
    sse-starlette and manual SSE formatting.

    Args:
        event_type: Event type (e.g., "token", "turn_start", "done")
        data: Event data (dict or string)

    Returns:
        Event dictionary with "event" and "data" keys.

    Example:
        event = sse_event("token", {"content": "Hello"})
        # Returns: {"event": "token", "data": "{\"content\": \"Hello\"}"}
    """
    if isinstance(data, dict):
        # Include type in data for backward compatibility
        data_with_type = {"type": event_type, **data}
        data_str = json.dumps(data_with_type)
    else:
        data_str = data

    return {
        "event": event_type,
        "data": data_str,
    }


def format_sse_manual(event: dict[str, Any]) -> str:
    """Format an SSE event for manual streaming.

    Args:
        event: Event dictionary with "event" and "data" keys.

    Returns:
        Formatted SSE string with proper newlines.
    """
    # The data field contains the full JSON including type
    return f"data: {event['data']}\n\n"


def format_done_event() -> str:
    """Format the special [DONE] event for manual streaming.

    Returns:
        Formatted [DONE] SSE string.
    """
    return "data: [DONE]\n\n"


async def _convert_to_manual_sse(
    generator: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[str]:
    """Convert event dictionaries to manual SSE format.

    Args:
        generator: Async generator yielding event dictionaries.

    Yields:
        Formatted SSE strings.
    """
    async for event in generator:
        if event.get("event") == "done":
            yield format_done_event()
        else:
            yield format_sse_manual(event)


def create_sse_response(
    generator: AsyncIterator[dict[str, Any]],
    use_sse_starlette: bool | None = None,
) -> StreamingResponse:
    """Create an SSE response from an async generator.

    Uses sse-starlette if available and enabled, otherwise falls back
    to manual SSE formatting with StreamingResponse.

    Args:
        generator: Async generator yielding event dictionaries.
        use_sse_starlette: Override automatic detection (None = auto-detect).

    Returns:
        StreamingResponse or EventSourceResponse configured for SSE.

    Example:
        async def generate():
            yield sse_event("token", {"content": "Hello"})
            yield sse_event("done", {})

        return create_sse_response(generate())
    """
    # Determine whether to use sse-starlette
    if use_sse_starlette is None:
        # Check feature flag
        try:
            from src.features import features

            use_sse_starlette = features().streaming and SSE_STARLETTE_AVAILABLE
        except ImportError:
            use_sse_starlette = SSE_STARLETTE_AVAILABLE

    if use_sse_starlette and SSE_STARLETTE_AVAILABLE:
        # Use sse-starlette's EventSourceResponse
        async def sse_generator():
            async for event in generator:
                if event.get("event") == "done":
                    yield {"event": "done", "data": "[DONE]"}
                else:
                    yield event

        return EventSourceResponse(
            sse_generator(),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Fallback to manual SSE with StreamingResponse
    return StreamingResponse(
        _convert_to_manual_sse(generator),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Convenience functions for common event types


def token_event(content: str) -> dict[str, Any]:
    """Create a token event.

    Args:
        content: Token content.

    Returns:
        Token event dictionary.
    """
    return sse_event("token", {"content": content})


def thinking_event(content: str) -> dict[str, Any]:
    """Create a thinking event (for chain-of-thought display).

    Args:
        content: Thinking step content.

    Returns:
        Thinking event dictionary.
    """
    return sse_event("thinking", {"content": content})


def turn_start_event(turn: int, role: str) -> dict[str, Any]:
    """Create a turn start event.

    Args:
        turn: Turn number (1-indexed).
        role: Role executing this turn.

    Returns:
        Turn start event dictionary.
    """
    return sse_event("turn_start", {"turn": turn, "role": role})


def turn_end_event(tokens: int, elapsed_ms: int) -> dict[str, Any]:
    """Create a turn end event.

    Args:
        tokens: Number of tokens processed.
        elapsed_ms: Elapsed time in milliseconds.

    Returns:
        Turn end event dictionary.
    """
    return sse_event("turn_end", {"tokens": tokens, "elapsed_ms": elapsed_ms})


def error_event(message: str) -> dict[str, Any]:
    """Create an error event.

    Args:
        message: Error message.

    Returns:
        Error event dictionary.
    """
    return sse_event("error", {"message": message})


def final_event(answer: str) -> dict[str, Any]:
    """Create a final answer event.

    Args:
        answer: Final answer content.

    Returns:
        Final event dictionary.
    """
    return sse_event("final", {"answer": answer})


def done_event() -> dict[str, Any]:
    """Create a done event (signals end of stream).

    Returns:
        Done event dictionary.
    """
    return {"event": "done", "data": "[DONE]"}
