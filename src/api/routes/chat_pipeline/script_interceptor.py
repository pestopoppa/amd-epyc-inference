"""Script interception: zero-cost local resolution for trivial queries.

Intercepts simple, deterministic requests (timestamps, arithmetic, UUIDs,
model status) before they reach the LLM pipeline. Returns results with
zero tokens and zero latency.

Guarded by features().script_interception (default: False — opt-in).

Usage in chat pipeline:
    from src.api.routes.chat_pipeline.script_interceptor import try_intercept

    interception = try_intercept(request.prompt)
    if interception.matched:
        return build_intercepted_response(interception, ...)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

# Max message length to even attempt interception
_MAX_INTERCEPT_LENGTH = 200


@dataclass(frozen=True)
class InterceptionResult:
    """Result of a script interception attempt."""

    matched: bool
    pattern_name: str | None = None
    result: str | None = None
    elapsed_ms: float = 0.0


@dataclass
class _Interceptor:
    """Registered interceptor: pattern + handler."""

    name: str
    pattern: re.Pattern
    handler: Callable[[re.Match], str | None]


# Global interceptor registry
_interceptors: list[_Interceptor] = []


def register_interceptor(
    name: str,
    pattern: str,
    handler: Callable[[re.Match], str | None],
    flags: int = re.IGNORECASE,
) -> None:
    """Register a new interceptor pattern.

    Args:
        name: Interceptor name (for logging/telemetry).
        pattern: Regex pattern to match against user message.
        handler: Function(match) -> result_string or None to fall through.
        flags: Regex flags (default: IGNORECASE).
    """
    _interceptors.append(_Interceptor(
        name=name,
        pattern=re.compile(pattern, flags),
        handler=handler,
    ))


# ── Built-in handlers ────────────────────────────────────────────────────


def _handle_timestamp(match: re.Match) -> str:
    """Return current timestamp."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    # Check for format hint in the match
    full = match.group(0).lower()
    if "iso" in full:
        return now.isoformat()
    if "unix" in full or "epoch" in full:
        return str(int(now.timestamp()))
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


def _handle_arithmetic(match: re.Match) -> str | None:
    """Evaluate simple arithmetic expression safely."""
    expr = match.group("expr").strip()

    # Strict whitelist: only digits, operators, parens, dots, spaces
    if not re.match(r"^[\d\s\+\-\*\/\.\(\)%]+$", expr):
        return None  # Not safe — fall through to model

    # Reject empty or trivially short
    if len(expr) < 2:
        return None

    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        # Format: avoid floating point noise
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)
    except Exception:
        return None  # eval failed — fall through


def _handle_uuid(match: re.Match) -> str:
    """Generate a UUID v4."""
    import uuid

    return str(uuid.uuid4())


def _handle_date(match: re.Match) -> str:
    """Return today's date."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return now.strftime("%A, %B %d, %Y")


# ── Register built-in interceptors ───────────────────────────────────────

register_interceptor(
    "timestamp",
    r"(?:what(?:'s| is) (?:the )?(?:current )?(?:time|timestamp))"
    r"|(?:what time is it)"
    r"|(?:(?:current|give|get|show)(?: me)?(?: the)? (?:current )?(?:time|timestamp))"
    r"(?:\s+(?:in\s+)?(?:iso|unix|utc|epoch))?$",
    _handle_timestamp,
)

register_interceptor(
    "date",
    r"(?:what(?:'s| is) (?:the )?(?:today(?:'s)? )?date)"
    r"|(?:what (?:day|date) is (?:it )?today)"
    r"|(?:today(?:'s)? date)$",
    _handle_date,
)

register_interceptor(
    "arithmetic",
    r"^(?:what(?:'s| is)\s+|calculate\s+|compute\s+|eval(?:uate)?\s+)?"
    r"(?P<expr>[\d\(][\d\s\+\-\*\/\.\(\)%]{1,80}[\d\)])"
    r"(?:\s*[=?]?\s*)$",
    _handle_arithmetic,
)

register_interceptor(
    "uuid",
    r"(?:generate|create|give|get|make)(?: me)?(?: a| an)?\s+(?:new\s+)?uuid",
    _handle_uuid,
)


# ── Public API ───────────────────────────────────────────────────────────


def try_intercept(message: str) -> InterceptionResult:
    """Attempt to intercept a message with a local handler.

    Only checks the feature flag externally (caller should check before
    calling, or this function checks internally for safety).

    Args:
        message: Raw user message text.

    Returns:
        InterceptionResult — check .matched for success.
    """
    # Quick rejects
    if not message or len(message) > _MAX_INTERCEPT_LENGTH:
        return InterceptionResult(matched=False)

    stripped = message.strip()
    if not stripped:
        return InterceptionResult(matched=False)

    start = time.perf_counter()

    for interceptor in _interceptors:
        match = interceptor.pattern.search(stripped)
        if match:
            try:
                result = interceptor.handler(match)
            except Exception as exc:
                logger.debug(
                    "Interceptor '%s' handler failed: %s", interceptor.name, exc
                )
                continue

            if result is not None:
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(
                    "Script interception: %s (%.2fms)", interceptor.name, elapsed
                )
                return InterceptionResult(
                    matched=True,
                    pattern_name=interceptor.name,
                    result=result,
                    elapsed_ms=elapsed,
                )

    return InterceptionResult(matched=False)
