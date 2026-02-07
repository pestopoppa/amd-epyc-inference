"""Intent classification and mode selection for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: direct mode detection, mode selection (REPL as unified default),
MemRL-informed mode selection, and proactive intent classification.

NOTE (2026-02-04): React mode has been unified into REPL with structured_mode=True.
The _select_mode() function now only returns "direct" or "repl".
"""

from __future__ import annotations

import logging
from typing import Any


log = logging.getLogger(__name__)



def _select_mode(
    prompt: str,
    context: str,
    state: "Any",
) -> str:
    """Select execution mode: direct or repl.

    React mode has been unified into REPL with structured_mode=True.
    This function now only returns "direct" or "repl".

    Uses MemRL route_with_mode() if available, falls back to REPL as default.
    REPL is the universal superset: models can FINAL() immediately for simple
    questions, or use tools/escalate/delegate for complex ones.

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        state: Application state (may have hybrid_router).

    Returns:
        One of "direct" or "repl".
    """
    # Try MemRL-based mode selection if available
    if hasattr(state, "hybrid_router") and state.hybrid_router is not None:
        try:
            task_ir = {
                "task_type": "chat",
                "objective": prompt[:200],
                "priority": "interactive",
                "context_length": len(context) if context else 0,
            }
            _routing, _strategy, mode = state.hybrid_router.route_with_mode(task_ir)
            # Map legacy "react" to "repl" (React is now unified)
            if mode == "react":
                mode = "repl"
            if mode in ("direct", "repl"):
                return mode
        except Exception as exc:
            log.debug("MemRL route_with_mode failed, using heuristic: %s", exc)

    # Heuristic fallback: REPL is the default (superset of direct + react)
    # Model can FINAL("answer") immediately for simple questions, or use
    # tools/escalate/delegate for complex ones. The REPL can also operate
    # in structured_mode=True for React-style one-tool-per-turn execution.
    #
    # Direct mode is only used when explicitly forced via request.force_mode.
    # The model-initiated mode selection defaults to REPL for MemRL exposure.
    return "repl"


def _classify_and_route(prompt: str, context: str = "", has_image: bool = False) -> tuple[str, str]:
    """Classify prompt intent and proactively route to the best specialist.

    Zero-latency keyword heuristic. Returns (role, strategy).
    Falls back to frontdoor if no strong signal.

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        has_image: Whether the request includes an image.

    Returns:
        Tuple of (role_name, routing_strategy).
    """
    from src.classifiers import classify_and_route

    result = classify_and_route(prompt, context, has_image)
    return result.role, result.strategy


# ============================================================================
# Confidence-Based Routing (Phase 3)
# ============================================================================


def _parse_confidence_response(response: str) -> dict[str, float]:
    """Parse CONF|SELF:0.7|ARCHITECT:0.9|WORKER:0.2 format.

    Args:
        response: Model response containing CONF| line.

    Returns:
        Dict mapping approach names (lowercase) to confidence values.
        Returns empty dict if parsing fails.

    Example:
        >>> _parse_confidence_response("CONF|SELF:0.85|ARCHITECT:0.60|WORKER:0.30")
        {'self': 0.85, 'architect': 0.6, 'worker': 0.3}
    """
    import re

    result: dict[str, float] = {}

    # Find the CONF| line
    match = re.search(r"CONF\|(.+)", response)
    if not match:
        return result

    pairs_str = match.group(1)
    # Parse KEY:VALUE pairs separated by |
    for pair in pairs_str.split("|"):
        pair = pair.strip()
        if ":" not in pair:
            continue
        key, _, value = pair.partition(":")
        key = key.strip().lower()
        try:
            result[key] = float(value.strip())
        except ValueError:
            continue

    return result


def _is_coding_task(prompt: str) -> bool:
    """Determine if a prompt is primarily a coding task.

    Used to select architect_coding vs architect_general when escalating.

    Args:
        prompt: The user's prompt.

    Returns:
        True if the task is coding-related.
    """
    from src.classifiers import is_coding_task

    return is_coding_task(prompt)


def _select_role_by_confidence(
    confidences: dict[str, float],
    threshold: float = 0.7,
    default_role: str = "frontdoor",
    is_coding: bool = False,
) -> tuple[str, float]:
    """Select the best role based on confidence scores.

    Picks the approach with highest confidence above threshold.
    Falls back to architect if no approach meets threshold (needs help).

    Args:
        confidences: Dict from _parse_confidence_response.
        threshold: Minimum confidence to trust the approach.
        default_role: Role to use if confidences is empty.
        is_coding: If True and ARCHITECT selected, use architect_coding.

    Returns:
        Tuple of (role_name, confidence_score).
    """
    if not confidences:
        return default_role, 0.0

    # Find the highest confidence
    best_key = max(confidences.keys(), key=lambda k: confidences.get(k, 0))
    best_conf = confidences.get(best_key, 0)

    # Map confidence keys to actual role names
    # ARCHITECT maps to architect_coding or architect_general based on task
    def key_to_role(key: str) -> str:
        if key == "self":
            return "frontdoor"
        elif key == "architect":
            return "architect_coding" if is_coding else "architect_general"
        elif key == "worker":
            return "worker_explore"
        else:
            return default_role

    if best_conf >= threshold:
        return key_to_role(best_key), best_conf

    # Below threshold: escalate to architect for complex reasoning
    architect_role = "architect_coding" if is_coding else "architect_general"
    return architect_role, best_conf


async def get_confidence_routing(
    prompt: str,
    context: str,
    primitives: "Any",
    threshold: float = 0.7,
) -> tuple[str, float, str]:
    """Get confidence-based routing decision from frontdoor.

    Asks the frontdoor model to estimate confidence for different approaches,
    then selects the best one above threshold.

    When ARCHITECT is selected, uses architect_coding for coding tasks
    and architect_general for non-coding tasks.

    Args:
        prompt: The user's question.
        context: Optional context text.
        primitives: LLMPrimitives instance for inference.
        threshold: Minimum confidence threshold (default 0.7).

    Returns:
        Tuple of (role, confidence, strategy).
        Strategy is "confidence" if routing was based on confidence,
        or "fallback" if the model didn't provide valid confidence.
    """
    from src.prompt_builders import build_confidence_estimation_prompt

    # Build and send confidence estimation prompt
    conf_prompt = build_confidence_estimation_prompt(prompt, context)

    try:
        response = primitives.llm_call(
            conf_prompt,
            role="frontdoor",
            n_tokens=64,  # Only need CONF|...|... line
            skip_suffix=True,
        )
    except Exception as exc:
        log.warning("Confidence estimation failed: %s", exc)
        return "frontdoor", 0.0, "fallback"

    # Parse the confidence response
    confidences = _parse_confidence_response(response)

    if not confidences:
        log.debug("No valid confidence scores in response: %s", response[:100])
        return "frontdoor", 0.0, "fallback"

    # Determine if this is a coding task (affects architect selection)
    is_coding = _is_coding_task(prompt)

    # Select role based on confidence
    role, conf = _select_role_by_confidence(
        confidences, threshold, is_coding=is_coding
    )
    log.info(
        "Confidence routing: %s (%.2f, coding=%s) from %s",
        role,
        conf,
        is_coding,
        {k: f"{v:.2f}" for k, v in confidences.items()},
    )

    return role, conf, "confidence"
