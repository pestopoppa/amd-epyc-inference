"""Utility functions and constants for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: token estimation, stub detection, answer resolution,
output quality heuristics, format enforcement, and shared constants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.features import features
from src.prompt_builders import (
    build_formalizer_prompt,
    detect_format_constraints,
)
from src.roles import Role

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives
    from src.services.document_preprocessor import PreprocessingResult


# ── Role-specific timeouts (Phase 1b: KV cache bug mitigation) ──────────
# Sourced from centralized config (src.config.TimeoutsConfig).
from src.config import get_config as _get_config

_timeouts = _get_config().timeouts
ROLE_TIMEOUTS: dict[str, int] = _timeouts.role_timeouts_dict()
DEFAULT_TIMEOUT_S: int = _timeouts.default_request


@dataclass
class RoutingResult:
    """Encapsulates all routing decisions made before execution.

    Created by _route_request(), consumed by mode handlers and response builder.
    Frozen after creation — routing is a read-only decision.
    """

    task_id: str
    task_ir: dict
    use_mock: bool
    routing_decision: list = field(default_factory=list)
    routing_strategy: str = ""
    formalization_applied: bool = False
    timeout_s: int = DEFAULT_TIMEOUT_S
    document_result: PreprocessingResult | None = None

    @property
    def role(self) -> str:
        """Primary role for this request."""
        if self.routing_decision:
            return str(self.routing_decision[0])
        return str(Role.FRONTDOOR)

    def timeout_for_role(self, role: str) -> int:
        """Get timeout for a specific role (used during escalation)."""
        return ROLE_TIMEOUTS.get(str(role), DEFAULT_TIMEOUT_S)

# Three-stage summarization configuration — values sourced from centralized config
_chat_cfg = _get_config().chat
THREE_STAGE_CONFIG = {
    "enabled": True,
    "threshold_tokens": _chat_cfg.summarization_threshold_tokens,
    "multi_doc_discount": _chat_cfg.multi_doc_discount,
    "stage1_role": Role.FRONTDOOR,
    "stage2_role": Role.INGEST_LONG_CONTEXT,
    "compression": {
        "enabled": _chat_cfg.compression_enabled,
        "min_chars": _chat_cfg.compression_min_chars,
        "target_ratio": _chat_cfg.compression_target_ratio,
        "stage1_context_limit": _chat_cfg.stage1_context_limit,
    },
}

# Backwards compatibility alias
TWO_STAGE_CONFIG = THREE_STAGE_CONFIG

# Qwen chat-template stop token — prevents runaway generation past turn boundary
QWEN_STOP = _get_config().llm.qwen_stop_token

# Long context exploration configuration
LONG_CONTEXT_CONFIG = {
    "enabled": _chat_cfg.long_context_enabled,
    "threshold_chars": _chat_cfg.long_context_threshold_chars,
    "max_turns": _chat_cfg.long_context_max_turns,
}

_STUB_PATTERNS = {
    "complete", "see above", "analysis complete", "estimation complete",
    "done", "finished", "see results above", "see output above",
    "see structured output above", "see integrated results above",
    "see the structured output above",
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough: 4 chars per token)."""
    return len(text) // 4


def _is_stub_final(text: str) -> bool:
    """Detect when FINAL() arg is a stub pointing to printed output.

    Models often print their analysis via print(), then call
    FINAL("Analysis complete. See above.") — the real content
    is in result.output, not result.final_answer.
    """
    normalized = text.strip().rstrip(".").lower()
    return any(p in normalized for p in _STUB_PATTERNS)


def _strip_tool_outputs(text: str, tool_outputs: list[str]) -> str:
    """Strip known tool outputs from captured REPL output.

    Routing tools (my_role, route_advice, list_dir, recall) return JSON/TOON
    strings that get captured in stdout. When the model prints these, they
    contaminate the final answer. Strip them.

    Uses structured delimiters (<<<TOOL_OUTPUT>>>...<<<END_TOOL_OUTPUT>>>)
    for reliable regex-based stripping, with legacy exact-string matching
    as fallback for outputs without delimiters.

    Args:
        text: The captured stdout text.
        tool_outputs: List of exact tool output strings to strip.

    Returns:
        Text with tool outputs removed, cleaned up.
    """
    if not text:
        return text

    result = text

    # Primary: strip structured delimiters (reliable, regex-based)
    result = re.sub(
        r"<<<TOOL_OUTPUT>>>.*?<<<END_TOOL_OUTPUT>>>",
        "",
        result,
        flags=re.DOTALL,
    )

    # Fallback: legacy exact-string matching for pre-delimiter tool outputs
    if tool_outputs:
        for output in tool_outputs:
            if output in result:
                result = result.replace(output, "")

    # Also strip common prefixes the model adds around tool outputs
    for prefix in [
        r"Current [Rr]ole:\s*",
        r"Available files:\s*",
        r"Routing advice:\s*",
        r"Could not get routing advice:\s*",
        r"Creating a ticket for further investigation\s*",
    ]:
        result = re.sub(prefix, "", result)

    # Clean up: collapse multiple blank lines, strip
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def _resolve_answer(result: "ExecutionResult", tool_outputs: list[str] | None = None) -> str:
    """Extract the best answer from an ExecutionResult.

    Handles cases where the model prints content then uses a stub FINAL().
    Strips tool outputs (my_role, route_advice, list_dir) from captured output.
    """
    captured = result.output.strip() if result.output else ""
    final = result.final_answer or ""

    # Strip tool outputs from captured stdout
    if tool_outputs:
        captured = _strip_tool_outputs(captured, tool_outputs)

    if captured and _is_stub_final(final):
        return captured
    elif captured and final and captured != final:
        # Prepend captured output if FINAL() doesn't already contain it
        if final not in captured:
            return f"{captured}\n\n{final}"
        return final
    else:
        return final


def _truncate_looped_answer(answer: str, prompt: str) -> str:
    """Defense-in-depth: truncate answer if prompt text reappears in it.

    Some models loop back to echoing the prompt after completing their answer.
    Detect this and truncate before the repeated prompt content.

    Args:
        answer: The model's raw output.
        prompt: The original prompt sent to the model.

    Returns:
        Truncated answer if loop detected, original answer otherwise.
    """
    if not answer or not prompt or len(prompt) < 40:
        return answer

    # Use a suffix of the prompt (last 80 chars) as the probe — if this
    # appears in the answer, the model is looping back to the prompt.
    probe = prompt[-80:].strip()
    if not probe:
        return answer

    idx = answer.find(probe)
    if idx > 0:
        # Truncate everything from the repeated prompt onwards
        truncated = answer[:idx].rstrip()
        if len(truncated) > 20:  # Only truncate if we keep a meaningful answer
            return truncated

    return answer


def _should_formalize(prompt: str) -> tuple[bool, str]:
    """Detect if the prompt has format constraints that need enforcement.

    Args:
        prompt: The user's prompt.

    Returns:
        Tuple of (should_formalize, format_spec_description).
    """
    if not features().output_formalizer:
        return False, ""

    constraints = detect_format_constraints(prompt)
    if constraints:
        return True, "; ".join(constraints)
    return False, ""


def _formalize_output(
    answer: str,
    prompt: str,
    format_spec: str,
    primitives: "LLMPrimitives",
) -> str:
    """Reformat an answer to satisfy detected format constraints.

    Uses worker_explore (7B, 44 t/s) for fast reformatting.
    The answer content is correct — only format needs fixing.

    Args:
        answer: The correct-content answer to reformat.
        prompt: The original user prompt.
        format_spec: Description of format constraints to satisfy.
        primitives: LLM primitives for inference.

    Returns:
        Reformatted answer, or original if formalization fails.
    """
    import logging
    log = logging.getLogger(__name__)

    formalizer_prompt = build_formalizer_prompt(answer, prompt, format_spec)
    try:
        result = primitives.llm_call(
            formalizer_prompt,
            role="worker_explore",
            n_tokens=2000,
            skip_suffix=True,
        )
        reformatted = result.strip()
        if reformatted and len(reformatted) > 5:
            log.info(f"Formalized output for constraint: {format_spec}")
            return reformatted
        return answer
    except Exception as e:
        log.warning(f"Output formalization failed: {e}")
        return answer
