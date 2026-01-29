"""Orchestrator service for Root LM operations.

This module contains the core orchestration logic including:
- Prompt building for the Root LM (delegated to src.prompt_builders)
- Code extraction from LLM responses (delegated to src.prompt_builders)
- Error classification for escalation decisions (delegated to src.prompt_builders)
- Escalation prompt building (delegated to src.prompt_builders)

DEPRECATION NOTICE:
    The prompt building functions in this module are deprecated.
    Import from src.prompt_builders instead:

        from src.prompt_builders import (
            build_root_lm_prompt,
            build_escalation_prompt,
            extract_code_from_response,
            classify_error,
        )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.roles import Role

# Import from unified prompt builders module
from src.prompt_builders import (
    build_root_lm_prompt as _build_root_lm_prompt,
    build_escalation_prompt as _build_escalation_prompt,
    extract_code_from_response as _extract_code_from_response,
    classify_error as _classify_error,
    auto_wrap_final as _auto_wrap_final,
)
from src.failure_router import FailureContext, ErrorCategory, RoutingDecision

if TYPE_CHECKING:
    pass


def build_root_lm_prompt(
    state: str,
    original_prompt: str,
    last_output: str = "",
    last_error: str = "",
    turn: int = 0,
    routing_context: str = "",
) -> str:
    """Build the prompt for the Root LM (frontdoor).

    DEPRECATED: Import from src.prompt_builders instead.

    Delegates to src.prompt_builders.build_root_lm_prompt().
    """
    return _build_root_lm_prompt(
        state=state,
        original_prompt=original_prompt,
        last_output=last_output,
        last_error=last_error,
        turn=turn,
        routing_context=routing_context,
    )


def extract_code_from_response(response: str) -> str:
    """Extract Python code from an LLM response.

    DEPRECATED: Import from src.prompt_builders instead.

    Delegates to src.prompt_builders.extract_code_from_response().
    """
    return _extract_code_from_response(response)


def auto_wrap_final(code: str) -> str:
    """Auto-wrap code in FINAL() if it looks like a final answer.

    DEPRECATED: Import from src.prompt_builders instead.

    Delegates to src.prompt_builders.auto_wrap_final().
    """
    return _auto_wrap_final(code)


def classify_error(error_message: str, gate_name: str = "") -> ErrorCategory:
    """Classify an error message into an ErrorCategory.

    DEPRECATED: Import from src.prompt_builders instead.

    Delegates to src.prompt_builders.classify_error().
    """
    return _classify_error(error_message, gate_name)


def build_escalation_prompt(
    original_prompt: str,
    state: str,
    failure_context: FailureContext,
    decision: RoutingDecision,
) -> str:
    """Build a prompt for an escalated role.

    DEPRECATED: Import from src.prompt_builders instead.

    Delegates to src.prompt_builders.build_escalation_prompt().
    """
    return _build_escalation_prompt(
        original_prompt=original_prompt,
        state=state,
        failure_context=failure_context,
        decision=decision,
    )


# Escalation role mapping (from lower to higher tier)
# NOTE: Consider migrating to src.escalation module for unified escalation logic
# Uses Role enum for type safety - str() returns the role value
ESCALATION_ROLES: dict[Role, Role] = {
    Role.WORKER_GENERAL: Role.CODER_PRIMARY,
    Role.CODER_PRIMARY: Role.ARCHITECT_GENERAL,
    Role.FRONTDOOR: Role.CODER_PRIMARY,
    Role.INGEST_LONG_CONTEXT: Role.ARCHITECT_GENERAL,
}
