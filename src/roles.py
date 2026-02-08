#!/usr/bin/env python3
"""Role definitions for hierarchical orchestration.

This module defines all agent roles in the orchestration system. Roles are
organized into tiers:

    Tier A (Frontdoor): Interactive chat, intent classification, task routing
    Tier B (Specialists): Domain-specific processing (coder, ingest, architect)
    Tier C (Workers): Parallel file-level implementation
    Tier D (Draft): Speculative decoding draft models

Usage:
    from src.roles import Role, Tier, get_tier

    # Use enum instead of strings
    role = Role.CODER_PRIMARY
    print(role.value)  # "coder_primary"

    # Get tier for a role
    tier = get_tier(Role.CODER_PRIMARY)  # Tier.B

    # Check if role is valid
    if Role.is_valid("coder_primary"):
        role = Role("coder_primary")

    # Get escalation target
    target = Role.CODER_PRIMARY.escalates_to()  # Role.ARCHITECT_GENERAL
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class FailoverReason(str, Enum):
    """Reason for triggering a model fallback (distinct from task escalation).

    Fallback is for infrastructure failures — the model is unavailable.
    Escalation is for task complexity — the model can't solve the problem.
    """

    CIRCUIT_OPEN = "circuit_open"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    OOM = "oom"


class Tier(str, Enum):
    """Agent tier in the orchestration hierarchy.

    Tiers define the capability and cost level of agents:
    - A: Frontdoor (always resident, low latency)
    - B: Specialists (loaded on demand, higher capability)
    - C: Workers (parallel execution, stateless)
    - D: Draft (speculative decoding support)
    """

    A = "A"  # Frontdoor
    B = "B"  # Specialists
    C = "C"  # Workers
    D = "D"  # Draft models


class Role(str, Enum):
    """Agent role identifiers.

    All agent roles in the orchestration system. Using this enum instead of
    raw strings enables:
    - IDE autocomplete
    - Compile-time typo detection
    - Centralized documentation of all roles
    - Type safety in function signatures

    Role naming convention:
    - {tier}_{function}: e.g., worker_math, coder_primary
    - Specific variants: e.g., architect_general vs architect_coding
    """

    # =========================================================================
    # Tier A: Frontdoor
    # =========================================================================
    FRONTDOOR = "frontdoor"
    """Interactive chat, intent classification, task routing.

    The frontdoor receives all user requests and either handles them directly
    or routes them to appropriate specialists. Always resident in memory.
    """

    # =========================================================================
    # Tier B: Specialists
    # =========================================================================
    CODER_PRIMARY = "coder_primary"
    """Primary code generation specialist.

    Handles most coding tasks: implementation, refactoring, debugging.
    Uses speculative decoding with Qwen2.5-Coder-0.5B draft for 11x speedup.
    """

    CODER_ESCALATION = "coder_escalation"
    """Escalation target for complex coding tasks.

    Used when coder_primary fails or task exceeds its capability.
    May use larger model or different acceleration strategy.
    """

    INGEST_LONG_CONTEXT = "ingest_long_context"
    """Long-context document ingestion.

    Processes documents >8K tokens. Uses SSM architecture (Qwen3-Next)
    which is incompatible with speculative decoding - expert reduction only.
    """

    ARCHITECT_GENERAL = "architect_general"
    """General architecture and system design.

    Handles high-level design, invariants, system architecture decisions.
    Top of escalation chain for most tasks.
    """

    ARCHITECT_CODING = "architect_coding"
    """Ultimate coding escalation target.

    Used for the most complex coding challenges that defeat other specialists.
    Typically uses largest available model (Qwen3-Coder-480B).
    """

    THINKING_REASONING = "thinking_reasoning"
    """Chain-of-thought reasoning specialist.

    Used for tasks requiring explicit reasoning chains.
    May output <think> tags before final answer.
    """

    # =========================================================================
    # Tier C: Workers
    # =========================================================================
    WORKER_GENERAL = "worker_general"
    """General-purpose worker for parallel tasks.

    Handles file-level implementation, boilerplate generation, documentation.
    Stateless, many can run concurrently.
    """

    WORKER_MATH = "worker_math"
    """Mathematical reasoning worker.

    Specialized for mathematical computations, proofs, step verification.
    Uses Qwen2.5-Math model.
    """

    WORKER_SUMMARIZE = "worker_summarize"
    """Summarization worker.

    Generates summaries, extracts key points from documents.
    Good candidate for prompt lookup acceleration.
    """

    WORKER_VISION = "worker_vision"
    """Vision-language worker.

    Handles tasks involving images: OCR, image understanding, UI analysis.
    Uses multimodal model (Qwen2.5-VL).
    """

    TOOLRUNNER = "toolrunner"
    """Tool execution worker.

    Runs external tools (lint, test, search) and processes results.
    Works with tool registry for permission checking.
    """

    # =========================================================================
    # Tier D: Draft Models
    # =========================================================================
    DRAFT_CODER = "draft_coder"
    """Draft model for code generation speculative decoding.

    Qwen2.5-Coder-0.5B - compatible with Qwen2.5-Coder family targets.
    """

    DRAFT_GENERAL = "draft_general"
    """Draft model for general speculative decoding.

    Qwen2.5-0.5B - compatible with Qwen2.5 family targets.
    """

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __str__(self) -> str:
        """Return the role value string.

        This ensures str(Role.CODER_PRIMARY) returns "coder_primary"
        instead of "Role.CODER_PRIMARY".
        """
        return self.value

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string is a valid role.

        Args:
            value: String to check.

        Returns:
            True if value is a valid Role.
        """
        return value in cls._value2member_map_

    @classmethod
    def from_string(cls, value: str, default: "Role | None" = None) -> "Role | None":
        """Convert string to Role, returning default if invalid.

        Args:
            value: String to convert.
            default: Default to return if invalid.

        Returns:
            Role enum member or default.
        """
        try:
            return cls(value)
        except ValueError:
            return default

    def escalates_to(self) -> "Role | None":
        """Get the escalation target for this role.

        Returns:
            The role to escalate to, or None if at top of chain.
        """
        return _ESCALATION_MAP.get(self)

    @property
    def tier(self) -> Tier:
        """Get the tier for this role.

        Returns:
            Tier enum member.
        """
        return _TIER_MAP.get(self, Tier.C)

    @property
    def is_specialist(self) -> bool:
        """Check if this is a specialist (Tier B) role."""
        return self.tier == Tier.B

    @property
    def is_worker(self) -> bool:
        """Check if this is a worker (Tier C) role."""
        return self.tier == Tier.C

    @property
    def is_draft(self) -> bool:
        """Check if this is a draft (Tier D) role."""
        return self.tier == Tier.D


# Role -> Tier mapping
_TIER_MAP: dict[Role, Tier] = {
    # Tier A
    Role.FRONTDOOR: Tier.A,
    # Tier B
    Role.CODER_PRIMARY: Tier.B,
    Role.CODER_ESCALATION: Tier.B,
    Role.INGEST_LONG_CONTEXT: Tier.B,
    Role.ARCHITECT_GENERAL: Tier.B,
    Role.ARCHITECT_CODING: Tier.B,
    Role.THINKING_REASONING: Tier.B,
    # Tier C
    Role.WORKER_GENERAL: Tier.C,
    Role.WORKER_MATH: Tier.C,
    Role.WORKER_SUMMARIZE: Tier.C,
    Role.WORKER_VISION: Tier.C,
    Role.TOOLRUNNER: Tier.C,
    # Tier D
    Role.DRAFT_CODER: Tier.D,
    Role.DRAFT_GENERAL: Tier.D,
}


# Role -> Escalation target mapping
_ESCALATION_MAP: dict[Role, Role] = {
    # Workers escalate to coder
    Role.WORKER_GENERAL: Role.CODER_PRIMARY,
    Role.WORKER_MATH: Role.CODER_PRIMARY,
    Role.WORKER_SUMMARIZE: Role.CODER_PRIMARY,
    Role.WORKER_VISION: Role.CODER_PRIMARY,
    Role.TOOLRUNNER: Role.CODER_PRIMARY,
    # Frontdoor escalates to coder
    Role.FRONTDOOR: Role.CODER_PRIMARY,
    # Coder escalates to architect
    Role.CODER_PRIMARY: Role.ARCHITECT_GENERAL,
    Role.CODER_ESCALATION: Role.ARCHITECT_CODING,
    Role.THINKING_REASONING: Role.ARCHITECT_GENERAL,
    # Ingest escalates to architect
    Role.INGEST_LONG_CONTEXT: Role.ARCHITECT_GENERAL,
    # Architects have no escalation (top of chain)
    # Draft models don't escalate (they support other models)
}


# Role -> Fallback alternatives (infrastructure failure, NOT task escalation)
# Used when model_fallback feature is enabled and primary backend is circuit-open.
_FALLBACK_MAP: dict[Role, list[Role]] = {
    Role.ARCHITECT_GENERAL: [Role.ARCHITECT_CODING, Role.CODER_PRIMARY],
    Role.ARCHITECT_CODING: [Role.ARCHITECT_GENERAL, Role.CODER_ESCALATION],
    Role.CODER_PRIMARY: [Role.CODER_ESCALATION],
    Role.CODER_ESCALATION: [Role.CODER_PRIMARY],
    Role.WORKER_MATH: [Role.WORKER_GENERAL],
    Role.INGEST_LONG_CONTEXT: [Role.ARCHITECT_GENERAL],
    Role.FRONTDOOR: [],  # Always-on, no fallback
    Role.WORKER_VISION: [],  # Hardware-specific, no fallback
}


def get_fallback_roles(role: Role | str) -> list[Role]:
    """Get fallback roles for infrastructure failure (NOT task escalation).

    Args:
        role: Role whose backend is unavailable.

    Returns:
        List of alternative roles to try, in priority order.
    """
    if isinstance(role, str):
        role = Role.from_string(role)
        if role is None:
            return []
    return list(_FALLBACK_MAP.get(role, []))


def get_tier(role: Role | str) -> Tier:
    """Get the tier for a role.

    Args:
        role: Role enum or string.

    Returns:
        Tier enum member.

    Example:
        >>> get_tier(Role.CODER_PRIMARY)
        Tier.B
        >>> get_tier("worker_math")
        Tier.C
    """
    if isinstance(role, str):
        role = Role.from_string(role)
        if role is None:
            return Tier.C  # Default to worker tier for unknown roles

    return _TIER_MAP.get(role, Tier.C)


def get_escalation_chain(role: Role | str) -> list[Role]:
    """Get the full escalation chain starting from a role.

    Args:
        role: Starting role.

    Returns:
        List of roles in escalation order (including starting role).

    Example:
        >>> get_escalation_chain(Role.WORKER_GENERAL)
        [Role.WORKER_GENERAL, Role.CODER_PRIMARY, Role.ARCHITECT_GENERAL]
    """
    if isinstance(role, str):
        role = Role.from_string(role)
        if role is None:
            return []

    chain = [role]
    current = role
    seen = {current}

    while True:
        next_role = current.escalates_to()
        if next_role is None or next_role in seen:
            break
        chain.append(next_role)
        seen.add(next_role)
        current = next_role

    return chain


# Generic chain names (used by graph node selection and routing)
CHAIN_NAMES = {
    "worker": Role.WORKER_GENERAL,
    "coder": Role.CODER_PRIMARY,
    "architect": Role.ARCHITECT_GENERAL,
    "ingest": Role.INGEST_LONG_CONTEXT,
    "frontdoor": Role.FRONTDOOR,
}


def chain_name_to_role(chain_name: str) -> Role | None:
    """Convert a generic chain name to a specific role.

    Args:
        chain_name: Generic name like "coder" or "architect".

    Returns:
        Specific Role enum member.
    """
    return CHAIN_NAMES.get(chain_name)


def role_to_chain_name(role: Role) -> str:
    """Convert a specific role to its generic chain name.

    Args:
        role: Specific Role enum member.

    Returns:
        Generic chain name like "coder" or "architect".
    """
    for name, r in CHAIN_NAMES.items():
        if role == r:
            return name

    # For non-primary roles, find their chain by escalation
    if role in {Role.CODER_ESCALATION}:
        return "coder"
    if role in {Role.ARCHITECT_CODING}:
        return "architect"
    if role in {Role.WORKER_MATH, Role.WORKER_SUMMARIZE, Role.WORKER_VISION, Role.TOOLRUNNER}:
        return "worker"
    if role in {Role.THINKING_REASONING}:
        return "coder"

    return role.value
