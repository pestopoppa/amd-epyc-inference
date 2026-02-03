#!/usr/bin/env python3
"""Parsing configuration for orchestrator model outputs.

This module defines which parsing strategy (Instructor, GBNF, Regex, None)
should be used for each role based on model capability and schema complexity.

Design principles:
- GBNF for smaller models (<32B) emitting schemas (100% compliance guaranteed)
- Instructor (Pydantic) for larger models (>32B) with retries
- Regex for simple pattern extraction (toolrunner)
- None for plain text outputs (workers, draft models)
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ParsingMode(str, Enum):
    """Parsing strategy for model outputs."""

    INSTRUCTOR = "instructor"  # Pydantic + retries (for high-quality models)
    GBNF = "gbnf"  # llama.cpp grammar constraint (guaranteed compliance)
    REGEX = "regex"  # Simple pattern extraction
    NONE = "none"  # Plain text, no parsing


# Role → Parsing mode mapping
PARSING_CONFIG: dict[str, ParsingMode] = {
    # Tier A - High quality models, use Instructor
    "frontdoor": ParsingMode.INSTRUCTOR,
    # Tier B - Mixed based on model size and schema complexity
    "formalizer": ParsingMode.GBNF,  # 8B model, ensure compliance
    "coder_primary": ParsingMode.INSTRUCTOR,
    "coder_escalation": ParsingMode.INSTRUCTOR,
    "architect_general": ParsingMode.INSTRUCTOR,
    "architect_coding": ParsingMode.INSTRUCTOR,
    "ingest_long_context": ParsingMode.NONE,  # Returns summaries
    "thinking_reasoning": ParsingMode.NONE,  # Returns reasoning chains
    # Tier C - Workers, mostly plain text
    "worker_general": ParsingMode.NONE,
    "worker_math": ParsingMode.NONE,
    "worker_vision": ParsingMode.NONE,
    "worker_summarize": ParsingMode.NONE,
    "toolrunner": ParsingMode.REGEX,  # Extract tool names/status
    # Tier D - Draft models, no parsing
    "draft_coder": ParsingMode.NONE,
    "draft_general": ParsingMode.NONE,
}


def get_parsing_mode(role: str) -> ParsingMode:
    """Get the parsing mode for a role.

    Args:
        role: Role name (e.g., "frontdoor", "coder_primary").

    Returns:
        ParsingMode for the role, defaults to NONE if unknown.
    """
    # Handle wildcard patterns
    if role.startswith("draft_"):
        return ParsingMode.NONE
    if role.startswith("worker_"):
        return ParsingMode.NONE

    return PARSING_CONFIG.get(role, ParsingMode.NONE)


# --- Pydantic schemas for Instructor ---


class ToolCall(BaseModel):
    """Single tool invocation."""

    tool: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    args: dict[str, str]


class TaskIR(BaseModel):
    """Task specification from frontdoor.

    Emitted by the frontdoor model to specify what task to perform,
    which agents to use, and what gates must pass.
    """

    task_id: str
    task_type: Literal["code", "doc", "ingest", "manage", "chat"]
    priority: Literal["interactive", "batch"]
    objective: str
    agents: list[dict]  # Agent specifications
    gates: list[str]  # Required gates (e.g., ["lint", "typecheck"])
    tool_sequence: list[ToolCall] | None = None
    definition_of_done: list[str] | None = None
    escalation: dict | None = None


class FormalizationIR(BaseModel):
    """Formal specification from formalizer.

    Converts natural language requirements into formal constraints,
    edge cases, and acceptance criteria.
    """

    problem_type: Literal["algorithm", "proof", "optimization", "validation", "tool_orchestration"]
    variables: list[dict]  # {name: str, type: str, constraints: list[str]}
    constraints: list[str]
    objective: str | None = None
    edge_cases: list[dict]  # {input: str, expected: str}
    acceptance_criteria: list[str]
    tool_sequence: list[ToolCall] | None = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class ArchitectureIR(BaseModel):
    """Architecture specification from architect models.

    Describes system design, component relationships, and invariants.
    """

    components: list[dict]  # {name: str, responsibility: str, interfaces: list}
    relationships: list[dict]  # {from: str, to: str, type: str}
    invariants: list[str]  # System-wide invariants
    constraints: list[str]  # Design constraints
    rationale: str  # Why this architecture
    alternatives_considered: list[dict] | None = None


# --- Toolrunner regex patterns ---

TOOLRUNNER_PATTERNS: dict[str, str] = {
    "tool_status": r"^(SUCCESS|FAILURE|TIMEOUT):\s*(\w+)",
    "output_summary": r"Output:\s*(.+?)(?:\n|$)",
    "next_action": r"Next:\s*(RETRY|ESCALATE|COMPLETE|ABORT)",
    "error_message": r"Error:\s*(.+?)(?:\n|$)",
}


def parse_toolrunner_output(output: str) -> dict[str, str | None]:
    """Parse toolrunner output using regex patterns.

    Args:
        output: Raw toolrunner output text.

    Returns:
        Dict with matched fields (status, summary, next_action, error).
    """
    import re

    result: dict[str, str | None] = {
        "status": None,
        "tool_name": None,
        "summary": None,
        "next_action": None,
        "error": None,
    }

    # Extract status and tool name
    status_match = re.search(TOOLRUNNER_PATTERNS["tool_status"], output)
    if status_match:
        result["status"] = status_match.group(1)
        result["tool_name"] = status_match.group(2)

    # Extract output summary
    summary_match = re.search(TOOLRUNNER_PATTERNS["output_summary"], output)
    if summary_match:
        result["summary"] = summary_match.group(1)

    # Extract next action
    action_match = re.search(TOOLRUNNER_PATTERNS["next_action"], output)
    if action_match:
        result["next_action"] = action_match.group(1)

    # Extract error message
    error_match = re.search(TOOLRUNNER_PATTERNS["error_message"], output)
    if error_match:
        result["error"] = error_match.group(1)

    return result
