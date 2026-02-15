"""Prompt data types and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PromptStyle(Enum):
    """Prompt formatting styles."""

    MINIMAL = "minimal"
    """Minimal formatting, just the essentials."""

    STRUCTURED = "structured"
    """Structured with headers and sections."""

    DETAILED = "detailed"
    """Full detail with examples and context."""


@dataclass
class PromptConfig:
    """Configuration for prompt building."""

    style: PromptStyle = PromptStyle.STRUCTURED
    """Default prompt style."""

    max_context_chars: int = 4000
    """Maximum characters for context sections."""

    max_output_preview: int = 500
    """Maximum characters for output previews."""

    max_error_preview: int = 500
    """Maximum characters for error previews."""

    use_toon_encoding: bool = True
    """Use TOON encoding for structured data. Benchmark: 55.6% token reduction, 41.8% TTFT improvement."""

    include_examples: bool = False
    """Include usage examples in prompts."""

    tools_file: str | None = None
    """Path to external tools prompt file. Overrides style-based selection when set."""

    rules_file: str | None = None
    """Path to external rules prompt file. Overrides DEFAULT_ROOT_LM_RULES when set."""


@dataclass
class RootLMPrompt:
    """Structured prompt for the Root LM.

    Contains all sections that make up a Root LM prompt.
    """

    system: str = ""
    """System instructions."""

    tools: str = ""
    """Available tools description."""

    rules: str = ""
    """Rules the LLM must follow."""

    state: str = ""
    """Current REPL state."""

    context: str = ""
    """Additional context (error, output, etc.)."""

    reference_code: str = ""
    """Corpus-retrieved code snippets for prompt-lookup acceleration."""

    task: str = ""
    """The user's task."""

    instruction: str = ""
    """Final instruction."""

    def to_string(self) -> str:
        """Convert to a single prompt string."""
        parts = []
        if self.system:
            parts.append(self.system)
        if self.tools:
            parts.extend(["", "## Available Tools", self.tools])
        if self.rules:
            parts.extend(["", "## Rules", self.rules])
        if self.state:
            parts.extend(["", "## Current State", self.state])
        if self.context:
            parts.extend(["", self.context])
        if self.reference_code:
            parts.extend(["", "## Reference Code", self.reference_code])
        if self.task:
            parts.extend(["", "## Task", self.task])
        if self.instruction:
            parts.extend(["", "## Your Code", self.instruction])
        return "\n".join(parts)


@dataclass
class EscalationPrompt:
    """Structured prompt for escalation scenarios.

    Contains all sections for an escalation prompt.
    """

    header: str = ""
    """Escalation header with source role."""

    failure_info: str = ""
    """Information about the failure."""

    error_details: str = ""
    """Detailed error information."""

    state: str = ""
    """Current state."""

    task: str = ""
    """Original task."""

    instructions: str = ""
    """Instructions for the escalated role."""

    def to_string(self) -> str:
        """Convert to a single prompt string."""
        parts = []
        if self.header:
            parts.append(self.header)
        if self.failure_info:
            parts.extend(["", self.failure_info])
        if self.error_details:
            parts.extend(["", "## Error Details", self.error_details])
        if self.state:
            parts.extend(["", "## Current State", self.state])
        if self.task:
            parts.extend(["", "## Original Task", self.task])
        if self.instructions:
            parts.extend(["", "## Instructions", self.instructions])
        return "\n".join(parts)


@dataclass
class StepPrompt:
    """Structured prompt for step execution.

    Contains all sections for a step execution prompt.
    """

    action: str = ""
    """The action to perform."""

    inputs: str = ""
    """Input context."""

    outputs: str = ""
    """Expected outputs."""

    constraints: str = ""
    """Any constraints or requirements."""

    def to_string(self) -> str:
        """Convert to a single prompt string."""
        parts = []
        if self.action:
            parts.append(f"Task: {self.action}")
        if self.inputs:
            parts.append(f"\nInputs:{self.inputs}")
        if self.outputs:
            parts.append(f"\nExpected outputs: {self.outputs}")
        if self.constraints:
            parts.append(f"\nConstraints: {self.constraints}")
        return "\n".join(parts)
