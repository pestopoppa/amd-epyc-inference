"""Unified prompt building for the orchestration system.

This module provides all prompt construction functionality in one place:
- Root LM prompts for the orchestrator
- Escalation prompts for tier upgrades
- Step execution prompts for workers
- Role-specific system prompts

Usage:
    from src.prompt_builders import PromptBuilder, RootLMPrompt, EscalationPrompt

    builder = PromptBuilder()
    prompt = builder.build_root_lm_prompt(state="...", task="...", turn=1)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.roles import Role, get_tier

if TYPE_CHECKING:
    from src.context_manager import ContextManager
    # Support both legacy and new escalation types
    from src.failure_router import ErrorCategory, FailureContext, RoutingDecision
    from src.escalation import EscalationContext, EscalationDecision


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

    include_examples: bool = False
    """Include usage examples in prompts."""


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


# Default tool descriptions for Root LM
DEFAULT_ROOT_LM_TOOLS = """### Context & Files
- `context`: str - The full input context (large, do not send to LLM)
- `artifacts`: dict - Store intermediate results
- `peek(n, file_path=None)`: Return first n characters of context or file
- `grep(pattern, file_path=None)`: Search context or file with regex
- `list_dir(path)`: List directory contents, returns JSON with files/dirs
- `file_info(path)`: Get file metadata (size, type, modified date)

### Document Processing (all return JSON strings - use json.loads())
- `ocr_document(path)`: Extract text from PDF, returns JSON with full_text, pages, figures
- `analyze_figure(image_path, prompt)`: Analyze image with vision model
- `extract_figure(pdf_path, page, bbox)`: Crop figure from PDF, returns image path

### Web & Shell
- `web_fetch(url)`: Fetch web content
- `run_shell(cmd)`: Run sandboxed shell command (ls, grep, git status only)

### LLM Delegation
- `llm_call(prompt, role='worker')`: Call a sub-LM for a task
- `llm_batch(prompts, role='worker')`: Parallel sub-LM calls
- `escalate(reason)`: Request escalation to higher-tier model
- `recall(query)`: Search episodic memory for similar past tasks

### Completion
- `FINAL(answer)`: Signal completion with the final answer (REQUIRED)"""

# Default rules for Root LM
DEFAULT_ROOT_LM_RULES = """## CRITICAL
1. **NO IMPORTS** - import/from are BLOCKED. The `json` module is pre-loaded, just use `json.loads()` directly.
2. **USE list_dir()** for files - NOT os.listdir or pathlib
3. **ALWAYS call FINAL(answer)** to complete the task

## Examples
List files: `result = list_dir('/path'); FINAL(result)`
Read file: `text = peek(1000, file_path='/path'); FINAL(text)`
Summarize PDF: `doc = json.loads(ocr_document('/path.pdf')); summary = llm_call(f"Summarize: {doc['full_text'][:6000]}", role='worker'); FINAL(summary)`

## Other Rules
4. NEVER send full context to llm_call - use peek() or grep() first
5. Output only valid Python code - no markdown, no explanations"""


class PromptBuilder:
    """Builder for all prompt types in the orchestration system.

    This class provides a unified interface for building prompts for:
    - Root LM (orchestrator) prompts
    - Escalation prompts
    - Step execution prompts
    - Role-specific system prompts

    Example:
        builder = PromptBuilder()

        # Build Root LM prompt
        prompt = builder.build_root_lm_prompt(
            state="artifacts = {}",
            original_prompt="Summarize the document",
            turn=0,
        )

        # Build escalation prompt
        esc_prompt = builder.build_escalation_prompt(
            original_prompt="...",
            state="...",
            failure_context=context,
            decision=decision,
        )
    """

    def __init__(self, config: PromptConfig | None = None):
        """Initialize the prompt builder.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or PromptConfig()

    def build_root_lm_prompt(
        self,
        state: str,
        original_prompt: str,
        last_output: str = "",
        last_error: str = "",
        turn: int = 0,
        *,
        as_structured: bool = False,
    ) -> str | RootLMPrompt:
        """Build the prompt for the Root LM (frontdoor).

        The Root LM generates Python code that executes in a sandboxed REPL.
        It has access to the context and can call sub-LMs for complex tasks.

        Args:
            state: Current REPL state from repl.get_state()
            original_prompt: The user's original prompt
            last_output: Output from the last code execution
            last_error: Error from the last code execution (if any)
            turn: Current turn number (0-indexed)
            as_structured: Return RootLMPrompt instead of string

        Returns:
            Prompt string or RootLMPrompt if as_structured=True
        """
        prompt = RootLMPrompt(
            system="You are an orchestrator that generates Python code to solve tasks.",
            tools=DEFAULT_ROOT_LM_TOOLS,
            rules=DEFAULT_ROOT_LM_RULES,
            state=f"Turn {turn + 1}\n{state}",
            task=original_prompt,
            instruction="Write Python code to complete the task. Output only the code:",
        )

        # Build context section based on last output/error
        context_parts = []
        if last_error:
            error_preview = last_error[:self.config.max_error_preview]
            if len(last_error) > self.config.max_error_preview:
                error_preview += "..."
            context_parts.extend([
                "## Last Error",
                "```",
                error_preview,
                "```",
                "Fix the error and try again.",
            ])
        elif last_output:
            output_preview = last_output[:self.config.max_output_preview]
            if len(last_output) > self.config.max_output_preview:
                output_preview += "..."
            context_parts.extend([
                "## Last Output",
                "```",
                output_preview,
                "```",
            ])

        if context_parts:
            prompt.context = "\n".join(context_parts)

        if as_structured:
            return prompt
        return prompt.to_string()

    def build_escalation_prompt(
        self,
        original_prompt: str,
        state: str,
        failure_context: "FailureContext | EscalationContext",
        decision: "RoutingDecision | EscalationDecision",
        *,
        as_structured: bool = False,
    ) -> str | EscalationPrompt:
        """Build a prompt for an escalated role.

        Includes failure context to help the higher-tier model understand
        what went wrong and what was tried.

        Supports both legacy (FailureContext/RoutingDecision) and new
        (EscalationContext/EscalationDecision) types for migration.

        Args:
            original_prompt: The user's original prompt.
            state: Current REPL state.
            failure_context: Context about the failure (legacy or new type).
            decision: The routing decision with escalation reason (legacy or new type).
            as_structured: Return EscalationPrompt instead of string

        Returns:
            Prompt string or EscalationPrompt if as_structured=True
        """
        # Handle both legacy (role) and new (current_role) attribute names
        role = getattr(failure_context, "current_role", None) or getattr(failure_context, "role", "unknown")

        prompt = EscalationPrompt(
            header=f"# Escalation from {role}",
            failure_info=(
                f"The {role} failed after "
                f"{failure_context.failure_count} attempts.\n"
                f"Reason: {decision.reason}"
            ),
            state=state,
            task=original_prompt,
            instructions=(
                "Fix the issue and complete the task. "
                "You have more capability than the previous role.\n"
                "Output Python code that will execute in the REPL environment."
            ),
        )

        # Build error details
        error_parts = [f"Category: {failure_context.error_category}"]
        if failure_context.gate_name:
            error_parts.append(f"Gate: {failure_context.gate_name}")
        if failure_context.error_message:
            error_preview = failure_context.error_message[:self.config.max_error_preview]
            error_parts.extend([
                "",
                "Error message:",
                "```",
                error_preview,
                "```",
            ])
        prompt.error_details = "\n".join(error_parts)

        if as_structured:
            return prompt
        return prompt.to_string()

    def build_step_prompt(
        self,
        action: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        context: ContextManager | None = None,
        constraints: str = "",
        *,
        as_structured: bool = False,
    ) -> str | StepPrompt:
        """Build a prompt for step execution.

        Uses ContextManager for rich context formatting if provided.

        Args:
            action: The action/task to perform.
            inputs: Input keys to include from context.
            outputs: Expected output keys.
            context: Optional ContextManager for input formatting.
            constraints: Additional constraints or requirements.
            as_structured: Return StepPrompt instead of string

        Returns:
            Prompt string or StepPrompt if as_structured=True
        """
        prompt = StepPrompt(action=action, constraints=constraints)

        # Build inputs section
        if inputs:
            if context:
                context_str = context.build_prompt_context(
                    input_keys=inputs,
                    max_chars=self.config.max_context_chars,
                )
                if context_str:
                    prompt.inputs = context_str
                else:
                    missing = [k for k in inputs if not context.has(k)]
                    if missing:
                        prompt.inputs = f"\n(Not available: {', '.join(missing)})"
            else:
                prompt.inputs = f"\n{', '.join(inputs)}"

        # Build outputs section
        if outputs:
            prompt.outputs = ", ".join(outputs)

        if as_structured:
            return prompt
        return prompt.to_string()

    def build_stage2_review_prompt(
        self,
        draft_summary: str,
        grep_hits: list[dict[str, Any]],
        figures: list[dict[str, Any]],
        original_task: str = "",
    ) -> str:
        """Build a prompt for Stage 2 of two-stage summarization.

        Stage 2 receives the frontdoor's draft summary along with key excerpts
        (grep hits) and figure descriptions. The large model refines the summary
        without reading the full document.

        Args:
            draft_summary: The draft summary from Stage 1 (frontdoor).
            grep_hits: List of grep hit records from REPL's get_grep_history().
                       Each record contains pattern, source, match_count, hits.
            figures: List of figure descriptions from document processing.
            original_task: Optional original summarization task.

        Returns:
            Prompt string for Stage 2 model.
        """
        parts = [
            "# Document Summary Review",
            "",
            "You are reviewing a draft summary produced by a fast model. Your task is to:",
            "1. Verify accuracy against the provided excerpts",
            "2. Add important details that may be missing",
            "3. Improve clarity and completeness",
            "4. Correct any inaccuracies",
            "",
        ]

        # Add original task if provided
        if original_task:
            parts.extend([
                "## Original Request",
                original_task,
                "",
            ])

        # Add draft summary
        parts.extend([
            "## Draft Summary (to review and refine)",
            "```",
            draft_summary[:8000],  # Cap draft length
            "```",
            "",
        ])

        # Add grep excerpts (key source material)
        if grep_hits:
            parts.extend([
                "## Key Excerpts from Source Document",
                "These are search results the draft model found relevant:",
                "",
            ])

            total_chars = 0
            max_excerpt_chars = 6000  # Cap total excerpt size

            for hit_record in grep_hits:
                if total_chars >= max_excerpt_chars:
                    parts.append(f"[... additional excerpts truncated at {max_excerpt_chars} chars]")
                    break

                pattern = hit_record.get("pattern", "")
                hits = hit_record.get("hits", [])

                if hits:
                    parts.append(f"### Search: `{pattern}`")
                    for hit in hits[:5]:  # Cap at 5 hits per pattern
                        context = hit.get("context", hit.get("match", ""))
                        line_num = hit.get("line_num", "?")
                        excerpt = f"Line {line_num}: {context[:500]}"
                        total_chars += len(excerpt)
                        parts.append(excerpt)
                        if total_chars >= max_excerpt_chars:
                            break
                    parts.append("")

        # Add figure descriptions
        if figures:
            parts.extend([
                "## Figures in Document",
                "",
            ])

            for i, fig in enumerate(figures[:10], 1):  # Cap at 10 figures
                desc = fig.get("description", fig.get("text", "No description"))
                page = fig.get("page", "?")
                parts.append(f"**Figure {i}** (Page {page}): {desc[:500]}")
            parts.append("")

        # Final instruction
        parts.extend([
            "## Your Task",
            "Write a refined, accurate summary based on the draft and excerpts above.",
            "Preserve the draft's structure but improve accuracy and completeness.",
            "Write the summary directly, without meta-commentary about your review process.",
        ])

        return "\n".join(parts)

    def get_system_prompt(self, role: Role) -> str:
        """Get the system prompt for a specific role.

        Args:
            role: The role to get the system prompt for.

        Returns:
            System prompt string.
        """
        tier = get_tier(role)
        role_name = role.value

        # Base system prompts by role
        system_prompts = {
            Role.FRONTDOOR: (
                "You are an orchestrator AI. Your job is to understand user requests, "
                "break them into tasks, and generate Python code that executes in a "
                "sandboxed REPL environment. You can call sub-LMs for complex subtasks."
            ),
            Role.CODER_PRIMARY: (
                "You are a senior software engineer. Write clean, efficient code that "
                "follows best practices. Focus on correctness and maintainability."
            ),
            Role.CODER_ARCHITECT: (
                "You are a software architect. Design robust systems and APIs. "
                "Consider scalability, security, and long-term maintainability."
            ),
            Role.ARCHITECT_GENERAL: (
                "You are a system architect. You handle complex problems that require "
                "deep reasoning and coordination across multiple domains."
            ),
            Role.ARCHITECT_CODING: (
                "You are a principal engineer with expertise in complex codebases. "
                "You solve the hardest coding problems and design critical systems."
            ),
            Role.INGEST_LONG_CONTEXT: (
                "You are a document analysis specialist. Process and synthesize "
                "information from long documents while maintaining accuracy."
            ),
            Role.WORKER_GENERAL: (
                "You are a general-purpose assistant. Complete tasks efficiently "
                "and accurately. Focus on the specific task at hand."
            ),
            Role.WORKER_MATH: (
                "You are a mathematical reasoning specialist. Solve math problems "
                "step by step, showing your work clearly."
            ),
            Role.WORKER_VISION: (
                "You are a vision-language specialist. Analyze images and provide "
                "accurate descriptions and interpretations."
            ),
        }

        # Return role-specific prompt or a generic one
        return system_prompts.get(
            role,
            f"You are a {tier.name.lower()}-tier {role_name} assistant. "
            f"Complete tasks efficiently and accurately.",
        )


# Module-level functions for backwards compatibility
_default_builder: PromptBuilder | None = None


def _get_builder() -> PromptBuilder:
    """Get the default prompt builder."""
    global _default_builder
    if _default_builder is None:
        _default_builder = PromptBuilder()
    return _default_builder


def build_root_lm_prompt(
    state: str,
    original_prompt: str,
    last_output: str = "",
    last_error: str = "",
    turn: int = 0,
) -> str:
    """Build the prompt for the Root LM (frontdoor).

    Module-level function for backwards compatibility.
    See PromptBuilder.build_root_lm_prompt() for full documentation.
    """
    result = _get_builder().build_root_lm_prompt(
        state=state,
        original_prompt=original_prompt,
        last_output=last_output,
        last_error=last_error,
        turn=turn,
    )
    return result if isinstance(result, str) else result.to_string()


def build_escalation_prompt(
    original_prompt: str,
    state: str,
    failure_context: "FailureContext | EscalationContext",
    decision: "RoutingDecision | EscalationDecision",
) -> str:
    """Build a prompt for an escalated role.

    Module-level function for backwards compatibility.
    Supports both legacy and new escalation types.
    See PromptBuilder.build_escalation_prompt() for full documentation.
    """
    result = _get_builder().build_escalation_prompt(
        original_prompt=original_prompt,
        state=state,
        failure_context=failure_context,
        decision=decision,
    )
    return result if isinstance(result, str) else result.to_string()


def build_step_prompt(
    action: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    context: ContextManager | None = None,
) -> str:
    """Build a prompt for step execution.

    Module-level function for backwards compatibility.
    See PromptBuilder.build_step_prompt() for full documentation.
    """
    result = _get_builder().build_step_prompt(
        action=action,
        inputs=inputs,
        outputs=outputs,
        context=context,
    )
    return result if isinstance(result, str) else result.to_string()


def build_stage2_review_prompt(
    draft_summary: str,
    grep_hits: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    original_task: str = "",
) -> str:
    """Build a prompt for Stage 2 of two-stage summarization.

    Module-level function for backwards compatibility.
    See PromptBuilder.build_stage2_review_prompt() for full documentation.
    """
    return _get_builder().build_stage2_review_prompt(
        draft_summary=draft_summary,
        grep_hits=grep_hits,
        figures=figures,
        original_task=original_task,
    )


# Code extraction utilities (moved from orchestrator.py)


def _strip_import_lines(code: str) -> str:
    """Strip import/from lines since all needed modules are pre-loaded in REPL globals.

    Models frequently generate 'import json' or 'import os' even when told not to.
    Safe modules (json) are already available; unsafe modules would be blocked anyway.
    """
    lines = code.split("\n")
    filtered = [
        line for line in lines
        if not line.strip().startswith("import ")
        and not line.strip().startswith("from ")
    ]
    return "\n".join(filtered).strip()


def extract_code_from_response(response: str) -> str:
    """Extract Python code from an LLM response.

    Handles responses that may be wrapped in markdown code blocks
    or contain explanatory text. Also strips import lines since
    all needed modules are pre-loaded in the REPL globals.
    """
    response = response.strip()

    # Remove trailing backticks that aren't properly paired
    # (model sometimes outputs code followed by ``` without opening)
    if response.endswith("```"):
        # Check if there's a matching opening
        if response.count("```") % 2 == 1:
            response = response[:-3].rstrip()

    # Try to extract from markdown code block
    code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        return _strip_import_lines(matches[0].strip())

    # If no code block, try to find code-like content
    lines = response.split("\n")
    code_lines = []
    in_code = False

    # Include REPL tool functions as code starters
    code_starters = [
        "import ", "from ", "def ", "class ", "if ", "for ", "while ",
        "try:", "except", "with ", "return ", "print(", "FINAL(",
        "artifacts[", "result =", "answer =", "output =",
        # REPL tools
        "peek(", "grep(", "list_dir(", "file_info(", "ocr_document(",
        "analyze_figure(", "extract_figure(", "web_fetch(", "run_shell(",
        "recall(", "escalate(", "llm_call(", "llm_batch(",
    ]

    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(kw) for kw in code_starters):
            in_code = True

        if in_code or stripped.startswith("#") or "=" in line or "()" in line:
            code_lines.append(line)

    if code_lines:
        # Strip common leading whitespace from all lines
        code = "\n".join(code_lines)
        # Dedent the code to remove consistent leading whitespace
        import textwrap
        code = textwrap.dedent(code).strip()
        # Strip import lines - modules like json are pre-loaded in REPL globals
        code = _strip_import_lines(code)
        return code

    # Fallback: return the whole response, dedented
    import textwrap
    code = textwrap.dedent(response).strip()
    code = _strip_import_lines(code)
    return code


def auto_wrap_final(code: str) -> str:
    """Auto-wrap code in FINAL() if it looks like a final answer.

    This is a deterministic wrapper that detects when the model has generated
    complete code but didn't wrap it in FINAL(). This allows models to generate
    code naturally while still signaling completion to the orchestrator.

    Args:
        code: Extracted code from the model's response.

    Returns:
        Code wrapped in FINAL() if it's a final answer, otherwise unchanged.
    """
    # Already has FINAL - no change
    if "FINAL(" in code:
        return code

    # Has exploration/continuation functions - not a final answer
    exploration_patterns = [
        "peek(",      # Exploring context
        "grep(",      # Searching context
        "llm_call(",  # Delegating to sub-LM
        "llm_batch(", # Batch delegation
        "artifacts[", # Storing intermediate results
    ]
    for pattern in exploration_patterns:
        if pattern in code:
            return code

    # Get non-empty, non-comment lines
    lines = [
        line.strip() for line in code.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return code

    # Code starting with def/class is likely a complete answer
    if lines[0].startswith(("def ", "class ")):
        # Escape triple quotes in the code to avoid breaking the wrapper
        escaped_code = code.replace("'''", r"\'\'\'")
        return f"FINAL('''{escaped_code}''')"

    # Single expression/value is likely a final answer
    # Exclude control flow and imports
    if len(lines) == 1:
        first_line = lines[0]
        non_final_patterns = [
            "import ", "from ", "for ", "while ", "if ", "try:", "with ",
        ]
        if not any(first_line.startswith(p) for p in non_final_patterns):
            return f"FINAL({first_line})"

    return code


# Error classification utilities (moved from orchestrator.py)
def classify_error(error_message: str, gate_name: str = "") -> ErrorCategory:
    """Classify an error message into an ErrorCategory.

    Args:
        error_message: The error message to classify.
        gate_name: Optional gate name if error came from a gate.

    Returns:
        ErrorCategory for the error.
    """
    # Import here to avoid circular imports
    from src.failure_router import ErrorCategory

    error_lower = error_message.lower()

    # Schema/format errors (from gates or parsing)
    if gate_name in ("schema", "format", "lint", "mdformat", "shfmt"):
        return ErrorCategory.FORMAT
    if "schema" in error_lower or "validation" in error_lower:
        return ErrorCategory.SCHEMA
    if "format" in error_lower or "style" in error_lower:
        return ErrorCategory.FORMAT

    # Code errors (syntax, type, import)
    code_keywords = [
        "syntaxerror", "indentationerror", "typeerror", "nameerror",
        "importerror", "modulenotfound", "attributeerror"
    ]
    if any(kw in error_lower for kw in code_keywords):
        return ErrorCategory.CODE

    # Logic errors (test failures, assertions)
    logic_keywords = ["assertionerror", "test failed", "expected", "actual"]
    if any(kw in error_lower for kw in logic_keywords):
        return ErrorCategory.LOGIC

    # Timeout errors
    if "timeout" in error_lower or "timed out" in error_lower:
        return ErrorCategory.TIMEOUT

    # Early abort (from generation monitor)
    if "early abort" in error_lower or "high entropy" in error_lower:
        return ErrorCategory.EARLY_ABORT

    return ErrorCategory.UNKNOWN
