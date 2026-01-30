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

    use_toon_encoding: bool = True
    """Use TOON encoding for structured data. Benchmark: 55.6% token reduction, 41.8% TTFT improvement."""

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

### Routing & Self-Assessment
- `my_role()`: Get your current role, tier, capabilities, and what you can delegate to
- `route_advice(task_description)`: Get MemRL routing recommendation with Q-values
- `delegate(prompt, target_role, reason)`: Delegate to a specific role with outcome tracking
- `escalate(reason, target_role=None)`: Request escalation (up-chain or to specific role)
- `recall(query)`: Search episodic memory — returns Q-values and past routing outcomes

### LLM Delegation (low-level, no tracking)
- `llm_call(prompt, role='worker')`: Raw sub-LM call
- `llm_batch(prompts, role='worker')`: Parallel raw sub-LM calls

### Long Context Exploration
- `context_len()`: Return character count of context
- `chunk_context(n_chunks=4, overlap=200)`: Split context into N chunks with metadata
- `summarize_chunks(task, n_chunks=4, role='worker_general')`: Chunk + parallel worker summaries

### Tool Invocation
- `TOOL(tool_name, **kwargs)`: Invoke a registered tool, returns raw Python object
- `CALL(tool_name, **kwargs)`: Invoke a registered tool, returns JSON string (simpler)
  Example: `result = CALL("search_arxiv", query="transformers"); data = json.loads(result)`
- `list_tools()`: List available tools for your role

### Completion
- `FINAL(answer)`: Signal completion with the final answer (REQUIRED)"""

# Default rules for Root LM
DEFAULT_ROOT_LM_RULES = """## CRITICAL
1. **NO IMPORTS** - import/from are BLOCKED. The `json` module is pre-loaded, just use `json.loads()` directly.
2. **USE list_dir()** for files - NOT os.listdir or pathlib
3. **ALWAYS call FINAL(answer)** when you have your answer — this is REQUIRED to complete the task.
   Do NOT keep calling tools after you have enough information.

## Examples
List files: `result = list_dir('/path'); FINAL(result)`
Read file: `text = peek(1000, file_path='/path'); FINAL(text)`
Summarize PDF: `doc = json.loads(ocr_document('/path.pdf')); summary = llm_call(f"Summarize: {doc['full_text'][:6000]}", role='worker'); FINAL(summary)`

## Routing (OPTIONAL — only for complex multi-model tasks)
4. Simple tasks: just answer directly — do NOT call my_role() or route_advice() first
5. Only call my_role() if genuinely unsure about your capabilities
6. Only call route_advice() before delegating complex subtasks to specialists
7. Use `delegate()` over `llm_call()` when making a conscious routing choice
8. Call `escalate(reason)` if the task exceeds your tier — don't guess

## Other Rules
9. NEVER send full context to llm_call - use peek() or grep() first
10. Output only valid Python code - no markdown, no explanations"""


# ── ReAct Tool Loop Constants ──────────────────────────────────────────────

# Read-only tools safe for ReAct mode (no shell, no filesystem writes)
REACT_TOOL_WHITELIST = frozenset({
    "web_search",
    "search_arxiv",
    "search_papers",
    "search_wikipedia",
    "get_wikipedia_article",
    "search_books",
    "calculate",
    "python_eval",
    "get_current_date",
    "get_current_time",
    "json_query",
    "fetch_wikipedia",
})

# ReAct format instructions
REACT_FORMAT = """You have access to the following tools:
{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: reason about what to do next
Action: tool_name(arg1="value1", arg2="value2")
Observation: the result of the action
... (repeat Thought/Action/Observation as needed, up to {max_turns} times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Important rules:
- Always start with a Thought before an Action
- Action arguments use keyword=value syntax (strings in quotes, numbers bare)
- After each Observation, decide if you have enough info for a Final Answer
- If no tools are needed, skip directly to Final Answer
- Be concise in your Final Answer — answer the question directly"""


def build_react_prompt(
    prompt: str,
    context: str = "",
    tool_registry: "Any | None" = None,
    max_turns: int = 5,
) -> str:
    """Build a ReAct-style prompt with tool descriptions.

    Args:
        prompt: The user's question.
        context: Optional context text.
        tool_registry: Optional tool registry for dynamic tool descriptions.
        max_turns: Maximum number of Thought/Action/Observation cycles.

    Returns:
        Formatted ReAct prompt string.
    """
    # Build tool descriptions from whitelist
    tool_descriptions = []
    if tool_registry is not None:
        for tool_info in tool_registry.list_tools():
            name = tool_info.get("name", "")
            if name in REACT_TOOL_WHITELIST:
                desc = tool_info.get("description", "No description")
                params = tool_info.get("parameters", {})
                param_strs = []
                for pname, pinfo in params.items():
                    ptype = pinfo.get("type", "string")
                    required = pinfo.get("required", False)
                    req_mark = " (required)" if required else ""
                    param_strs.append(f"  {pname}: {ptype}{req_mark}")
                param_block = "\n".join(param_strs) if param_strs else "  (no parameters)"
                tool_descriptions.append(f"- {name}: {desc}\n{param_block}")
    else:
        # Static fallback descriptions for common tools
        tool_descriptions = [
            "- calculate(expression=\"...\"): Evaluate a math expression",
            "- get_current_date(): Get today's date",
            "- get_current_time(): Get current time",
            "- web_search(query=\"...\"): Search the web",
            "- search_arxiv(query=\"...\", max_results=5): Search arXiv papers",
            "- search_wikipedia(query=\"...\"): Search Wikipedia articles",
            "- get_wikipedia_article(title=\"...\"): Get full Wikipedia article",
            "- python_eval(code=\"...\"): Evaluate Python expression safely",
            "- json_query(data=\"...\", query=\"...\"): Query JSON data with JMESPath",
        ]

    tool_desc_str = "\n".join(tool_descriptions)
    react_prompt = REACT_FORMAT.format(
        tool_descriptions=tool_desc_str,
        max_turns=max_turns,
    )

    if context:
        return f"{react_prompt}\n\nContext:\n{context}\n\nQuestion: {prompt}"
    return f"{react_prompt}\n\nQuestion: {prompt}"


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
        routing_context: str = "",
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

        if routing_context:
            context_parts.append(f"## Routing Intelligence\n{routing_context}")

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

            # NOTE: TOON encoding was tested for grep hits but found to be LESS
            # efficient than Markdown due to pattern repetition. Keeping Markdown.
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
            "Write a refined, accurate executive summary based on the draft and excerpts above.",
            "",
            "IMPORTANT OUTPUT FORMAT:",
            "- Write the summary directly - no thinking, analysis, or meta-commentary",
            "- Do not explain your reasoning or discuss terminology",
            "- Do not include phrases like 'here is' or 'based on the draft'",
            "- Start directly with the summary content",
            "- Keep the same structure as the draft but improve accuracy and completeness",
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
    routing_context: str = "",
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
        routing_context=routing_context,
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


def build_routing_context(
    role: str,
    hybrid_router: Any,
    task_description: str,
    max_chars: int = 300,
) -> str:
    """Build compact routing context for injection into Root LM prompt.

    Called on turn 0 only to give the model initial routing intelligence
    from MemRL. Kept very compact to minimize token overhead (~75 tokens).

    Args:
        role: Current model's role.
        hybrid_router: HybridRouter instance (may be None).
        task_description: The user's task.
        max_chars: Maximum characters for routing context.

    Returns:
        Compact routing context string, or "" if unavailable.
    """
    if hybrid_router is None:
        return ""

    from src.roles import get_tier

    try:
        task_ir = {
            "task_type": "chat",
            "objective": task_description[:200],
        }
        results = hybrid_router.retriever.retrieve_for_routing(task_ir)

        tier = get_tier(role)

        if not results:
            return f"Role: {role} (Tier {tier.value})\nNo similar past tasks found."[:max_chars]

        # Build structured routing data for TOON encoding
        similar = []
        for r in results[:3]:
            ctx = r.memory.context or {}
            similar.append({
                "role": ctx.get("role", r.memory.action),
                "Q": round(r.q_value, 2),
                "outcome": r.memory.outcome or "?",
            })

        best = hybrid_router.retriever.get_best_action(results)
        routing_data = {
            "self": {"role": role, "tier": tier.value},
            "similar": similar,
        }
        if best:
            routing_data["suggested"] = {"role": best[0], "conf": round(best[1], 2)}

        # TOON encoding: ~50% reduction on uniform similar-tasks array
        try:
            from src.services.toon_encoder import is_available, encode
            if is_available() and len(similar) >= 2:
                return encode(routing_data)[:max_chars]
        except Exception:
            pass

        # Fallback: compact text format
        lines = [f"Role: {role} (Tier {tier.value})"]
        for s in similar:
            lines.append(f"  Similar → {s['role']}: Q={s['Q']:.2f} ({s['outcome']})")
        if best:
            lines.append(f"Suggested: {best[0]} (conf={best[1]:.2f})")
        return "\n".join(lines)[:max_chars]

    except Exception:
        return ""


def build_long_context_exploration_prompt(
    original_prompt: str,
    context_chars: int,
    state: str = "",
) -> str:
    """Build a prompt that instructs the model to explore large context via REPL tools.

    Instead of dumping the full context into the LLM's input window, this prompt
    tells the model to use REPL exploration tools (peek, grep, chunk_context,
    summarize_chunks) to process it piece by piece.

    Args:
        original_prompt: The user's original task/question.
        context_chars: Character count of the full context.
        state: Current REPL state.

    Returns:
        Prompt string for the Root LM.
    """
    est_tokens = context_chars // 4
    n_chunks = max(2, min(8, est_tokens // 4000))  # ~4K tokens per chunk

    # Detect search/needle tasks
    search_keywords = ["find", "search", "locate", "identify", "extract", "detect", "where"]
    is_search = any(kw in original_prompt.lower().split() for kw in search_keywords)

    if is_search:
        first_step = f"""# Step 1: Search for relevant items
matches = grep(r"key|secret|password|token|credential|api_key")
for m in matches:
    print(f"Line {{m['line']}}: {{m['text']}}")"""
    else:
        first_step = f"""# Step 1: Inspect document structure
header = peek(2000)
print(header)"""

    return f"""You are an orchestrator processing a large document ({context_chars:,} chars, ~{est_tokens:,} tokens).

## Task
{original_prompt}

## Strategy
The full document is stored in the `context` variable.
These functions are ALREADY AVAILABLE — call them directly:

```python
{first_step}

# Step 2: Search for specific terms
matches = grep(r"pattern_here")  # returns list of {{"line": int, "text": str, "position": int}}

# Step 3: Parallel analysis (dispatches to workers)
summaries = summarize_chunks(
    task="{original_prompt[:80]}",
    n_chunks={n_chunks}
)  # returns list of {{"index": int, "summary": str}}

# Step 4: Synthesize and return
FINAL("your answer here")
```

**DO NOT** define your own functions. `peek`, `grep`, `summarize_chunks`, `FINAL` are built-ins.
**DO NOT** try to read files from disk — the context is already loaded in memory.

{f'## Current State{chr(10)}{state}' if state else ''}

## Rules
- NEVER import anything — all tools are pre-loaded
- NEVER send full context to llm_call() (too large)
- Use grep() to find specific needles, summarize_chunks() for broad analysis
- Workers return section summaries — you synthesize the final answer
- Output only valid Python code — no markdown, no explanations

Code:"""


# ─── Quality Review Prompts ──────────────────────────────────────────────────


def build_review_verdict_prompt(
    question: str,
    answer: str,
    context_digest: str = "",
    worker_digests: list[dict] | None = None,
) -> str:
    """Build architect verdict prompt — forces hyper-concise output.

    Uses TOON encoding for worker_digests (uniform array of section summaries)
    to minimize tokens sent to architect. TOON achieves 40-65% reduction on
    structured arrays vs JSON.

    Args:
        question: Original user question (truncated to 300 chars).
        answer: The answer to review (truncated to 1500 chars).
        context_digest: Optional compact text digest for context-dependent claims.
        worker_digests: Optional list of worker digest dicts for TOON encoding.

    Returns:
        Prompt string for architect verdict.
    """
    digest_section = ""
    if worker_digests:
        # TOON-encode structured worker digests (uniform array → 40-65% savings)
        try:
            from src.services.toon_encoder import encode, is_available
            if is_available():
                digest_section = f"\nEvidence:\n{encode(worker_digests)}\n"
            else:
                import json
                digest_section = f"\nEvidence:\n{json.dumps(worker_digests)}\n"
        except Exception:
            import json
            digest_section = f"\nEvidence:\n{json.dumps(worker_digests)}\n"
    elif context_digest:
        digest_section = f"\nContext: {context_digest[:800]}\n"

    return f"""Judge this answer. Respond with ONLY one line:
- "OK" if correct and complete
- "WRONG: <what to fix>" if incorrect (max 30 words)
{digest_section}
Q: {question[:300]}
A: {answer[:1500]}

Verdict:"""


def build_revision_prompt(
    question: str, original: str, corrections: str
) -> str:
    """Build fast model revision prompt — expands architect corrections.

    Args:
        question: Original user question.
        original: The original answer to revise.
        corrections: Architect's correction notes.

    Returns:
        Prompt string for the revision model.
    """
    return f"""Rewrite this answer applying the corrections below.
Keep the same style and depth. Only change what the corrections require.

Question: {question[:300]}

Original answer: {original[:1500]}

Corrections: {corrections}

Revised answer:"""


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


# ── Plan Review Prompts ───────────────────────────────────────────────────


def build_plan_review_prompt(
    objective: str,
    task_type: str,
    plan_steps: list[dict[str, Any]],
) -> str:
    """Build architect plan review prompt — forces hyper-concise JSON output.

    The architect reviews the frontdoor's tentative plan and can confirm,
    reroute steps, reorder, drop, or add missing steps.

    Args:
        objective: Task objective (truncated to 200 chars).
        task_type: Task type (e.g., "code", "chat").
        plan_steps: List of plan step dicts with id, actor/role, action, deps.

    Returns:
        Prompt string for architect plan review (~100-120 tokens input).
    """
    # Format steps compactly: S1:coder:Implement handler->output.py
    step_lines = []
    for step in plan_steps[:8]:  # Cap at 8 steps
        step_id = step.get("id", "S?")
        actor = step.get("actor", step.get("role", "worker"))
        action = step.get("action", "")[:50]
        outputs = step.get("outputs", step.get("out", []))
        out_str = ",".join(str(o) for o in outputs[:2]) if outputs else ""
        deps = step.get("deps", step.get("inputs", []))
        dep_str = f"({','.join(str(d) for d in deps[:2])})" if deps else ""

        line = f"{step_id}:{actor}:{action}"
        if out_str:
            line += f"->{out_str}"
        if dep_str:
            line += dep_str
        step_lines.append(line)

    steps_block = "\n".join(step_lines)

    return f"""Review plan. Reply JSON ONLY:
{{"d":"ok|reorder|drop|add|reroute","s":0.0-1.0,"f":"<15 words","p":[]}}

d=decision, s=confidence, f=feedback, p=patches (optional)
Patch format: {{"step":"S1","op":"reroute|drop|add|reorder","v":"new_value"}}

Task: {objective[:200]}
Type: {task_type}
Plan:
{steps_block}

Verdict:"""


# ── Output Formalizer ──────────────────────────────────────────────────────

# Patterns that indicate format constraints in a prompt
_FORMAT_CONSTRAINT_PATTERNS = [
    (r"exactly\s+(\d+)\s+words?", "exactly {0} words"),
    (r"in\s+(\d+)\s+words?\s+or\s+(fewer|less)", "at most {0} words"),
    (r"no\s+more\s+than\s+(\d+)\s+words?", "at most {0} words"),
    (r"(?:in|as)\s+JSON(?:\s+format)?", "JSON format"),
    (r"(?:as\s+a?\s*)?numbered\s+list", "numbered list"),
    (r"(?:as\s+a?\s*)?bullet(?:ed)?\s+list", "bullet list"),
    (r"comma[- ]separated", "comma-separated list"),
    (r"(?:all\s+)?(?:in\s+)?(?:upper|UPPER)\s*case", "uppercase"),
    (r"(?:all\s+)?(?:in\s+)?(?:lower)\s*case", "lowercase"),
    (r"one\s+(?:single\s+)?sentence", "one sentence"),
    (r"single\s+paragraph", "single paragraph"),
    (r"(?:as\s+a?\s*)?(?:markdown\s+)?table", "table format"),
    (r"(?:as\s+a?\s*)?(?:YAML|yaml)\s+(?:format)?", "YAML format"),
    (r"(?:as\s+a?\s*)?(?:XML|xml)\s+(?:format)?", "XML format"),
]


def detect_format_constraints(prompt: str) -> list[str]:
    """Detect format constraints in a prompt.

    Args:
        prompt: The user's prompt text.

    Returns:
        List of detected format constraint descriptions (empty if none).
    """
    constraints = []
    for pattern, template in _FORMAT_CONSTRAINT_PATTERNS:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            # Substitute captured groups into template
            try:
                desc = template.format(*match.groups())
            except (IndexError, KeyError):
                desc = template
            constraints.append(desc)
    return constraints


def build_formalizer_prompt(answer: str, prompt: str, format_spec: str) -> str:
    """Build a prompt for the output formalizer.

    Args:
        answer: The original answer to reformat.
        prompt: The original user prompt (for context).
        format_spec: Description of the format constraint to satisfy.

    Returns:
        Formatted formalizer prompt string.
    """
    return (
        f"Reformat the following answer to strictly satisfy this format constraint: {format_spec}\n\n"
        f"Original question: {prompt[:500]}\n\n"
        f"Original answer:\n{answer}\n\n"
        f"Output ONLY the reformatted answer. Do not add explanations or preamble."
    )
