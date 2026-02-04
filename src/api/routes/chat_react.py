"""ReAct tool loop for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: ReAct argument parsing, mode detection, and the
Thought/Action/Observation loop with whitelisted read-only tools.

DEPRECATION NOTICE (2026-02-04):
    This module is deprecated. React mode is superseded by REPL with
    structured_mode=True. The _react_mode_answer function redirects to
    the unified REPL implementation. New code should use REPLEnvironment
    directly with structured_mode=True.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, TYPE_CHECKING

from src.features import features
from src.api.routes.chat_utils import QWEN_STOP

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives


def _parse_react_args(args_str: str) -> dict[str, Any]:
    """Parse ReAct action arguments safely using ast.literal_eval.

    Parses key="value", key=3 format into a dict.
    No eval, no imports — uses ast.literal_eval on individual values.

    Args:
        args_str: The argument string from a ReAct Action line.
            e.g. 'query="quantum computing", max_results=5'

    Returns:
        Dictionary of parsed arguments.
    """
    import ast

    result = {}
    if not args_str or not args_str.strip():
        return result

    # Split on commas, but respect quoted strings
    # Simple state machine: track whether we're inside quotes
    parts = []
    current = []
    in_quotes = False
    quote_char = None
    for ch in args_str:
        if ch in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = ch
            current.append(ch)
        elif ch == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current.append(ch)
        elif ch == "," and not in_quotes:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())

    for part in parts:
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        try:
            result[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # If literal_eval fails, treat as string
            result[key] = val
    return result


def _should_use_react_mode(prompt: str, context: str = "") -> bool:
    """Decide if the prompt should use ReAct tool loop instead of direct answer.

    Triggers on tool-needing keywords that don't require REPL
    (search, calculate, date, lookup). Returns False for large context,
    file ops, format-only prompts.

    Args:
        prompt: The user's prompt.
        context: Optional context text.

    Returns:
        True if ReAct mode should be used.
    """
    # Feature flag check
    if not features().react_mode:
        return False

    # REPL handles large context better
    if context and len(context) > 5000:
        return False

    prompt_lower = prompt.lower()

    # Detect tool-needing keywords
    react_indicators = [
        "search for",
        "look up",
        "find information",
        "what is the current",
        "today's date",
        "what time",
        "calculate",
        "compute",
        "evaluate",
        "search arxiv",
        "search papers",
        "search wikipedia",
        "look up on wikipedia",
        "web search",
        "what year",
        "when did",
        "how many",
    ]

    return any(ind in prompt_lower for ind in react_indicators)


def _react_mode_answer(
    prompt: str,
    context: str,
    primitives: "LLMPrimitives",
    role: str,
    tool_registry: "Any | None" = None,
    max_turns: int = 5,
    tool_whitelist: "frozenset[str] | None" = None,
) -> "tuple[str, int, list[str]]":
    """Execute a ReAct-style tool loop for direct-mode prompts needing tools.

    DEPRECATED: Use REPLEnvironment with structured_mode=True instead.
    This function now redirects to the unified REPL implementation.

    Builds a ReAct prompt, then loops: LLM generates Thought/Action,
    we execute the Action tool and append Observation, until Final Answer
    is found or max_turns reached.

    Args:
        prompt: The user's question.
        context: Optional context text.
        primitives: LLM primitives for inference.
        role: The LLM role to use.
        tool_registry: Optional tool registry for tool execution.
        max_turns: Maximum Thought/Action/Observation cycles.
        tool_whitelist: Optional override for REACT_TOOL_WHITELIST.
            If None, uses the module-level default.

    Returns:
        Tuple of (final_answer, tools_used_count, tools_called_names).
    """
    warnings.warn(
        "_react_mode_answer is deprecated. Use REPLEnvironment(structured_mode=True) instead. "
        "React mode has been unified into REPL for consistency.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.prompt_builders import build_react_prompt, REACT_TOOL_WHITELIST

    active_whitelist = tool_whitelist if tool_whitelist is not None else REACT_TOOL_WHITELIST
    tools_used = 0
    tools_called: list[str] = []

    react_prompt = build_react_prompt(
        prompt=prompt,
        context=context,
        tool_registry=tool_registry,
        max_turns=max_turns,
        tool_whitelist=tool_whitelist,
    )

    conversation = react_prompt

    for turn in range(max_turns):
        # Generate next Thought/Action or Final Answer
        response = primitives.llm_call(
            conversation,
            role=role,
            n_tokens=2048,
            skip_suffix=True,
            stop_sequences=["Observation:", "\n\n\n", QWEN_STOP],
        )
        response = response.strip()

        if not response:
            log.warning(f"ReAct turn {turn}: empty response")
            break

        conversation += "\n" + response

        # Check for Final Answer
        if "Final Answer:" in response:
            # Extract everything after "Final Answer:"
            idx = response.index("Final Answer:")
            answer = response[idx + len("Final Answer:") :].strip()
            log.info(f"ReAct completed in {turn + 1} turns, {tools_used} tools used")
            return answer, tools_used, tools_called

        # Parse Action line
        action_match = None
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("Action:"):
                action_match = line[len("Action:") :].strip()
                break

        if not action_match:
            # No action and no final answer — treat entire response as answer
            log.info(f"ReAct turn {turn}: no Action found, treating as answer")
            # Try to extract useful text (skip Thought: prefix)
            lines = response.split("\n")
            answer_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("Thought:"):
                    stripped = stripped[len("Thought:") :].strip()
                answer_lines.append(stripped)
            return "\n".join(answer_lines).strip(), tools_used, tools_called

        # Parse tool name and args: "tool_name(arg1="val1", arg2=val2)"
        import re as _re

        tool_match = _re.match(r"(\w+)\((.*)\)$", action_match, _re.DOTALL)
        if not tool_match:
            observation = f"[ERROR: Could not parse action: {action_match}]"
        else:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2)

            # Safety: only whitelisted tools
            if tool_name not in active_whitelist:
                observation = f"[ERROR: Tool '{tool_name}' is not available in ReAct mode]"
            elif tool_registry is None:
                observation = f"[ERROR: No tool registry available to execute {tool_name}]"
            else:
                try:
                    args = _parse_react_args(args_str)
                    result = tool_registry.invoke(tool_name, "frontdoor", **args)
                    tools_used += 1
                    tools_called.append(tool_name)
                    # Truncate large results
                    result_str = str(result)
                    if len(result_str) > 2000:
                        result_str = result_str[:2000] + "... [truncated]"
                    observation = result_str
                except Exception as e:
                    observation = f"[ERROR: {tool_name} failed: {e}]"

        conversation += f"\nObservation: {observation}\n"

    # Max turns reached — extract best answer from conversation
    log.warning(f"ReAct reached max turns ({max_turns}), {tools_used} tools used")
    # Look for last Thought that contains useful info
    last_thought = ""
    for line in conversation.split("\n"):
        if line.strip().startswith("Thought:"):
            last_thought = line.strip()[len("Thought:") :].strip()

    if last_thought:
        return f"[ReAct max turns reached]\n{last_thought}", tools_used, tools_called
    return f"[ReAct: Could not determine answer after {max_turns} turns]", tools_used, tools_called
