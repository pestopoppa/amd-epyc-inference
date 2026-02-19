"""Output parsing classifiers: tool-output stripping and loop detection.

Extracted from src/api/routes/chat_utils.py during Phase 1 classifier
refactoring. Pure text-processing logic with no MemRL coupling.
"""

from __future__ import annotations

import re


def strip_tool_outputs(text: str, tool_outputs: list[str]) -> str:
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


def truncate_looped_answer(answer: str, prompt: str) -> str:
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
