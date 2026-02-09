"""Format constraint detection and output formalizer prompts."""

from __future__ import annotations

import re

from src.prompt_builders.resolver import resolve_prompt

_FORMALIZER_FALLBACK = (
    "Reformat the following answer to strictly satisfy this format constraint: {format_spec}\n\n"
    "Original question: {prompt}\n\n"
    "Original answer:\n{answer}\n\n"
    "Output ONLY the reformatted answer. Do not add explanations or preamble."
)


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
    return resolve_prompt(
        "formalizer", _FORMALIZER_FALLBACK,
        format_spec=format_spec,
        prompt=prompt[:500],
        answer=answer,
    )
