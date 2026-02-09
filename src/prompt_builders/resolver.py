"""Prompt hot-swap resolver: file → fallback constant, with A/B variant support.

Same pattern as _resolve_tools()/_resolve_rules() in builder.py: uncached file
read (~1ms) so edits take effect on the next request without API restart.

Usage:
    from src.prompt_builders.resolver import resolve_prompt

    _MY_FALLBACK = "You are a {role}. Answer: {question}"

    def build_my_prompt(role, question):
        return resolve_prompt(
            "my_prompt", _MY_FALLBACK,
            role=role, question=question,
        )

Variant selection (A/B testing):
    # Per-prompt: PROMPT_VARIANT__my_prompt=v2
    # Global:     PROMPT_VARIANT=v2
    # Creates lookup: orchestration/prompts/my_prompt.v2.md
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "orchestration" / "prompts"


class _SafeDict(dict):
    """Dict that returns '{key}' for missing keys instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _safe_format(template: str, variables: dict[str, str]) -> str:
    """Format a template string, leaving unmatched placeholders intact.

    Handles malformed templates gracefully (no crash on bad syntax).
    """
    try:
        return template.format_map(_SafeDict(variables))
    except (ValueError, IndexError):
        # Malformed format string (e.g. unmatched braces) — return as-is
        _log.debug("Malformed template, returning raw: %.80s", template)
        return template


def _get_variant(name: str) -> str | None:
    """Get variant suffix for a prompt name.

    Priority:
        1. PROMPT_VARIANT__{name} (per-prompt, e.g. PROMPT_VARIANT__architect_investigate=v2)
        2. PROMPT_VARIANT (global, e.g. PROMPT_VARIANT=v2)
        3. None (no variant)
    """
    # Per-prompt override (dots in name replaced with _ for env var compat)
    env_key = f"PROMPT_VARIANT__{name.replace('.', '_').replace('/', '_')}"
    variant = os.environ.get(env_key)
    if variant:
        return variant

    # Global variant
    variant = os.environ.get("PROMPT_VARIANT")
    if variant:
        return variant

    return None


def resolve_prompt(
    name: str,
    fallback: str,
    *,
    variant: str | None = None,
    subdir: str = "",
    **template_vars: str,
) -> str:
    """Resolve a prompt: file (variant) -> file (default) -> fallback constant.

    File read is uncached (~1ms) to enable hot-swap: edit the .md file and the
    next request picks it up without restarting the API.

    Args:
        name: Prompt name (e.g. "architect_investigate", "frontdoor").
        fallback: Fallback prompt string if no file found.
        variant: Explicit variant override (takes priority over env vars).
        subdir: Subdirectory under PROMPT_DIR (e.g. "roles").
        **template_vars: Variables to interpolate into the template.

    Returns:
        Resolved and interpolated prompt string.
    """
    base_dir = PROMPT_DIR / subdir if subdir else PROMPT_DIR

    # Determine variant
    effective_variant = variant or _get_variant(name)

    # Try variant file first
    if effective_variant:
        variant_path = base_dir / f"{name}.{effective_variant}.md"
        try:
            template = variant_path.read_text()
            _log.debug("Loaded variant prompt: %s", variant_path)
            return _safe_format(template, template_vars) if template_vars else template
        except OSError:
            _log.debug("Variant file not found: %s", variant_path)

    # Try default file
    default_path = base_dir / f"{name}.md"
    try:
        template = default_path.read_text()
        _log.debug("Loaded prompt: %s", default_path)
        return _safe_format(template, template_vars) if template_vars else template
    except OSError:
        _log.debug("Prompt file not found: %s, using fallback", default_path)

    # Fallback to constant
    return _safe_format(fallback, template_vars) if template_vars else fallback
