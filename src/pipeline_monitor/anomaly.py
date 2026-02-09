"""Anomaly signal detection for pipeline diagnostics.

Pure functions, no side effects. Each signal detector takes answer text
and/or metadata and returns a boolean. The score function aggregates
weighted signals into a [0, 1] anomaly score.
"""

from __future__ import annotations

import re


# ── Signal weights ──

SIGNAL_WEIGHTS: dict[str, float] = {
    "repetition_loop": 1.0,
    "comment_only": 0.5,
    "template_echo": 1.0,
    "self_doubt_loop": 0.5,
    "format_violation": 1.0,
    "think_tag_leak": 0.5,
    "near_empty": 1.0,
    "excessive_tokens": 0.5,
    "delegation_format_error": 1.0,
    "self_escalation": 0.5,
    "vision_blindness": 1.0,
    "silent_execution": 0.5,
}

# ── Restart phrases for self-doubt detection ──

_RESTART_PHRASES = re.compile(
    r"\b(?:Actually|Wait|Let me reconsider|Let me rethink|"
    r"On second thought|I was wrong|Hmm|No,? wait)\b",
    re.IGNORECASE,
)

# ── Delegation format patterns ──

_DELEGATION_PREFIX_RE = re.compile(r"^I\|", re.MULTILINE)
_DELEGATION_BRIEF_RE = re.compile(r"^I\|brief:", re.MULTILINE)
_DELEGATION_TO_RE = re.compile(r"\bto:", re.IGNORECASE)

# ── Architect answer prefixes ──

_DIRECT_ANSWER_RE = re.compile(r"^D\|", re.MULTILINE)
_INVESTIGATE_RE = re.compile(r"^I\|", re.MULTILINE)


# ── Individual signal detectors ──


def detect_repetition_loop(answer: str, threshold: float = 0.4) -> bool:
    """Trigram unique ratio below threshold indicates degeneration loop."""
    words = answer.split()
    if len(words) < 9:
        return False
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return False
    unique_ratio = len(set(trigrams)) / len(trigrams)
    return unique_ratio < threshold


def detect_comment_only(answer: str) -> bool:
    """All code lines are comments or blank — no executable content."""
    lines = answer.split("\n")
    has_content = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return False
        if stripped:
            has_content = True
    return has_content


def detect_template_echo(answer: str) -> bool:
    """Both D| and I| prefixes appear — model echoed template instead of choosing."""
    has_direct = bool(_DIRECT_ANSWER_RE.search(answer))
    has_investigate = bool(_INVESTIGATE_RE.search(answer))
    return has_direct and has_investigate


def detect_self_doubt_loop(answer: str, threshold: int = 3) -> bool:
    """More than N restart phrases indicate indecisive reasoning loop."""
    matches = _RESTART_PHRASES.findall(answer)
    return len(matches) > threshold


def detect_format_violation(
    answer: str, role: str, mode: str, role_history: list[str] | None = None,
) -> bool:
    """Architect role in delegated mode but no D| or I| prefix in answer.

    Skips when delegation occurred (role_history >1 entry) or when the
    answer is short enough to be a post-extraction result — the D|/I|
    prefix is stripped by _parse_architect_decision before we see it.
    """
    if "architect" not in role:
        return False
    if mode not in ("delegated", "direct"):
        return False
    # If delegation actually occurred (role_history has >1 role), the D|/I|
    # prefix was already parsed and stripped by _parse_architect_decision.
    if role_history and len(role_history) > 1:
        return False
    # Short answers are post-extraction (e.g. "A", "42") — prefix already stripped.
    if len(answer) < 50:
        return False
    has_direct = bool(_DIRECT_ANSWER_RE.search(answer))
    has_investigate = bool(_INVESTIGATE_RE.search(answer))
    return not has_direct and not has_investigate


def detect_think_tag_leak(answer: str) -> bool:
    """<think> tag leaked into final answer (should be stripped by backend)."""
    return "<think>" in answer


def detect_near_empty(
    answer: str, error: str | None, scoring_method: str = "", threshold: int = 5,
) -> bool:
    """Answer has fewer than N tokens and no error — model produced almost nothing."""
    if error:
        return False
    if scoring_method == "multiple_choice":
        # Single-letter answers are valid for MCQ
        return False
    tokens = answer.split()
    return len(tokens) < threshold


def detect_excessive_tokens(
    tokens_generated: int,
    scoring_method: str,
    threshold: int = 2000,
) -> bool:
    """Over N tokens for a multiple-choice question — model is rambling."""
    if scoring_method != "multiple_choice":
        return False
    return tokens_generated > threshold


def detect_delegation_format_error(answer: str) -> bool:
    """I| prefix present but missing required brief: or to: fields."""
    if not _DELEGATION_PREFIX_RE.search(answer):
        return False
    # Must have brief: after I|
    has_brief = bool(_DELEGATION_BRIEF_RE.search(answer))
    # Valid delegation needs at least brief:
    return not has_brief


def detect_self_escalation(role_history: list[str]) -> bool:
    """Consecutive duplicate roles in history — model escalated to itself."""
    if len(role_history) < 2:
        return False
    for i in range(1, len(role_history)):
        if role_history[i] == role_history[i - 1]:
            return True
    return False


def detect_vision_blindness(
    answer: str, role: str, threshold: int = 10,
) -> bool:
    """Vision role but answer has fewer than N tokens — model didn't see the image."""
    if "vision" not in role:
        return False
    tokens = answer.split()
    return len(tokens) < threshold


def detect_silent_execution(
    answer: str, tools_used: int, error: str | None,
) -> bool:
    """Tools were used, no error, but answer is empty — silent execution."""
    if tools_used <= 0:
        return False
    if error:
        return False
    return not answer.strip()


# ── Aggregation ──


def compute_anomaly_signals(
    answer: str,
    role: str = "",
    mode: str = "",
    error: str | None = None,
    tokens_generated: int = 0,
    scoring_method: str = "",
    role_history: list[str] | None = None,
    tools_used: int = 0,
) -> dict[str, bool]:
    """Run all 12 anomaly detectors, return dict of signal_name → bool."""
    return {
        "repetition_loop": detect_repetition_loop(answer),
        "comment_only": detect_comment_only(answer),
        "template_echo": detect_template_echo(answer),
        "self_doubt_loop": detect_self_doubt_loop(answer),
        "format_violation": detect_format_violation(answer, role, mode, role_history),
        "think_tag_leak": detect_think_tag_leak(answer),
        "near_empty": detect_near_empty(answer, error, scoring_method),
        "excessive_tokens": detect_excessive_tokens(
            tokens_generated, scoring_method,
        ),
        "delegation_format_error": detect_delegation_format_error(answer),
        "self_escalation": detect_self_escalation(role_history or []),
        "vision_blindness": detect_vision_blindness(answer, role),
        "silent_execution": detect_silent_execution(answer, tools_used, error),
    }


def anomaly_score(signals: dict[str, bool]) -> float:
    """Compute anomaly score from signal dict. Returns max weight of triggered signals, [0, 1]."""
    triggered = [
        SIGNAL_WEIGHTS.get(name, 0.0)
        for name, active in signals.items()
        if active
    ]
    if not triggered:
        return 0.0
    return max(triggered)
