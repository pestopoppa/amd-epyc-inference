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
    "repl_no_tools": 0.5,
    "slow_delegation": 0.5,
    "misrouted_to_coder": 1.0,
    "function_repr_leak": 1.0,
    "status_phrase_final": 1.0,
    "wasteful_delegation": 0.5,
    "repl_max_turns": 1.0,
    "escalation_cycle": 1.0,
    "assistant_help_request": 1.0,
    "prose_only_code_task": 1.0,
    # SkillBank signals
    "skill_mismatch": 0.5,
    "no_skills_available": 0.3,
    # Distillation latency signal
    "distill_batch_latency": 0.5,
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
    if scoring_method != "code_execution":
        # Only code_execution should produce substantial output;
        # all other scoring methods (MCQ, exact_match, substring, f1,
        # programmatic) can legitimately have short answers.
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
    answer: str, role: str, mode: str = "", threshold: int = 10,
) -> bool:
    """Vision role but answer has fewer than N tokens — model didn't see the image."""
    if "vision" not in role:
        return False
    # REPL mode produces concise FINAL() answers — a single-word answer
    # like "Cancer" is valid, not blind.
    if mode == "repl" and answer.strip():
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


def detect_repl_no_tools(mode: str, tools_used: int) -> bool:
    """REPL mode but no tools were used — model ignored its tool capability."""
    if mode != "repl":
        return False
    return tools_used <= 0


def detect_slow_delegation(
    delegation_events: list[dict],
    threshold_ms: int = 120_000,
) -> bool:
    """Any delegation hop took longer than threshold — bottleneck in chain."""
    for ev in delegation_events:
        if ev.get("elapsed_ms", 0) > threshold_ms:
            return True
    return False


_FUNCTION_REPR_RE = re.compile(r"<function \w+ at 0x[0-9a-fA-F]+>")


def detect_function_repr_leak(answer: str) -> bool:
    """Answer contains a Python function repr — callable passed to FINAL instead of called."""
    return bool(_FUNCTION_REPR_RE.search(answer))


_STATUS_PHRASES = {
    "code execution complete", "execution complete",
    "done", "complete", "completed", "implemented",
    "implementation complete", "finished", "success",
    "task complete", "task completed", "code complete",
    "answer", "your answer", "your_answer",
    "your answer here", "your_answer_here",
    "result", "the answer", "the result",
    "your_computed_value", "your computed value",
    "code", "explanation of code or reasoning",
    "code execution complete. check output",
}


def detect_status_phrase_final(answer: str) -> bool:
    """Answer is a status phrase rather than an actual answer (e.g. 'Done', 'Complete')."""
    stripped = answer.strip().rstrip(".!").lower()
    if not stripped:
        return False
    return stripped in _STATUS_PHRASES


def detect_misrouted_to_coder(
    scoring_method: str,
    role: str,
    delegation_events: list[dict],
) -> bool:
    """Architect delegated to coder for a non-code question (factual/MCQ/QA)."""
    if scoring_method in ("code_execution", "substring"):
        return False
    if "architect" not in role:
        return False
    for ev in delegation_events:
        if ev.get("to_role", "") == "coder_escalation":
            return True
    return False


def detect_wasteful_delegation(
    answer: str,
    role: str,
    delegation_events: list[dict],
    scoring_method: str = "",
) -> bool:
    """Delegation occurred but final answer is short/numeric — specialist added no value.

    The architect already computed the answer (visible in its <think> block or
    brief text), then delegated to a specialist who round-tripped it back
    unchanged.  Wastes 50-300s of specialist time.

    Triggers when:
    - Architect role delegated at least once
    - Final answer is short (<30 chars) or purely numeric
    - Scoring method is NOT code_execution (code tasks legitimately delegate)
    """
    if scoring_method == "code_execution":
        return False
    if "architect" not in role:
        return False
    if not delegation_events:
        return False
    stripped = answer.strip()
    if not stripped:
        return False
    # Short answer after delegation = likely wasteful
    is_short = len(stripped) < 30
    is_numeric = bool(re.fullmatch(r"-?[\d,]+\.?\d*\s*%?", stripped))
    return is_short or is_numeric


def detect_escalation_cycle(role_history: list[str]) -> bool:
    """A→B→A→B or A→B→C→A→B→C bouncing pattern in role history."""
    if len(role_history) < 4:
        return False
    # Period-2: ABAB
    if role_history[-1] == role_history[-3] and role_history[-2] == role_history[-4]:
        return True
    # Period-3: ABCABC
    if len(role_history) >= 6:
        if (
            role_history[-1] == role_history[-4]
            and role_history[-2] == role_history[-5]
            and role_history[-3] == role_history[-6]
        ):
            return True
    return False


def detect_repl_max_turns(answer: str, mode: str) -> bool:
    """REPL exhausted all turns without calling FINAL() — model never submitted an answer."""
    if mode != "repl":
        return False
    return "[Max turns" in answer and "without FINAL()" in answer


_HELP_REQUEST_RE = re.compile(
    r"\b(?:Can you (?:help|assist|clarify|provide)|"
    r"Could you (?:help|assist|clarify|provide)|"
    r"Would you (?:like me to|help)|"
    r"Please (?:provide|share|clarify)|"
    r"I(?:'d| would) need (?:more|additional|the)|"
    r"help me (?:fix|solve|debug|understand))\b",
    re.IGNORECASE,
)


def detect_assistant_help_request(answer: str) -> bool:
    """Model asks the user for help instead of answering — role reversal failure."""
    matches = _HELP_REQUEST_RE.findall(answer)
    return len(matches) >= 2


_CODE_INDICATOR_RE = re.compile(
    r"(?:def |class |import |from .+ import |for .+ in |while |"
    r"if __name__|print\(|return |CALL\(|FINAL\(|```)",
)


def detect_prose_only_code_task(
    answer: str, scoring_method: str,
) -> bool:
    """Code execution task answered with pure prose — no executable code at all."""
    if scoring_method != "code_execution":
        return False
    if not answer.strip():
        return False
    return not bool(_CODE_INDICATOR_RE.search(answer))


def detect_skill_mismatch(passed: bool, skills_retrieved: int) -> bool:
    """Skills were retrieved but the task still failed — skill quality issue."""
    return not passed and skills_retrieved > 0


def detect_no_skills_available(skills_retrieved: int, skill_coverage: bool) -> bool:
    """SkillBank is enabled but returned nothing for this task type — coverage gap."""
    return skill_coverage and skills_retrieved == 0


def detect_distill_batch_latency(
    batch_latencies: list[dict],
    threshold_ms: int = 5_000,
) -> bool:
    """Any distillation teacher batch exceeded threshold — model transition bottleneck.

    Fires on per-batch latency spikes (default 5s) in the distillation pipeline,
    much more granular than slow_delegation's 120s threshold.
    """
    for bl in batch_latencies:
        if bl.get("elapsed_ms", 0) > threshold_ms:
            return True
    return False


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
    delegation_events: list[dict] | None = None,
    # SkillBank signals (optional, backward-compatible)
    skills_retrieved: int = 0,
    skill_coverage: bool = False,
    passed: bool = True,
    # Distillation latency (optional, backward-compatible)
    batch_latencies: list[dict] | None = None,
) -> dict[str, bool]:
    """Run all anomaly detectors, return dict of signal_name → bool."""
    deleg = delegation_events or []
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
        "vision_blindness": detect_vision_blindness(answer, role, mode),
        "silent_execution": detect_silent_execution(answer, tools_used, error),
        "repl_no_tools": detect_repl_no_tools(mode, tools_used),
        "slow_delegation": detect_slow_delegation(deleg),
        "function_repr_leak": detect_function_repr_leak(answer),
        "status_phrase_final": detect_status_phrase_final(answer),
        "misrouted_to_coder": detect_misrouted_to_coder(
            scoring_method, role, deleg,
        ),
        "wasteful_delegation": detect_wasteful_delegation(
            answer, role, deleg, scoring_method,
        ),
        "repl_max_turns": detect_repl_max_turns(answer, mode),
        "escalation_cycle": detect_escalation_cycle(role_history or []),
        "assistant_help_request": detect_assistant_help_request(answer),
        "prose_only_code_task": detect_prose_only_code_task(
            answer, scoring_method,
        ),
        # SkillBank signals
        "skill_mismatch": detect_skill_mismatch(passed, skills_retrieved),
        "no_skills_available": detect_no_skills_available(
            skills_retrieved, skill_coverage,
        ),
        # Distillation latency
        "distill_batch_latency": detect_distill_batch_latency(
            batch_latencies or [],
        ),
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
