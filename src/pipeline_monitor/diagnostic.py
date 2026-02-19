"""Diagnostic record builder and JSONL writer for pipeline monitoring.

Builds structured diagnostic dicts from RoleResult + metadata,
writes them to an append-only JSONL file with fcntl locking.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline_monitor.anomaly import compute_anomaly_signals, anomaly_score

DIAGNOSTICS_PATH = Path("/mnt/raid0/llm/claude/logs/seeding_diagnostics.jsonl")


def build_diagnostic(
    question_id: str,
    suite: str,
    config: str,
    role: str,
    mode: str,
    passed: bool,
    answer: str,
    expected: str,
    scoring_method: str,
    error: str | None,
    error_type: str,
    tokens_generated: int,
    elapsed_s: float,
    role_history: list[str],
    delegation_events: list[dict],
    tools_used: int,
    tools_called: list[str],
    delegation_diagnostics: dict[str, Any] | None = None,
    tap_offset_bytes: int = 0,
    tap_length_bytes: int = 0,
    repl_tap_offset_bytes: int = 0,
    repl_tap_length_bytes: int = 0,
    # Orchestrator intelligence tunable fields
    cost_dimensions: dict[str, float] | None = None,
    think_harder_attempted: bool = False,
    think_harder_succeeded: bool | None = None,
    cheap_first_attempted: bool = False,
    cheap_first_passed: bool | None = None,
    grammar_enforced: bool = False,
    parallel_tools_used: bool = False,
    cache_affinity_bonus: float = 0.0,
    # SkillBank retrieval data (None = SkillBank not loaded, 0 = loaded but no hits)
    skills_retrieved: int | None = None,
    skill_types: list[str] | None = None,
    skill_context_tokens: int = 0,
    # Context window management (C1/C3) and budget tracking (R1)
    budget_diagnostics: dict[str, Any] | None = None,
    tool_results_cleared: int = 0,
    compaction_triggered: bool = False,
    compaction_tokens_saved: int = 0,
    think_harder_expected_roi: float = 0.0,
) -> dict[str, Any]:
    """Build a diagnostic record from evaluation results.

    Returns a flat dict suitable for JSONL serialization.
    """
    signals = compute_anomaly_signals(
        answer=answer,
        role=role,
        mode=mode,
        error=error,
        tokens_generated=tokens_generated,
        scoring_method=scoring_method,
        role_history=role_history,
        tools_used=tools_used,
        delegation_events=delegation_events,
        skills_retrieved=skills_retrieved if skills_retrieved is not None else 0,
        skill_coverage=(skills_retrieved is not None),
        passed=passed,
    )
    score = anomaly_score(signals)

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question_id": question_id,
        "suite": suite,
        "config": config,
        "role": role,
        "mode": mode,
        "passed": passed,
        "answer": answer,
        "expected": expected,
        "scoring_method": scoring_method,
        "error": error,
        "error_type": error_type,
        "tokens_generated": tokens_generated,
        "elapsed_s": elapsed_s,
        "role_history": role_history,
        "delegation_events": delegation_events,
        "delegation_diagnostics": delegation_diagnostics or {},
        "tools_used": tools_used,
        "tools_called": tools_called,
        "anomaly_signals": signals,
        "anomaly_score": score,
        "tap_offset_bytes": tap_offset_bytes,
        "tap_length_bytes": tap_length_bytes,
        "repl_tap_offset_bytes": repl_tap_offset_bytes,
        "repl_tap_length_bytes": repl_tap_length_bytes,
        # Orchestrator intelligence tunables
        "cost_dimensions": cost_dimensions or {},
        "think_harder_attempted": think_harder_attempted,
        "think_harder_succeeded": think_harder_succeeded,
        "cheap_first_attempted": cheap_first_attempted,
        "cheap_first_passed": cheap_first_passed,
        "grammar_enforced": grammar_enforced,
        "parallel_tools_used": parallel_tools_used,
        "cache_affinity_bonus": cache_affinity_bonus,
        # SkillBank retrieval
        "skill_retrieval": {
            "skills_retrieved": skills_retrieved,
            "skill_types": skill_types or [],
            "skill_context_tokens": skill_context_tokens,
        } if skills_retrieved else {},
        # Context window management (C1/C3) and budget tracking (R1)
        "budget_diagnostics": budget_diagnostics or {},
        "tool_results_cleared": tool_results_cleared,
        "compaction_triggered": compaction_triggered,
        "compaction_tokens_saved": compaction_tokens_saved,
        "think_harder_expected_roi": think_harder_expected_roi,
    }


def append_diagnostic(
    diag: dict[str, Any],
    path: Path | None = None,
) -> None:
    """Append a diagnostic record to the JSONL file with fcntl locking."""
    target = path or DIAGNOSTICS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(diag, default=str) + "\n"
    with open(target, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
