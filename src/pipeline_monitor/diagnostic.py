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
    tap_offset_bytes: int = 0,
    tap_length_bytes: int = 0,
    repl_tap_offset_bytes: int = 0,
    repl_tap_length_bytes: int = 0,
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
        "tools_used": tools_used,
        "tools_called": tools_called,
        "anomaly_signals": signals,
        "anomaly_score": score,
        "tap_offset_bytes": tap_offset_bytes,
        "tap_length_bytes": tap_length_bytes,
        "repl_tap_offset_bytes": repl_tap_offset_bytes,
        "repl_tap_length_bytes": repl_tap_length_bytes,
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
