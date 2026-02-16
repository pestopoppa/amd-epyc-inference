"""Tests for TaskIR canonicalization budgets and deterministic ordering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task_ir import canonicalize_task_ir, canonicalize_task_ir_json


def test_canonicalize_task_ir_applies_budgets():
    task_ir = {
        "task_type": "chat",
        "objective": "  solve   this   problem " * 50,
        "priority": "interactive",
        "context": "x " * 1000,
        "constraints": [f"c{i}" for i in range(30)],
        "invariants": [f"i{i}" for i in range(30)],
        "retrieval_snippets": [f"s{i}" for i in range(20)],
        "plan": {"steps": [{"id": str(i), "actor": "worker", "action": "a" * 500} for i in range(20)]},
    }
    out = canonicalize_task_ir(task_ir)

    assert len(out["objective"]) <= 200
    assert len(out["context_preview"]) <= 400
    assert len(out["constraints"]) <= 8
    assert len(out["invariants"]) <= 8
    assert len(out["retrieval_snippets"]) <= 4
    assert len(out["plan"]["steps"]) <= 8
    assert all(len(step["action"]) <= 180 for step in out["plan"]["steps"])


def test_canonicalize_task_ir_json_is_deterministic():
    a = {
        "objective": "Do thing",
        "task_type": "chat",
        "constraints": ["b", "a"],
        "priority": "interactive",
    }
    b = {
        "priority": "interactive",
        "constraints": ["b", "a"],
        "task_type": "chat",
        "objective": "Do thing",
    }
    ja = canonicalize_task_ir_json(a)
    jb = canonicalize_task_ir_json(b)
    assert ja == jb

    parsed = json.loads(ja)
    assert parsed["task_type"] == "chat"
    assert parsed["objective"] == "Do thing"
