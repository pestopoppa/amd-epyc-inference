"""TaskIR canonicalization utilities for stable routing and prompt budgets."""

from __future__ import annotations

import json
import re
from typing import Any

from src.constants import TASK_IR_OBJECTIVE_LEN

MAX_CONTEXT_PREVIEW_LEN = 400
MAX_LIST_ITEMS = 8
MAX_LIST_ITEM_LEN = 120
MAX_SNIPPETS = 4
MAX_SNIPPET_LEN = 180
MAX_PLAN_STEPS = 8
MAX_PLAN_ACTION_LEN = 180


def _clean_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _as_string_list(value: Any, *, item_limit: int, max_items: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    cleaned = [_clean_text(v, item_limit) for v in items if str(v).strip()]
    return cleaned[:max_items]


def canonicalize_task_ir(task_ir: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized, bounded TaskIR dictionary."""
    src = task_ir or {}

    out: dict[str, Any] = {
        "task_type": _clean_text(src.get("task_type", "chat"), 40) or "chat",
        "objective": _clean_text(src.get("objective", src.get("prompt", "")), TASK_IR_OBJECTIVE_LEN),
        "priority": _clean_text(src.get("priority", "interactive"), 24) or "interactive",
    }

    task_id = _clean_text(src.get("task_id", ""), 64)
    if task_id:
        out["task_id"] = task_id

    context_preview = _clean_text(
        src.get("context_preview", src.get("context", "")),
        MAX_CONTEXT_PREVIEW_LEN,
    )
    if context_preview:
        out["context_preview"] = context_preview

    constraints = _as_string_list(
        src.get("constraints"),
        item_limit=MAX_LIST_ITEM_LEN,
        max_items=MAX_LIST_ITEMS,
    )
    if constraints:
        out["constraints"] = constraints

    invariants = _as_string_list(
        src.get("invariants"),
        item_limit=MAX_LIST_ITEM_LEN,
        max_items=MAX_LIST_ITEMS,
    )
    if invariants:
        out["invariants"] = invariants

    snippets = _as_string_list(
        src.get("retrieval_snippets"),
        item_limit=MAX_SNIPPET_LEN,
        max_items=MAX_SNIPPETS,
    )
    if snippets:
        out["retrieval_snippets"] = snippets

    plan = src.get("plan")
    if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
        normalized_steps: list[dict[str, Any]] = []
        for raw in plan["steps"][:MAX_PLAN_STEPS]:
            if not isinstance(raw, dict):
                continue
            step = {
                "id": _clean_text(raw.get("id", ""), 24),
                "actor": _clean_text(raw.get("actor", ""), 24),
                "action": _clean_text(raw.get("action", ""), MAX_PLAN_ACTION_LEN),
                "depends_on": _as_string_list(raw.get("depends_on"), item_limit=24, max_items=4),
            }
            if step["action"]:
                normalized_steps.append(step)
        if normalized_steps:
            out["plan"] = {"steps": normalized_steps}

    return out


def canonicalize_task_ir_json(task_ir: dict[str, Any] | None) -> str:
    """Return deterministic JSON for TaskIR cache/prompt stability."""
    return json.dumps(canonicalize_task_ir(task_ir), sort_keys=True, separators=(",", ":"))

