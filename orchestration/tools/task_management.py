"""In-request task tracking and budget management tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolBudget:
    """Per-task tool call budget buckets."""

    read_calls: int = 5
    write_calls: int = 3
    exec_calls: int = 2
    llm_calls: int = 1
    overrides_used: int = 0

    def can_use(self, category: str) -> bool:
        if category == "read":
            return self.read_calls > 0
        if category == "write":
            return self.write_calls > 0
        if category == "exec":
            return self.exec_calls > 0
        if category == "llm":
            return self.llm_calls > 0
        return True

    def consume(self, category: str) -> bool:
        if not self.can_use(category):
            return False
        if category == "read":
            self.read_calls -= 1
        elif category == "write":
            self.write_calls -= 1
        elif category == "exec":
            self.exec_calls -= 1
        elif category == "llm":
            self.llm_calls -= 1
        return True

    def override(self, category: str, amount: int = 3, max_overrides: int = 2) -> bool:
        if self.overrides_used >= max_overrides:
            return False
        if category == "read":
            self.read_calls += amount
        elif category == "write":
            self.write_calls += amount
        elif category == "exec":
            self.exec_calls += amount
        elif category == "llm":
            self.llm_calls += amount
        else:
            return False
        self.overrides_used += 1
        return True

    def remaining_summary(self) -> str:
        return (
            f"read={self.read_calls}, write={self.write_calls}, "
            f"exec={self.exec_calls}, llm={self.llm_calls}"
        )


def budget_for_task_type(task_type: str) -> ToolBudget:
    """Default budget map by task type."""
    normalized = (task_type or "chat").strip().lower()
    if normalized in {"bugfix", "bug_fix", "bug"}:
        return ToolBudget(read_calls=5, write_calls=3, exec_calls=3, llm_calls=1)
    if normalized in {"feature", "implementation"}:
        return ToolBudget(read_calls=8, write_calls=5, exec_calls=2, llm_calls=1)
    if normalized in {"refactor", "cleanup"}:
        return ToolBudget(read_calls=10, write_calls=8, exec_calls=2, llm_calls=0)
    if normalized in {"research", "analysis"}:
        return ToolBudget(read_calls=15, write_calls=0, exec_calls=0, llm_calls=2)
    return ToolBudget()


@dataclass
class ManagedTask:
    """Tracked work unit for a single request lifecycle."""

    id: str
    subject: str
    description: str
    active_form: str
    status: str = "pending"  # pending|in_progress|completed|deleted
    created_at: str = field(default_factory=_now_iso)
    completed_at: str | None = None
    blocked_by: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    budget: ToolBudget = field(default_factory=ToolBudget)


@dataclass
class TaskManager:
    """In-request task tracking with lightweight budget support."""

    tasks: list[ManagedTask] = field(default_factory=list)
    current_task_id: str | None = None
    _counter: int = 0

    def create(
        self,
        subject: str,
        description: str,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
        task_type: str = "chat",
    ) -> ManagedTask:
        self._counter += 1
        task = ManagedTask(
            id=f"T{self._counter}",
            subject=(subject or "").strip()[:120],
            description=(description or "").strip()[:400],
            active_form=(active_form or f"Working on {subject}").strip()[:120],
            metadata=dict(metadata or {}),
            budget=budget_for_task_type(task_type),
        )
        self.tasks.append(task)
        if self.current_task_id is None:
            self.current_task_id = task.id
        return task

    def update(self, task_id: str, **kwargs: Any) -> ManagedTask:
        task = self.get(task_id)
        if "status" in kwargs and kwargs["status"] is not None:
            status = str(kwargs["status"]).strip()
            if status in {"pending", "in_progress", "completed", "deleted"}:
                task.status = status
                if status == "completed":
                    task.completed_at = _now_iso()
                if status == "in_progress":
                    self.current_task_id = task.id
        if "subject" in kwargs and kwargs["subject"]:
            task.subject = str(kwargs["subject"]).strip()[:120]
        if "description" in kwargs and kwargs["description"]:
            task.description = str(kwargs["description"]).strip()[:400]
        return task

    def get(self, task_id: str) -> ManagedTask:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"Unknown task_id: {task_id}")

    def list_all(self) -> list[ManagedTask]:
        return list(self.tasks)

    def has_tasks(self) -> bool:
        return any(t.status != "deleted" for t in self.tasks)

    def current_task_budget(self) -> ToolBudget | None:
        if not self.current_task_id:
            return None
        try:
            return self.get(self.current_task_id).budget
        except KeyError:
            return None

    def override_budget(self, category: str, reason: str = "") -> dict[str, Any]:
        budget = self.current_task_budget()
        if budget is None:
            return {"ok": False, "error": "No active task for budget override"}
        ok = budget.override(category=category, amount=3, max_overrides=2)
        return {
            "ok": ok,
            "category": category,
            "reason": reason[:160],
            "remaining": budget.remaining_summary(),
            "overrides_used": budget.overrides_used,
        }

    def summary_block(self, limit: int = 8) -> list[str]:
        lines: list[str] = []
        for task in self.tasks[:limit]:
            if task.status == "deleted":
                continue
            icon = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}.get(task.status, "[?]")
            lines.append(f"{icon} {task.id} {task.subject}")
        return lines


_MANAGER: TaskManager | None = None


def set_active_task_manager(manager: TaskManager | None) -> None:
    """Set process-local active manager for registry-invoked task tools."""
    global _MANAGER
    _MANAGER = manager


def _require_manager() -> TaskManager:
    if _MANAGER is None:
        raise RuntimeError("Task manager not attached")
    return _MANAGER


def _task_to_dict(task: ManagedTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "subject": task.subject,
        "status": task.status,
        "active_form": task.active_form,
        "description": task.description,
        "completed_at": task.completed_at,
    }


def tool_task_create(subject: str, description: str, active_form: str | None = None) -> dict[str, Any]:
    manager = _require_manager()
    task = manager.create(subject=subject, description=description, active_form=active_form)
    return _task_to_dict(task)


def tool_task_update(task_id: str, status: str | None = None) -> dict[str, Any]:
    manager = _require_manager()
    task = manager.update(task_id, status=status)
    return _task_to_dict(task)


def tool_task_list() -> list[dict[str, Any]]:
    manager = _require_manager()
    return [_task_to_dict(task) for task in manager.list_all() if task.status != "deleted"]


def tool_budget_override(category: str, reason: str = "") -> dict[str, Any]:
    manager = _require_manager()
    return manager.override_budget(category=category, reason=reason)
