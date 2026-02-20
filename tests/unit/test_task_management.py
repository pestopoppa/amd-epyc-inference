"""Unit tests for in-request task manager utilities."""

from orchestration.tools.task_management import TaskManager


def test_task_manager_create_update_list():
    manager = TaskManager()
    t1 = manager.create(subject="Fix bug", description="Investigate timeout", task_type="bugfix")
    assert t1.id == "T1"
    assert manager.has_tasks() is True
    assert manager.current_task_id == "T1"

    manager.update(t1.id, status="in_progress")
    assert manager.get(t1.id).status == "in_progress"

    manager.update(t1.id, status="completed")
    listed = manager.list_all()
    assert len(listed) == 1
    assert listed[0].completed_at is not None


def test_budget_override_cap():
    manager = TaskManager()
    manager.create(subject="Task", description="Desc", task_type="chat")

    first = manager.override_budget("read", reason="need more files")
    second = manager.override_budget("read", reason="need more files")
    third = manager.override_budget("read", reason="need more files")

    assert first["ok"] is True
    assert second["ok"] is True
    assert third["ok"] is False
