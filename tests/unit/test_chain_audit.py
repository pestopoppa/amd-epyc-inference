"""Unit tests for chain audit grouping and metadata propagation."""

from __future__ import annotations

from src.repl_environment import REPLEnvironment, REPLConfig


class _AuditRegistry:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, tool_name: str, role: str, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}

    def list_tools(self, role: str | None = None):
        return []

    def get_read_only_tools(self):
        return set()

    def get_chainable_tools(self):
        return {"audit_tool"}


def test_chain_id_and_index_monotonic_for_wrapped_tool_chain():
    registry = _AuditRegistry()
    repl = REPLEnvironment(
        context="x",
        config=REPLConfig(structured_mode=True),
        tool_registry=registry,
    )
    result = repl.execute('a = TOOL("audit_tool")\nb = TOOL("audit_tool")')

    assert result.error is None
    assert len(registry.calls) == 2
    first, second = registry.calls
    assert first["caller_type"] == "chain"
    assert second["caller_type"] == "chain"
    assert first["chain_id"] == second["chain_id"]
    assert first["chain_index"] == 0
    assert second["chain_index"] == 1

    chain_log = repl.get_chain_execution_log()
    assert chain_log
    assert chain_log[-1]["chain_id"] == first["chain_id"]
