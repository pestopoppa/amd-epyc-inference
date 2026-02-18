"""Unit tests for structured-mode tool chaining gates and metadata."""

from __future__ import annotations

from src.repl_environment import REPLConfig, REPLEnvironment


class _DummyRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def invoke(self, tool_name: str, role: str, **kwargs):
        self.calls.append((tool_name, role, kwargs))
        return {"ok": True, "tool": tool_name}

    def list_tools(self, role: str | None = None):
        return [{"name": "mock_tool"}]

    def get_read_only_tools(self):
        return set()

    def get_chainable_tools(self):
        return {"mock_tool"}


def test_structured_blocks_non_chainable_multi_tool():
    repl = REPLEnvironment(context="x", config=REPLConfig(structured_mode=True))
    result = repl.execute('delegate("do work")\nrun_shell("echo hi")')

    assert result.error is not None
    assert "Blocked tools: delegate" in result.error


def test_structured_chain_sets_invocation_chain_metadata():
    registry = _DummyRegistry()
    repl = REPLEnvironment(
        context="x",
        config=REPLConfig(structured_mode=True),
        tool_registry=registry,
        role="frontdoor",
    )
    result = repl.execute('a = TOOL("mock_tool")\nb = TOOL("mock_tool")')

    assert result.error is None
    assert len(registry.calls) == 2
    first = registry.calls[0][2]
    second = registry.calls[1][2]
    assert first["caller_type"] == "chain"
    assert second["caller_type"] == "chain"
    assert first["chain_id"] == second["chain_id"]
    assert first["chain_id"] is not None
    assert first["chain_index"] == 0
    assert second["chain_index"] == 1


def test_dependency_mode_executes_mixed_chain(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TOOL_CHAIN_MODE", "dep")
    repl = REPLEnvironment(context="xyz", config=REPLConfig(structured_mode=True))
    result = repl.execute('a = peek(1)\nb = grep("x")\nc = run_shell("echo dep-mode")')

    assert result.error is None
    assert result.output.startswith("Observation:")
    assert "[a]:" in result.output
    assert "[b]:" in result.output
    assert "[c]:" in result.output
    chain_log = repl.get_chain_execution_log()
    assert len(chain_log) >= 1
    last = chain_log[-1]
    assert last["mode_requested"] == "dep"
    assert last["mode_used"] == "dep"
    assert last["fallback_to_seq"] is False
    assert int(last["waves"]) >= 1


def test_dependency_mode_falls_back_to_seq_exec_on_non_call_stmt(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TOOL_CHAIN_MODE", "dep")
    repl = REPLEnvironment(context="xyz", config=REPLConfig(structured_mode=True))
    result = repl.execute('a = peek(1)\nmsg = "ok"\nb = run_shell("echo dep-mode")')

    assert result.error is None
    assert result.output.startswith("Observation:")
    chain_log = repl.get_chain_execution_log()
    assert len(chain_log) >= 1
    last = chain_log[-1]
    assert last["mode_requested"] == "dep"
    assert last["mode_used"] == "seq"
    assert last["fallback_to_seq"] is True
