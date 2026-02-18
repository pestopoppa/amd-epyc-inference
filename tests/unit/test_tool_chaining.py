"""Unit tests for structured-mode tool chaining gates and metadata."""

from __future__ import annotations

import time

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


def test_dependency_mode_parallel_mutations_reduce_latency(monkeypatch):
    code = (
        'a = run_shell("python3 -c \\"import time; time.sleep(0.2); print(1)\\"")\n'
        'b = run_shell("python3 -c \\"import time; time.sleep(0.2); print(2)\\"")'
    )

    monkeypatch.setenv("ORCHESTRATOR_TOOL_CHAIN_MODE", "dep")
    monkeypatch.setenv("ORCHESTRATOR_TOOL_CHAIN_PARALLEL_MUTATIONS", "0")
    seq_repl = REPLEnvironment(context="x", config=REPLConfig(structured_mode=True))
    t0 = time.perf_counter()
    seq_result = seq_repl.execute(code)
    seq_elapsed = time.perf_counter() - t0
    assert seq_result.error is None

    monkeypatch.setenv("ORCHESTRATOR_TOOL_CHAIN_PARALLEL_MUTATIONS", "1")
    par_repl = REPLEnvironment(context="x", config=REPLConfig(structured_mode=True))
    t1 = time.perf_counter()
    par_result = par_repl.execute(code)
    par_elapsed = time.perf_counter() - t1
    assert par_result.error is None

    # Parallel mutation waves should be materially faster than serialized execution.
    assert par_elapsed < seq_elapsed * 0.8
    log = par_repl.get_chain_execution_log()
    assert log[-1]["mode_used"] == "dep"
    assert log[-1]["fallback_to_seq"] is False
    assert int(log[-1]["waves"]) == 1
