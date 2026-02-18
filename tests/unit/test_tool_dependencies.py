"""Unit tests for dependency-aware chain analysis behavior."""

from __future__ import annotations

from src.repl_environment import REPLEnvironment, REPLConfig


def test_dep_mode_builds_multiple_waves_for_data_dependencies(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TOOL_CHAIN_MODE", "dep")
    repl = REPLEnvironment(context="abc", config=REPLConfig(structured_mode=True))
    result = repl.execute('a = peek(1)\nb = grep(a)\nc = grep("a")')

    assert result.error is None
    chain_log = repl.get_chain_execution_log()
    assert chain_log
    # b depends on a, so at least two waves are required.
    assert int(chain_log[-1]["waves"]) >= 2
    assert chain_log[-1]["mode_used"] == "dep"
