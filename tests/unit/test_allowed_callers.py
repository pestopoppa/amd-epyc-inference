"""Unit tests for allowed_callers gating in structured chains."""

from __future__ import annotations

from src.repl_environment import REPLEnvironment, REPLConfig


class _Registry:
    def __init__(self, chainable: set[str]) -> None:
        self.chainable = chainable

    def invoke(self, tool_name: str, role: str, **kwargs):
        return {"ok": True, "tool": tool_name}

    def list_tools(self, role: str | None = None):
        return []

    def get_read_only_tools(self):
        return set()

    def get_chainable_tools(self):
        return set(self.chainable)


def test_structured_blocks_tool_call_when_wrapped_tool_not_chainable():
    repl = REPLEnvironment(
        context="x",
        config=REPLConfig(structured_mode=True),
        tool_registry=_Registry(chainable=set()),
    )
    result = repl.execute('a = TOOL("unsafe")\nb = TOOL("unsafe")')
    assert result.error is not None
    assert "TOOL(unsafe)" in result.error


def test_structured_allows_tool_chain_when_wrapped_tool_is_chainable():
    repl = REPLEnvironment(
        context="x",
        config=REPLConfig(structured_mode=True),
        tool_registry=_Registry(chainable={"safe_tool"}),
    )
    result = repl.execute('a = TOOL("safe_tool")\nb = TOOL("safe_tool")')
    assert result.error is None
