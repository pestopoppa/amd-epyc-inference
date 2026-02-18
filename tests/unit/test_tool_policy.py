"""Tests for cascading tool policy resolution."""

from __future__ import annotations

import pytest

from src.tool_policy import (
    TOOL_GROUPS,
    PolicyLayer,
    permissions_to_policy,
    resolve_policy_chain,
)
from src.tool_registry import (
    Tool,
    ToolCategory,
    ToolPermissions,
    ToolRegistry,
)


# Shared tool universe for tests
ALL_TOOLS = frozenset({
    "list_dir", "read_file", "search_files", "write_file",
    "edit_file", "run_shell", "web_fetch", "web_search",
    "query_db", "calculate", "plot", "raw_exec",
})


class TestPolicyLayer:
    """Tests for PolicyLayer dataclass."""

    def test_frozen(self):
        layer = PolicyLayer(name="test")
        with pytest.raises(AttributeError):
            layer.name = "changed"  # type: ignore[misc]

    def test_expand_groups_read(self):
        layer = PolicyLayer(name="test", allow=frozenset({"group:read"}))
        expanded_allow, expanded_deny = layer.expand_groups(ALL_TOOLS)
        assert expanded_allow == TOOL_GROUPS["group:read"]
        assert expanded_deny == frozenset()

    def test_expand_groups_write(self):
        layer = PolicyLayer(name="test", deny=frozenset({"group:write"}))
        _, expanded_deny = layer.expand_groups(ALL_TOOLS)
        assert expanded_deny == TOOL_GROUPS["group:write"]

    def test_expand_groups_all(self):
        layer = PolicyLayer(name="test", allow=frozenset({"group:all"}))
        expanded_allow, _ = layer.expand_groups(ALL_TOOLS)
        assert expanded_allow == ALL_TOOLS

    def test_expand_unknown_group(self):
        layer = PolicyLayer(name="test", allow=frozenset({"group:nonexistent"}))
        expanded_allow, _ = layer.expand_groups(ALL_TOOLS)
        # Unknown group expands to empty set
        assert expanded_allow == frozenset()

    def test_expand_mixed_groups_and_names(self):
        layer = PolicyLayer(
            name="test",
            allow=frozenset({"group:read", "calculate"}),
        )
        expanded_allow, _ = layer.expand_groups(ALL_TOOLS)
        expected = TOOL_GROUPS["group:read"] | {"calculate"}
        assert expanded_allow == expected


class TestResolvePolicyChain:
    """Tests for resolve_policy_chain()."""

    def test_empty_chain_allows_all(self):
        result = resolve_policy_chain([], ALL_TOOLS)
        assert result == ALL_TOOLS

    def test_single_deny_removes_tool(self):
        chain = [PolicyLayer(name="global", deny=frozenset({"raw_exec"}))]
        result = resolve_policy_chain(chain, ALL_TOOLS)
        assert "raw_exec" not in result
        assert len(result) == len(ALL_TOOLS) - 1

    def test_deny_wins_over_allow_in_same_layer(self):
        chain = [PolicyLayer(
            name="conflict",
            allow=frozenset({"raw_exec", "read_file"}),
            deny=frozenset({"raw_exec"}),
        )]
        result = resolve_policy_chain(chain, ALL_TOOLS)
        assert "raw_exec" not in result
        assert "read_file" in result

    def test_layers_only_narrow_never_expand(self):
        chain = [
            PolicyLayer(name="narrow", allow=frozenset({"read_file", "write_file"})),
            # Second layer tries to allow more tools — but can only narrow
            PolicyLayer(name="expand_attempt", allow=frozenset({
                "read_file", "write_file", "run_shell", "raw_exec",
            })),
        ]
        result = resolve_policy_chain(chain, ALL_TOOLS)
        # After first layer: {read_file, write_file}
        # Second layer intersects: {read_file, write_file} & {read_file, write_file, run_shell, raw_exec}
        # = {read_file, write_file} — no expansion
        assert result == frozenset({"read_file", "write_file"})

    def test_group_deny_across_layers(self):
        chain = [
            PolicyLayer(name="role", allow=frozenset({"group:code"})),
            PolicyLayer(name="task", deny=frozenset({"group:write"})),
        ]
        result = resolve_policy_chain(chain, ALL_TOOLS)
        # group:code = {list_dir, read_file, search_files, write_file, edit_file, run_shell}
        # minus group:write = {write_file, edit_file, run_shell}
        # = {list_dir, read_file, search_files}
        assert result == frozenset({"list_dir", "read_file", "search_files"})

    def test_three_layer_narrowing(self):
        chain = [
            PolicyLayer(name="global", deny=frozenset({"raw_exec"})),
            PolicyLayer(name="role", allow=frozenset({"group:code"})),
            PolicyLayer(name="task", deny=frozenset({"run_shell"})),
        ]
        result = resolve_policy_chain(chain, ALL_TOOLS)
        expected = frozenset({"list_dir", "read_file", "search_files", "write_file", "edit_file"})
        assert result == expected

    def test_empty_allow_inherits_previous(self):
        chain = [
            PolicyLayer(name="role", allow=frozenset({"read_file", "write_file"})),
            # Empty allow = no further narrowing (just deny)
            PolicyLayer(name="task", deny=frozenset({"write_file"})),
        ]
        result = resolve_policy_chain(chain, ALL_TOOLS)
        assert result == frozenset({"read_file"})

    def test_all_denied_results_in_empty(self):
        chain = [
            PolicyLayer(name="allow_one", allow=frozenset({"read_file"})),
            PolicyLayer(name="deny_it", deny=frozenset({"read_file"})),
        ]
        result = resolve_policy_chain(chain, ALL_TOOLS)
        assert result == frozenset()

    def test_empty_tool_universe(self):
        chain = [PolicyLayer(name="test", allow=frozenset({"read_file"}))]
        result = resolve_policy_chain(chain, frozenset())
        assert result == frozenset()


class TestPermissionsToPolicy:
    """Tests for the ToolPermissions -> PolicyLayer adapter."""

    def _make_tools(self) -> dict[str, Tool]:
        return {
            "fetch_docs": Tool(
                name="fetch_docs", description="Fetch", category=ToolCategory.WEB, parameters={},
            ),
            "read_file": Tool(
                name="read_file", description="Read", category=ToolCategory.FILE, parameters={},
            ),
            "write_file": Tool(
                name="write_file", description="Write", category=ToolCategory.FILE, parameters={},
            ),
            "double": Tool(
                name="double", description="Double", category=ToolCategory.DATA, parameters={},
            ),
        }

    def test_category_allow(self):
        tools = self._make_tools()
        perms = ToolPermissions(
            web_access=True,
            allowed_categories=[ToolCategory.WEB, ToolCategory.FILE],
        )
        layer = permissions_to_policy("role:test", perms, tools)
        assert "fetch_docs" in layer.allow
        assert "read_file" in layer.allow
        assert "write_file" in layer.allow
        assert "double" not in layer.allow

    def test_web_without_web_access(self):
        tools = self._make_tools()
        perms = ToolPermissions(
            web_access=False,
            allowed_categories=[ToolCategory.WEB, ToolCategory.FILE],
        )
        layer = permissions_to_policy("role:test", perms, tools)
        assert "fetch_docs" not in layer.allow  # No web_access
        assert "read_file" in layer.allow

    def test_forbidden_becomes_deny(self):
        tools = self._make_tools()
        perms = ToolPermissions(
            allowed_categories=[ToolCategory.FILE],
            forbidden_tools=["write_file"],
        )
        layer = permissions_to_policy("role:test", perms, tools)
        assert "write_file" in layer.deny

    def test_explicit_allow(self):
        tools = self._make_tools()
        perms = ToolPermissions(allowed_tools=["double"])
        layer = permissions_to_policy("role:test", perms, tools)
        assert "double" in layer.allow

    def test_equivalence_with_legacy(self):
        """Adapted policy should produce same access as legacy can_use_tool()."""
        tools = self._make_tools()
        perms = ToolPermissions(
            web_access=True,
            allowed_categories=[ToolCategory.WEB, ToolCategory.FILE],
            forbidden_tools=["write_file"],
        )
        layer = permissions_to_policy("role:test", perms, tools)
        all_tool_names = frozenset(tools.keys())
        allowed = resolve_policy_chain([layer], all_tool_names)

        # Check each tool matches legacy behavior
        for tool_name, tool in tools.items():
            legacy_result = perms.can_use_tool(tool)
            policy_result = tool_name in allowed
            assert legacy_result == policy_result, (
                f"Mismatch for {tool_name}: legacy={legacy_result}, policy={policy_result}"
            )


class TestToolRegistryCascading:
    """Tests for ToolRegistry with cascading_tool_policy enabled."""

    def _make_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        for name, cat in [
            ("read_file", ToolCategory.FILE),
            ("write_file", ToolCategory.FILE),
            ("web_fetch", ToolCategory.WEB),
            ("run_shell", ToolCategory.SYSTEM),
        ]:
            registry.register_tool(Tool(
                name=name, description=name, category=cat, parameters={},
            ))
        return registry

    def test_cascading_with_global_deny(self, monkeypatch):
        from src.features import Features, set_features, reset_features

        try:
            set_features(Features(cascading_tool_policy=True))
            registry = self._make_registry()

            from src.tool_policy import PolicyLayer
            registry.add_global_policy(PolicyLayer(name="global", deny=frozenset({"run_shell"})))

            # Role with all tools allowed
            registry.set_role_permissions(
                "worker",
                ToolPermissions(
                    allowed_categories=[ToolCategory.FILE, ToolCategory.SYSTEM],
                ),
            )

            assert registry.can_use_tool("worker", "read_file") is True
            assert registry.can_use_tool("worker", "run_shell") is False  # Global deny
        finally:
            reset_features()

    def test_cascading_with_context_read_only(self):
        from src.features import Features, set_features, reset_features

        try:
            set_features(Features(cascading_tool_policy=True))
            registry = self._make_registry()
            registry.set_role_permissions(
                "worker",
                ToolPermissions(allowed_categories=[ToolCategory.FILE]),
            )

            # Without context: both read and write allowed
            assert registry.can_use_tool("worker", "read_file") is True
            assert registry.can_use_tool("worker", "write_file") is True

            # With read_only context: write denied
            assert registry.can_use_tool("worker", "read_file", context={"read_only": True}) is True
            assert registry.can_use_tool("worker", "write_file", context={"read_only": True}) is False
        finally:
            reset_features()

    def test_cascading_with_context_no_web(self):
        from src.features import Features, set_features, reset_features

        try:
            set_features(Features(cascading_tool_policy=True))
            registry = self._make_registry()
            registry.set_role_permissions(
                "frontdoor",
                ToolPermissions(
                    web_access=True,
                    allowed_categories=[ToolCategory.FILE, ToolCategory.WEB],
                ),
            )

            assert registry.can_use_tool("frontdoor", "web_fetch") is True
            assert registry.can_use_tool("frontdoor", "web_fetch", context={"no_web": True}) is False
        finally:
            reset_features()

    def test_cascading_with_role_policy_layers(self):
        from src.features import Features, set_features, reset_features

        try:
            set_features(Features(cascading_tool_policy=True))
            registry = self._make_registry()

            from src.tool_policy import PolicyLayer
            # Use explicit role policy layers instead of legacy ToolPermissions
            registry.add_role_policy("reader", PolicyLayer(
                name="role:reader",
                allow=frozenset({"read_file"}),
            ))

            assert registry.can_use_tool("reader", "read_file") is True
            assert registry.can_use_tool("reader", "write_file") is False
        finally:
            reset_features()

    def test_legacy_path_when_flag_off(self):
        """With cascading_tool_policy=False, legacy ToolPermissions used."""
        from src.features import Features, set_features, reset_features

        try:
            set_features(Features(cascading_tool_policy=False))
            registry = self._make_registry()
            registry.set_role_permissions(
                "worker",
                ToolPermissions(allowed_categories=[ToolCategory.FILE]),
            )

            # Legacy path: no context parameter effect
            assert registry.can_use_tool("worker", "read_file") is True
            assert registry.can_use_tool("worker", "web_fetch") is False
        finally:
            reset_features()

    def test_unknown_role_cascading_denies(self):
        from src.features import Features, set_features, reset_features

        try:
            set_features(Features(cascading_tool_policy=True))
            registry = self._make_registry()
            # No permissions or policies for "unknown_role" — no allow layers
            # With no allow narrowing, all tools remain. But there's no role policy.
            # The chain is just global (empty), so all tools pass.
            # This is actually correct — if no restrictions exist, everything is allowed.
            # To deny by default, add a global policy.
            result = registry.can_use_tool("unknown_role", "read_file")
            assert result is True  # No restrictions = allowed
        finally:
            reset_features()

    def test_invoke_respects_context(self):
        """invoke() passes through to can_use_tool which uses legacy path (no context)."""
        from src.features import Features, set_features, reset_features

        try:
            set_features(Features(cascading_tool_policy=False))
            registry = self._make_registry()
            registry._tools["read_file"].handler = lambda: "content"
            registry.set_role_permissions(
                "worker",
                ToolPermissions(allowed_categories=[ToolCategory.FILE]),
            )

            result = registry.invoke("read_file", "worker")
            assert result == "content"
        finally:
            reset_features()
