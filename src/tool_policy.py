"""Cascading tool policy resolution.

Implements a layered policy chain where each layer can only narrow —
never expand — the allowed tool set. Deny always wins at every layer.

Policy chain: Global → Role → Task → Delegation
Each layer is a PolicyLayer with allow/deny sets that support group:
prefixes for common tool bundles.

Guarded by features().cascading_tool_policy (default: False — opt-in).

Usage:
    from src.tool_policy import PolicyLayer, resolve_policy_chain

    chain = [
        PolicyLayer(name="global", deny=frozenset({"raw_exec"})),
        PolicyLayer(name="role:worker", allow=frozenset({"group:read"})),
        PolicyLayer(name="task:readonly", deny=frozenset({"group:write"})),
    ]
    allowed = resolve_policy_chain(chain, all_tools)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)

# Tool group expansion — shorthand for common tool sets.
# Group names use "group:" prefix in allow/deny sets.
# "group:all" is special: expands to the full tool universe at resolution time.
TOOL_GROUPS: dict[str, frozenset[str]] = {
    "group:read": frozenset({
        "list_dir", "read_file", "search_files", "web_fetch",
    }),
    "group:write": frozenset({
        "write_file", "edit_file", "run_shell",
    }),
    "group:code": frozenset({
        "list_dir", "read_file", "search_files",
        "write_file", "edit_file", "run_shell",
    }),
    "group:data": frozenset({
        "query_db", "read_file", "search_files",
    }),
    "group:web": frozenset({
        "web_fetch", "web_search",
    }),
    "group:math": frozenset({
        "calculate", "plot",
    }),
    "group:all": frozenset(),  # Special: expanded to all_tools at resolution time
}


@dataclass(frozen=True)
class PolicyLayer:
    """A single layer in the cascading policy chain.

    Rules:
    - ``allow`` intersects with the current set (can only narrow).
    - ``deny`` always removes, regardless of allow.
    - Empty allow = "inherit everything from previous layer" (no narrowing).
    - Both allow and deny can use ``group:`` prefixes.
    """

    name: str
    allow: frozenset[str] = field(default_factory=frozenset)
    deny: frozenset[str] = field(default_factory=frozenset)

    def expand_groups(
        self, all_tools: frozenset[str],
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Expand group: prefixes in allow/deny sets.

        Args:
            all_tools: Universe of all registered tool names.

        Returns:
            Tuple of (expanded_allow, expanded_deny).
        """
        expanded_allow: set[str] = set()
        for item in self.allow:
            if item.startswith("group:"):
                if item == "group:all":
                    expanded_allow.update(all_tools)
                else:
                    group = TOOL_GROUPS.get(item, frozenset())
                    if not group and item != "group:all":
                        logger.debug("Unknown tool group: %s", item)
                    expanded_allow.update(group)
            else:
                expanded_allow.add(item)

        expanded_deny: set[str] = set()
        for item in self.deny:
            if item.startswith("group:"):
                group = TOOL_GROUPS.get(item, frozenset())
                if not group:
                    logger.debug("Unknown tool group in deny: %s", item)
                expanded_deny.update(group)
            else:
                expanded_deny.add(item)

        return frozenset(expanded_allow), frozenset(expanded_deny)


def resolve_policy_chain(
    layers: Sequence[PolicyLayer],
    all_tools: frozenset[str],
) -> frozenset[str]:
    """Resolve a chain of policy layers into a final allowed tool set.

    Each layer narrows (never expands beyond) the previous result.
    Deny always wins at every layer.

    Args:
        layers: Ordered policy layers (outermost first).
        all_tools: Universe of all registered tool names.

    Returns:
        Final set of allowed tool names.
    """
    current = all_tools

    for layer in layers:
        expanded_allow, expanded_deny = layer.expand_groups(all_tools)

        # Apply allow (intersect — can only narrow)
        if expanded_allow:
            current = current & expanded_allow

        # Apply deny (always removes)
        current = current - expanded_deny

    return current


def permissions_to_policy(
    name: str,
    permissions: "ToolPermissions",  # noqa: F821 — forward ref to avoid circular import
    all_tools: dict[str, "Tool"],  # noqa: F821
) -> PolicyLayer:
    """Convert a legacy ToolPermissions instance to a PolicyLayer.

    This adapter allows existing model_registry.yaml tool_permissions
    to work with the cascading policy system without YAML changes.

    Args:
        name: Layer name (typically "role:{role_name}").
        permissions: Legacy ToolPermissions instance.
        all_tools: Dict of tool_name -> Tool for category resolution.

    Returns:
        Equivalent PolicyLayer.
    """
    allow: set[str] = set()
    deny: set[str] = set(permissions.forbidden_tools)

    # Explicit allow list
    if permissions.allowed_tools:
        allow.update(permissions.allowed_tools)

    # Category-based allows: resolve to concrete tool names
    if permissions.allowed_categories:
        from src.tool_registry import ToolCategory

        for tool_name, tool in all_tools.items():
            if tool.category in permissions.allowed_categories:
                # Web tools require web_access
                if tool.category == ToolCategory.WEB and not permissions.web_access:
                    continue
                allow.add(tool_name)

    return PolicyLayer(
        name=name,
        allow=frozenset(allow) if allow else frozenset(),
        deny=frozenset(deny),
    )
