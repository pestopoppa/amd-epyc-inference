#!/usr/bin/env python3
"""Tests for the tool registry module."""

import pytest
from src.tool_registry import (
    Tool,
    ToolCategory,
    ToolPermissions,
    ToolRegistry,
    get_registry,
)


class TestToolPermissions:
    """Tests for ToolPermissions class."""

    def test_can_use_tool_allowed_category(self):
        """Tool in allowed category should be accessible."""
        perms = ToolPermissions(
            web_access=True,
            allowed_categories=[ToolCategory.WEB, ToolCategory.FILE],
        )
        tool = Tool(
            name="fetch_docs",
            description="Fetch docs",
            category=ToolCategory.WEB,
            parameters={},
        )
        assert perms.can_use_tool(tool) is True

    def test_can_use_tool_forbidden(self):
        """Tool in forbidden list should be blocked."""
        perms = ToolPermissions(
            allowed_categories=[ToolCategory.FILE],
            forbidden_tools=["write_file"],
        )
        tool = Tool(
            name="write_file",
            description="Write file",
            category=ToolCategory.FILE,
            parameters={},
        )
        assert perms.can_use_tool(tool) is False

    def test_can_use_tool_explicit_allow(self):
        """Explicitly allowed tool should be accessible."""
        perms = ToolPermissions(
            allowed_tools=["special_tool"],
        )
        tool = Tool(
            name="special_tool",
            description="Special",
            category=ToolCategory.SPECIALIZED,
            parameters={},
        )
        assert perms.can_use_tool(tool) is True

    def test_can_use_tool_web_requires_web_access(self):
        """Web tools require web_access flag."""
        perms = ToolPermissions(
            web_access=False,
            allowed_categories=[ToolCategory.WEB],
        )
        tool = Tool(
            name="fetch_docs",
            description="Fetch docs",
            category=ToolCategory.WEB,
            parameters={},
        )
        assert perms.can_use_tool(tool) is False

        perms.web_access = True
        assert perms.can_use_tool(tool) is True


class TestTool:
    """Tests for Tool class."""

    def test_validate_args_required_missing(self):
        """Missing required argument should fail validation."""
        tool = Tool(
            name="test",
            description="Test",
            category=ToolCategory.DATA,
            parameters={
                "url": {"type": "string", "required": True},
            },
        )
        errors = tool.validate_args({})
        assert "Missing required parameter: url" in errors

    def test_validate_args_unknown_param(self):
        """Unknown parameter should fail validation."""
        tool = Tool(
            name="test",
            description="Test",
            category=ToolCategory.DATA,
            parameters={
                "url": {"type": "string"},
            },
        )
        errors = tool.validate_args({"url": "test", "unknown": "value"})
        assert "Unknown parameter: unknown" in errors

    def test_validate_args_valid(self):
        """Valid arguments should pass validation."""
        tool = Tool(
            name="test",
            description="Test",
            category=ToolCategory.DATA,
            parameters={
                "url": {"type": "string", "required": True},
                "limit": {"type": "integer"},
            },
        )
        errors = tool.validate_args({"url": "http://example.com", "limit": 10})
        assert errors == []


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self):
        """Tools can be registered."""
        registry = ToolRegistry()
        tool = Tool(
            name="test_tool",
            description="Test",
            category=ToolCategory.DATA,
            parameters={},
        )
        registry.register_tool(tool)
        assert "test_tool" in registry._tools

    def test_register_duplicate_raises(self):
        """Registering duplicate tool raises error."""
        registry = ToolRegistry()
        tool = Tool(
            name="test_tool",
            description="Test",
            category=ToolCategory.DATA,
            parameters={},
        )
        registry.register_tool(tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_tool(tool)

    def test_can_use_tool_with_permissions(self):
        """Permission checking works correctly."""
        registry = ToolRegistry()
        tool = Tool(
            name="fetch_docs",
            description="Fetch",
            category=ToolCategory.WEB,
            parameters={},
        )
        registry.register_tool(tool)

        # Set permissions for role
        registry.set_role_permissions(
            "frontdoor",
            ToolPermissions(web_access=True, allowed_categories=[ToolCategory.WEB]),
        )
        registry.set_role_permissions(
            "worker",
            ToolPermissions(web_access=False, allowed_categories=[ToolCategory.FILE]),
        )

        assert registry.can_use_tool("frontdoor", "fetch_docs") is True
        assert registry.can_use_tool("worker", "fetch_docs") is False

    def test_list_tools_filtered_by_role(self):
        """List tools should filter by role permissions."""
        registry = ToolRegistry()

        web_tool = Tool(
            name="fetch",
            description="Fetch",
            category=ToolCategory.WEB,
            parameters={},
        )
        file_tool = Tool(
            name="read_file",
            description="Read",
            category=ToolCategory.FILE,
            parameters={},
        )
        registry.register_tool(web_tool)
        registry.register_tool(file_tool)

        registry.set_role_permissions(
            "worker",
            ToolPermissions(web_access=False, allowed_categories=[ToolCategory.FILE]),
        )

        tools = registry.list_tools(role="worker")
        tool_names = [t["name"] for t in tools]
        assert "read_file" in tool_names
        assert "fetch" not in tool_names

    def test_invoke_with_handler(self):
        """Invoke should call handler function."""
        registry = ToolRegistry()

        def my_handler(value: int) -> int:
            return value * 2

        tool = Tool(
            name="double",
            description="Double",
            category=ToolCategory.DATA,
            parameters={"value": {"type": "integer", "required": True}},
            handler=my_handler,
        )
        registry.register_tool(tool)
        registry.set_role_permissions(
            "test",
            ToolPermissions(allowed_categories=[ToolCategory.DATA]),
        )

        result = registry.invoke("double", "test", value=5)
        assert result == 10
        inv = registry.get_invocation_log()[-1]
        assert inv.caller_type == "direct"
        assert inv.chain_id is None
        assert inv.chain_index == 0

    def test_invoke_records_chain_metadata(self):
        registry = ToolRegistry()

        tool = Tool(
            name="double",
            description="Double",
            category=ToolCategory.DATA,
            parameters={"value": {"type": "integer", "required": True}},
            handler=lambda value: value * 2,
        )
        registry.register_tool(tool)
        registry.set_role_permissions(
            "test",
            ToolPermissions(allowed_categories=[ToolCategory.DATA]),
        )

        result = registry.invoke(
            "double",
            "test",
            caller_type="chain",
            chain_id="ch_123",
            chain_index=2,
            value=5,
        )
        assert result == 10
        inv = registry.get_invocation_log()[-1]
        assert inv.caller_type == "chain"
        assert inv.chain_id == "ch_123"
        assert inv.chain_index == 2

    def test_invoke_permission_denied(self):
        """Invoke should raise PermissionError when role can't use tool."""
        registry = ToolRegistry()
        tool = Tool(
            name="test",
            description="Test",
            category=ToolCategory.WEB,
            parameters={},
            handler=lambda: "test",
        )
        registry.register_tool(tool)
        registry.set_role_permissions(
            "worker",
            ToolPermissions(web_access=False),
        )

        with pytest.raises(PermissionError):
            registry.invoke("test", "worker")

    def test_get_chainable_tools(self):
        registry = ToolRegistry()
        registry.register_tool(
            Tool(
                name="chainable",
                description="Chainable",
                category=ToolCategory.DATA,
                parameters={},
                allowed_callers=["direct", "chain"],
            )
        )
        registry.register_tool(
            Tool(
                name="direct_only",
                description="Direct",
                category=ToolCategory.DATA,
                parameters={},
                allowed_callers=["direct"],
            )
        )

        assert registry.get_chainable_tools() == {"chainable"}


class TestToolRegistryMCP:
    """Tests for MCP integration in ToolRegistry."""

    def test_invoke_mcp_tool(self):
        """_invoke_mcp should delegate to call_mcp_tool."""
        from unittest.mock import patch
        from src.mcp_client import MCPServerConfig

        registry = ToolRegistry()
        # Pre-load a config so it doesn't hit disk
        registry._mcp_configs = {
            "test-server": MCPServerConfig(
                name="test-server",
                command="echo",
                timeout=5,
            )
        }

        with patch("src.mcp_client.call_mcp_tool", return_value="mcp result") as mock_call:
            result = registry._invoke_mcp("test-server", "my_tool", {"key": "val"})

        assert result == "mcp result"
        mock_call.assert_called_once_with(
            registry._mcp_configs["test-server"], "my_tool", {"key": "val"}
        )

    def test_invoke_mcp_unknown_server(self):
        """_invoke_mcp with unknown server should raise RuntimeError."""
        registry = ToolRegistry()
        registry._mcp_configs = {}  # No servers configured

        with pytest.raises(RuntimeError, match="Unknown MCP server"):
            registry._invoke_mcp("nonexistent", "tool", {})


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_singleton(self):
        """get_registry should return same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
