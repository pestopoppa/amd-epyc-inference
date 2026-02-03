#!/usr/bin/env python3
"""Tests for the script registry module."""

import json
import tempfile
from pathlib import Path

import pytest
from src.script_registry import Script, ScriptRegistry, get_registry


class TestScript:
    """Tests for Script class."""

    def test_validate_args_required_missing(self):
        """Missing required argument should fail validation."""
        script = Script(
            id="test",
            description="Test",
            category="data",
            tags=[],
            parameters={
                "url": {"type": "string", "required": True},
            },
        )
        errors = script.validate_args({})
        assert "Missing required parameter: url" in errors

    def test_validate_args_unknown_param(self):
        """Unknown parameter should fail validation."""
        script = Script(
            id="test",
            description="Test",
            category="data",
            tags=[],
            parameters={
                "url": {"type": "string"},
            },
        )
        errors = script.validate_args({"url": "test", "unknown": "value"})
        assert "Unknown parameter: unknown" in errors

    def test_get_merged_args(self):
        """Merged args should combine defaults and user args."""
        script = Script(
            id="test",
            description="Test",
            category="data",
            tags=[],
            parameters={
                "url": {"type": "string", "required": True},
                "limit": {"type": "integer", "default": 10},
            },
            default_args={"timeout": 30},
        )
        merged = script.get_merged_args({"url": "http://example.com"})
        assert merged == {
            "url": "http://example.com",
            "limit": 10,
            "timeout": 30,
        }


class TestScriptRegistry:
    """Tests for ScriptRegistry class."""

    def test_register_script(self):
        """Scripts can be registered."""
        registry = ScriptRegistry()
        script = Script(
            id="test_script",
            description="Test",
            category="data",
            tags=["test"],
            parameters={},
        )
        registry.register_script(script)
        assert "test_script" in registry._scripts

    def test_register_duplicate_raises(self):
        """Registering duplicate script raises error."""
        registry = ScriptRegistry()
        script = Script(
            id="test_script",
            description="Test",
            category="data",
            tags=[],
            parameters={},
        )
        registry.register_script(script)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_script(script)

    def test_load_from_json(self):
        """Scripts can be loaded from JSON files."""
        registry = ScriptRegistry()

        script_data = {
            "id": "json_script",
            "description": "Loaded from JSON",
            "category": "test",
            "tags": ["json", "test"],
            "parameters": {"value": {"type": "string", "required": True}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(script_data, f)
            temp_path = f.name

        try:
            registry.load_from_json(temp_path)
            assert "json_script" in registry._scripts
            assert registry._scripts["json_script"].category == "test"
        finally:
            Path(temp_path).unlink()

    def test_find_scripts_by_description(self):
        """find_scripts should match on description."""
        registry = ScriptRegistry()

        script1 = Script(
            id="fetch_docs",
            description="Fetch documentation from URL",
            category="web",
            tags=["fetch", "docs"],
            parameters={},
        )
        script2 = Script(
            id="run_tests",
            description="Run pytest tests",
            category="code",
            tags=["test", "pytest"],
            parameters={},
        )
        registry.register_script(script1)
        registry.register_script(script2)

        matches = registry.find_scripts("fetch documentation")
        assert len(matches) >= 1
        assert matches[0].script.id == "fetch_docs"

    def test_find_scripts_by_tag(self):
        """find_scripts should match on tags."""
        registry = ScriptRegistry()

        script = Script(
            id="pytest_runner",
            description="Execute tests",
            category="code",
            tags=["pytest", "testing"],
            parameters={},
        )
        registry.register_script(script)

        matches = registry.find_scripts("pytest")
        assert len(matches) >= 1
        assert matches[0].script.id == "pytest_runner"

    def test_find_scripts_category_filter(self):
        """find_scripts should filter by category."""
        registry = ScriptRegistry()

        web_script = Script(
            id="fetch",
            description="Fetch URL",
            category="web",
            tags=["fetch"],
            parameters={},
        )
        code_script = Script(
            id="lint",
            description="Lint code",
            category="code",
            tags=["lint"],
            parameters={},
        )
        registry.register_script(web_script)
        registry.register_script(code_script)

        matches = registry.find_scripts("fetch", category="web")
        script_ids = [m.script.id for m in matches]
        assert "fetch" in script_ids
        assert "lint" not in script_ids

    def test_invoke_with_code(self):
        """invoke should execute embedded code."""
        registry = ScriptRegistry()

        script = Script(
            id="double",
            description="Double a value",
            category="data",
            tags=[],
            parameters={"value": {"type": "integer", "required": True}},
            code="result = args['value'] * 2",
        )
        registry.register_script(script)

        result = registry.invoke("double", value=5)
        assert result == 10

    def test_invoke_unknown_script_raises(self):
        """invoke with unknown script raises ValueError."""
        registry = ScriptRegistry()
        with pytest.raises(ValueError, match="Unknown script"):
            registry.invoke("nonexistent")

    def test_list_scripts(self):
        """list_scripts should return script info."""
        registry = ScriptRegistry()

        script = Script(
            id="test",
            description="Test script",
            category="test",
            tags=["a", "b"],
            parameters={},
            token_savings="90%",
        )
        registry.register_script(script)

        scripts = registry.list_scripts()
        assert len(scripts) == 1
        assert scripts[0]["id"] == "test"
        assert scripts[0]["token_savings"] == "90%"

    def test_list_scripts_with_tag_filter(self):
        """list_scripts should filter by tags."""
        registry = ScriptRegistry()

        script1 = Script(
            id="script1",
            description="Has tag_a",
            category="test",
            tags=["tag_a"],
            parameters={},
        )
        script2 = Script(
            id="script2",
            description="Has tag_b",
            category="test",
            tags=["tag_b"],
            parameters={},
        )
        registry.register_script(script1)
        registry.register_script(script2)

        scripts = registry.list_scripts(tags=["tag_a"])
        assert len(scripts) == 1
        assert scripts[0]["id"] == "script1"

    def test_get_categories(self):
        """get_categories should return all categories."""
        registry = ScriptRegistry()

        registry.register_script(
            Script(id="web1", description="W", category="web", tags=[], parameters={})
        )
        registry.register_script(
            Script(id="code1", description="C", category="code", tags=[], parameters={})
        )

        categories = registry.get_categories()
        assert "web" in categories
        assert "code" in categories


class TestScriptRegistryMCP:
    """Tests for MCP integration in ScriptRegistry."""

    def test_invoke_mcp_script(self):
        """_execute_mcp should delegate to call_mcp_tool."""
        from unittest.mock import patch
        from src.mcp_client import MCPServerConfig

        registry = ScriptRegistry()
        registry._mcp_configs = {
            "@anthropic/fetch": MCPServerConfig(
                name="@anthropic/fetch",
                command="npx",
                timeout=10,
            )
        }

        script = Script(
            id="fetch_docs",
            description="Fetch docs",
            category="web",
            tags=[],
            parameters={"url": {"type": "string", "required": True}},
            mcp_server="@anthropic/fetch",
            mcp_tool="fetch",
        )
        registry.register_script(script)

        with patch("src.mcp_client.call_mcp_tool", return_value="fetched content") as mock_call:
            result = registry.invoke("fetch_docs", url="https://example.com")

        assert result == "fetched content"
        mock_call.assert_called_once_with(
            registry._mcp_configs["@anthropic/fetch"],
            "fetch",
            {"url": "https://example.com"},
        )

    def test_invoke_mcp_unknown_server(self):
        """_execute_mcp with unknown server should raise RuntimeError."""
        registry = ScriptRegistry()
        registry._mcp_configs = {}  # No servers

        script = Script(
            id="bad_script",
            description="Bad",
            category="web",
            tags=[],
            parameters={},
            mcp_server="nonexistent-server",
        )
        registry.register_script(script)

        with pytest.raises(RuntimeError, match="Unknown MCP server"):
            registry.invoke("bad_script")


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_singleton(self):
        """get_registry should return same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
