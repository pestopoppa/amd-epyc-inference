"""Tests for tool_loader module."""

from __future__ import annotations

import json

import pytest

from src.tool_loader import (
    MANIFEST_SCHEMA_VERSION,
    ToolDefinition,
    ToolManifest,
    ToolPluginLoader,
    ToolSettings,
    create_manifest_template,
    parse_manifest,
    validate_manifest,
)


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_basic_definition(self):
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            module="src.tools.test",
            function="test_func",
        )
        assert tool.name == "test_tool"
        assert tool.category == "specialized"
        assert tool.enabled is True

    def test_to_dict(self):
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            module="src.tools.test",
            function="test_func",
            category="data",
            parameters={"arg": {"type": "string"}},
        )
        result = tool.to_dict()

        assert result["name"] == "test_tool"
        assert result["module"] == "src.tools.test"
        assert result["category"] == "data"
        assert "arg" in result["parameters"]


class TestToolManifest:
    """Tests for ToolManifest dataclass."""

    def test_basic_manifest(self):
        manifest = ToolManifest(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
        )
        assert manifest.name == "test_plugin"
        assert manifest.enabled is True
        assert manifest.tools == []

    def test_to_dict(self):
        manifest = ToolManifest(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            tools=[
                ToolDefinition(
                    name="tool1",
                    description="Tool 1",
                    module="mod",
                    function="func",
                )
            ],
        )
        result = manifest.to_dict()

        assert result["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert result["name"] == "test_plugin"
        assert len(result["tools"]) == 1


class TestToolSettings:
    """Tests for ToolSettings dataclass."""

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", tmp_path)

        settings = ToolSettings.load("nonexistent_plugin")

        assert settings.plugin_name == "nonexistent_plugin"
        assert settings.enabled is True

    def test_load_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", tmp_path)

        settings_file = tmp_path / "test_plugin.json"
        settings_file.write_text(json.dumps({
            "enabled": False,
            "tool_overrides": {"tool1": {"enabled": False}},
            "custom_config": {"key": "value"},
        }))

        settings = ToolSettings.load("test_plugin")

        assert settings.enabled is False
        assert settings.tool_overrides["tool1"]["enabled"] is False
        assert settings.custom_config["key"] == "value"

    def test_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", tmp_path)

        settings = ToolSettings(
            plugin_name="test_plugin",
            enabled=False,
            custom_config={"test": True},
        )
        settings.save()

        saved_file = tmp_path / "test_plugin.json"
        assert saved_file.exists()

        loaded = json.loads(saved_file.read_text())
        assert loaded["enabled"] is False
        assert loaded["custom_config"]["test"] is True


class TestValidateManifest:
    """Tests for validate_manifest function."""

    def test_valid_manifest(self):
        data = {
            "name": "test",
            "version": "1.0.0",
            "description": "Test plugin",
            "tools": [
                {
                    "name": "tool1",
                    "description": "Tool 1",
                    "module": "mod",
                    "function": "func",
                }
            ],
        }
        errors = validate_manifest(data)
        assert errors == []

    def test_missing_required_fields(self):
        data = {"version": "1.0.0"}
        errors = validate_manifest(data)

        assert any("name" in e for e in errors)
        assert any("description" in e for e in errors)

    def test_invalid_tools_type(self):
        data = {
            "name": "test",
            "version": "1.0.0",
            "description": "Test",
            "tools": "not a list",
        }
        errors = validate_manifest(data)
        assert any("must be a list" in e for e in errors)

    def test_tool_missing_fields(self):
        data = {
            "name": "test",
            "version": "1.0.0",
            "description": "Test",
            "tools": [{"name": "tool1"}],
        }
        errors = validate_manifest(data)

        assert any("description" in e for e in errors)
        assert any("module" in e for e in errors)
        assert any("function" in e for e in errors)


class TestParseManifest:
    """Tests for parse_manifest function."""

    def test_parse_valid_manifest(self, tmp_path):
        manifest_data = {
            "name": "test_plugin",
            "version": "1.0.0",
            "description": "Test plugin",
            "tools": [
                {
                    "name": "tool1",
                    "description": "Tool 1",
                    "module": "src.tools.test",
                    "function": "test_func",
                    "category": "data",
                }
            ],
            "dependencies": ["dep1"],
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data))

        manifest = parse_manifest(manifest_file)

        assert manifest.name == "test_plugin"
        assert len(manifest.tools) == 1
        assert manifest.tools[0].name == "tool1"
        assert manifest.tools[0].category == "data"
        assert manifest.dependencies == ["dep1"]
        assert manifest.path == manifest_file

    def test_parse_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_manifest(tmp_path / "nonexistent.json")

    def test_parse_invalid_manifest(self, tmp_path):
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps({"invalid": "data"}))

        with pytest.raises(ValueError):
            parse_manifest(manifest_file)


class TestToolPluginLoader:
    """Tests for ToolPluginLoader class."""

    @pytest.fixture
    def plugin_dir(self, tmp_path):
        """Create a mock plugin directory structure."""
        # Plugin 1
        plugin1_dir = tmp_path / "plugin1"
        plugin1_dir.mkdir()
        (plugin1_dir / "manifest.json").write_text(json.dumps({
            "name": "plugin1",
            "version": "1.0.0",
            "description": "Plugin 1",
            "tools": [
                {
                    "name": "tool1",
                    "description": "Tool 1",
                    "module": "src.tools.plugin1.impl",
                    "function": "tool1_func",
                }
            ],
        }))

        # Plugin 2
        plugin2_dir = tmp_path / "plugin2"
        plugin2_dir.mkdir()
        (plugin2_dir / "manifest.json").write_text(json.dumps({
            "name": "plugin2",
            "version": "2.0.0",
            "description": "Plugin 2",
            "enabled": False,
            "tools": [
                {
                    "name": "tool2",
                    "description": "Tool 2",
                    "module": "src.tools.plugin2.impl",
                    "function": "tool2_func",
                }
            ],
        }))

        return tmp_path

    def test_discover_plugins(self, plugin_dir, monkeypatch):
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", plugin_dir / "settings")

        loader = ToolPluginLoader()
        count = loader.discover_plugins(plugin_dir)

        assert count == 2
        assert "plugin1" in loader.plugins
        assert "plugin2" in loader.plugins

    def test_list_tools_enabled_only(self, plugin_dir, monkeypatch):
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", plugin_dir / "settings")

        loader = ToolPluginLoader()
        loader.discover_plugins(plugin_dir)

        tools = loader.list_tools(enabled_only=True)

        # Only plugin1's tools should be listed (plugin2 is disabled)
        assert len(tools) == 1
        assert tools[0]["name"] == "tool1"

    def test_list_tools_all(self, plugin_dir, monkeypatch):
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", plugin_dir / "settings")

        loader = ToolPluginLoader()
        loader.discover_plugins(plugin_dir)

        tools = loader.list_tools(enabled_only=False)

        assert len(tools) == 2

    def test_check_for_changes(self, plugin_dir, monkeypatch):
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", plugin_dir / "settings")

        loader = ToolPluginLoader()
        loader.discover_plugins(plugin_dir)

        # No changes initially
        changed = loader.check_for_changes()
        assert changed == []

        # Modify manifest
        import time
        time.sleep(0.1)  # Ensure mtime changes
        manifest_path = plugin_dir / "plugin1" / "manifest.json"
        manifest_path.write_text(manifest_path.read_text())  # Touch file

        changed = loader.check_for_changes()
        assert "plugin1" in changed

    def test_reload_plugin(self, plugin_dir, monkeypatch):
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", plugin_dir / "settings")

        loader = ToolPluginLoader()
        loader.discover_plugins(plugin_dir)

        # Modify manifest
        manifest_path = plugin_dir / "plugin1" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        data["version"] = "1.1.0"
        manifest_path.write_text(json.dumps(data))

        result = loader.reload_plugin("plugin1")

        assert result is True
        assert loader.plugins["plugin1"].version == "1.1.0"

    def test_get_settings(self, plugin_dir, monkeypatch):
        settings_dir = plugin_dir / "settings"
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", settings_dir)

        loader = ToolPluginLoader()
        loader.discover_plugins(plugin_dir)

        settings = loader.get_settings("plugin1")

        assert settings.plugin_name == "plugin1"
        assert settings.enabled is True

    def test_update_settings(self, plugin_dir, monkeypatch):
        settings_dir = plugin_dir / "settings"
        monkeypatch.setattr("src.tool_loader.SETTINGS_DIR", settings_dir)

        loader = ToolPluginLoader()
        loader.discover_plugins(plugin_dir)

        loader.update_settings("plugin1", enabled=False, custom_config={"key": "value"})

        settings = loader.get_settings("plugin1")
        assert settings.enabled is False
        assert settings.custom_config["key"] == "value"

        # Check saved to file
        settings_file = settings_dir / "plugin1.json"
        assert settings_file.exists()


class TestCreateManifestTemplate:
    """Tests for create_manifest_template function."""

    def test_basic_template(self):
        template = create_manifest_template(
            name="new_plugin",
            description="A new plugin",
        )

        assert template["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert template["name"] == "new_plugin"
        assert template["description"] == "A new plugin"
        assert template["version"] == "1.0.0"
        assert template["enabled"] is True
        assert template["tools"] == []

    def test_template_with_tools(self):
        template = create_manifest_template(
            name="plugin",
            description="Plugin",
            tools=[
                {
                    "name": "tool1",
                    "description": "Tool",
                    "module": "mod",
                    "function": "func",
                }
            ],
        )

        assert len(template["tools"]) == 1
        assert template["tools"][0]["name"] == "tool1"
