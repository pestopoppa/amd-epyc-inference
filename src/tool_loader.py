"""Plugin-based tool loading with hot-reload support.

Discovers and loads tools from directory-based plugins with manifest files.
Each plugin directory contains:
- manifest.json: Plugin metadata, tool list, configuration
- Python modules with tool implementations

Features:
- Automatic discovery of plugins in src/tools/
- Hot-reload when manifest files change
- Per-tool settings in src/tool_settings/ (gitignored)
- Validation of manifest schema
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Schema version for manifests
MANIFEST_SCHEMA_VERSION = "1.0"

# Default tools directory
DEFAULT_TOOLS_DIR = Path(__file__).parent / "tools"

# Settings directory (gitignored)
SETTINGS_DIR = Path(__file__).parent / "tool_settings"


@dataclass
class ToolDefinition:
    """A tool defined in a manifest."""

    name: str
    description: str
    module: str
    function: str
    category: str = "specialized"
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "module": self.module,
            "function": self.function,
            "category": self.category,
            "parameters": self.parameters,
            "enabled": self.enabled,
        }


@dataclass
class ToolManifest:
    """Manifest describing a tool plugin."""

    name: str
    version: str
    description: str
    tools: list[ToolDefinition] = field(default_factory=list)
    enabled: bool = True
    dependencies: list[str] = field(default_factory=list)
    settings_schema: dict[str, Any] = field(default_factory=dict)

    # Internal fields
    path: Path | None = None
    last_modified: float = 0.0
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled,
            "dependencies": self.dependencies,
            "settings_schema": self.settings_schema,
            "tools": [t.to_dict() for t in self.tools],
        }


@dataclass
class ToolSettings:
    """Per-plugin settings loaded from tool_settings/."""

    plugin_name: str
    enabled: bool = True
    tool_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    custom_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, plugin_name: str) -> "ToolSettings":
        """Load settings for a plugin from file.

        Args:
            plugin_name: Name of the plugin

        Returns:
            ToolSettings instance (defaults if file doesn't exist)
        """
        settings_file = SETTINGS_DIR / f"{plugin_name}.json"
        if not settings_file.exists():
            return cls(plugin_name=plugin_name)

        try:
            data = json.loads(settings_file.read_text())
            return cls(
                plugin_name=plugin_name,
                enabled=data.get("enabled", True),
                tool_overrides=data.get("tool_overrides", {}),
                custom_config=data.get("custom_config", {}),
            )
        except Exception as e:
            logger.warning("Failed to load settings for %s: %s", plugin_name, e)
            return cls(plugin_name=plugin_name)

    def save(self) -> None:
        """Save settings to file."""
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        settings_file = SETTINGS_DIR / f"{self.plugin_name}.json"
        data = {
            "enabled": self.enabled,
            "tool_overrides": self.tool_overrides,
            "custom_config": self.custom_config,
        }
        settings_file.write_text(json.dumps(data, indent=2))


def validate_manifest(data: dict[str, Any]) -> list[str]:
    """Validate a manifest dictionary against the schema.

    Args:
        data: Manifest data to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    for required in ("name", "version", "description"):
        if required not in data:
            errors.append(f"Missing required field: {required}")

    # Validate tools list
    tools = data.get("tools", [])
    if not isinstance(tools, list):
        errors.append("'tools' must be a list")
    else:
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                errors.append(f"Tool {i}: must be a dict")
                continue

            for required in ("name", "description", "module", "function"):
                if required not in tool:
                    errors.append(f"Tool {i}: missing '{required}'")

            # Validate parameters
            params = tool.get("parameters", {})
            if not isinstance(params, dict):
                errors.append(f"Tool {i}: 'parameters' must be a dict")

    # Validate dependencies
    deps = data.get("dependencies", [])
    if not isinstance(deps, list):
        errors.append("'dependencies' must be a list")

    return errors


def parse_manifest(path: Path) -> ToolManifest:
    """Parse a manifest file.

    Args:
        path: Path to manifest.json

    Returns:
        ToolManifest instance

    Raises:
        ValueError: If manifest is invalid
        FileNotFoundError: If manifest doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    content = path.read_text()
    data = json.loads(content)

    errors = validate_manifest(data)
    if errors:
        raise ValueError(f"Invalid manifest {path}: {'; '.join(errors)}")

    # Parse tools
    tools = []
    for tool_data in data.get("tools", []):
        tools.append(
            ToolDefinition(
                name=tool_data["name"],
                description=tool_data["description"],
                module=tool_data["module"],
                function=tool_data["function"],
                category=tool_data.get("category", "specialized"),
                parameters=tool_data.get("parameters", {}),
                enabled=tool_data.get("enabled", True),
            )
        )

    manifest = ToolManifest(
        name=data["name"],
        version=data["version"],
        description=data["description"],
        tools=tools,
        enabled=data.get("enabled", True),
        dependencies=data.get("dependencies", []),
        settings_schema=data.get("settings_schema", {}),
        path=path,
        last_modified=path.stat().st_mtime,
        content_hash=hashlib.md5(content.encode()).hexdigest(),
    )

    return manifest


class ToolPluginLoader:
    """Loads and manages tool plugins with hot-reload support.

    Usage:
        loader = ToolPluginLoader()
        loader.discover_plugins(Path("src/tools"))

        # Check for changes and reload
        if loader.check_for_changes():
            loader.reload_changed()

        # Get all enabled tools
        for plugin in loader.plugins.values():
            if plugin.enabled:
                for tool in plugin.tools:
                    print(f"{tool.name}: {tool.description}")
    """

    def __init__(self):
        """Initialize the plugin loader."""
        self.plugins: dict[str, ToolManifest] = {}
        self._handlers: dict[str, Callable] = {}
        self._settings: dict[str, ToolSettings] = {}

    def discover_plugins(self, tools_dir: Path = DEFAULT_TOOLS_DIR) -> int:
        """Discover and load all plugins in a directory.

        Args:
            tools_dir: Directory containing plugin subdirectories

        Returns:
            Number of plugins loaded
        """
        if not tools_dir.exists():
            logger.warning("Tools directory not found: %s", tools_dir)
            return 0

        count = 0
        for subdir in tools_dir.iterdir():
            if not subdir.is_dir():
                continue

            manifest_path = subdir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                manifest = parse_manifest(manifest_path)
                self.plugins[manifest.name] = manifest

                # Load settings
                self._settings[manifest.name] = ToolSettings.load(manifest.name)

                # Apply settings overrides
                if not self._settings[manifest.name].enabled:
                    manifest.enabled = False

                logger.info("Loaded plugin: %s v%s (%d tools)", manifest.name, manifest.version, len(manifest.tools))
                count += 1

            except (ValueError, json.JSONDecodeError) as e:
                logger.warning("Failed to load plugin from %s: %s", subdir, e)
                continue

        return count

    def check_for_changes(self) -> list[str]:
        """Check if any manifest files have changed.

        Returns:
            List of plugin names that have changed
        """
        changed = []

        for name, manifest in self.plugins.items():
            if manifest.path is None:
                continue

            if not manifest.path.exists():
                # Plugin was removed
                changed.append(name)
                continue

            current_mtime = manifest.path.stat().st_mtime
            if current_mtime > manifest.last_modified:
                changed.append(name)

        return changed

    def reload_plugin(self, name: str) -> bool:
        """Reload a specific plugin.

        Args:
            name: Plugin name to reload

        Returns:
            True if reload succeeded
        """
        if name not in self.plugins:
            logger.warning("Plugin not found: %s", name)
            return False

        old_manifest = self.plugins[name]
        if old_manifest.path is None:
            return False

        try:
            new_manifest = parse_manifest(old_manifest.path)
            self.plugins[name] = new_manifest

            # Clear cached handlers for this plugin's tools
            for tool in old_manifest.tools:
                self._handlers.pop(tool.name, None)

            logger.info("Reloaded plugin: %s", name)
            return True

        except Exception as e:
            logger.error("Failed to reload plugin %s: %s", name, e)
            return False

    def reload_changed(self) -> int:
        """Reload all changed plugins.

        Returns:
            Number of plugins reloaded
        """
        changed = self.check_for_changes()
        count = 0

        for name in changed:
            if self.reload_plugin(name):
                count += 1

        return count

    def get_handler(self, tool_name: str) -> Callable | None:
        """Get the handler function for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Handler function or None if not found/loadable
        """
        if tool_name in self._handlers:
            return self._handlers[tool_name]

        # Find the tool
        tool_def = None
        plugin_name = None
        for name, manifest in self.plugins.items():
            for tool in manifest.tools:
                if tool.name == tool_name:
                    tool_def = tool
                    plugin_name = name
                    break
            if tool_def:
                break

        if tool_def is None:
            logger.debug("Tool not found: %s", tool_name)
            return None

        # Check if enabled
        settings = self._settings.get(plugin_name, ToolSettings(plugin_name=plugin_name or ""))
        tool_override = settings.tool_overrides.get(tool_name, {})
        if not tool_override.get("enabled", tool_def.enabled):
            logger.debug("Tool disabled: %s", tool_name)
            return None

        # Import and cache handler
        try:
            module = importlib.import_module(tool_def.module)
            handler = getattr(module, tool_def.function)
            self._handlers[tool_name] = handler
            return handler

        except (ImportError, AttributeError) as e:
            logger.warning("Failed to load handler for %s: %s", tool_name, e)
            return None

    def list_tools(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """List all available tools.

        Args:
            enabled_only: Only include enabled tools

        Returns:
            List of tool info dicts
        """
        tools = []

        for name, manifest in self.plugins.items():
            if enabled_only and not manifest.enabled:
                continue

            settings = self._settings.get(name, ToolSettings(plugin_name=name))

            for tool in manifest.tools:
                tool_override = settings.tool_overrides.get(tool.name, {})
                tool_enabled = tool_override.get("enabled", tool.enabled)

                if enabled_only and not tool_enabled:
                    continue

                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "plugin": name,
                    "enabled": tool_enabled,
                    "parameters": tool.parameters,
                })

        return tools

    def get_settings(self, plugin_name: str) -> ToolSettings:
        """Get settings for a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            ToolSettings instance
        """
        if plugin_name not in self._settings:
            self._settings[plugin_name] = ToolSettings.load(plugin_name)
        return self._settings[plugin_name]

    def update_settings(self, plugin_name: str, **kwargs) -> None:
        """Update settings for a plugin.

        Args:
            plugin_name: Name of the plugin
            **kwargs: Settings to update (enabled, tool_overrides, custom_config)
        """
        settings = self.get_settings(plugin_name)

        if "enabled" in kwargs:
            settings.enabled = kwargs["enabled"]
        if "tool_overrides" in kwargs:
            settings.tool_overrides.update(kwargs["tool_overrides"])
        if "custom_config" in kwargs:
            settings.custom_config.update(kwargs["custom_config"])

        settings.save()


def create_manifest_template(
    name: str,
    description: str,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a manifest template for a new plugin.

    Args:
        name: Plugin name
        description: Plugin description
        tools: Optional list of tool definitions

    Returns:
        Manifest dictionary
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": name,
        "version": "1.0.0",
        "description": description,
        "enabled": True,
        "dependencies": [],
        "settings_schema": {},
        "tools": tools or [],
    }
