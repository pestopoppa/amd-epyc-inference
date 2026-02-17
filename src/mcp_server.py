#!/usr/bin/env python3
"""Read-only MCP server exposing orchestrator info to Claude Code.

Provides introspection tools for querying model configurations,
orchestrator roles, server status, benchmark results, and canvas operations.

Uses plugin-based tool loading with hot-reload support.

Usage (stdio transport, launched by Claude Code):
    python src/mcp_server.py

Configuration (.mcp.json):
    {
      "mcpServers": {
        "orchestrator": {
          "command": "python",
          "args": ["src/mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

mcp = FastMCP("orchestrator-info")

# Plugin loader for dynamic tool discovery
_plugin_loader = None


def _get_plugin_loader():
    """Get or create the plugin loader."""
    global _plugin_loader
    if _plugin_loader is None:
        from src.tool_loader import ToolPluginLoader

        _plugin_loader = ToolPluginLoader()
        tools_dir = PROJECT_ROOT / "src" / "tools"
        count = _plugin_loader.discover_plugins(tools_dir)
        logger.info("Discovered %d plugins from %s", count, tools_dir)
    return _plugin_loader


@mcp.tool()
def lookup_model(role: str) -> str:
    """Look up model config for an orchestrator role.

    Args:
        role: Role name (e.g., "coder_escalation", "architect_general").

    Returns:
        Formatted string with role configuration details.
    """
    try:
        from src.registry_loader import RegistryLoader

        registry = RegistryLoader(validate_paths=False)
        role_config = registry.get_role(role)

        speed = role_config.performance.optimized_tps or role_config.performance.baseline_tps or "?"
        speedup = role_config.performance.speedup or "N/A"

        lines = [
            f"Role: {role_config.name}",
            f"Tier: {role_config.tier}",
            f"Description: {role_config.description}",
            f"Model: {role_config.model.name}",
            f"Quant: {role_config.model.quant}",
            f"Size: {role_config.model.size_gb} GB",
            f"Acceleration: {role_config.acceleration.type}",
            f"Speed: {speed} t/s",
            f"Speedup: {speedup}",
        ]

        if role_config.constraints and role_config.constraints.forbid:
            lines.append(f"Forbidden: {', '.join(role_config.constraints.forbid)}")
        if role_config.notes:
            lines.append(f"Notes: {role_config.notes}")

        return "\n".join(lines)

    except KeyError:
        logger.debug("Role not found: %s", role)
        return f"Role not found: {role}"
    except Exception as e:
        logger.warning("Error loading role '%s': %s: %s", role, type(e).__name__, e)
        return f"Error loading role '{role}': {type(e).__name__}: {e}"


@mcp.tool()
def list_roles() -> str:
    """List all configured orchestrator roles by tier.

    Returns:
        Formatted string with all roles grouped by tier.
    """
    try:
        from src.registry_loader import RegistryLoader

        registry = RegistryLoader(validate_paths=False)
        lines = []

        for tier in ["A", "B", "C", "D"]:
            roles = registry.get_roles_by_tier(tier)
            if roles:
                lines.append(f"\n--- Tier {tier} ---")
                for r in roles:
                    speed = r.performance.optimized_tps or r.performance.baseline_tps or "?"
                    accel = r.acceleration.type
                    lines.append(f"  {r.name}: {r.model.name} ({accel}, {speed} t/s)")

        return "\n".join(lines) if lines else "No roles configured."

    except Exception as e:
        logger.warning("Error listing roles: %s: %s", type(e).__name__, e)
        return f"Error listing roles: {type(e).__name__}: {e}"


@mcp.tool()
def server_status() -> str:
    """Get current status of all orchestrator services.

    Returns:
        Formatted string with running service information.
    """
    state_file = PROJECT_ROOT / "logs" / "orchestrator_state.json"

    if not state_file.exists():
        return "No orchestrator state file found. Stack may not be running."

    try:
        state = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Error reading state file %s: %s", state_file, e)
        return f"Error reading state file: {e}"

    if not state:
        return "No services running."

    lines = []
    for name, info in state.items():
        if isinstance(info, dict):
            pid = info.get("pid", "?")
            port = info.get("port", "?")
            started = info.get("started_at", "?")
            lines.append(f"{name}: PID {pid} on port {port} (started {started})")
        else:
            lines.append(f"{name}: {info}")

    return "\n".join(lines)


@mcp.tool()
def query_benchmarks(model_name: str = "", suite: str = "") -> str:
    """Query benchmark results from the summary CSV.

    Args:
        model_name: Filter by model name (case-insensitive substring match).
                    Empty string returns all models.
        suite: Filter by suite name (e.g., "thinking", "coder", "math").
               Empty string returns all suites.

    Returns:
        Formatted benchmark results.
    """
    csv_path = PROJECT_ROOT / "benchmarks" / "results" / "reviews" / "summary.csv"

    if not csv_path.exists():
        return "No benchmark summary found."

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except (OSError, csv.Error) as e:
        logger.warning("Error reading benchmark CSV %s: %s", csv_path, e)
        return f"Error reading benchmark CSV: {e}"

    # Filter
    if model_name:
        model_lower = model_name.lower()
        rows = [r for r in rows if model_lower in r.get("model", "").lower()]

    if not rows:
        return f"No results matching model='{model_name}'"

    # Format output
    lines = []
    for r in rows:
        model = r.get("model", "?")
        pct = r.get("pct_str", "?")
        tps = r.get("avg_tps", "?")

        if suite and suite in r:
            score = r.get(suite, "N/A")
            lines.append(f"{model}: {suite}={score}, overall={pct} ({tps} t/s)")
        else:
            lines.append(f"{model}: {pct} ({tps} t/s)")

    return "\n".join(lines)


# Canvas tools - integrated via plugin system
@mcp.tool()
def export_reasoning_canvas(
    graph_type: str = "hypothesis",
    include_evidence: bool = True,
    output_path: str = "",
) -> str:
    """Export a reasoning graph to JSON Canvas format.

    Args:
        graph_type: Type of graph: 'hypothesis', 'failure', or 'session'
        include_evidence: Whether to include evidence/symptom nodes
        output_path: Optional custom output path

    Returns:
        Path to the created canvas file, or error message.
    """
    try:
        from src.tools.canvas_tools import export_reasoning_canvas as _export

        return _export(
            graph_type=graph_type,  # type: ignore
            include_evidence=include_evidence,
            output_path=output_path or None,
        )
    except Exception as e:
        logger.exception("Canvas export failed: %s", e)
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
def import_canvas_edits(canvas_path: str, baseline_path: str = "") -> str:
    """Import an edited canvas and extract planning constraints.

    Args:
        canvas_path: Path to the edited canvas file
        baseline_path: Optional path to baseline canvas for diff

    Returns:
        JSON-encoded constraints extracted from the canvas.
    """
    try:
        from src.tools.canvas_tools import import_canvas_edits as _import

        return _import(canvas_path, baseline_path or None)
    except Exception as e:
        logger.exception("Canvas import failed: %s", e)
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})


@mcp.tool()
def list_canvases(directory: str = "") -> str:
    """List available canvas files.

    Args:
        directory: Directory to search. Defaults to logs/canvases.

    Returns:
        Formatted list of canvas files with metadata.
    """
    try:
        from src.tools.canvas_tools import list_canvases as _list

        return _list(directory or None)
    except Exception as e:
        logger.exception("Canvas list failed: %s", e)
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})


# Plugin management tools
@mcp.tool()
def list_plugins(enabled_only: bool = True) -> str:
    """List all available tool plugins.

    Args:
        enabled_only: Only show enabled plugins

    Returns:
        JSON-formatted list of plugins and their tools.
    """
    try:
        loader = _get_plugin_loader()
        tools = loader.list_tools(enabled_only=enabled_only)

        # Group by plugin
        by_plugin: dict[str, list] = {}
        for tool in tools:
            plugin = tool.get("plugin", "unknown")
            by_plugin.setdefault(plugin, []).append(tool)

        result = {
            "status": "success",
            "plugin_count": len(by_plugin),
            "tool_count": len(tools),
            "plugins": {
                name: {
                    "tools": [t["name"] for t in tools_list],
                    "enabled": all(t.get("enabled", True) for t in tools_list),
                }
                for name, tools_list in by_plugin.items()
            },
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception("List plugins failed: %s", e)
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})


@mcp.tool()
def reload_plugins() -> str:
    """Reload any changed plugins (hot-reload).

    Returns:
        Status message with number of plugins reloaded.
    """
    try:
        loader = _get_plugin_loader()
        count = loader.reload_changed()
        return json.dumps({
            "status": "success",
            "reloaded": count,
            "message": f"Reloaded {count} plugin(s)",
        })
    except Exception as e:
        logger.exception("Reload plugins failed: %s", e)
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    # Initialize plugin loader on startup
    _get_plugin_loader()
    mcp.run(transport="stdio")
