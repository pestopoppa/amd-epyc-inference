#!/usr/bin/env python3
"""Read-only MCP server exposing orchestrator info to Claude Code.

Provides introspection tools for querying model configurations,
orchestrator roles, server status, and benchmark results.

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
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

mcp = FastMCP("orchestrator-info")


@mcp.tool()
def lookup_model(role: str) -> str:
    """Look up model config for an orchestrator role.

    Args:
        role: Role name (e.g., "coder_primary", "architect_general").

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
        return f"Role not found: {role}"
    except Exception as e:
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
