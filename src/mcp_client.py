#!/usr/bin/env python3
"""Shared MCP client for tool and script registries.

Provides configuration loading and a synchronous wrapper around the
async MCP client SDK. Both ToolRegistry._invoke_mcp() and
ScriptRegistry._execute_mcp() delegate here.

Usage:
    from src.mcp_client import load_server_configs, call_mcp_tool

    configs = load_server_configs("orchestration/mcp_servers.yaml")
    result = call_mcp_tool(configs["orchestrator"], "list_roles", {})
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml

from src.config import _registry_timeout

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for launching an MCP server subprocess.

    Timeout default from model_registry.yaml (runtime_defaults.timeouts.external.mcp_client).
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    timeout: int = field(
        default_factory=lambda: int(_registry_timeout("external", "mcp_client", 30))
    )


def load_server_configs(yaml_path: str | Path) -> dict[str, MCPServerConfig]:
    """Parse mcp_servers.yaml into a dict of server configs.

    Args:
        yaml_path: Path to the YAML config file.

    Returns:
        Mapping of server identifier to MCPServerConfig.
        Returns empty dict if the file doesn't exist.
    """
    path = Path(yaml_path)
    if not path.exists():
        logger.warning(f"MCP server config not found: {path}")
        return {}

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data or "servers" not in data:
        return {}

    configs: dict[str, MCPServerConfig] = {}
    for server_id, spec in data["servers"].items():
        configs[server_id] = MCPServerConfig(
            name=server_id,
            command=spec.get("command", ""),
            args=spec.get("args", []),
            env=spec.get("env"),
            cwd=spec.get("cwd"),
            timeout=spec.get("timeout", 30),
        )

    logger.debug(f"Loaded {len(configs)} MCP server configs from {path}")
    return configs


def _extract_text(content: list[Any]) -> str:
    """Extract text from MCP CallToolResult content blocks.

    Args:
        content: List of content objects from CallToolResult.

    Returns:
        Concatenated text from TextContent blocks.
    """
    from mcp.types import TextContent

    parts = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "\n".join(parts)


async def _call_async(
    config: MCPServerConfig,
    tool_name: str,
    args: dict[str, Any] | None,
) -> str:
    """Async implementation of MCP tool call.

    Spawns the MCP server as a subprocess, initializes a session,
    calls the tool, and returns the text result.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Build environment: inherit current env, overlay config env
    env = dict(os.environ)
    if config.env:
        env.update(config.env)

    params = StdioServerParameters(
        command=config.command,
        args=config.args,
        env=env,
        cwd=config.cwd,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                name=tool_name,
                arguments=args,
                read_timeout_seconds=timedelta(seconds=config.timeout),
            )

            if result.isError:
                text = _extract_text(result.content) if result.content else "Unknown error"
                raise RuntimeError(f"MCP tool '{tool_name}' returned error: {text}")

            return _extract_text(result.content) if result.content else ""


def call_mcp_tool(
    config: MCPServerConfig,
    tool_name: str,
    args: dict[str, Any] | None = None,
) -> str:
    """Call an MCP tool synchronously.

    Handles the sync/async bridge: uses asyncio.run() when no event loop
    is running, or a ThreadPoolExecutor when called from within an
    existing event loop (e.g., FastAPI).

    Args:
        config: Server configuration.
        tool_name: Name of the tool to call on the MCP server.
        args: Arguments to pass to the tool.

    Returns:
        Text result from the tool.

    Raises:
        RuntimeError: If the tool returns an error or the server can't be reached.
        FileNotFoundError: If the server command is not found.
    """
    try:
        asyncio.get_running_loop()
        # Already in async context — run in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _call_async(config, tool_name, args))
            return future.result(timeout=config.timeout + 5)
    except RuntimeError:
        # No event loop running — safe to use asyncio.run()
        return asyncio.run(_call_async(config, tool_name, args))
