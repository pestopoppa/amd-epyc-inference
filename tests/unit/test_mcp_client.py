#!/usr/bin/env python3
"""Tests for the MCP client module."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from src.mcp_client import (
    MCPServerConfig,
    _call_async,
    _extract_text,
    call_mcp_tool,
    load_server_configs,
)


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_defaults(self):
        """Default values should be sensible."""
        config = MCPServerConfig(name="test", command="echo")
        assert config.args == []
        assert config.env is None
        assert config.cwd is None
        assert config.timeout == 30

    def test_all_fields(self):
        """All fields should be settable."""
        config = MCPServerConfig(
            name="orch",
            command="python3",
            args=["/path/to/server.py"],
            env={"PYTHONPATH": "/app"},
            cwd="/app",
            timeout=10,
        )
        assert config.name == "orch"
        assert config.command == "python3"
        assert config.args == ["/path/to/server.py"]
        assert config.env == {"PYTHONPATH": "/app"}
        assert config.cwd == "/app"
        assert config.timeout == 10


class TestLoadServerConfigs:
    """Tests for load_server_configs()."""

    def test_load_valid_yaml(self, tmp_path):
        """Should parse YAML and return MCPServerConfig objects."""
        config_data = {
            "servers": {
                "test-server": {
                    "command": "npx",
                    "args": ["-y", "@test/mcp-server"],
                    "timeout": 15,
                },
                "local-server": {
                    "command": "python3",
                    "args": ["/path/to/server.py"],
                    "env": {"KEY": "value"},
                    "cwd": "/tmp",
                    "timeout": 5,
                },
            }
        }

        yaml_path = tmp_path / "mcp_servers.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        configs = load_server_configs(yaml_path)

        assert len(configs) == 2
        assert "test-server" in configs
        assert "local-server" in configs

        ts = configs["test-server"]
        assert ts.name == "test-server"
        assert ts.command == "npx"
        assert ts.args == ["-y", "@test/mcp-server"]
        assert ts.timeout == 15
        assert ts.env is None

        ls = configs["local-server"]
        assert ls.env == {"KEY": "value"}
        assert ls.cwd == "/tmp"
        assert ls.timeout == 5

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing YAML file should return empty dict."""
        configs = load_server_configs(tmp_path / "nonexistent.yaml")
        assert configs == {}

    def test_empty_file_returns_empty(self, tmp_path):
        """Empty YAML file should return empty dict."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")

        configs = load_server_configs(yaml_path)
        assert configs == {}

    def test_no_servers_key_returns_empty(self, tmp_path):
        """YAML without 'servers' key should return empty dict."""
        yaml_path = tmp_path / "no_servers.yaml"
        yaml_path.write_text(yaml.dump({"other_key": "value"}))

        configs = load_server_configs(yaml_path)
        assert configs == {}

    def test_default_timeout(self, tmp_path):
        """Missing timeout should default to 30."""
        config_data = {
            "servers": {
                "minimal": {
                    "command": "echo",
                },
            }
        }

        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        configs = load_server_configs(yaml_path)
        assert configs["minimal"].timeout == 30
        assert configs["minimal"].args == []


class TestExtractText:
    """Tests for _extract_text()."""

    def test_extracts_text_content(self):
        """Should extract text from TextContent blocks."""
        from mcp.types import TextContent

        content = [
            TextContent(type="text", text="Hello"),
            TextContent(type="text", text="World"),
        ]
        result = _extract_text(content)
        assert result == "Hello\nWorld"

    def test_empty_content(self):
        """Empty content list should return empty string."""
        result = _extract_text([])
        assert result == ""

    def test_filters_non_text(self):
        """Should skip non-TextContent blocks."""
        from mcp.types import TextContent

        content = [
            TextContent(type="text", text="Keep this"),
            MagicMock(spec=[]),  # Not a TextContent
        ]
        result = _extract_text(content)
        assert result == "Keep this"


class TestCallMCPTool:
    """Tests for call_mcp_tool() and _call_async()."""

    @patch("mcp.client.stdio.stdio_client")
    def test_success(self, mock_stdio_client):
        """Successful MCP tool call should return text result."""
        from mcp.types import TextContent

        # Build mock result
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [TextContent(type="text", text="result text")]

        # Build mock session
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        # Build mock context managers
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_rw = (MagicMock(), MagicMock())
        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_rw)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        mock_stdio_client.return_value = mock_client_cm

        config = MCPServerConfig(name="test", command="echo", timeout=5)

        with patch("mcp.ClientSession", return_value=mock_session_cm):
            result = call_mcp_tool(config, "my_tool", {"key": "value"})

        assert result == "result text"

    @patch("mcp.client.stdio.stdio_client")
    def test_error_result_raises(self, mock_stdio_client):
        """MCP tool returning isError=True should raise RuntimeError."""
        from mcp.types import TextContent

        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = [TextContent(type="text", text="tool failed")]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_rw = (MagicMock(), MagicMock())
        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_rw)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        mock_stdio_client.return_value = mock_client_cm

        config = MCPServerConfig(name="test", command="echo", timeout=5)

        with patch("mcp.ClientSession", return_value=mock_session_cm):
            with pytest.raises(RuntimeError, match="returned error"):
                call_mcp_tool(config, "failing_tool", {})

    def test_server_not_found_raises(self):
        """Non-existent command should raise FileNotFoundError."""
        config = MCPServerConfig(
            name="bad",
            command="/nonexistent/binary",
            timeout=3,
        )

        with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
            call_mcp_tool(config, "any_tool", {})
