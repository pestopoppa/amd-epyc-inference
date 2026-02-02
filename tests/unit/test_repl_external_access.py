#!/usr/bin/env python3
"""Unit tests for REPL external access tools (_ExternalAccessMixin)."""

import subprocess
from unittest.mock import patch, MagicMock
import pytest

from src.repl_environment import REPLConfig, REPLEnvironment, ExecutionResult


class TestWebFetchTool:
    """Test the web_fetch() tool (_web_fetch)."""

    @patch('requests.get')
    def test_web_fetch_successful_response(self, mock_get):
        """Test web_fetch with successful response."""
        mock_response = MagicMock()
        mock_response.text = "This is the fetched content"
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = web_fetch("https://example.com")')

        assert result.error is None
        assert "This is the fetched content" in repl.artifacts["result"]

    @patch('requests.get')
    def test_web_fetch_html_content(self, mock_get):
        """Test web_fetch handles HTML content (extracts text)."""
        html_content = """
        <html>
            <head><title>Test</title></head>
            <body>
                <nav>Navigation</nav>
                <p>Main content here</p>
                <script>alert('test');</script>
                <footer>Footer</footer>
            </body>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = web_fetch("https://example.com")')

        assert result.error is None
        # Should extract main content - BeautifulSoup may or may not strip scripts depending on availability
        assert "Main content here" in repl.artifacts["result"]

    def test_web_fetch_rejects_non_http_urls(self):
        """Test web_fetch rejects non-http URLs."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = web_fetch("file:///etc/passwd")')

        assert result.error is None
        assert "[ERROR: Only http/https URLs are allowed]" in repl.artifacts["result"]

    @patch('requests.get')
    def test_web_fetch_handles_timeout(self, mock_get):
        """Test web_fetch handles timeout error."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = web_fetch("https://example.com")')

        assert result.error is None
        assert "[ERROR: Request timed out after 30s]" in repl.artifacts["result"]

    @patch('requests.get')
    def test_web_fetch_handles_request_error(self, mock_get):
        """Test web_fetch handles request error."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = web_fetch("https://example.com")')

        assert result.error is None
        assert "[ERROR: Request failed:" in repl.artifacts["result"]

    @patch('requests.get')
    def test_web_fetch_truncates_at_max_chars(self, mock_get):
        """Test web_fetch truncates at max_chars."""
        long_content = "X" * 15000
        mock_response = MagicMock()
        mock_response.text = long_content
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = web_fetch("https://example.com", max_chars=10000)')

        assert result.error is None
        assert "truncated at 10000 chars" in repl.artifacts["result"]


class TestRunShellTool:
    """Test the run_shell() tool (_run_shell)."""

    def test_run_shell_safe_command(self):
        """Test run_shell with safe command (e.g. echo) captures output."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = run_shell("echo hello")')

        assert result.error is None
        assert "hello" in repl.artifacts["result"]

    def test_run_shell_blocked_command(self):
        """Test run_shell with blocked command (e.g. rm) returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = run_shell("rm -rf /")')

        assert result.error is None
        assert "[ERROR: Command 'rm' is blocked for security]" in repl.artifacts["result"]

    def test_run_shell_unlisted_command(self):
        """Test run_shell with unlisted command returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = run_shell("nonexistent_cmd")')

        assert result.error is None
        assert "[ERROR: Command 'nonexistent_cmd' not in allowlist:" in repl.artifacts["result"]

    def test_run_shell_git_safe_subcommand(self):
        """Test run_shell with git safe subcommand succeeds."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = run_shell("git status")')

        assert result.error is None
        # Should execute (may fail if not in git repo, but should not be blocked)
        assert "[ERROR: Command 'git' is blocked" not in repl.artifacts["result"]
        assert "[ERROR: Command 'git' not in allowlist" not in repl.artifacts["result"]

    def test_run_shell_git_unsafe_subcommand(self):
        """Test run_shell with git unsafe subcommand returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = run_shell("git push")')

        assert result.error is None
        assert "[ERROR: git push not allowed" in repl.artifacts["result"]

    def test_run_shell_invalid_syntax(self):
        """Test run_shell with invalid syntax returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = run_shell("echo \\"unclosed quote")')

        assert result.error is None
        assert "[ERROR: Invalid command syntax:" in repl.artifacts["result"]

    def test_run_shell_empty_command(self):
        """Test run_shell with empty command returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = run_shell("")')

        assert result.error is None
        assert "[ERROR: Empty command]" in repl.artifacts["result"]

    @patch('subprocess.run')
    def test_run_shell_timeout(self, mock_run):
        """Test run_shell handles timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)

        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = run_shell("echo test", timeout=30)')

        assert result.error is None
        assert "[ERROR: Command timed out after 30s]" in repl.artifacts["result"]

    def test_run_shell_output_truncation(self):
        """Test run_shell truncates output at 8000 chars."""
        repl = REPLEnvironment(context="test")
        # Generate a long string in the REPL, then use it in shell command
        # Using printf with many repetitions to create long output
        result = repl.execute('artifacts["result"] = run_shell("printf \'X%.0s\' {1..10000}")')

        assert result.error is None
        # Output should be truncated if it exceeds 8000 chars
        if len(repl.artifacts.get("result", "")) > 8000:
            assert "truncated at 8000 chars" in repl.artifacts["result"]
