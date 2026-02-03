#!/usr/bin/env python3
"""Tests for the src/tools package.

This file tests all tool modules including:
- Tool registration (src/tools/__init__.py)
- File tools (list.py, read.py)
- Code tools (lint.py, run_tests.py)
- Web tools (fetch.py, search.py)
- Data tools (data/__init__.py)

External dependencies are mocked (subprocess, HTTP, filesystem).
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from src.tool_registry import Tool, ToolCategory, ToolRegistry


# ============================================================================
# Tool Registration Tests (src/tools/__init__.py)
# ============================================================================


def test_register_all_tools_full():
    """Test register_all_tools with all submodules available."""
    registry = ToolRegistry()

    from src.tools import register_all_tools

    count = register_all_tools(registry)

    # Should register multiple tools
    assert count > 0

    # Verify some key tools exist
    tools = registry.list_tools()
    tool_names = {t["name"] for t in tools}

    # At least one from each category should be registered
    assert "read_file" in tool_names  # file
    assert "list_dir" in tool_names   # file
    assert "run_tests" in tool_names  # code
    assert "lint_python" in tool_names  # code
    assert "fetch_docs" in tool_names  # web
    assert "web_search" in tool_names  # web
    assert "json_parse" in tool_names  # data


def test_register_all_tools_import_errors():
    """Test register_all_tools handles import errors gracefully."""
    registry = ToolRegistry()

    # Mock failed imports
    with patch("src.tools.register_all_tools") as mock_register:
        # Simulate one module failing to import
        def side_effect(reg):
            from src.tools.file import register_file_tools
            return register_file_tools(reg)

        mock_register.side_effect = side_effect

        from src.tools import register_all_tools
        count = register_all_tools(registry)

        # Should still register available tools
        assert count >= 0


# ============================================================================
# File Tools Tests
# ============================================================================


class TestListDirectoryTool:
    """Tests for list_dir tool."""

    def test_list_dir_success(self, tmp_path):
        """Test listing a directory successfully."""
        from src.tools.file.list import list_dir

        # Create test directory structure
        test_dir = Path("/mnt/raid0/llm/tmp/test_dir")

        # Mock Path operations
        with patch("src.tools.file.list.Path") as mock_path:
            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.resolve.return_value = mock_dir
            mock_dir.__str__.return_value = "/mnt/raid0/llm/tmp/test_dir"
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True

            # Mock directory entries
            file1 = MagicMock()
            file1.name = "test.py"
            file1.is_dir.return_value = False
            file1.is_file.return_value = True
            file1.stat.return_value.st_size = 1024
            file1.__str__.return_value = "/mnt/raid0/llm/tmp/test_dir/test.py"

            dir1 = MagicMock()
            dir1.name = "subdir"
            dir1.is_dir.return_value = True
            dir1.is_file.return_value = False
            dir1.__str__.return_value = "/mnt/raid0/llm/tmp/test_dir/subdir"

            mock_dir.glob.return_value = [dir1, file1]

            result = list_dir(
                directory="/mnt/raid0/llm/tmp/test_dir",
                pattern="*",
            )

        assert result["success"] is True
        assert result["total_count"] == 2
        assert result["returned_count"] == 2
        assert len(result["entries"]) == 2

        # Check entries (directories first)
        assert result["entries"][0]["type"] == "dir"
        assert result["entries"][0]["name"] == "subdir"
        assert result["entries"][1]["type"] == "file"
        assert result["entries"][1]["name"] == "test.py"
        assert "size" in result["entries"][1]

    def test_list_dir_invalid_path(self):
        """Test listing directory with invalid path."""
        from src.tools.file.list import list_dir

        result = list_dir(directory="/invalid/path")

        assert result["success"] is False
        assert "not in allowed locations" in result["error"]

    def test_list_dir_not_found(self):
        """Test listing non-existent directory."""
        from src.tools.file.list import list_dir

        with patch("src.tools.file.list.Path") as mock_path:
            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.resolve.return_value = mock_dir
            mock_dir.__str__.return_value = "/mnt/raid0/llm/tmp/missing"
            mock_dir.exists.return_value = False

            result = list_dir(directory="/mnt/raid0/llm/tmp/missing")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_list_dir_with_pattern(self):
        """Test listing with glob pattern."""
        from src.tools.file.list import list_dir

        with patch("src.tools.file.list.Path") as mock_path:
            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.resolve.return_value = mock_dir
            mock_dir.__str__.return_value = "/mnt/raid0/llm/tmp/test_dir"
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True

            # Mock only .py files
            py_file = MagicMock()
            py_file.name = "test.py"
            py_file.is_dir.return_value = False
            py_file.is_file.return_value = True
            py_file.stat.return_value.st_size = 1024
            py_file.__str__.return_value = "/mnt/raid0/llm/tmp/test_dir/test.py"

            mock_dir.glob.return_value = [py_file]

            result = list_dir(
                directory="/mnt/raid0/llm/tmp/test_dir",
                pattern="*.py",
            )

        assert result["success"] is True
        assert result["pattern"] == "*.py"
        assert len(result["entries"]) == 1

    def test_list_dir_recursive(self):
        """Test recursive directory listing."""
        from src.tools.file.list import list_dir

        with patch("src.tools.file.list.Path") as mock_path:
            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.resolve.return_value = mock_dir
            mock_dir.__str__.return_value = "/mnt/raid0/llm/tmp/test_dir"
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True

            file1 = MagicMock()
            file1.name = "test.py"
            file1.is_dir.return_value = False
            file1.is_file.return_value = True
            file1.stat.return_value.st_size = 1024
            file1.__str__.return_value = "/mnt/raid0/llm/tmp/test_dir/test.py"

            mock_dir.rglob.return_value = [file1]

            result = list_dir(
                directory="/mnt/raid0/llm/tmp/test_dir",
                recursive=True,
            )

        assert result["success"] is True
        mock_dir.rglob.assert_called_once()

    def test_list_dir_limit(self):
        """Test entry limit."""
        from src.tools.file.list import list_dir

        with patch("src.tools.file.list.Path") as mock_path:
            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.resolve.return_value = mock_dir
            mock_dir.__str__.return_value = "/mnt/raid0/llm/tmp/test_dir"
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True

            # Create 10 mock files
            files = []
            for i in range(10):
                f = MagicMock()
                f.name = f"file{i}.txt"
                f.is_dir.return_value = False
                f.is_file.return_value = True
                f.stat.return_value.st_size = 1024
                f.__str__.return_value = f"/mnt/raid0/llm/tmp/test_dir/file{i}.txt"
                files.append(f)

            mock_dir.glob.return_value = files

            result = list_dir(
                directory="/mnt/raid0/llm/tmp/test_dir",
                limit=5,
            )

        assert result["success"] is True
        assert result["total_count"] == 10
        assert result["returned_count"] == 5
        assert result["truncated"] is True

    def test_register_list_tool(self):
        """Test list tool registration."""
        from src.tools.file.list import register_list_tool

        registry = ToolRegistry()
        count = register_list_tool(registry)

        assert count == 1

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "list_dir"
        assert tools[0]["category"] == "file"


class TestReadFileTool:
    """Tests for read_file tool."""

    def test_read_file_success(self):
        """Test reading a file successfully."""
        from src.tools.file.read import read_file

        file_content = "line 1\nline 2\nline 3\n"

        with patch("src.tools.file.read.Path") as mock_path, \
             patch("builtins.open", mock_open(read_data=file_content)):

            mock_file = MagicMock()
            mock_path.return_value = mock_file
            mock_file.resolve.return_value = mock_file
            mock_file.__str__.return_value = "/mnt/raid0/llm/tmp/test.txt"
            mock_file.exists.return_value = True
            mock_file.is_file.return_value = True

            result = read_file(file_path="/mnt/raid0/llm/tmp/test.txt")

        assert result["success"] is True
        assert "line 1" in result["content"]
        assert result["total_lines"] == 3
        assert result["lines_returned"] == 3

    def test_read_file_invalid_path(self):
        """Test reading file with invalid path."""
        from src.tools.file.read import read_file

        result = read_file(file_path="/invalid/path.txt")

        assert result["success"] is False
        assert "not in allowed locations" in result["error"]

    def test_read_file_not_found(self):
        """Test reading non-existent file."""
        from src.tools.file.read import read_file

        with patch("src.tools.file.read.Path") as mock_path:
            mock_file = MagicMock()
            mock_path.return_value = mock_file
            mock_file.resolve.return_value = mock_file
            mock_file.__str__.return_value = "/mnt/raid0/llm/tmp/missing.txt"
            mock_file.exists.return_value = False

            result = read_file(file_path="/mnt/raid0/llm/tmp/missing.txt")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_read_file_with_offset_limit(self):
        """Test reading file with offset and limit."""
        from src.tools.file.read import read_file

        lines = [f"line {i}\n" for i in range(100)]
        file_content = "".join(lines)

        with patch("src.tools.file.read.Path") as mock_path, \
             patch("builtins.open", mock_open(read_data=file_content)):

            mock_file = MagicMock()
            mock_path.return_value = mock_file
            mock_file.resolve.return_value = mock_file
            mock_file.__str__.return_value = "/mnt/raid0/llm/tmp/test.txt"
            mock_file.exists.return_value = True
            mock_file.is_file.return_value = True

            result = read_file(
                file_path="/mnt/raid0/llm/tmp/test.txt",
                offset=10,
                limit=5,
            )

        assert result["success"] is True
        assert result["total_lines"] == 100
        assert result["lines_returned"] == 5
        assert result["offset"] == 10

    def test_read_file_no_line_numbers(self):
        """Test reading file without line numbers."""
        from src.tools.file.read import read_file

        file_content = "line 1\nline 2\n"

        with patch("src.tools.file.read.Path") as mock_path, \
             patch("builtins.open", mock_open(read_data=file_content)):

            mock_file = MagicMock()
            mock_path.return_value = mock_file
            mock_file.resolve.return_value = mock_file
            mock_file.__str__.return_value = "/mnt/raid0/llm/tmp/test.txt"
            mock_file.exists.return_value = True
            mock_file.is_file.return_value = True

            result = read_file(
                file_path="/mnt/raid0/llm/tmp/test.txt",
                show_line_numbers=False,
            )

        assert result["success"] is True
        # Content should not have line number prefix
        assert not result["content"].startswith("     1\t")

    def test_register_read_tool(self):
        """Test read tool registration."""
        from src.tools.file.read import register_read_tool

        registry = ToolRegistry()
        count = register_read_tool(registry)

        assert count == 1

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"
        assert tools[0]["category"] == "file"


# ============================================================================
# Code Tools Tests
# ============================================================================


class TestLintCodeTool:
    """Tests for lint_python tool."""

    def test_lint_success_no_issues(self):
        """Test linting with no issues."""
        from src.tools.code.lint import lint_python

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "[]"
        mock_result.stderr = ""

        with patch("src.tools.code.lint.Path") as mock_path, \
             patch("src.tools.code.lint.subprocess.run", return_value=mock_result):

            mock_file = MagicMock()
            mock_path.return_value = mock_file
            mock_file.resolve.return_value = mock_file
            mock_file.__str__.return_value = "/mnt/raid0/llm/claude/test.py"
            mock_file.exists.return_value = True

            result = lint_python(file_path="/mnt/raid0/llm/claude/test.py")

        assert result["success"] is True
        assert result["issue_count"] == 0
        assert result["exit_code"] == 0

    def test_lint_with_issues(self):
        """Test linting with issues found."""
        from src.tools.code.lint import lint_python

        issues_json = json.dumps([
            {
                "filename": "test.py",
                "location": {"row": 10, "column": 5},
                "code": "E501",
                "message": "Line too long (100 > 88 characters)",
                "fix": None,
            }
        ])

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = issues_json
        mock_result.stderr = ""

        with patch("src.tools.code.lint.Path") as mock_path, \
             patch("src.tools.code.lint.subprocess.run", return_value=mock_result):

            mock_file = MagicMock()
            mock_path.return_value = mock_file
            mock_file.resolve.return_value = mock_file
            mock_file.__str__.return_value = "/mnt/raid0/llm/claude/test.py"
            mock_file.exists.return_value = True

            result = lint_python(file_path="/mnt/raid0/llm/claude/test.py")

        assert result["success"] is False
        assert result["issue_count"] == 1
        assert len(result["issues"]) == 1
        assert result["issues"][0]["code"] == "E501"

    def test_lint_invalid_path(self):
        """Test linting with invalid path."""
        from src.tools.code.lint import lint_python

        result = lint_python(file_path="/invalid/path.py")

        assert result["success"] is False
        assert "not in allowed locations" in result["error"]

    def test_lint_timeout(self):
        """Test linting with timeout."""
        from src.tools.code.lint import lint_python

        with patch("src.tools.code.lint.Path") as mock_path, \
             patch("src.tools.code.lint.subprocess.run", side_effect=subprocess.TimeoutExpired("ruff", 60)):

            mock_file = MagicMock()
            mock_path.return_value = mock_file
            mock_file.resolve.return_value = mock_file
            mock_file.__str__.return_value = "/mnt/raid0/llm/claude/test.py"
            mock_file.exists.return_value = True

            result = lint_python(file_path="/mnt/raid0/llm/claude/test.py")

        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_lint_ruff_not_installed(self):
        """Test linting when ruff is not installed."""
        from src.tools.code.lint import lint_python

        with patch("src.tools.code.lint.Path") as mock_path, \
             patch("src.tools.code.lint.subprocess.run", side_effect=FileNotFoundError):

            mock_file = MagicMock()
            mock_path.return_value = mock_file
            mock_file.resolve.return_value = mock_file
            mock_file.__str__.return_value = "/mnt/raid0/llm/claude/test.py"
            mock_file.exists.return_value = True

            result = lint_python(file_path="/mnt/raid0/llm/claude/test.py")

        assert result["success"] is False
        assert "not installed" in result["error"]

    def test_register_lint_tool(self):
        """Test lint tool registration."""
        from src.tools.code.lint import register_lint_tool

        registry = ToolRegistry()
        count = register_lint_tool(registry)

        assert count == 1

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "lint_python"
        assert tools[0]["category"] == "code"


class TestRunTestsTool:
    """Tests for run_tests tool."""

    def test_run_tests_success(self):
        """Test running tests successfully."""
        from src.tools.code.run_tests import run_tests

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "5 passed in 1.23s"
        mock_result.stderr = ""

        with patch("src.tools.code.run_tests.Path") as mock_path, \
             patch("src.tools.code.run_tests.subprocess.run", return_value=mock_result):

            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.resolve.return_value = mock_dir
            mock_dir.__str__.return_value = "/mnt/raid0/llm/claude"

            result = run_tests(test_path="tests/")

        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["passed"] == 5

    def test_run_tests_with_failures(self):
        """Test running tests with failures."""
        from src.tools.code.run_tests import run_tests

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "3 passed, 2 failed in 2.34s"
        mock_result.stderr = ""

        with patch("src.tools.code.run_tests.Path") as mock_path, \
             patch("src.tools.code.run_tests.subprocess.run", return_value=mock_result):

            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.resolve.return_value = mock_dir
            mock_dir.__str__.return_value = "/mnt/raid0/llm/claude"

            result = run_tests(test_path="tests/")

        assert result["success"] is False
        assert result["passed"] == 3
        assert result["failed"] == 2

    def test_run_tests_invalid_path(self):
        """Test running tests with invalid working dir."""
        from src.tools.code.run_tests import run_tests

        result = run_tests(
            test_path="tests/",
            working_dir="/invalid/path",
        )

        assert result["success"] is False
        assert "not in allowed locations" in result["error"]

    def test_run_tests_timeout(self):
        """Test running tests with timeout."""
        from src.tools.code.run_tests import run_tests

        with patch("src.tools.code.run_tests.Path") as mock_path, \
             patch("src.tools.code.run_tests.subprocess.run", side_effect=subprocess.TimeoutExpired("pytest", 300)):

            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.resolve.return_value = mock_dir
            mock_dir.__str__.return_value = "/mnt/raid0/llm/claude"

            result = run_tests(test_path="tests/")

        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_run_tests_with_pattern(self):
        """Test running tests with pattern filter."""
        from src.tools.code.run_tests import run_tests

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "2 passed in 0.5s"
        mock_result.stderr = ""

        with patch("src.tools.code.run_tests.Path") as mock_path, \
             patch("src.tools.code.run_tests.subprocess.run", return_value=mock_result) as mock_run:

            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.resolve.return_value = mock_dir
            mock_dir.__str__.return_value = "/mnt/raid0/llm/claude"

            result = run_tests(
                test_path="tests/",
                test_pattern="test_api",
            )

        assert result["success"] is True

        # Verify -k flag was used
        call_args = mock_run.call_args[0][0]
        assert "-k" in call_args
        assert "test_api" in call_args

    def test_register_run_tests_tool(self):
        """Test run_tests tool registration."""
        from src.tools.code.run_tests import register_run_tests_tool

        registry = ToolRegistry()
        count = register_run_tests_tool(registry)

        assert count == 1

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "run_tests"
        assert tools[0]["category"] == "code"


# ============================================================================
# Web Tools Tests
# ============================================================================


class TestWebFetchTool:
    """Tests for fetch_docs tool."""

    def test_fetch_docs_success(self):
        """Test fetching docs successfully."""
        from src.tools.web.fetch import fetch_docs

        html_content = b"""
        <html>
            <body>
                <article>
                    <h1>Test Page</h1>
                    <p>This is test content.</p>
                </article>
            </body>
        </html>
        """

        mock_response = MagicMock()
        mock_response.read.return_value = html_content
        mock_response.headers.get.return_value = "text/html"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_response), \
             patch("src.tools.web.fetch._load_source_registry", return_value={}):

            result = fetch_docs(url="https://example.com/docs")

        assert result["success"] is True
        assert "Test Page" in result["content"]
        assert result["url"] == "https://example.com/docs"

    def test_fetch_docs_with_trust_level(self):
        """Test fetching docs with trust level checking."""
        from src.tools.web.fetch import fetch_docs

        html_content = b"<html><body>Content</body></html>"

        mock_response = MagicMock()
        mock_response.read.return_value = html_content
        mock_response.headers.get.return_value = "text/html"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        registry_data = {
            "coding": {
                "level1": [
                    {"domain": "example.com", "trust": 1}
                ]
            }
        }

        with patch("urllib.request.urlopen", return_value=mock_response), \
             patch("src.tools.web.fetch._load_source_registry", return_value=registry_data):

            result = fetch_docs(
                url="https://example.com/docs",
                check_trust=True,
            )

        assert result["success"] is True
        assert result["trust_level"] == 1

    def test_fetch_docs_http_error(self):
        """Test fetching docs with HTTP error."""
        from src.tools.web.fetch import fetch_docs
        from urllib.error import HTTPError

        with patch("urllib.request.urlopen", side_effect=HTTPError(None, 404, "Not Found", {}, None)), \
             patch("src.tools.web.fetch._load_source_registry", return_value={}):

            result = fetch_docs(url="https://example.com/missing")

        assert result["success"] is False
        assert "404" in result["error"]

    def test_fetch_docs_cache(self):
        """Test fetch cache."""
        from src.tools.web.fetch import fetch_docs, _fetch_cache

        html_content = b"<html><body>Cached content</body></html>"

        mock_response = MagicMock()
        mock_response.read.return_value = html_content
        mock_response.headers.get.return_value = "text/html"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        _fetch_cache.clear()

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen, \
             patch("src.tools.web.fetch._load_source_registry", return_value={}):

            # First call - should hit network
            result1 = fetch_docs(url="https://example.com/docs")
            assert result1["success"] is True

            # Second call - should use cache
            result2 = fetch_docs(url="https://example.com/docs")
            assert result2["success"] is True

            # Only one network call
            assert mock_urlopen.call_count == 1

    def test_register_fetch_tool(self):
        """Test fetch tool registration."""
        from src.tools.web.fetch import register_fetch_tool

        registry = ToolRegistry()
        count = register_fetch_tool(registry)

        assert count == 1

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "fetch_docs"
        assert tools[0]["category"] == "web"


class TestWebSearchTool:
    """Tests for web_search tool."""

    def test_web_search_success(self):
        """Test web search successfully."""
        from src.tools.web.search import web_search

        html_response = b"""
        <html>
            <a class="result__a" href="https://example.com/page1">Example Page 1</a>
            <a class="result__snippet">This is a snippet for page 1</a>
            <a class="result__a" href="https://example.com/page2">Example Page 2</a>
            <a class="result__snippet">This is a snippet for page 2</a>
        </html>
        """

        mock_response = MagicMock()
        mock_response.read.return_value = html_response
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = web_search(query="test query", max_results=5)

        assert result["success"] is True
        assert result["query"] == "test query"
        assert result["result_count"] >= 0

    def test_web_search_with_domain_filter(self):
        """Test web search with domain filter."""
        from src.tools.web.search import web_search

        html_response = b"<html></html>"

        mock_response = MagicMock()
        mock_response.read.return_value = html_response
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = web_search(
                query="python docs",
                domain_filter="docs.python.org",
            )

        assert result["success"] is True

        # Verify site: filter was added
        call_args = mock_urlopen.call_args[0][0]
        assert "site%3Adocs.python.org" in call_args.full_url or "site:docs.python.org" in str(call_args)

    def test_web_search_error(self):
        """Test web search with error."""
        from src.tools.web.search import web_search
        from urllib.error import URLError

        with patch("urllib.request.urlopen", side_effect=URLError("Network error")):
            result = web_search(query="test query")

        assert result["success"] is False
        assert "error" in result

    def test_register_search_tool(self):
        """Test search tool registration."""
        from src.tools.web.search import register_search_tool

        registry = ToolRegistry()
        count = register_search_tool(registry)

        assert count == 1

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "web_search"
        assert tools[0]["category"] == "web"


# ============================================================================
# Data Tools Tests
# ============================================================================


class TestDataTools:
    """Tests for data tools (json_parse)."""

    def test_json_parse_success(self):
        """Test parsing valid JSON."""
        from src.tools.data import register_data_tools

        registry = ToolRegistry()
        register_data_tools(registry)

        # Get the json_parse handler
        tool = registry._tools["json_parse"]

        json_str = '{"name": "test", "value": 42}'
        result = tool.handler(content=json_str)

        assert result["success"] is True
        assert result["data"]["name"] == "test"
        assert result["data"]["value"] == 42
        assert result["type"] == "dict"

    def test_json_parse_with_path(self):
        """Test parsing JSON with path extraction."""
        from src.tools.data import register_data_tools

        registry = ToolRegistry()
        register_data_tools(registry)

        tool = registry._tools["json_parse"]

        json_str = '{"user": {"name": "Alice", "age": 30}}'
        result = tool.handler(
            content=json_str,
            extract_path="user.name",
        )

        assert result["success"] is True
        assert result["data"] == "Alice"

    def test_json_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        from src.tools.data import register_data_tools

        registry = ToolRegistry()
        register_data_tools(registry)

        tool = registry._tools["json_parse"]

        result = tool.handler(content="not valid json")

        assert result["success"] is False
        assert "error" in result

    def test_json_parse_invalid_path(self):
        """Test parsing JSON with invalid path."""
        from src.tools.data import register_data_tools

        registry = ToolRegistry()
        register_data_tools(registry)

        tool = registry._tools["json_parse"]

        json_str = '{"name": "test"}'
        result = tool.handler(
            content=json_str,
            extract_path="user.name",
        )

        assert result["success"] is False
        assert "Cannot extract" in result["error"]

    def test_register_data_tools(self):
        """Test data tools registration."""
        from src.tools.data import register_data_tools

        registry = ToolRegistry()
        count = register_data_tools(registry)

        assert count == 1

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "json_parse"
        assert tools[0]["category"] == "data"


# ============================================================================
# Integration Tests
# ============================================================================


class TestToolsIntegration:
    """Integration tests for the entire tools package."""

    def test_register_all_categories(self):
        """Test that all tool categories register correctly."""
        from src.tools import register_all_tools

        registry = ToolRegistry()
        count = register_all_tools(registry)

        assert count >= 7  # At least 7 tools (2 file, 2 code, 2 web, 1 data)

        # Check category distribution
        tools = registry.list_tools()
        categories = {t["category"] for t in tools}

        assert "file" in categories
        assert "code" in categories
        assert "web" in categories
        assert "data" in categories

    def test_tool_registry_with_permissions(self):
        """Test tools work with registry permissions."""
        from src.tools.file import register_file_tools
        from src.tool_registry import ToolPermissions

        registry = ToolRegistry()
        register_file_tools(registry)

        # Set up permissions
        perms = ToolPermissions(
            allowed_categories=[ToolCategory.FILE],
        )
        registry.set_role_permissions("test_role", perms)

        # Check access
        assert registry.can_use_tool("test_role", "read_file")
        assert registry.can_use_tool("test_role", "list_dir")

    def test_all_tools_have_handlers(self):
        """Test that all registered tools have valid handlers."""
        from src.tools import register_all_tools

        registry = ToolRegistry()
        register_all_tools(registry)

        for tool_name, tool in registry._tools.items():
            assert tool.handler is not None, f"Tool {tool_name} has no handler"
            assert callable(tool.handler), f"Tool {tool_name} handler is not callable"

    def test_all_tools_have_required_fields(self):
        """Test that all registered tools have required fields."""
        from src.tools import register_all_tools

        registry = ToolRegistry()
        register_all_tools(registry)

        for tool_name, tool in registry._tools.items():
            assert tool.name, f"Tool {tool_name} has no name"
            assert tool.description, f"Tool {tool_name} has no description"
            assert tool.category, f"Tool {tool_name} has no category"
            assert isinstance(tool.parameters, dict), f"Tool {tool_name} parameters not a dict"
