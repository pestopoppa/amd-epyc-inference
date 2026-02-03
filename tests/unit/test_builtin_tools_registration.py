#!/usr/bin/env python3
"""Tests for builtin_tools registration module."""

import json
from unittest.mock import patch, MagicMock


from src.builtin_tools import register_builtin_tools
from src.tool_registry import ToolRegistry, ToolCategory


class TestRegisterBuiltinTools:
    """Tests for register_builtin_tools function."""

    def test_register_builtin_tools_registers_all(self):
        """register_builtin_tools should register all built-in tools."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Should have registered multiple tools
        assert len(registry._tools) > 0

        # Check that specific tools are registered
        expected_tools = [
            "lint_python",
            "run_tests",
            "format_python",
            "read_json",
            "write_json",
            "read_file",
            "list_files",
        ]
        for tool_name in expected_tools:
            assert tool_name in registry._tools, f"Tool {tool_name} not registered"

    def test_register_builtin_tools_sets_categories(self):
        """Registered tools should have correct categories."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # CODE tools
        assert registry._tools["lint_python"].category == ToolCategory.CODE
        assert registry._tools["run_tests"].category == ToolCategory.CODE
        assert registry._tools["format_python"].category == ToolCategory.CODE

        # DATA tools
        assert registry._tools["read_json"].category == ToolCategory.DATA
        assert registry._tools["write_json"].category == ToolCategory.DATA

        # FILE tools
        assert registry._tools["read_file"].category == ToolCategory.FILE
        assert registry._tools["list_files"].category == ToolCategory.FILE

    def test_register_builtin_tools_has_descriptions(self):
        """Registered tools should have descriptions."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        for tool in registry._tools.values():
            assert tool.description, f"Tool {tool.name} missing description"
            assert len(tool.description) > 10, f"Tool {tool.name} has too short description"

    def test_register_builtin_tools_has_handlers(self):
        """Registered tools should have handler functions."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        for tool in registry._tools.values():
            assert tool.handler is not None, f"Tool {tool.name} missing handler"
            assert callable(tool.handler), f"Tool {tool.name} handler not callable"


class TestLintPythonTool:
    """Tests for lint_python tool."""

    def test_lint_python_valid_code(self):
        """lint_python should validate correct Python code."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        result = registry._tools["lint_python"].handler(code="x = 1 + 1\nprint(x)")

        assert result["syntax_valid"] is True
        assert result["syntax_error"] is None

    def test_lint_python_invalid_syntax(self):
        """lint_python should detect syntax errors."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        result = registry._tools["lint_python"].handler(code="if True print('bad')")

        assert result["syntax_valid"] is False
        assert result["syntax_error"] is not None
        assert "Line" in result["syntax_error"]

    def test_lint_python_from_file(self, tmp_path):
        """lint_python should accept file_path parameter."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Create temp file with valid code
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42")

        result = registry._tools["lint_python"].handler(file_path=str(test_file))

        assert result["syntax_valid"] is True

    def test_lint_python_file_not_found(self):
        """lint_python should handle missing file."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        result = registry._tools["lint_python"].handler(file_path="/nonexistent/file.py")

        assert "error" in result
        assert "Could not read file" in result["error"]

    def test_lint_python_requires_code_or_file(self):
        """lint_python should require either code or file_path."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        result = registry._tools["lint_python"].handler()

        assert "error" in result
        assert "code" in result["error"] or "file_path" in result["error"]


class TestRunTestsTool:
    """Tests for run_tests tool."""

    def test_run_tests_success(self, tmp_path):
        """run_tests should run pytest on valid test file."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Create a simple passing test
        test_file = tmp_path / "test_simple.py"
        test_file.write_text("def test_pass():\n    assert True")

        result = registry._tools["run_tests"].handler(path=str(test_file))

        assert result["passed"] is True
        assert result["return_code"] == 0
        assert "stdout" in result

    def test_run_tests_failure(self, tmp_path):
        """run_tests should detect failing tests."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Create a failing test
        test_file = tmp_path / "test_fail.py"
        test_file.write_text("def test_fail():\n    assert False")

        result = registry._tools["run_tests"].handler(path=str(test_file))

        assert result["passed"] is False
        assert result["return_code"] != 0

    def test_run_tests_verbose_mode(self, tmp_path):
        """run_tests should support verbose parameter."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        test_file = tmp_path / "test_simple.py"
        test_file.write_text("def test_pass():\n    assert True")

        result = registry._tools["run_tests"].handler(path=str(test_file), verbose=True)

        assert "stdout" in result
        # Verbose output should be present (exact format varies by pytest version)

    def test_run_tests_truncates_output(self, tmp_path):
        """run_tests should truncate long output."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Create a test that produces lots of output
        test_file = tmp_path / "test_verbose.py"
        test_file.write_text(
            "def test_verbose():\n"
            + "    for i in range(1000):\n"
            + "        print('X' * 100)\n"
            + "    assert True"
        )

        result = registry._tools["run_tests"].handler(path=str(test_file))

        # Output should be capped at 2000 chars
        assert len(result.get("stdout", "")) <= 2000


class TestFormatPythonTool:
    """Tests for format_python tool."""

    def test_format_python_already_formatted(self):
        """format_python should handle already formatted code."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        code = "x = 1\ny = 2\n"
        result = registry._tools["format_python"].handler(code=code)

        assert "formatted" in result
        # May or may not have changed depending on formatter

    def test_format_python_unformatted_code(self):
        """format_python should format unformatted code."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        code = "x=1;y=2"  # Poorly formatted
        result = registry._tools["format_python"].handler(code=code)

        assert "formatted" in result
        assert "changed" in result

    def test_format_python_preserves_validity(self):
        """format_python should preserve code validity."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        code = "def foo(x, y):\n    return x + y"
        result = registry._tools["format_python"].handler(code=code)

        # Check that formatted code is still valid
        assert "formatted" in result
        formatted = result["formatted"]
        try:
            compile(formatted, "<string>", "exec")
            valid = True
        except SyntaxError:
            valid = False
        assert valid


class TestReadJsonTool:
    """Tests for read_json tool."""

    def test_read_json_valid_file(self, tmp_path):
        """read_json should read valid JSON file."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Create JSON file
        json_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}
        json_file.write_text(json.dumps(test_data))

        result = registry._tools["read_json"].handler(path=str(json_file))

        assert result["success"] is True
        assert result["data"] == test_data

    def test_read_json_invalid_json(self, tmp_path):
        """read_json should handle invalid JSON."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        json_file = tmp_path / "bad.json"
        json_file.write_text("{invalid json")

        result = registry._tools["read_json"].handler(path=str(json_file))

        assert result["success"] is False
        assert "error" in result
        assert "Invalid JSON" in result["error"]

    def test_read_json_file_not_found(self):
        """read_json should handle missing file."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        result = registry._tools["read_json"].handler(path="/nonexistent/file.json")

        assert result["success"] is False
        assert "error" in result


class TestWriteJsonTool:
    """Tests for write_json tool."""

    def test_write_json_to_raid(self, tmp_path, monkeypatch):
        """write_json should write data to allowed path."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Mock get_config to return our temp path as raid_prefix
        mock_config = MagicMock()
        mock_config.paths.raid_prefix = str(tmp_path)

        with patch("src.config.get_config", return_value=mock_config):
            json_file = tmp_path / "output.json"
            test_data = {"key": "value", "number": 42}

            result = registry._tools["write_json"].handler(path=str(json_file), data=test_data)

            assert result["success"] is True
            assert result["path"] == str(json_file)
            assert json_file.exists()

            # Verify content
            with open(json_file) as f:
                written_data = json.load(f)
            assert written_data == test_data

    def test_write_json_blocks_non_raid_path(self):
        """write_json should block writing outside RAID."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Mock config with specific raid prefix
        mock_config = MagicMock()
        mock_config.paths.raid_prefix = "/mnt/raid0/"

        with patch("src.config.get_config", return_value=mock_config):
            result = registry._tools["write_json"].handler(
                path="/tmp/test.json", data={"key": "val"}
            )

            assert result["success"] is False
            assert "error" in result
            assert "Can only write to" in result["error"]

    def test_write_json_with_indent(self, tmp_path):
        """write_json should support custom indentation."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        mock_config = MagicMock()
        mock_config.paths.raid_prefix = str(tmp_path)

        with patch("src.config.get_config", return_value=mock_config):
            json_file = tmp_path / "indented.json"
            test_data = {"key": "value"}

            result = registry._tools["write_json"].handler(
                path=str(json_file), data=test_data, indent=4
            )

            assert result["success"] is True

            # Check that indentation was applied
            content = json_file.read_text()
            assert "    " in content  # 4-space indent


class TestReadFileTool:
    """Tests for read_file tool."""

    def test_read_file_complete(self, tmp_path):
        """read_file should read entire file."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        test_file = tmp_path / "test.txt"
        content = "Line 1\nLine 2\nLine 3"
        test_file.write_text(content)

        result = registry._tools["read_file"].handler(path=str(test_file))

        assert result["success"] is True
        assert result["content"] == content
        assert result["truncated"] is False

    def test_read_file_with_max_lines(self, tmp_path):
        """read_file should respect max_lines parameter."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        test_file = tmp_path / "test.txt"
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        test_file.write_text(content)

        result = registry._tools["read_file"].handler(path=str(test_file), max_lines=3)

        assert result["success"] is True
        assert result["truncated"] is True
        assert result["content"].count("\n") <= 3

    def test_read_file_not_found(self):
        """read_file should handle missing file."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        result = registry._tools["read_file"].handler(path="/nonexistent/file.txt")

        assert result["success"] is False
        assert "error" in result


class TestListFilesTool:
    """Tests for list_files tool."""

    def test_list_files_basic(self, tmp_path):
        """list_files should list files in directory."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Create some files
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "file3.py").touch()

        result = registry._tools["list_files"].handler(path=str(tmp_path))

        assert result["success"] is True
        assert result["count"] == 3
        assert result["truncated"] is False
        assert len(result["files"]) == 3

    def test_list_files_with_pattern(self, tmp_path):
        """list_files should support glob pattern filtering."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "file3.py").touch()

        result = registry._tools["list_files"].handler(path=str(tmp_path), pattern="*.txt")

        assert result["success"] is True
        assert result["count"] == 2
        txt_files = [f for f in result["files"] if f.endswith(".txt")]
        assert len(txt_files) == 2

    def test_list_files_truncates_large_directories(self, tmp_path):
        """list_files should truncate very large directories."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Create 150 files
        for i in range(150):
            (tmp_path / f"file{i}.txt").touch()

        result = registry._tools["list_files"].handler(path=str(tmp_path))

        assert result["success"] is True
        assert result["count"] == 150
        assert result["truncated"] is True
        assert len(result["files"]) == 100  # Capped at 100

    def test_list_files_not_directory(self, tmp_path):
        """list_files should handle non-directory path."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Create a file, not a directory
        test_file = tmp_path / "not_a_dir.txt"
        test_file.touch()

        result = registry._tools["list_files"].handler(path=str(test_file))

        assert result["success"] is False
        assert "error" in result
        assert "Not a directory" in result["error"]


class TestToolParameters:
    """Tests for tool parameter definitions."""

    def test_lint_python_parameters(self):
        """lint_python should have correct parameter definitions."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        tool = registry._tools["lint_python"]
        params = tool.parameters

        assert "code" in params
        assert "file_path" in params
        assert params["code"]["required"] is False
        assert params["file_path"]["required"] is False

    def test_run_tests_parameters(self):
        """run_tests should have correct parameter definitions."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        tool = registry._tools["run_tests"]
        params = tool.parameters

        assert "path" in params
        assert "verbose" in params
        assert params["path"]["required"] is True
        assert params["verbose"]["required"] is False

    def test_write_json_parameters(self):
        """write_json should have correct parameter definitions."""
        registry = ToolRegistry()
        register_builtin_tools(registry)

        tool = registry._tools["write_json"]
        params = tool.parameters

        assert "path" in params
        assert "data" in params
        assert "indent" in params
        assert params["path"]["required"] is True
        assert params["data"]["required"] is True
        assert params["indent"]["required"] is False
