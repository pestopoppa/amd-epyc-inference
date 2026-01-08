"""Code tools for code execution and quality.

Tools for running tests and linting:
- run_tests: Execute pytest tests
- lint: Run linting tools
"""

from src.tool_registry import ToolRegistry


def register_code_tools(registry: ToolRegistry) -> int:
    """Register all code tools with the registry.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered.
    """
    from src.tools.code.run_tests import register_run_tests_tool
    from src.tools.code.lint import register_lint_tool

    count = 0
    count += register_run_tests_tool(registry)
    count += register_lint_tool(registry)
    return count
