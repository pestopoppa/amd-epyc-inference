"""Tool Executor - Load and execute tools from registry."""

import importlib
import yaml
from pathlib import Path
from typing import Any, Callable


class ToolExecutor:
    """Execute tools registered in tool_registry.yaml."""

    def __init__(self, registry_path: Path | None = None):
        if registry_path is None:
            registry_path = Path(__file__).parent.parent / "tool_registry.yaml"

        self.registry_path = registry_path
        self._tools: dict[str, dict] = {}
        self._functions: dict[str, Callable] = {}
        self._load_registry()

    def _load_registry(self):
        """Load tool definitions from registry."""
        with open(self.registry_path) as f:
            data = yaml.safe_load(f)

        self._tools = data.get("tools", {})

    def _load_function(self, tool_name: str) -> Callable | None:
        """Lazy-load the implementation function for a tool."""
        if tool_name in self._functions:
            return self._functions[tool_name]

        tool = self._tools.get(tool_name)
        if not tool:
            return None

        impl = tool.get("implementation", {})
        if impl.get("type") != "python":
            return None

        module_name = impl.get("module")
        func_name = impl.get("function")

        if not module_name or not func_name:
            return None

        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            self._functions[tool_name] = func
            return func
        except Exception as e:
            print(f"Error loading tool {tool_name}: {e}")
            return None

    def list_tools(self, category: str | None = None) -> list[dict]:
        """List available tools, optionally filtered by category."""
        tools = []
        for name, spec in self._tools.items():
            if category and spec.get("category") != category:
                continue
            tools.append({
                "name": name,
                "category": spec.get("category", ""),
                "description": spec.get("description", ""),
                "parameters": list(spec.get("parameters", {}).keys()),
            })
        return tools

    def get_tool_spec(self, tool_name: str) -> dict | None:
        """Get full specification for a tool."""
        return self._tools.get(tool_name)

    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool with given parameters.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool

        Returns:
            Tool output or error dict
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Tool not found: {tool_name}"}

        # Validate required parameters
        params = tool.get("parameters", {})
        for param_name, param_spec in params.items():
            if param_spec.get("required", False) and param_name not in kwargs:
                # Check for default
                if "default" not in param_spec:
                    return {"error": f"Missing required parameter: {param_name}"}
                kwargs[param_name] = param_spec["default"]
            elif param_name not in kwargs and "default" in param_spec:
                kwargs[param_name] = param_spec["default"]

        # Load and execute function
        func = self._load_function(tool_name)
        if not func:
            return {"error": f"Tool implementation not found: {tool_name}"}

        try:
            return func(**kwargs)
        except Exception as e:
            return {"error": f"Tool execution failed: {e}"}

    def __call__(self, tool_name: str, **kwargs) -> Any:
        """Shortcut for execute()."""
        return self.execute(tool_name, **kwargs)


# Global executor instance
_executor: ToolExecutor | None = None


def get_executor() -> ToolExecutor:
    """Get or create global tool executor."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor


def TOOL(name: str, **kwargs) -> Any:
    """Execute a tool by name - for use in REPL.

    Example:
        result = TOOL("web_search", query="Python asyncio")
        result = TOOL("calculate", expression="np.pi * 2")
    """
    return get_executor().execute(name, **kwargs)


def list_tools(category: str | None = None) -> list[dict]:
    """List available tools."""
    return get_executor().list_tools(category)
