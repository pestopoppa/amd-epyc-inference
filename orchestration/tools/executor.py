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


class ToolRegistry:
    """Tool registry adapter for REPLEnvironment integration.

    Wraps ToolExecutor to provide role-based access control.
    """

    # Role permissions: which roles can use which tool categories
    ROLE_PERMISSIONS: dict[str, set[str]] = {
        "frontdoor": {"web", "data", "llm"},  # High-level tools only
        "coder_primary": {"web", "data", "code", "math", "system", "llm"},
        "coder_escalation": {"web", "data", "code", "math", "system", "llm"},
        "worker_general": {"web", "data", "math", "llm"},
        "worker_math": {"math", "data"},
        "worker_summarize": {"web", "data", "llm"},
        "architect_general": {"web", "data", "llm"},
        "architect_coding": {"web", "data", "code", "llm"},
        "toolrunner": {"web", "data", "code", "math", "system", "llm"},  # All tools
    }

    def __init__(self, executor: ToolExecutor | None = None):
        self._executor = executor or get_executor()

    def _check_permission(self, tool_name: str, role: str) -> bool:
        """Check if role can use this tool."""
        tool = self._executor.get_tool_spec(tool_name)
        if not tool:
            return False

        category = tool.get("category", "")
        allowed = self.ROLE_PERMISSIONS.get(role, set())

        # toolrunner can use all, others need category match
        if role == "toolrunner":
            return True

        return category in allowed

    def invoke(self, tool_name: str, role: str, **kwargs) -> Any:
        """Invoke a tool with role-based permission check.

        Args:
            tool_name: Name of the tool.
            role: Role requesting the tool.
            **kwargs: Tool parameters.

        Returns:
            Tool result.

        Raises:
            PermissionError: If role cannot use this tool.
        """
        if not self._check_permission(tool_name, role):
            raise PermissionError(
                f"Role '{role}' cannot use tool '{tool_name}'"
            )

        return self._executor.execute(tool_name, **kwargs)

    def list_tools(self, role: str | None = None) -> list[dict]:
        """List tools available to a role.

        Args:
            role: Role to filter by. If None, returns all tools.

        Returns:
            List of tool info dicts.
        """
        all_tools = self._executor.list_tools()

        if role is None:
            return all_tools

        allowed_categories = self.ROLE_PERMISSIONS.get(role, set())
        if role == "toolrunner":
            return all_tools

        return [
            t for t in all_tools
            if t.get("category", "") in allowed_categories
        ]


class ScriptRegistry:
    """Registry for prepared scripts that can be invoked in REPL.

    Scripts are pre-built code snippets that can be invoked by ID.
    They run within the REPL sandbox with access to context.
    """

    def __init__(self, scripts_dir: Path | None = None):
        if scripts_dir is None:
            scripts_dir = Path(__file__).parent.parent / "scripts"

        self.scripts_dir = scripts_dir
        self._scripts: dict[str, dict] = {}
        self._load_scripts()

    def _load_scripts(self):
        """Load script definitions from scripts directory."""
        if not self.scripts_dir.exists():
            return

        # Look for script_registry.yaml
        registry_file = self.scripts_dir / "script_registry.yaml"
        if registry_file.exists():
            with open(registry_file) as f:
                data = yaml.safe_load(f) or {}
                self._scripts = data.get("scripts", {})

    def invoke(
        self,
        script_id: str,
        sandbox_globals: dict | None = None,
        **kwargs
    ) -> Any:
        """Invoke a script by ID.

        Args:
            script_id: Script identifier.
            sandbox_globals: Globals dict from REPL (contains context, etc).
            **kwargs: Script parameters.

        Returns:
            Script result.

        Raises:
            ValueError: If script doesn't exist.
        """
        script = self._scripts.get(script_id)
        if not script:
            raise ValueError(f"Script not found: {script_id}")

        code = script.get("code", "")
        if not code:
            raise ValueError(f"Script has no code: {script_id}")

        # Prepare execution environment
        exec_globals = dict(sandbox_globals or {})
        exec_globals.update(kwargs)

        # Execute and capture result
        exec_locals: dict[str, Any] = {}
        exec(code, exec_globals, exec_locals)

        return exec_locals.get("result", None)

    def find_scripts(self, query: str, limit: int = 5) -> list[dict]:
        """Find scripts matching a query.

        Args:
            query: Search query (matched against name/description).
            limit: Maximum results to return.

        Returns:
            List of matching script info dicts.
        """
        query_lower = query.lower()
        matches = []

        for script_id, script in self._scripts.items():
            name = script.get("name", script_id)
            description = script.get("description", "")

            if (query_lower in name.lower() or
                query_lower in description.lower()):
                matches.append({
                    "id": script_id,
                    "name": name,
                    "description": description,
                    "parameters": list(script.get("parameters", {}).keys()),
                })

            if len(matches) >= limit:
                break

        return matches

    def list_scripts(self) -> list[dict]:
        """List all available scripts."""
        return [
            {
                "id": sid,
                "name": s.get("name", sid),
                "description": s.get("description", ""),
            }
            for sid, s in self._scripts.items()
        ]


# Global registry instances
_tool_registry: ToolRegistry | None = None
_script_registry: ScriptRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def get_script_registry() -> ScriptRegistry:
    """Get or create global script registry."""
    global _script_registry
    if _script_registry is None:
        _script_registry = ScriptRegistry()
    return _script_registry
