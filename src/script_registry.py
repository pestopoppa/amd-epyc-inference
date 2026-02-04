#!/usr/bin/env python3
"""Script Registry for prepared tooling scripts.

This module provides a registry of prepared scripts that models can invoke by ID
instead of generating code from scratch. This achieves ~92% token savings.

Design principles:
- Scripts stored as JSON with metadata and code/MCP mapping
- Fuzzy search for script discovery
- SHA256 integrity checking for code-based scripts
- MCP server aliasing for common operations

Usage:
    from src.script_registry import ScriptRegistry

    registry = ScriptRegistry()
    registry.load_from_directory("orchestration/script_registry")

    # Find relevant scripts
    matches = registry.find_scripts("fetch python documentation")

    # Invoke a script by ID
    result = registry.invoke("fetch_docs", url="https://docs.python.org")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_default_working_dir() -> str:
    """Get default working directory from config with fallback."""
    try:
        from src.config import get_config

        return str(get_config().paths.project_root)
    except Exception:
        return "/mnt/raid0/llm/claude"


@dataclass
class Script:
    """A prepared script that can be invoked by ID."""

    id: str
    description: str
    category: str
    tags: list[str]
    parameters: dict[str, dict[str, Any]]  # name -> {type, description, required, default}
    default_args: dict[str, Any] = field(default_factory=dict)

    # Execution - one of these should be set
    code: str | None = None  # Python code to exec
    code_hash: str | None = None  # SHA256 of code for integrity
    mcp_server: str | None = None  # MCP server for delegation
    mcp_tool: str | None = None  # Tool name on MCP server
    command: str | None = None  # Shell command template

    # Metadata
    token_savings: str | None = None  # e.g., "92%"
    examples: list[dict[str, Any]] = field(default_factory=list)

    def validate_args(self, args: dict[str, Any]) -> list[str]:
        """Validate arguments against parameter schema.

        Args:
            args: Arguments to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Merge with defaults
        merged = {**self.default_args, **args}

        # Check required parameters
        for param_name, param_spec in self.parameters.items():
            if param_spec.get("required", False) and param_name not in merged:
                errors.append(f"Missing required parameter: {param_name}")

        # Check for unknown parameters
        for arg_name in args:
            if arg_name not in self.parameters:
                errors.append(f"Unknown parameter: {arg_name}")

        return errors

    def get_merged_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get arguments merged with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Merged arguments dict.
        """
        merged = {}

        # Add defaults
        for param_name, param_spec in self.parameters.items():
            if "default" in param_spec:
                merged[param_name] = param_spec["default"]

        # Add script-level defaults
        merged.update(self.default_args)

        # Add user args (override)
        merged.update(args)

        return merged


@dataclass
class ScriptMatch:
    """Result from fuzzy script search."""

    script: Script
    score: float
    matched_on: str  # What triggered the match


class ScriptRegistry:
    """Registry of prepared scripts for token-efficient operations.

    Scripts can be backed by:
    - Embedded Python code (executed in sandbox)
    - MCP server delegation
    - Shell command templates
    """

    def __init__(self):
        """Initialize an empty script registry."""
        self._scripts: dict[str, Script] = {}
        self._by_category: dict[str, list[str]] = {}  # category -> [script_ids]
        self._by_tag: dict[str, list[str]] = {}  # tag -> [script_ids]
        self._mcp_configs: dict | None = None

    def register_script(self, script: Script) -> None:
        """Register a script in the registry.

        Args:
            script: Script to register.

        Raises:
            ValueError: If script with same ID already exists.
        """
        if script.id in self._scripts:
            raise ValueError(f"Script '{script.id}' already registered")

        # Validate code hash if code-based
        if script.code is not None:
            computed_hash = hashlib.sha256(script.code.encode()).hexdigest()[:16]
            if script.code_hash and script.code_hash != computed_hash:
                raise ValueError(
                    f"Script '{script.id}' code hash mismatch: "
                    f"expected {script.code_hash}, got {computed_hash}"
                )
            script.code_hash = computed_hash

        self._scripts[script.id] = script

        # Index by category
        if script.category not in self._by_category:
            self._by_category[script.category] = []
        self._by_category[script.category].append(script.id)

        # Index by tags
        for tag in script.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(script.id)

        logger.info(f"Registered script: {script.id} ({script.category})")

    def load_from_json(self, path: str | Path) -> None:
        """Load a script from a JSON file.

        Args:
            path: Path to script JSON file.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        script = Script(
            id=data["id"],
            description=data["description"],
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            parameters=data.get("parameters", {}),
            default_args=data.get("default_args", {}),
            code=data.get("code"),
            code_hash=data.get("code_hash"),
            mcp_server=data.get("mcp_server"),
            mcp_tool=data.get("mcp_tool"),
            command=data.get("command"),
            token_savings=data.get("token_savings"),
            examples=data.get("examples", []),
        )

        self.register_script(script)

    def load_from_directory(self, directory: str | Path) -> int:
        """Load all scripts from a directory (recursive).

        Args:
            directory: Directory containing script JSON files.

        Returns:
            Number of scripts loaded.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Script directory not found: {directory}")
            return 0

        count = 0
        for json_path in directory.rglob("*.json"):
            try:
                self.load_from_json(json_path)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load script {json_path}: {e}")

        logger.info(f"Loaded {count} scripts from {directory}")
        return count

    def find_scripts(
        self,
        query: str,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[ScriptMatch]:
        """Find scripts matching a natural language query.

        Uses fuzzy matching on description, ID, and tags.

        Args:
            query: Natural language query (e.g., "fetch python documentation").
            category: Optional category filter.
            tags: Optional tag filters (OR logic).
            limit: Maximum results to return.

        Returns:
            List of ScriptMatch sorted by relevance.
        """
        query_lower = query.lower()
        query_words = set(re.split(r"\W+", query_lower))
        matches: list[ScriptMatch] = []

        # Filter candidates
        candidates = list(self._scripts.values())
        if category:
            candidates = [s for s in candidates if s.category == category]
        if tags:
            tag_set = set(tags)
            candidates = [s for s in candidates if tag_set & set(s.tags)]

        for script in candidates:
            best_score = 0.0
            matched_on = ""

            # Match on ID
            id_score = SequenceMatcher(None, query_lower, script.id.lower()).ratio()
            if id_score > best_score:
                best_score = id_score
                matched_on = "id"

            # Match on description
            desc_lower = script.description.lower()
            desc_score = SequenceMatcher(None, query_lower, desc_lower).ratio()
            if desc_score > best_score:
                best_score = desc_score
                matched_on = "description"

            # Boost for exact word matches in tags
            for tag in script.tags:
                if tag.lower() in query_words:
                    tag_score = 0.9
                    if tag_score > best_score:
                        best_score = tag_score
                        matched_on = f"tag:{tag}"

            # Boost for word overlap with description
            desc_words = set(re.split(r"\W+", desc_lower))
            overlap = query_words & desc_words
            if overlap:
                word_score = len(overlap) / max(len(query_words), 1) * 0.8
                if word_score > best_score:
                    best_score = word_score
                    matched_on = f"words:{','.join(overlap)}"

            if best_score > 0.3:  # Minimum threshold
                matches.append(
                    ScriptMatch(
                        script=script,
                        score=best_score,
                        matched_on=matched_on,
                    )
                )

        # Sort by score and limit
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:limit]

    def get_script(self, script_id: str) -> Script | None:
        """Get a script by ID.

        Args:
            script_id: Script identifier.

        Returns:
            Script if found, None otherwise.
        """
        return self._scripts.get(script_id)

    def invoke(
        self,
        script_id: str,
        sandbox_globals: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a script by ID.

        Args:
            script_id: Script identifier.
            sandbox_globals: Optional globals dict for code execution.
            **kwargs: Script arguments.

        Returns:
            Script result.

        Raises:
            ValueError: If script doesn't exist or args are invalid.
            RuntimeError: If script execution fails.
        """
        import time

        script = self.get_script(script_id)
        if script is None:
            raise ValueError(f"Unknown script: {script_id}")

        # Validate arguments
        errors = script.validate_args(kwargs)
        if errors:
            raise ValueError(f"Invalid arguments: {'; '.join(errors)}")

        # Merge with defaults
        merged_args = script.get_merged_args(kwargs)

        start = time.perf_counter()

        try:
            if script.code is not None:
                result = self._execute_code(script, merged_args, sandbox_globals)
            elif script.mcp_server is not None:
                result = self._execute_mcp(script, merged_args)
            elif script.command is not None:
                result = self._execute_command(script, merged_args)
            else:
                raise RuntimeError(f"Script '{script_id}' has no execution method")

            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"Script {script_id} completed in {elapsed:.1f}ms")

            return result

        except Exception as e:
            logger.error(f"Script {script_id} failed: {e}")
            raise RuntimeError(f"Script execution failed: {e}") from e

    def _execute_code(
        self,
        script: Script,
        args: dict[str, Any],
        sandbox_globals: dict[str, Any] | None,
    ) -> Any:
        """Execute embedded Python code.

        Args:
            script: Script with code to execute.
            args: Merged arguments.
            sandbox_globals: Optional globals for sandbox.

        Returns:
            Code execution result.
        """
        # Build execution environment
        if sandbox_globals is None:
            sandbox_globals = {}

        local_vars = {"args": args, "result": None}
        exec_globals = {**sandbox_globals, **local_vars}

        # Execute the code
        exec(script.code, exec_globals)

        # Return the result variable
        return exec_globals.get("result")

    def _execute_mcp(self, script: Script, args: dict[str, Any]) -> Any:
        """Execute via MCP server delegation.

        Args:
            script: Script with MCP configuration.
            args: Merged arguments.

        Returns:
            MCP tool result as text string.

        Raises:
            RuntimeError: If server is unknown or tool call fails.
        """
        from src.mcp_client import call_mcp_tool, load_server_configs

        if self._mcp_configs is None:
            try:
                from src.config import get_config

                config_path = get_config().paths.project_root / "orchestration" / "mcp_servers.yaml"
            except Exception:
                config_path = Path(__file__).parent.parent / "orchestration" / "mcp_servers.yaml"
            self._mcp_configs = load_server_configs(config_path)

        server_id = script.mcp_server
        if server_id not in self._mcp_configs:
            raise RuntimeError(
                f"Unknown MCP server: {server_id}. Configure in orchestration/mcp_servers.yaml"
            )

        tool_name = script.mcp_tool or script.id
        return call_mcp_tool(self._mcp_configs[server_id], tool_name, args)

    def _execute_command(self, script: Script, args: dict[str, Any]) -> Any:
        """Execute a shell command template.

        Args:
            script: Script with command template.
            args: Merged arguments (for template substitution).

        Returns:
            Command output.
        """
        import subprocess

        # Simple template substitution
        command = script.command
        for key, value in args.items():
            command = command.replace(f"{{{key}}}", str(value))

        # Execute (with safety constraints, no shell injection)
        import shlex

        result = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
            cwd=args.get("working_dir", _get_default_working_dir()),
        )

        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")

        return result.stdout

    def list_scripts(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List available scripts.

        Args:
            category: Optional category filter.
            tags: Optional tag filters.

        Returns:
            List of script info dicts.
        """
        result = []
        for script in self._scripts.values():
            if category and script.category != category:
                continue
            if tags and not (set(tags) & set(script.tags)):
                continue

            result.append(
                {
                    "id": script.id,
                    "description": script.description,
                    "category": script.category,
                    "tags": script.tags,
                    "parameters": script.parameters,
                    "token_savings": script.token_savings,
                }
            )

        return result

    def get_categories(self) -> list[str]:
        """Get list of all categories."""
        return list(self._by_category.keys())

    def get_tags(self) -> list[str]:
        """Get list of all tags."""
        return list(self._by_tag.keys())


# Default global registry
_default_registry: ScriptRegistry | None = None


def get_registry() -> ScriptRegistry:
    """Get or create the default script registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ScriptRegistry()
    return _default_registry
