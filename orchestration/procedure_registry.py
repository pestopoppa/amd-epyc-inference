#!/usr/bin/env python3
"""Load, validate and execute deterministic self-management procedures.

This module provides the ProcedureRegistry for loading procedure definitions
from YAML files, validating them against the schema, and executing them
with proper input validation, step execution, and rollback support.

Usage:
    from orchestration.procedure_registry import ProcedureRegistry

    registry = ProcedureRegistry()
    result = registry.execute("benchmark_model", model_path="/path/to/model.gguf")
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

try:
    from jsonschema import Draft202012Validator, ValidationError
except ImportError:
    Draft202012Validator = None
    ValidationError = Exception

# Optional memory integration
try:
    from orchestration.repl_memory.episodic_store import EpisodicStore
    from orchestration.repl_memory.embedder import TaskEmbedder
    MEMORY_AVAILABLE = True
except ImportError:
    EpisodicStore = None
    TaskEmbedder = None
    MEMORY_AVAILABLE = False


# Default paths
DEFAULT_PROCEDURES_DIR = Path(__file__).resolve().parent / "procedures"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "procedure.schema.json"
DEFAULT_STATE_DIR = Path(__file__).resolve().parent / "procedures" / "state"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


@dataclass
class StepResult:
    """Result of executing a single procedure step."""

    step_id: str
    success: bool
    output: str = ""
    error: str | None = None
    elapsed_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class ProcedureResult:
    """Result of executing a complete procedure."""

    procedure_id: str
    success: bool
    step_results: list[StepResult] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_seconds: float = 0.0
    checkpoint_id: str | None = None
    rolled_back: bool = False


@dataclass
class ProcedureInput:
    """Definition of a procedure input parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    validation: dict[str, Any] | None = None


@dataclass
class ProcedureStep:
    """Definition of a procedure step."""

    id: str
    name: str
    action: dict[str, Any]
    description: str | None = None
    condition: str | None = None
    on_failure: str = "abort"
    max_retries: int = 0
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Procedure:
    """Complete procedure definition."""

    id: str
    name: str
    version: str
    description: str
    category: str
    steps: list[ProcedureStep]
    verification: dict[str, Any]
    inputs: list[ProcedureInput] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    permissions: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 350


class ProcedureValidationError(Exception):
    """Error validating a procedure definition."""

    pass


class ProcedureExecutionError(Exception):
    """Error executing a procedure."""

    pass


class ProcedureRegistry:
    """Registry for loading, validating and executing procedures.

    The registry loads procedure definitions from YAML files, validates
    them against the JSON schema, and provides methods to execute
    procedures with proper input validation and error handling.
    """

    def __init__(
        self,
        procedures_dir: Path | str | None = None,
        schema_path: Path | str | None = None,
        state_dir: Path | str | None = None,
        checkpoint_dir: Path | str | None = None,
        validate_on_load: bool = True,
        enable_memory: bool = False,
    ):
        """Initialize the procedure registry.

        Args:
            procedures_dir: Directory containing procedure YAML files.
            schema_path: Path to procedure.schema.json.
            state_dir: Directory for storing procedure state.
            checkpoint_dir: Directory for storing checkpoints.
            validate_on_load: Whether to validate procedures against schema.
            enable_memory: Whether to enable episodic memory logging.
        """
        self.procedures_dir = Path(procedures_dir or DEFAULT_PROCEDURES_DIR)
        self.schema_path = Path(schema_path or DEFAULT_SCHEMA_PATH)
        self.state_dir = Path(state_dir or DEFAULT_STATE_DIR)
        self.checkpoint_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
        self.validate_on_load = validate_on_load

        # Ensure directories exist
        self.procedures_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load schema
        self._schema: dict[str, Any] | None = None
        self._validator: Draft202012Validator | None = None
        if self.schema_path.exists():
            self._load_schema()

        # Load procedures
        self.procedures: dict[str, Procedure] = {}
        self._load_procedures()

        # Execution context
        self._context: dict[str, Any] = {}
        self._tool_handlers: dict[str, Callable] = {}

        # Memory integration (optional)
        self._enable_memory = enable_memory and MEMORY_AVAILABLE
        self._episodic_store: EpisodicStore | None = None
        self._embedder: TaskEmbedder | None = None
        if self._enable_memory:
            try:
                self._episodic_store = EpisodicStore()
                self._embedder = TaskEmbedder()
            except Exception as e:
                logger.warning("Failed to initialize memory: %s", e)
                self._enable_memory = False

    def _load_schema(self) -> None:
        """Load the procedure JSON schema."""
        try:
            with open(self.schema_path, encoding="utf-8") as f:
                self._schema = json.load(f)
            if Draft202012Validator is not None:
                self._validator = Draft202012Validator(self._schema)
        except Exception as e:
            raise ProcedureValidationError(f"Failed to load schema: {e}") from e

    def _load_procedures(self) -> None:
        """Load all procedure definitions from the procedures directory."""
        if not self.procedures_dir.exists():
            return

        # Load both YAML and JSON files
        patterns = ["*.yaml", "*.json"] if YAML_AVAILABLE else ["*.json"]
        for pattern in patterns:
            for file_path in self.procedures_dir.glob(pattern):
                try:
                    procedure = self._load_procedure_file(file_path)
                    self.procedures[procedure.id] = procedure
                except Exception as e:
                    # Log but don't fail - allow partial loading
                    logger.warning("Failed to load %s: %s", file_path, e)

    def _load_procedure_file(self, path: Path) -> Procedure:
        """Load a single procedure from a YAML or JSON file.

        Args:
            path: Path to the procedure file.

        Returns:
            Parsed Procedure object.

        Raises:
            ProcedureValidationError: If validation fails.
        """
        with open(path, encoding="utf-8") as f:
            if path.suffix == ".json":
                data = json.load(f)
            elif path.suffix in (".yaml", ".yml") and YAML_AVAILABLE:
                data = yaml.safe_load(f)
            else:
                raise ProcedureValidationError(
                    f"Unsupported file format: {path.suffix}"
                )

        # Validate against schema
        if self.validate_on_load and self._validator is not None:
            errors = list(self._validator.iter_errors(data))
            if errors:
                error_msgs = [f"{e.json_path}: {e.message}" for e in errors[:5]]
                raise ProcedureValidationError(
                    f"Validation errors in {path.name}:\n" + "\n".join(error_msgs)
                )

        # Parse into dataclass
        return self._parse_procedure(data)

    def _parse_procedure(self, data: dict[str, Any]) -> Procedure:
        """Parse a procedure dict into a Procedure object."""
        # Parse inputs
        inputs = []
        for inp in data.get("inputs", []):
            inputs.append(
                ProcedureInput(
                    name=inp["name"],
                    type=inp["type"],
                    description=inp["description"],
                    required=inp.get("required", True),
                    default=inp.get("default"),
                    validation=inp.get("validation"),
                )
            )

        # Parse steps
        steps = []
        for step in data["steps"]:
            steps.append(
                ProcedureStep(
                    id=step["id"],
                    name=step["name"],
                    action=step["action"],
                    description=step.get("description"),
                    condition=step.get("condition"),
                    on_failure=step.get("on_failure", "abort"),
                    max_retries=step.get("max_retries", 0),
                    depends_on=step.get("depends_on", []),
                )
            )

        return Procedure(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data["description"],
            category=data["category"],
            steps=steps,
            verification=data["verification"],
            inputs=inputs,
            outputs=data.get("outputs", []),
            permissions=data.get("permissions", {}),
            rollback=data.get("rollback"),
            metadata=data.get("metadata", {}),
            estimated_tokens=data.get("estimated_tokens", 350),
        )

    def get(self, procedure_id: str) -> Procedure | None:
        """Get a procedure by ID.

        Args:
            procedure_id: The procedure identifier.

        Returns:
            Procedure object or None if not found.
        """
        return self.procedures.get(procedure_id)

    def list_procedures(
        self,
        category: str | None = None,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        """List available procedures with optional filtering.

        Args:
            category: Filter by category.
            role: Filter by role permission.

        Returns:
            List of procedure summaries.
        """
        result = []
        for proc in self.procedures.values():
            # Filter by category
            if category and proc.category != category:
                continue

            # Filter by role permission
            if role:
                allowed_roles = proc.permissions.get("roles", [])
                if allowed_roles and role not in allowed_roles:
                    continue

            result.append(
                {
                    "id": proc.id,
                    "name": proc.name,
                    "category": proc.category,
                    "description": proc.description[:100],
                    "estimated_tokens": proc.estimated_tokens,
                    "input_count": len(proc.inputs),
                    "step_count": len(proc.steps),
                }
            )

        return result

    def validate_inputs(
        self,
        procedure: Procedure,
        inputs: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate inputs for a procedure.

        Args:
            procedure: The procedure to validate inputs for.
            inputs: Input values to validate.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        for inp in procedure.inputs:
            value = inputs.get(inp.name)

            # Check required
            if inp.required and value is None and inp.default is None:
                errors.append(f"Missing required input: {inp.name}")
                continue

            # Use default if not provided
            if value is None:
                value = inp.default

            if value is None:
                continue

            # Type validation
            if inp.type == "path":
                if not isinstance(value, str):
                    errors.append(f"{inp.name}: expected path string")
                elif inp.validation and inp.validation.get("path_must_exist"):
                    if not Path(value).exists():
                        errors.append(f"{inp.name}: path does not exist: {value}")
                elif inp.validation and inp.validation.get("path_prefix"):
                    prefix = inp.validation["path_prefix"]
                    if not value.startswith(prefix):
                        errors.append(f"{inp.name}: path must start with {prefix}")

            elif inp.type == "integer":
                if not isinstance(value, int):
                    errors.append(f"{inp.name}: expected integer")
                elif inp.validation:
                    if "min" in inp.validation and value < inp.validation["min"]:
                        errors.append(f"{inp.name}: must be >= {inp.validation['min']}")
                    if "max" in inp.validation and value > inp.validation["max"]:
                        errors.append(f"{inp.name}: must be <= {inp.validation['max']}")

            elif inp.type == "string":
                if not isinstance(value, str):
                    errors.append(f"{inp.name}: expected string")
                elif inp.validation and inp.validation.get("pattern"):
                    if not re.match(inp.validation["pattern"], value):
                        errors.append(f"{inp.name}: does not match pattern")
                elif inp.validation and inp.validation.get("enum"):
                    if value not in inp.validation["enum"]:
                        errors.append(f"{inp.name}: must be one of {inp.validation['enum']}")

        return len(errors) == 0, errors

    def register_tool_handler(
        self,
        tool_name: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register a handler for REPL tool actions.

        Args:
            tool_name: Name of the REPL tool.
            handler: Callable to execute the tool.
        """
        self._tool_handlers[tool_name] = handler

    def execute(
        self,
        procedure_id: str,
        role: str | None = None,
        dry_run: bool = False,
        **inputs: Any,
    ) -> ProcedureResult:
        """Execute a procedure.

        Args:
            procedure_id: ID of the procedure to execute.
            role: Role executing the procedure (for permission check).
            dry_run: If True, validate but don't execute.
            **inputs: Input values for the procedure.

        Returns:
            ProcedureResult with execution details.

        Raises:
            ProcedureExecutionError: If execution fails.
        """
        start_time = time.perf_counter()

        # Get procedure
        procedure = self.get(procedure_id)
        if procedure is None:
            return ProcedureResult(
                procedure_id=procedure_id,
                success=False,
                error=f"Procedure not found: {procedure_id}",
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Check permissions
        if role is not None:
            allowed_roles = procedure.permissions.get("roles", [])
            if allowed_roles and role not in allowed_roles:
                return ProcedureResult(
                    procedure_id=procedure_id,
                    success=False,
                    error=f"Role '{role}' not allowed to execute {procedure_id}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

        # Validate inputs
        is_valid, errors = self.validate_inputs(procedure, inputs)
        if not is_valid:
            return ProcedureResult(
                procedure_id=procedure_id,
                success=False,
                error=f"Input validation failed: {'; '.join(errors)}",
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Apply defaults
        resolved_inputs = {}
        for inp in procedure.inputs:
            value = inputs.get(inp.name, inp.default)
            resolved_inputs[inp.name] = value

        # Dry run - just return validation success
        if dry_run:
            return ProcedureResult(
                procedure_id=procedure_id,
                success=True,
                outputs={"dry_run": True, "inputs": resolved_inputs},
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Create checkpoint if rollback enabled
        checkpoint_id = None
        if procedure.rollback and procedure.rollback.get("checkpoint_before", True):
            checkpoint_id = self._create_checkpoint(procedure_id)

        # Initialize execution context
        self._context = {
            "inputs": resolved_inputs,
            "outputs": {},
            "step_outputs": {},
            "procedure": procedure,
        }

        # Execute steps
        step_results = []
        success = True
        error = None

        for step in procedure.steps:
            # Check dependencies
            for dep in step.depends_on:
                dep_result = next(
                    (r for r in step_results if r.step_id == dep), None
                )
                if dep_result is None or not dep_result.success:
                    step_results.append(
                        StepResult(
                            step_id=step.id,
                            success=False,
                            skipped=True,
                            skip_reason=f"Dependency {dep} not satisfied",
                        )
                    )
                    continue

            # Check condition
            if step.condition:
                try:
                    if not self._eval_condition(step.condition):
                        step_results.append(
                            StepResult(
                                step_id=step.id,
                                success=True,
                                skipped=True,
                                skip_reason="Condition evaluated to False",
                            )
                        )
                        continue
                except Exception as e:
                    step_results.append(
                        StepResult(
                            step_id=step.id,
                            success=False,
                            error=f"Condition evaluation failed: {e}",
                        )
                    )
                    if step.on_failure == "abort":
                        success = False
                        error = f"Step {step.id} condition failed"
                        break
                    continue

            # Execute step with retries
            step_result = self._execute_step(step)
            step_results.append(step_result)

            if not step_result.success:
                if step.on_failure == "abort":
                    success = False
                    error = f"Step {step.id} failed: {step_result.error}"
                    break
                elif step.on_failure == "rollback":
                    success = False
                    error = f"Step {step.id} failed, rolling back"
                    if checkpoint_id:
                        self._restore_checkpoint(checkpoint_id)
                    break
                # "continue" - just keep going

        # Build result
        result = ProcedureResult(
            procedure_id=procedure_id,
            success=success,
            step_results=step_results,
            outputs=self._context.get("outputs", {}),
            error=error,
            elapsed_seconds=time.perf_counter() - start_time,
            checkpoint_id=checkpoint_id,
        )

        # Save execution state
        self._save_execution_state(procedure_id, result)

        # Log to episodic memory if enabled
        if self._enable_memory:
            self._log_to_memory(procedure, result, kwargs)

        return result

    def _log_to_memory(
        self,
        procedure: Procedure,
        result: ProcedureResult,
        inputs: dict[str, Any],
    ) -> None:
        """Log procedure execution to episodic memory.

        Args:
            procedure: The executed procedure.
            result: The execution result.
            inputs: The input parameters used.
        """
        if not self._episodic_store or not self._embedder:
            return

        try:
            # Create embedding from procedure + context
            text = f"procedure:{procedure.id} | category:{procedure.category} | inputs:{list(inputs.keys())}"
            embedding = self._embedder.embed_text(text)

            # Determine outcome and Q-value
            outcome = "success" if result.success else "failure"
            initial_q = 0.9 if result.success else 0.3

            # Build context
            context = {
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "category": procedure.category,
                "inputs": {k: str(v)[:100] for k, v in inputs.items()},  # Truncate values
                "steps_completed": len([s for s in result.step_results if s.success]),
                "total_steps": len(result.step_results),
                "elapsed_seconds": result.elapsed_seconds,
                "error": result.error,
            }

            # Store to memory
            self._episodic_store.store(
                embedding=embedding,
                action=f"procedure:{procedure.id}",
                action_type="procedure_execution",
                context=context,
                outcome=outcome,
                initial_q=initial_q,
            )
        except Exception as e:
            logger.warning("Failed to log procedure to memory: %s", e)

    def _execute_step(self, step: ProcedureStep) -> StepResult:
        """Execute a single procedure step.

        Args:
            step: The step to execute.

        Returns:
            StepResult with execution details.
        """
        start_time = time.perf_counter()
        action = step.action
        action_type = action.get("type")

        # Interpolate variables in action
        action = self._interpolate_action(action)

        try:
            if action_type == "shell":
                output = self._execute_shell(action)
            elif action_type == "python":
                output = self._execute_python(action)
            elif action_type == "repl_tool":
                output = self._execute_repl_tool(action)
            elif action_type == "file_write":
                output = self._execute_file_write(action)
            elif action_type == "file_read":
                output = self._execute_file_read(action)
            elif action_type == "registry_update":
                output = self._execute_registry_update(action)
            elif action_type == "checkpoint":
                output = self._execute_checkpoint(action)
            elif action_type == "gate":
                output = self._execute_gate(action)
            elif action_type == "wait":
                output = self._execute_wait(action)
            else:
                return StepResult(
                    step_id=step.id,
                    success=False,
                    error=f"Unknown action type: {action_type}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            # Capture output if specified
            if action.get("capture_output"):
                self._context["step_outputs"][action["capture_output"]] = output

            return StepResult(
                step_id=step.id,
                success=True,
                output=str(output)[:1000],  # Cap output
                elapsed_seconds=time.perf_counter() - start_time,
            )

        except Exception as e:
            # Retry logic
            retries = 0
            while retries < step.max_retries:
                retries += 1
                try:
                    time.sleep(1)  # Brief delay between retries
                    # Re-execute based on type
                    if action_type == "shell":
                        output = self._execute_shell(action)
                        return StepResult(
                            step_id=step.id,
                            success=True,
                            output=str(output)[:1000],
                            elapsed_seconds=time.perf_counter() - start_time,
                        )
                except Exception as e:
                    logger.debug("Step %s retry %d failed: %s", step.id, retries, e)
                    continue

            return StepResult(
                step_id=step.id,
                success=False,
                error=str(e),
                elapsed_seconds=time.perf_counter() - start_time,
            )

    def _interpolate_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Interpolate variables in action parameters.

        Supports ${inputs.name}, ${outputs.name}, ${step_outputs.name}.
        """
        result = {}
        for key, value in action.items():
            if isinstance(value, str):
                result[key] = self._interpolate_string(value)
            elif isinstance(value, dict):
                result[key] = self._interpolate_action(value)
            else:
                result[key] = value
        return result

    def _interpolate_string(self, s: str) -> str:
        """Interpolate variables in a string."""
        pattern = r"\$\{(\w+)\.(\w+)\}"

        def replacer(match: re.Match) -> str:
            scope = match.group(1)
            name = match.group(2)
            if scope == "inputs":
                return str(self._context.get("inputs", {}).get(name, ""))
            elif scope == "outputs":
                return str(self._context.get("outputs", {}).get(name, ""))
            elif scope == "step_outputs":
                return str(self._context.get("step_outputs", {}).get(name, ""))
            return match.group(0)

        return re.sub(pattern, replacer, s)

    def _eval_condition(self, condition: str) -> bool:
        """Evaluate a condition expression."""
        # Build safe eval context
        eval_context = {
            "inputs": self._context.get("inputs", {}),
            "outputs": self._context.get("outputs", {}),
            "step_outputs": self._context.get("step_outputs", {}),
            "Path": Path,
            "os": os,
        }
        return bool(eval(condition, {"__builtins__": {}}, eval_context))

    def _execute_shell(self, action: dict[str, Any]) -> str:
        """Execute a shell command using shlex for safe argument splitting."""
        command = action.get("command", "")
        timeout = action.get("timeout_seconds", 300)
        working_dir = action.get("working_dir", "/mnt/raid0/llm/claude")

        # Validate working directory is on RAID
        if not working_dir.startswith("/mnt/raid0/"):
            raise ProcedureExecutionError(
                f"Working directory must be on /mnt/raid0/: {working_dir}"
            )

        try:
            cmd_parts = shlex.split(command)
        except ValueError as e:
            raise ProcedureExecutionError(f"Invalid command syntax: {e}") from e

        logger.debug("Executing shell: %s (cwd=%s)", cmd_parts, working_dir)
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )

        if result.returncode != 0:
            raise ProcedureExecutionError(
                f"Command failed (exit {result.returncode}): {result.stderr}"
            )

        return result.stdout

    def _execute_python(self, action: dict[str, Any]) -> Any:
        """Execute a Python expression."""
        command = action.get("command", "")
        eval_context = {
            "inputs": self._context.get("inputs", {}),
            "outputs": self._context.get("outputs", {}),
            "step_outputs": self._context.get("step_outputs", {}),
            "Path": Path,
            "os": os,
            "json": json,
            "yaml": yaml,
            "datetime": datetime,
        }
        return eval(command, {"__builtins__": __builtins__}, eval_context)

    def _execute_repl_tool(self, action: dict[str, Any]) -> Any:
        """Execute a REPL tool."""
        tool_name = action.get("tool")
        args = action.get("args", {})

        if tool_name not in self._tool_handlers:
            raise ProcedureExecutionError(f"Unknown REPL tool: {tool_name}")

        return self._tool_handlers[tool_name](**args)

    def _execute_file_write(self, action: dict[str, Any]) -> str:
        """Write content to a file."""
        path = action.get("args", {}).get("path")
        content = action.get("args", {}).get("content")

        if not path or content is None:
            raise ProcedureExecutionError("file_write requires path and content")

        # Security check
        if not path.startswith("/mnt/raid0/"):
            raise ProcedureExecutionError(f"Path not on RAID: {path}")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Wrote {len(content)} bytes to {path}"

    def _execute_file_read(self, action: dict[str, Any]) -> str:
        """Read content from a file."""
        path = action.get("args", {}).get("path")

        if not path:
            raise ProcedureExecutionError("file_read requires path")

        with open(path, encoding="utf-8") as f:
            return f.read()

    def _execute_registry_update(self, action: dict[str, Any]) -> str:
        """Update a registry entry."""
        registry_path = action.get("args", {}).get("registry_path")
        key_path = action.get("args", {}).get("key_path")
        value = action.get("args", {}).get("value")

        if not all([registry_path, key_path]):
            raise ProcedureExecutionError("registry_update requires registry_path and key_path")

        # Load registry
        with open(registry_path, encoding="utf-8") as f:
            if registry_path.endswith(".json"):
                registry = json.load(f)
            elif YAML_AVAILABLE:
                registry = yaml.safe_load(f)
            else:
                raise ProcedureExecutionError("YAML support not available")

        # Navigate to key
        keys = key_path.split(".")
        obj = registry
        for key in keys[:-1]:
            if key not in obj:
                obj[key] = {}
            obj = obj[key]

        # Update value
        obj[keys[-1]] = value

        # Write back
        with open(registry_path, "w", encoding="utf-8") as f:
            if registry_path.endswith(".json"):
                json.dump(registry, f, indent=2)
            elif YAML_AVAILABLE:
                yaml.dump(registry, f, default_flow_style=False)
            else:
                raise ProcedureExecutionError("YAML support not available")

        return f"Updated {key_path} in {registry_path}"

    def _execute_checkpoint(self, action: dict[str, Any]) -> str:
        """Create or restore a checkpoint."""
        checkpoint_action = action.get("args", {}).get("action", "create")
        checkpoint_id = action.get("args", {}).get("id")

        if checkpoint_action == "create":
            return self._create_checkpoint(checkpoint_id or "auto")
        elif checkpoint_action == "restore":
            if not checkpoint_id:
                raise ProcedureExecutionError("restore requires checkpoint id")
            return self._restore_checkpoint(checkpoint_id)
        else:
            raise ProcedureExecutionError(f"Unknown checkpoint action: {checkpoint_action}")

    def _execute_gate(self, action: dict[str, Any]) -> str:
        """Run a verification gate."""
        gate_name = action.get("args", {}).get("gate")

        cwd = "/mnt/raid0/llm/claude"

        # Map gates to argument lists (no shell=True needed)
        gate_commands: dict[str, list[str]] = {
            "schema": ["python3", "orchestration/validate_ir.py", "task",
                        "orchestration/last_task_ir.json"],
            "lint": ["ruff", "check", "src/"],
            "format": ["ruff", "format", "--check", "src/"],
            "shellcheck": ["shellcheck"]
                          + sorted(glob.glob("scripts/**/*.sh",
                                             root_dir=cwd, recursive=True)),
            "unit": ["pytest", "tests/unit/", "-q"],
        }

        if gate_name not in gate_commands:
            raise ProcedureExecutionError(f"Unknown gate: {gate_name}")

        logger.debug("Running gate %s: %s", gate_name, gate_commands[gate_name])
        result = subprocess.run(
            gate_commands[gate_name],
            capture_output=True,
            text=True,
            cwd=cwd,
        )

        if result.returncode != 0:
            raise ProcedureExecutionError(f"Gate {gate_name} failed: {result.stderr}")

        return f"Gate {gate_name} passed"

    def _execute_wait(self, action: dict[str, Any]) -> str:
        """Wait for a specified duration."""
        seconds = action.get("args", {}).get("seconds", 1)
        time.sleep(seconds)
        return f"Waited {seconds} seconds"

    def _create_checkpoint(self, name: str) -> str:
        """Create a checkpoint of current state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{name}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        checkpoint_data = {
            "id": checkpoint_id,
            "created_at": datetime.now().isoformat(),
            "context": self._context,
        }

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        return checkpoint_id

    def _restore_checkpoint(self, checkpoint_id: str) -> str:
        """Restore state from a checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_path.exists():
            raise ProcedureExecutionError(f"Checkpoint not found: {checkpoint_id}")

        with open(checkpoint_path, encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        self._context = checkpoint_data.get("context", {})
        return f"Restored checkpoint {checkpoint_id}"

    def _save_execution_state(
        self,
        procedure_id: str,
        result: ProcedureResult,
    ) -> None:
        """Save execution state for debugging and logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_path = self.state_dir / f"{procedure_id}_{timestamp}.json"

        state_data = {
            "procedure_id": procedure_id,
            "success": result.success,
            "error": result.error,
            "elapsed_seconds": result.elapsed_seconds,
            "step_results": [
                {
                    "step_id": sr.step_id,
                    "success": sr.success,
                    "output": sr.output[:500] if sr.output else None,
                    "error": sr.error,
                    "skipped": sr.skipped,
                }
                for sr in result.step_results
            ],
            "outputs": result.outputs,
            "executed_at": datetime.now().isoformat(),
        }

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, default=str)
