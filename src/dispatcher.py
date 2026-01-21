#!/usr/bin/env python3
"""Dispatcher for hierarchical local-agent orchestration.

This module reads TaskIR JSON and routes work to appropriate models
based on the model registry configuration.

Usage:
    from src.dispatcher import Dispatcher

    dispatcher = Dispatcher()
    result = dispatcher.dispatch(task_ir)
    # result contains routing decisions and generated commands
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.registry_loader import RegistryLoader, RoleConfig

if TYPE_CHECKING:
    from orchestration.repl_memory.progress_logger import ProgressLogger
    from orchestration.repl_memory.retriever import HybridRouter


@dataclass
class StepExecution:
    """Execution plan for a single step."""

    step_id: str
    actor: str
    action: str
    role_config: RoleConfig | None
    command: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    parallel_group: str | None = None


@dataclass
class DispatchResult:
    """Result of dispatching a TaskIR."""

    task_id: str
    timestamp: str
    roles_used: list[str]
    steps: list[StepExecution]
    routing_strategy: str = "rules"  # "learned" or "rules"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "roles_used": self.roles_used,
            "routing_strategy": self.routing_strategy,
            "steps": [
                {
                    "step_id": s.step_id,
                    "actor": s.actor,
                    "action": s.action,
                    "role": s.role_config.name if s.role_config else None,
                    "command": s.command,
                    "inputs": s.inputs,
                    "outputs": s.outputs,
                    "depends_on": s.depends_on,
                    "parallel_group": s.parallel_group,
                }
                for s in self.steps
            ],
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DispatchError(Exception):
    """Error during dispatch."""

    pass


class Dispatcher:
    """Dispatch TaskIR to appropriate models and generate execution plans."""

    # Map TaskIR agent roles to registry role names
    ROLE_MAPPING = {
        "frontdoor": "frontdoor",
        "coder": "coder_primary",
        "ingest": "ingest_long_context",
        "architect": "architect_general",
        "worker": "worker_general",
        "docwriter": "worker_general",
        "math": "worker_math",
        "vision": "worker_vision",
        "toolrunner": "toolrunner",
        "draft": "draft_qwen25_coder",
        # Document processing roles
        "doc": "document_formalizer",
        "document": "document_formalizer",
        "document_formalizer": "document_formalizer",
        "ocr": "document_formalizer",
    }

    def __init__(
        self,
        registry: RegistryLoader | None = None,
        validate_paths: bool = True,
        progress_logger: "ProgressLogger | None" = None,
        hybrid_router: "HybridRouter | None" = None,
    ):
        """Initialize the dispatcher.

        Args:
            registry: Pre-loaded registry. Loads default if None.
            validate_paths: Validate model paths exist.
            progress_logger: Optional ProgressLogger for MemRL integration.
            hybrid_router: Optional HybridRouter for learned routing.
        """
        self.registry = registry or RegistryLoader(validate_paths=validate_paths)
        self.progress_logger = progress_logger
        self.hybrid_router = hybrid_router
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._routing_strategy: str = "rules"  # Track last routing strategy

    def dispatch(self, task_ir: dict[str, Any]) -> DispatchResult:
        """Dispatch a TaskIR to generate execution plan.

        Args:
            task_ir: Parsed TaskIR JSON.

        Returns:
            DispatchResult with execution plan and commands.
        """
        self._warnings = []
        self._errors = []
        self._routing_strategy = "rules"

        # Extract task info
        task_id = task_ir.get("task_id", str(uuid.uuid4()))
        timestamp = datetime.now().isoformat()

        # Determine roles needed (uses HybridRouter if available)
        roles_used = self._determine_roles(task_ir)

        # Log task start with routing decision (MemRL integration)
        if self.progress_logger:
            self.progress_logger.log_task_started(
                task_id=task_id,
                task_ir=task_ir,
                routing_decision=roles_used,
                routing_strategy=self._routing_strategy,
            )

        # Generate step executions
        steps = self._generate_steps(task_ir, roles_used)

        return DispatchResult(
            task_id=task_id,
            timestamp=timestamp,
            roles_used=roles_used,
            routing_strategy=self._routing_strategy,
            steps=steps,
            warnings=self._warnings,
            errors=self._errors,
        )

    def _determine_roles(self, task_ir: dict[str, Any]) -> list[str]:
        """Determine which registry roles to use based on TaskIR.

        First checks explicit agents in TaskIR, then falls back to routing hints.
        """
        roles = []

        # Check explicit agents in TaskIR
        agents = task_ir.get("agents", [])
        for agent in agents:
            ir_role = agent.get("role", "")
            model_hint = agent.get("model_hint")

            # Use model_hint if provided, otherwise map from IR role
            if model_hint and model_hint in self.registry.roles:
                registry_role = model_hint
            elif ir_role in self.ROLE_MAPPING:
                registry_role = self.ROLE_MAPPING[ir_role]
            else:
                self._warnings.append(f"Unknown role '{ir_role}', defaulting to worker_general")
                registry_role = "worker_general"

            if registry_role not in roles:
                roles.append(registry_role)

        # If no explicit agents, use HybridRouter (learned + rules) or fall back to registry
        if not roles:
            if self.hybrid_router:
                # Use learned routing with rule-based fallback
                roles, self._routing_strategy = self.hybrid_router.route(task_ir)
            else:
                # Pure rule-based routing from registry
                roles = self.registry.route_task(task_ir)
                self._routing_strategy = "rules"

        # Always include draft model if using speculative decoding
        for role_name in list(roles):
            try:
                role = self.registry.get_role(role_name)
                if role.acceleration.type == "speculative_decoding":
                    draft_role = role.acceleration.draft_role
                    if draft_role and draft_role not in roles:
                        roles.append(draft_role)
            except KeyError:
                pass

        return roles

    def _generate_steps(
        self, task_ir: dict[str, Any], roles_used: list[str]
    ) -> list[StepExecution]:
        """Generate step execution plans from TaskIR."""
        steps = []

        plan = task_ir.get("plan", {})
        plan_steps = plan.get("steps", [])

        for step in plan_steps:
            step_id = step.get("id", "S0")
            actor = step.get("actor", "")
            action = step.get("action", "")
            inputs = step.get("inputs", [])
            outputs = step.get("outputs", [])
            depends_on = step.get("depends_on", [])
            parallel_group = step.get("parallel_group")

            # Map actor to registry role
            registry_role = self._map_actor_to_role(actor, roles_used)

            # Get role config and generate command
            role_config = None
            command = ""

            if registry_role:
                try:
                    role_config = self.registry.get_role(registry_role)
                    command = self._generate_command_for_step(role_config, action, inputs)
                except KeyError:
                    self._errors.append(f"Role '{registry_role}' not found in registry")

            steps.append(
                StepExecution(
                    step_id=step_id,
                    actor=actor,
                    action=action,
                    role_config=role_config,
                    command=command,
                    inputs=inputs,
                    outputs=outputs,
                    depends_on=depends_on,
                    parallel_group=parallel_group,
                )
            )

        return steps

    def _map_actor_to_role(self, actor: str, roles_used: list[str]) -> str | None:
        """Map a step actor to a registry role name."""
        # Direct match in roles_used
        if actor in roles_used:
            return actor

        # Check role mapping
        if actor in self.ROLE_MAPPING:
            mapped = self.ROLE_MAPPING[actor]
            if mapped in roles_used or mapped in self.registry.roles:
                return mapped

        # Try to find a role in roles_used that matches
        for role_name in roles_used:
            if actor.lower() in role_name.lower():
                return role_name

        # Default fallback
        if "worker_general" in roles_used:
            return "worker_general"

        return None

    def _generate_command_for_step(
        self,
        role: RoleConfig,
        action: str,
        inputs: list[str],
    ) -> str:
        """Generate a llama.cpp command for a step.

        Creates a prompt from the action and generates the appropriate command
        based on the role's acceleration configuration.
        """
        # Build prompt from action
        prompt = f"Task: {action}"
        if inputs:
            prompt += "\n\nInputs:\n" + "\n".join(f"- {i}" for i in inputs)

        # Use registry's command generation
        return self.registry.generate_command(
            role.name,
            prompt=prompt,
            n_tokens=512,  # Default, can be overridden
        )

    def log_task_completed(
        self,
        task_id: str,
        success: bool,
        details: str | None = None,
    ) -> None:
        """Log task completion for MemRL Q-scoring.

        Args:
            task_id: The task ID from DispatchResult.
            success: Whether the task completed successfully.
            details: Optional details about the outcome.
        """
        if self.progress_logger:
            self.progress_logger.log_task_completed(
                task_id=task_id,
                success=success,
                details=details,
            )

    def dispatch_from_file(self, path: Path | str) -> DispatchResult:
        """Dispatch from a TaskIR JSON file."""
        path = Path(path)
        if not path.exists():
            raise DispatchError(f"TaskIR file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                task_ir = json.load(f)
        except json.JSONDecodeError as e:
            raise DispatchError(f"Invalid JSON in TaskIR file: {e}") from e

        return self.dispatch(task_ir)

    def dispatch_from_stdin(self) -> DispatchResult:
        """Dispatch from stdin."""
        import sys

        try:
            task_ir = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            raise DispatchError(f"Invalid JSON from stdin: {e}") from e

        return self.dispatch(task_ir)


def main() -> int:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: dispatcher.py <task_ir.json | ->")
        print("  Use '-' to read from stdin")
        return 1

    try:
        dispatcher = Dispatcher()

        if sys.argv[1] == "-":
            result = dispatcher.dispatch_from_stdin()
        else:
            result = dispatcher.dispatch_from_file(sys.argv[1])

        # Output as JSON
        print(json.dumps(result.to_dict(), indent=2))

        if result.errors:
            return 2
        return 0

    except DispatchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    import sys

    sys.exit(main())
