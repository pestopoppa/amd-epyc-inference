#!/usr/bin/env python3
"""Load and validate the model registry for hierarchical orchestration.

This module provides typed access to model configurations, acceleration
strategies, and command generation for llama.cpp inference.

Usage:
    from src.registry_loader import RegistryLoader

    registry = RegistryLoader()
    role = registry.get_role("coder_primary")
    cmd = registry.generate_command("coder_primary", prompt="Write a function...")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Default registry location
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "orchestration" / "model_registry.yaml"


@dataclass
class ModelConfig:
    """Model file configuration."""
    name: str
    path: str  # Relative to model_base_path
    quant: str
    size_gb: float
    full_path: str = ""  # Resolved absolute path
    architecture: str | None = None
    split_count: int | None = None
    mmproj_path: str | None = None


@dataclass
class AccelerationConfig:
    """Acceleration strategy configuration."""
    type: str  # speculative_decoding, moe_expert_reduction, prompt_lookup, none
    draft_role: str | None = None
    k: int | None = None  # Draft tokens for speculative
    experts: int | None = None  # Expert count for MoE reduction
    override_key: str | None = None
    ngram_min: int | None = None  # For prompt lookup
    temperature: float | None = None


@dataclass
class PerformanceMetrics:
    """Performance metrics from benchmarks."""
    baseline_tps: float | None = None
    optimized_tps: float | None = None
    speedup: str | None = None
    acceptance_rate: str | None = None
    raw_tps: float | None = None


@dataclass
class MemoryConfig:
    """Memory residency configuration."""
    residency: str  # hot, warm
    pinned: bool = False
    max_instances: int = 1


@dataclass
class Constraints:
    """Acceleration constraints for a role."""
    forbid: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass
class RoleConfig:
    """Complete configuration for an agent role."""
    name: str
    tier: str
    description: str
    model: ModelConfig
    acceleration: AccelerationConfig
    performance: PerformanceMetrics
    memory: MemoryConfig
    constraints: Constraints | None = None
    compatible_targets: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class RoutingHint:
    """Deterministic routing rule."""
    condition: str
    use: list[str]


class RegistryError(Exception):
    """Error loading or validating the registry."""
    pass


class RegistryLoader:
    """Load and provide access to the model registry."""

    def __init__(
        self,
        registry_path: Path | str | None = None,
        validate_paths: bool = True,
    ):
        """Initialize the registry loader.

        Args:
            registry_path: Path to model_registry.yaml. Uses default if None.
            validate_paths: If True, verify model files exist on disk.
        """
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self._raw: dict[str, Any] = {}
        self._roles: dict[str, RoleConfig] = {}
        self._routing_hints: list[RoutingHint] = []
        self._command_templates: dict[str, str] = {}
        self._model_base_path: Path = Path("/mnt/raid0/llm/lmstudio/models")
        self._runtime_defaults: dict[str, Any] = {}
        self._missing_models: list[str] = []

        self._load(validate_paths)

    def _load(self, validate_paths: bool) -> None:
        """Load and parse the registry YAML."""
        if not self.registry_path.exists():
            raise RegistryError(f"Registry not found: {self.registry_path}")

        try:
            with self.registry_path.open("r", encoding="utf-8") as f:
                self._raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RegistryError(f"Invalid YAML in registry: {e}") from e

        if not self._raw:
            raise RegistryError("Empty registry file")

        # Load runtime defaults
        self._runtime_defaults = self._raw.get("runtime_defaults", {})
        base_path = self._runtime_defaults.get("model_base_path")
        if base_path:
            self._model_base_path = Path(base_path)

        # Load roles
        roles_data = self._raw.get("roles", {})
        if not roles_data:
            raise RegistryError("No roles defined in registry")

        for role_name, role_data in roles_data.items():
            try:
                role_config = self._parse_role(role_name, role_data)
                if validate_paths:
                    self._validate_model_path(role_config)
                self._roles[role_name] = role_config
            except Exception as e:
                raise RegistryError(f"Error parsing role '{role_name}': {e}") from e

        # Load routing hints
        hints_data = self._raw.get("routing_hints", [])
        for hint in hints_data:
            self._routing_hints.append(RoutingHint(
                condition=hint.get("if", ""),
                use=hint.get("use", []),
            ))

        # Load command templates
        self._command_templates = self._raw.get("command_templates", {})

    def _parse_role(self, name: str, data: dict[str, Any]) -> RoleConfig:
        """Parse a role configuration from YAML data."""
        model_data = data.get("model", {})
        accel_data = data.get("acceleration", {})
        perf_data = data.get("performance", {})
        mem_data = data.get("memory", {})
        constraints_data = data.get("constraints")

        # Build model config
        model = ModelConfig(
            name=model_data.get("name", ""),
            path=model_data.get("path", ""),
            quant=model_data.get("quant", "Q4_K_M"),
            size_gb=model_data.get("size_gb", 0.0),
            architecture=model_data.get("architecture"),
            split_count=model_data.get("split_count"),
            mmproj_path=model_data.get("mmproj_path"),
        )

        # Resolve full path
        if model.path:
            model.full_path = str(self._model_base_path / model.path)

        # Build acceleration config
        acceleration = AccelerationConfig(
            type=accel_data.get("type", "none"),
            draft_role=accel_data.get("draft_role"),
            k=accel_data.get("k"),
            experts=accel_data.get("experts"),
            override_key=accel_data.get("override_key"),
            ngram_min=accel_data.get("ngram_min", 3),
            temperature=accel_data.get("temperature"),
        )

        # Build performance metrics
        performance = PerformanceMetrics(
            baseline_tps=perf_data.get("baseline_tps"),
            optimized_tps=perf_data.get("optimized_tps"),
            speedup=perf_data.get("speedup"),
            acceptance_rate=perf_data.get("acceptance_rate"),
            raw_tps=perf_data.get("raw_tps"),
        )

        # Build memory config
        memory = MemoryConfig(
            residency=mem_data.get("residency", "warm"),
            pinned=mem_data.get("pinned", False),
            max_instances=mem_data.get("max_instances", 1),
        )

        # Build constraints
        constraints = None
        if constraints_data:
            constraints = Constraints(
                forbid=constraints_data.get("forbid", []),
                reason=constraints_data.get("reason"),
            )

        return RoleConfig(
            name=name,
            tier=data.get("tier", "C"),
            description=data.get("description", ""),
            model=model,
            acceleration=acceleration,
            performance=performance,
            memory=memory,
            constraints=constraints,
            compatible_targets=data.get("compatible_targets", []),
            notes=data.get("notes"),
        )

    def _validate_model_path(self, role: RoleConfig) -> None:
        """Check if model file exists, record missing models."""
        if not role.model.full_path:
            return
        path = Path(role.model.full_path)
        if not path.exists():
            self._missing_models.append(f"{role.name}: {role.model.full_path}")

    @property
    def roles(self) -> dict[str, RoleConfig]:
        """All loaded role configurations."""
        return self._roles

    @property
    def routing_hints(self) -> list[RoutingHint]:
        """Deterministic routing rules."""
        return self._routing_hints

    @property
    def missing_models(self) -> list[str]:
        """List of roles with missing model files."""
        return self._missing_models

    @property
    def model_base_path(self) -> Path:
        """Base path for model files."""
        return self._model_base_path

    def get_role(self, name: str) -> RoleConfig:
        """Get configuration for a specific role.

        Raises:
            KeyError: If role not found.
        """
        if name not in self._roles:
            raise KeyError(f"Role not found: {name}. Available: {list(self._roles.keys())}")
        return self._roles[name]

    def get_roles_by_tier(self, tier: str) -> list[RoleConfig]:
        """Get all roles in a specific tier (A, B, C, D)."""
        return [r for r in self._roles.values() if r.tier == tier]

    def get_draft_for_role(self, role_name: str) -> RoleConfig | None:
        """Get the draft model configuration for a role using speculative decoding."""
        role = self.get_role(role_name)
        if role.acceleration.type != "speculative_decoding":
            return None
        draft_role = role.acceleration.draft_role
        if draft_role and draft_role in self._roles:
            return self._roles[draft_role]
        return None

    def generate_command(
        self,
        role_name: str,
        prompt: str | None = None,
        prompt_file: str | None = None,
        n_tokens: int = 128,
        extra_args: dict[str, Any] | None = None,
    ) -> str:
        """Generate a llama.cpp command for a role.

        Args:
            role_name: The role to generate a command for.
            prompt: Inline prompt text.
            prompt_file: Path to prompt file (overrides prompt).
            n_tokens: Number of tokens to generate.
            extra_args: Additional arguments to append.

        Returns:
            Shell command string.
        """
        role = self.get_role(role_name)
        accel_type = role.acceleration.type

        # Get template
        template_name = accel_type if accel_type != "none" else "baseline"
        if template_name not in self._command_templates:
            template_name = "baseline"
        template = self._command_templates.get(template_name, "")

        # Build substitution dict
        defaults = self._runtime_defaults
        subs = {
            "model_path": role.model.full_path,
            "threads": defaults.get("threads", 96),
            "context_length": defaults.get("context_length", 8192),
        }

        # Add acceleration-specific params
        if accel_type == "speculative_decoding":
            draft = self.get_draft_for_role(role_name)
            if draft:
                subs["draft_path"] = draft.model.full_path
            subs["k"] = role.acceleration.k or 16

        elif accel_type == "moe_expert_reduction":
            subs["override_key"] = role.acceleration.override_key or "moe_n_expert"
            subs["experts"] = role.acceleration.experts or 4

        elif accel_type == "prompt_lookup":
            subs["ngram_min"] = role.acceleration.ngram_min or 3

        # Vision model
        if role.model.mmproj_path:
            subs["mmproj_path"] = str(self._model_base_path / role.model.mmproj_path)
            template_name = "vision"
            template = self._command_templates.get("vision", template)

        # Apply substitutions
        cmd = template.strip()
        for key, value in subs.items():
            cmd = cmd.replace(f"{{{key}}}", str(value))

        # Add prompt
        if prompt_file:
            cmd += f" \\\n      -f {prompt_file}"
        elif prompt:
            # Escape for shell
            safe_prompt = prompt.replace("'", "'\\''")
            cmd += f" \\\n      -p '{safe_prompt}'"

        # Add token count and common flags
        cmd += f" \\\n      -n {n_tokens}"
        cmd += " \\\n      --no-display-prompt"
        cmd += " \\\n      --simple-io"

        # Add extra args
        if extra_args:
            for k, v in extra_args.items():
                cmd += f" \\\n      --{k} {v}"

        return cmd

    def route_task(self, task_ir: dict[str, Any]) -> list[str]:
        """Determine which roles to use based on TaskIR.

        Uses the routing_hints from the registry to match task properties.

        Args:
            task_ir: Parsed TaskIR JSON.

        Returns:
            List of role names to use.
        """
        # Extract relevant fields for matching
        task_type = task_ir.get("task_type", "")
        priority = task_ir.get("priority", "batch")
        objective = task_ir.get("objective", "").lower()
        constraints = [c.lower() for c in task_ir.get("constraints", [])]
        inputs = task_ir.get("inputs", [])
        input_types = [i.get("type", "") for i in inputs]
        escalation = task_ir.get("escalation", {})

        # Check each routing hint
        for hint in self._routing_hints:
            condition = hint.condition

            # Simple condition evaluation (not a full expression parser)
            # Supports: task_type == 'x', priority == 'x', 'x' in objective, etc.
            try:
                # Build evaluation context
                ctx = {
                    "task_type": task_type,
                    "priority": priority,
                    "objective": objective,
                    "constraints": constraints,
                    "inputs": input_types,
                    "escalation": escalation,
                }

                # Evaluate condition (limited, safe subset)
                if self._eval_condition(condition, ctx):
                    return hint.use
            except Exception:
                continue

        # Default: frontdoor only
        return ["frontdoor"]

    def _eval_condition(self, condition: str, ctx: dict[str, Any]) -> bool:
        """Safely evaluate a routing condition.

        Supports limited expressions like:
        - task_type == 'code'
        - 'refactor' in objective
        - priority == 'interactive' and task_type == 'chat'
        """
        # Handle 'and' by splitting
        if " and " in condition:
            parts = condition.split(" and ")
            return all(self._eval_condition(p.strip(), ctx) for p in parts)

        # Handle 'or'
        if " or " in condition:
            parts = condition.split(" or ")
            return any(self._eval_condition(p.strip(), ctx) for p in parts)

        # Handle equality: x == 'y'
        eq_match = re.match(r"(\w+(?:\.\w+)?)\s*==\s*['\"]([^'\"]+)['\"]", condition)
        if eq_match:
            key, value = eq_match.groups()
            actual = self._get_nested(ctx, key)
            return actual == value

        # Handle 'x' in y
        in_match = re.match(r"['\"]([^'\"]+)['\"]\s+in\s+(\w+)", condition)
        if in_match:
            needle, key = in_match.groups()
            haystack = ctx.get(key, "")
            if isinstance(haystack, str):
                return needle in haystack
            if isinstance(haystack, list):
                return needle in haystack
            return False

        return False

    def _get_nested(self, ctx: dict[str, Any], key: str) -> Any:
        """Get a potentially nested value from context (e.g., 'escalation.max_level')."""
        parts = key.split(".")
        value = ctx
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def summary(self) -> str:
        """Return a summary of loaded roles."""
        lines = [
            f"Registry: {self.registry_path}",
            f"Base path: {self._model_base_path}",
            f"Roles loaded: {len(self._roles)}",
            "",
        ]

        for tier in ["A", "B", "C", "D"]:
            tier_roles = self.get_roles_by_tier(tier)
            if tier_roles:
                lines.append(f"Tier {tier}:")
                for r in tier_roles:
                    status = "OK" if r.model.full_path and Path(r.model.full_path).exists() else "MISSING"
                    lines.append(f"  {r.name}: {r.model.name} [{status}]")
                lines.append("")

        if self._missing_models:
            lines.append("Missing models:")
            for m in self._missing_models:
                lines.append(f"  {m}")

        return "\n".join(lines)


def main() -> int:
    """CLI entry point for testing."""
    import sys

    try:
        loader = RegistryLoader(validate_paths=True)
        print(loader.summary())

        if loader.missing_models:
            print(f"\nWarning: {len(loader.missing_models)} model(s) not found on disk")
            return 1

        print("\nAll models validated successfully.")
        return 0

    except RegistryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
