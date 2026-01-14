#!/usr/bin/env python3
"""Gate Runner for automated quality gates.

This module runs quality gates (linting, testing, schema validation) and
parses their output to provide structured failure information for agent feedback.

Usage:
    from src.gate_runner import GateRunner

    runner = GateRunner()
    results = runner.run_all_gates()

    if not all(r.passed for r in results):
        failed = [r for r in results if not r.passed]
        for r in failed:
            print(f"Gate '{r.gate_name}' failed: {r.errors}")
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from orchestration.repl_memory.progress_logger import ProgressLogger


@dataclass
class GateConfig:
    """Configuration for a single gate."""

    name: str
    command: str
    timeout: int = 60
    required: bool = True
    retry_count: int = 0
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateConfig:
        """Create GateConfig from dictionary."""
        return cls(
            name=data["name"],
            command=data["command"],
            timeout=data.get("timeout", 60),
            required=data.get("required", True),
            retry_count=data.get("retry_count", 0),
            description=data.get("description", ""),
        )


@dataclass
class GateResult:
    """Result of running a single gate."""

    gate_name: str
    passed: bool
    exit_code: int
    output: str
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempt: int = 1
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "elapsed_seconds": self.elapsed_seconds,
            "errors": self.errors,
            "warnings": self.warnings,
            "attempt": self.attempt,
            "required": self.required,
        }

    @property
    def summary(self) -> str:
        """Get a one-line summary of the result."""
        status = "PASSED" if self.passed else "FAILED"
        return f"[{status}] {self.gate_name} ({self.elapsed_seconds:.1f}s)"


class GateRunnerError(Exception):
    """Error in gate runner."""

    pass


class GateRunner:
    """Runs quality gates and parses output.

    Loads gate configuration from a YAML file and executes gates
    in order, collecting structured results.
    """

    DEFAULT_CONFIG_PATH = Path("config/gates.yaml")

    def __init__(
        self,
        config_path: Path | str | None = None,
        working_dir: Path | str | None = None,
        progress_logger: "ProgressLogger | None" = None,
    ):
        """Initialize the gate runner.

        Args:
            config_path: Path to gates.yaml configuration file.
            working_dir: Working directory for running commands.
            progress_logger: Optional ProgressLogger for MemRL integration.
        """
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.progress_logger = progress_logger
        self.gates: list[GateConfig] = []
        self._load_config()

    def _load_config(self) -> None:
        """Load gate configuration from YAML file."""
        if not self.config_path.exists():
            # Use default gates if config doesn't exist
            self.gates = self._get_default_gates()
            return

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            self.gates = [
                GateConfig.from_dict(gate_data)
                for gate_data in config.get("gates", [])
            ]
        except Exception as e:
            raise GateRunnerError(f"Failed to load config: {e}")

    def _get_default_gates(self) -> list[GateConfig]:
        """Get default gate configuration."""
        return [
            GateConfig(
                name="format",
                command="make format-check",
                timeout=30,
                required=True,
                description="Check code formatting",
            ),
            GateConfig(
                name="lint",
                command="make lint",
                timeout=60,
                required=True,
                description="Run linters",
            ),
            GateConfig(
                name="typecheck",
                command="make typecheck",
                timeout=120,
                required=False,
                description="Run type checker",
            ),
            GateConfig(
                name="unit",
                command="make test-unit",
                timeout=180,
                required=True,
                description="Run unit tests",
            ),
        ]

    def run_gate(
        self,
        gate: GateConfig,
        attempt: int = 1,
        task_id: str | None = None,
        agent_tier: str = "C",
        agent_role: str = "worker",
    ) -> GateResult:
        """Run a single gate and parse output.

        Args:
            gate: Gate configuration.
            attempt: Current attempt number (for retries).
            task_id: Optional task ID for MemRL logging.
            agent_tier: Agent tier for logging (default "C" for workers).
            agent_role: Agent role for logging.

        Returns:
            GateResult with parsed output.
        """
        start_time = time.perf_counter()

        try:
            result = subprocess.run(
                gate.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=gate.timeout,
                cwd=self.working_dir,
            )

            elapsed = time.perf_counter() - start_time
            output = result.stdout + "\n" + result.stderr
            passed = result.returncode == 0

            # Parse errors and warnings from output
            errors, warnings = self._parse_output(output, gate.name)

            gate_result = GateResult(
                gate_name=gate.name,
                passed=passed,
                exit_code=result.returncode,
                output=output,
                elapsed_seconds=elapsed,
                errors=errors if not passed else [],
                warnings=warnings,
                attempt=attempt,
                required=gate.required,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_time
            gate_result = GateResult(
                gate_name=gate.name,
                passed=False,
                exit_code=-1,
                output=f"Timeout after {gate.timeout}s",
                elapsed_seconds=elapsed,
                errors=[f"Gate timed out after {gate.timeout}s"],
                attempt=attempt,
                required=gate.required,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            gate_result = GateResult(
                gate_name=gate.name,
                passed=False,
                exit_code=-1,
                output=str(e),
                elapsed_seconds=elapsed,
                errors=[f"Gate execution error: {e}"],
                attempt=attempt,
                required=gate.required,
            )

        # Log gate result for MemRL if configured
        if self.progress_logger and task_id:
            error_msg = gate_result.errors[0] if gate_result.errors else None
            self.progress_logger.log_gate_result(
                task_id=task_id,
                gate_name=gate.name,
                passed=gate_result.passed,
                agent_tier=agent_tier,
                agent_role=agent_role,
                error_message=error_msg,
            )

        return gate_result

    def _parse_output(self, output: str, gate_name: str) -> tuple[list[str], list[str]]:
        """Parse gate output for errors and warnings.

        Args:
            output: Raw gate output.
            gate_name: Name of the gate (for context-specific parsing).

        Returns:
            Tuple of (errors, warnings) lists.
        """
        errors = []
        warnings = []

        for line in output.split("\n"):
            line_lower = line.lower()

            # Skip common log prefixes
            if any(prefix in line_lower for prefix in ["build:", "main:", "info:"]):
                continue

            # Detect errors
            if any(pattern in line_lower for pattern in ["error:", "error ", "failed", "fatal"]):
                # Clean up the error message
                clean_line = line.strip()
                if clean_line and clean_line not in errors:
                    errors.append(clean_line)

            # Detect warnings
            elif "warning:" in line_lower or "warn " in line_lower:
                clean_line = line.strip()
                if clean_line and clean_line not in warnings:
                    warnings.append(clean_line)

        # Limit to most relevant errors
        if len(errors) > 20:
            errors = errors[:20] + [f"... and {len(errors) - 20} more errors"]

        return errors, warnings

    def run_all_gates(
        self,
        stop_on_first_failure: bool = True,
        required_only: bool = False,
        task_id: str | None = None,
        agent_tier: str = "C",
        agent_role: str = "worker",
    ) -> list[GateResult]:
        """Run all configured gates.

        Args:
            stop_on_first_failure: Stop after first required gate fails.
            required_only: Only run gates marked as required.
            task_id: Optional task ID for MemRL logging.
            agent_tier: Agent tier for logging.
            agent_role: Agent role for logging.

        Returns:
            List of GateResult objects.
        """
        results = []

        for gate in self.gates:
            if required_only and not gate.required:
                continue

            # Run with retries
            for attempt in range(1, gate.retry_count + 2):  # +2 for first attempt
                result = self.run_gate(
                    gate,
                    attempt=attempt,
                    task_id=task_id,
                    agent_tier=agent_tier,
                    agent_role=agent_role,
                )
                if result.passed:
                    break

            results.append(result)

            # Check if we should stop
            if stop_on_first_failure and not result.passed and gate.required:
                break

        return results

    def run_gates_by_name(self, gate_names: list[str]) -> list[GateResult]:
        """Run specific gates by name.

        Args:
            gate_names: List of gate names to run.

        Returns:
            List of GateResult objects.
        """
        results = []

        for name in gate_names:
            gate = next((g for g in self.gates if g.name == name), None)
            if gate is None:
                results.append(
                    GateResult(
                        gate_name=name,
                        passed=False,
                        exit_code=-1,
                        output="",
                        elapsed_seconds=0,
                        errors=[f"Unknown gate: {name}"],
                    )
                )
            else:
                results.append(self.run_gate(gate))

        return results

    def get_gate_names(self) -> list[str]:
        """Get list of configured gate names."""
        return [g.name for g in self.gates]

    def get_summary(self, results: list[GateResult]) -> str:
        """Get a summary of gate results.

        Args:
            results: List of GateResult objects.

        Returns:
            Multi-line summary string.
        """
        lines = ["Gate Results:"]
        lines.append("-" * 40)

        for r in results:
            lines.append(r.summary)
            if r.errors:
                for e in r.errors[:3]:
                    lines.append(f"  - {e[:100]}")

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        lines.append("-" * 40)
        lines.append(f"Total: {passed}/{total} passed, {failed} failed")

        return "\n".join(lines)
