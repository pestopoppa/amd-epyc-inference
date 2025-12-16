#!/usr/bin/env python3
"""Model Server for hierarchical local-agent orchestration.

This module manages model processes, handles inference requests, and tracks
model health and memory residency.

Usage:
    from src.model_server import ModelServer

    server = ModelServer()
    server.load("coder_primary")
    result = server.infer("coder_primary", prompt="Write a function")
    server.unload("coder_primary")
"""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.registry_loader import RegistryLoader, RoleConfig


class ModelState(Enum):
    """State of a model in the server."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ModelStatus:
    """Status of a loaded model."""

    role: str
    state: ModelState
    pid: int | None = None
    loaded_at: float | None = None
    last_inference: float | None = None
    inference_count: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "state": self.state.value,
            "pid": self.pid,
            "loaded_at": self.loaded_at,
            "last_inference": self.last_inference,
            "inference_count": self.inference_count,
            "error_message": self.error_message,
        }


@dataclass
class InferenceRequest:
    """Request for model inference."""

    role: str
    prompt: str | None = None
    prompt_file: Path | None = None
    n_tokens: int = 512
    temperature: float = 0.0
    context_length: int = 8192
    timeout: int = 300  # seconds


@dataclass
class InferenceResult:
    """Result from model inference."""

    role: str
    output: str
    tokens_generated: int
    generation_speed: float  # tokens/second
    elapsed_time: float
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "output": self.output,
            "tokens_generated": self.tokens_generated,
            "generation_speed": self.generation_speed,
            "elapsed_time": self.elapsed_time,
            "success": self.success,
            "error_message": self.error_message,
        }


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    @abstractmethod
    def load(self, role_config: RoleConfig) -> int:
        """Load a model and return its process ID."""
        pass

    @abstractmethod
    def unload(self, pid: int) -> bool:
        """Unload a model by its process ID."""
        pass

    @abstractmethod
    def infer(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Run inference on a model."""
        pass

    @abstractmethod
    def health_check(self, pid: int) -> bool:
        """Check if a model process is healthy."""
        pass


class LlamaCppBackend(ModelBackend):
    """Backend using llama.cpp for inference."""

    def __init__(self, registry: RegistryLoader):
        """Initialize the llama.cpp backend.

        Args:
            registry: Loaded model registry for command generation.
        """
        self.registry = registry
        self._processes: dict[int, subprocess.Popen] = {}

    def load(self, role_config: RoleConfig) -> int:
        """Load a model (stub - models are loaded per-inference in llama.cpp).

        In llama.cpp, models are typically loaded per-inference rather than
        kept resident. This method is a placeholder for potential future
        server mode support.

        Returns:
            Placeholder PID (0 for stub).
        """
        # TODO: Implement llama-server mode for persistent model loading
        return 0

    def unload(self, pid: int) -> bool:
        """Unload a model (stub).

        Returns:
            True (always succeeds for stub).
        """
        if pid in self._processes:
            self._processes[pid].terminate()
            del self._processes[pid]
        return True

    def infer(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Run inference using llama-cli.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters.

        Returns:
            InferenceResult with output and metrics.
        """
        start_time = time.time()

        # Generate command using registry
        cmd = self.registry.generate_command(
            role_config.name,
            prompt=request.prompt,
            prompt_file=str(request.prompt_file) if request.prompt_file else None,
            n_tokens=request.n_tokens,
        )

        try:
            # Run inference
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=request.timeout,
            )

            elapsed = time.time() - start_time
            output = result.stdout

            # Parse generation speed from output if available
            speed = self._parse_speed(output)
            tokens = self._estimate_tokens(output)

            return InferenceResult(
                role=role_config.name,
                output=output,
                tokens_generated=tokens,
                generation_speed=speed,
                elapsed_time=elapsed,
                success=result.returncode == 0,
                error_message=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=request.timeout,
                success=False,
                error_message=f"Inference timed out after {request.timeout}s",
            )

        except Exception as e:
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    def health_check(self, pid: int) -> bool:
        """Check if a process is running.

        Args:
            pid: Process ID to check.

        Returns:
            True if process is running.
        """
        if pid == 0:
            return True  # Stub PID always healthy

        if pid in self._processes:
            return self._processes[pid].poll() is None

        return False

    def _parse_speed(self, output: str) -> float:
        """Parse generation speed from llama.cpp output."""
        # Look for pattern like "Generation: 35.1 t/s"
        import re

        match = re.search(r"Generation:\s*([\d.]+)\s*t/s", output)
        if match:
            return float(match.group(1))
        return 0.0

    def _estimate_tokens(self, output: str) -> int:
        """Estimate token count from output."""
        # Rough estimate: ~4 chars per token
        return len(output) // 4


class ModelServerError(Exception):
    """Error from the model server."""

    pass


@dataclass
class ModelServer:
    """Server managing model lifecycle and inference.

    Attributes:
        registry: Model registry loader.
        backend: Backend for model operations.
        models: Currently tracked model statuses.
    """

    registry: RegistryLoader = field(default_factory=RegistryLoader)
    backend: ModelBackend | None = None
    models: dict[str, ModelStatus] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize backend if not provided."""
        if self.backend is None:
            self.backend = LlamaCppBackend(self.registry)

    def load(self, role: str) -> ModelStatus:
        """Load a model for a given role.

        Args:
            role: Registry role name (e.g., "coder_primary").

        Returns:
            Status of the loaded model.

        Raises:
            ModelServerError: If role not found or loading fails.
        """
        try:
            role_config = self.registry.get_role(role)
        except KeyError as e:
            raise ModelServerError(f"Role not found: {role}") from e

        status = ModelStatus(
            role=role,
            state=ModelState.LOADING,
        )
        self.models[role] = status

        try:
            pid = self.backend.load(role_config)
            status.pid = pid
            status.state = ModelState.READY
            status.loaded_at = time.time()
        except Exception as e:
            status.state = ModelState.ERROR
            status.error_message = str(e)
            raise ModelServerError(f"Failed to load {role}: {e}") from e

        return status

    def unload(self, role: str) -> bool:
        """Unload a model.

        Args:
            role: Role name to unload.

        Returns:
            True if successfully unloaded.
        """
        if role not in self.models:
            return False

        status = self.models[role]
        if status.pid is not None:
            self.backend.unload(status.pid)

        status.state = ModelState.UNLOADED
        del self.models[role]
        return True

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference on a model.

        Args:
            request: Inference request.

        Returns:
            Inference result.

        Raises:
            ModelServerError: If role not found.
        """
        try:
            role_config = self.registry.get_role(request.role)
        except KeyError as e:
            raise ModelServerError(f"Role not found: {request.role}") from e

        # Update status
        if request.role in self.models:
            self.models[request.role].state = ModelState.BUSY

        result = self.backend.infer(role_config, request)

        # Update status
        if request.role in self.models:
            status = self.models[request.role]
            status.state = ModelState.READY if result.success else ModelState.ERROR
            status.last_inference = time.time()
            status.inference_count += 1
            if not result.success:
                status.error_message = result.error_message

        return result

    def get_status(self, role: str) -> ModelStatus | None:
        """Get status of a model.

        Args:
            role: Role name.

        Returns:
            Model status or None if not loaded.
        """
        return self.models.get(role)

    def list_models(self) -> list[ModelStatus]:
        """List all loaded models.

        Returns:
            List of model statuses.
        """
        return list(self.models.values())

    def health_check(self) -> dict[str, Any]:
        """Run health check on all models.

        Returns:
            Health status dictionary.
        """
        results = {}
        for role, status in self.models.items():
            healthy = True
            if status.pid is not None:
                healthy = self.backend.health_check(status.pid)
            results[role] = {
                "healthy": healthy,
                "state": status.state.value,
                "inference_count": status.inference_count,
            }

        return {
            "status": "healthy" if all(r["healthy"] for r in results.values()) else "degraded",
            "models": results,
            "timestamp": time.time(),
        }


def main() -> int:
    """CLI entry point for testing."""
    import json

    server = ModelServer()

    print("Model Server initialized")
    print(f"Available roles: {list(server.registry.roles.keys())}")

    # Health check
    health = server.health_check()
    print(f"\nHealth check: {json.dumps(health, indent=2)}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
