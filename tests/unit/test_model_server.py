"""Unit tests for model_server module."""

from pathlib import Path

import pytest
import yaml

from src.model_server import (
    InferenceRequest,
    InferenceResult,
    ModelServer,
    ModelServerError,
    ModelState,
    ModelStatus,
)
from src.registry_loader import RegistryLoader


@pytest.fixture
def minimal_registry(tmp_path: Path) -> Path:
    """Create a minimal valid registry file."""
    registry = {
        "runtime_defaults": {
            "model_base_path": str(tmp_path),
            "threads": 96,
            "context_length": 8192,
        },
        "roles": {
            "test_role": {
                "tier": "C",
                "description": "Test role",
                "model": {
                    "name": "test-model",
                    "path": "test-model.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 1.0,
                },
                "acceleration": {"type": "none"},
                "performance": {"baseline_tps": 10.0},
                "memory": {"residency": "hot"},
            },
        },
        "command_templates": {
            "baseline": "echo 'test output'",
        },
    }

    (tmp_path / "test-model.gguf").touch()

    registry_path = tmp_path / "registry.yaml"
    with registry_path.open("w") as f:
        yaml.dump(registry, f)

    return registry_path


class TestModelStatus:
    """Tests for ModelStatus dataclass."""

    def test_status_creation(self):
        """Test creating a model status."""
        status = ModelStatus(role="test_role", state=ModelState.READY)

        assert status.role == "test_role"
        assert status.state == ModelState.READY
        assert status.pid is None
        assert status.inference_count == 0

    def test_status_to_dict(self):
        """Test converting status to dictionary."""
        status = ModelStatus(
            role="test_role",
            state=ModelState.READY,
            pid=1234,
            inference_count=5,
        )

        d = status.to_dict()
        assert d["role"] == "test_role"
        assert d["state"] == "ready"
        assert d["pid"] == 1234
        assert d["inference_count"] == 5


class TestInferenceRequest:
    """Tests for InferenceRequest dataclass."""

    def test_request_defaults(self):
        """Test default values."""
        request = InferenceRequest(role="test_role", prompt="Hello")

        assert request.role == "test_role"
        assert request.prompt == "Hello"
        assert request.n_tokens == 512
        assert request.temperature == 0.0
        assert request.timeout == 300


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = InferenceResult(
            role="test_role",
            output="Hello world",
            tokens_generated=10,
            generation_speed=50.0,
            elapsed_time=0.2,
            success=True,
        )

        d = result.to_dict()
        assert d["role"] == "test_role"
        assert d["output"] == "Hello world"
        assert d["success"] is True


class TestModelServer:
    """Tests for ModelServer class."""

    def test_server_creation(self, minimal_registry: Path):
        """Test creating a model server."""
        registry = RegistryLoader(minimal_registry)
        server = ModelServer(registry=registry)

        assert server.registry is not None
        assert server.backend is not None
        assert len(server.models) == 0

    def test_load_model(self, minimal_registry: Path):
        """Test loading a model."""
        registry = RegistryLoader(minimal_registry)
        server = ModelServer(registry=registry)

        status = server.load("test_role")

        assert status.role == "test_role"
        assert status.state == ModelState.READY
        assert "test_role" in server.models

    def test_load_nonexistent_role(self, minimal_registry: Path):
        """Test loading a nonexistent role."""
        registry = RegistryLoader(minimal_registry)
        server = ModelServer(registry=registry)

        with pytest.raises(ModelServerError, match="Role not found"):
            server.load("nonexistent")

    def test_unload_model(self, minimal_registry: Path):
        """Test unloading a model."""
        registry = RegistryLoader(minimal_registry)
        server = ModelServer(registry=registry)

        server.load("test_role")
        assert "test_role" in server.models

        result = server.unload("test_role")
        assert result is True
        assert "test_role" not in server.models

    def test_unload_nonexistent(self, minimal_registry: Path):
        """Test unloading a model that isn't loaded."""
        registry = RegistryLoader(minimal_registry)
        server = ModelServer(registry=registry)

        result = server.unload("nonexistent")
        assert result is False

    def test_list_models(self, minimal_registry: Path):
        """Test listing loaded models."""
        registry = RegistryLoader(minimal_registry)
        server = ModelServer(registry=registry)

        assert len(server.list_models()) == 0

        server.load("test_role")
        models = server.list_models()

        assert len(models) == 1
        assert models[0].role == "test_role"

    def test_get_status(self, minimal_registry: Path):
        """Test getting model status."""
        registry = RegistryLoader(minimal_registry)
        server = ModelServer(registry=registry)

        assert server.get_status("test_role") is None

        server.load("test_role")
        status = server.get_status("test_role")

        assert status is not None
        assert status.role == "test_role"

    def test_health_check(self, minimal_registry: Path):
        """Test health check."""
        registry = RegistryLoader(minimal_registry)
        server = ModelServer(registry=registry)

        health = server.health_check()

        assert "status" in health
        assert "models" in health
        assert "timestamp" in health

    def test_health_check_with_models(self, minimal_registry: Path):
        """Test health check with loaded models."""
        registry = RegistryLoader(minimal_registry)
        server = ModelServer(registry=registry)

        server.load("test_role")
        health = server.health_check()

        assert health["status"] == "healthy"
        assert "test_role" in health["models"]
        assert health["models"]["test_role"]["healthy"] is True


class TestModelState:
    """Tests for ModelState enum."""

    def test_state_values(self):
        """Test state enum values."""
        assert ModelState.UNLOADED.value == "unloaded"
        assert ModelState.LOADING.value == "loading"
        assert ModelState.READY.value == "ready"
        assert ModelState.BUSY.value == "busy"
        assert ModelState.ERROR.value == "error"
