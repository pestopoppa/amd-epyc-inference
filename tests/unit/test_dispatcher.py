"""Unit tests for dispatcher module."""

import json
from pathlib import Path

import pytest
import yaml

from src.dispatcher import (
    Dispatcher,
    DispatchError,
    DispatchResult,
)
from src.registry_loader import RegistryLoader


# Test fixtures
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
            "frontdoor": {
                "tier": "A",
                "description": "Front door routing",
                "model": {
                    "name": "frontdoor-model",
                    "path": "frontdoor.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 5.0,
                },
                "acceleration": {"type": "none"},
                "performance": {"baseline_tps": 15.0},
                "memory": {"residency": "hot"},
            },
            "coder_primary": {
                "tier": "B",
                "description": "Primary coder",
                "model": {
                    "name": "coder-model",
                    "path": "coder.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 18.0,
                },
                "acceleration": {
                    "type": "speculative_decoding",
                    "draft_role": "draft_qwen25_coder",
                    "k": 16,
                },
                "performance": {"baseline_tps": 5.0, "optimized_tps": 30.0},
                "memory": {"residency": "hot"},
            },
            "draft_qwen25_coder": {
                "tier": "D",
                "description": "Draft model",
                "model": {
                    "name": "draft-model",
                    "path": "draft.gguf",
                    "quant": "Q8_0",
                    "size_gb": 0.5,
                },
                "acceleration": {"type": "none"},
                "performance": {"raw_tps": 85},
                "memory": {"residency": "hot"},
            },
            "worker_general": {
                "tier": "C",
                "description": "General worker",
                "model": {
                    "name": "worker-model",
                    "path": "worker.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 10.0,
                },
                "acceleration": {"type": "none"},
                "performance": {"baseline_tps": 20.0},
                "memory": {"residency": "hot"},
            },
        },
        "routing_hints": [
            {"if": "task_type == 'code'", "use": ["coder_primary"]},
            {"if": "task_type == 'general'", "use": ["worker_general"]},
        ],
        "command_templates": {
            "baseline": "llama-cli -m {model_path} -t {threads}",
            "speculative_decoding": "llama-spec -m {model_path} -md {draft_path} --draft-max {k}",
        },
    }

    # Create dummy model files
    (tmp_path / "frontdoor.gguf").touch()
    (tmp_path / "coder.gguf").touch()
    (tmp_path / "draft.gguf").touch()
    (tmp_path / "worker.gguf").touch()

    registry_path = tmp_path / "registry.yaml"
    with registry_path.open("w") as f:
        yaml.dump(registry, f)

    return registry_path


@pytest.fixture
def sample_task_ir() -> dict:
    """Create a sample TaskIR for testing."""
    return {
        "task_id": "test-task-001",
        "task_type": "code",
        "priority": "interactive",
        "objective": "Write a function to sort a list",
        "constraints": ["must be efficient", "use Python"],
        "inputs": [{"type": "text", "content": "sort algorithm requirements"}],
        "agents": [
            {"role": "coder", "model_hint": None},
        ],
        "plan": {
            "steps": [
                {
                    "id": "S1",
                    "actor": "coder",
                    "action": "Write sorting function",
                    "inputs": ["requirements"],
                    "outputs": ["sort_function.py"],
                    "depends_on": [],
                },
                {
                    "id": "S2",
                    "actor": "coder",
                    "action": "Write unit tests",
                    "inputs": ["sort_function.py"],
                    "outputs": ["test_sort.py"],
                    "depends_on": ["S1"],
                },
            ],
        },
        "escalation": {"on_failure": "retry", "max_retries": 2},
    }


class TestDispatcherBasic:
    """Basic dispatcher tests."""

    def test_create_dispatcher(self, minimal_registry: Path):
        """Test creating a dispatcher with registry."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry, validate_paths=True)

        assert dispatcher.registry is not None
        assert len(dispatcher.registry.roles) == 4

    def test_dispatch_simple_task(self, minimal_registry: Path, sample_task_ir: dict):
        """Test dispatching a simple task."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        result = dispatcher.dispatch(sample_task_ir)

        assert isinstance(result, DispatchResult)
        assert result.task_id == "test-task-001"
        assert len(result.steps) == 2
        assert "coder_primary" in result.roles_used

    def test_dispatch_result_to_dict(self, minimal_registry: Path, sample_task_ir: dict):
        """Test converting dispatch result to dictionary."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        result = dispatcher.dispatch(sample_task_ir)
        result_dict = result.to_dict()

        assert "task_id" in result_dict
        assert "timestamp" in result_dict
        assert "roles_used" in result_dict
        assert "steps" in result_dict
        assert len(result_dict["steps"]) == 2


class TestRoleMapping:
    """Tests for role mapping."""

    def test_map_coder_role(self, minimal_registry: Path):
        """Test mapping 'coder' IR role to 'coder_primary'."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        task_ir = {
            "task_type": "code",
            "agents": [{"role": "coder"}],
            "plan": {"steps": []},
        }

        result = dispatcher.dispatch(task_ir)
        assert "coder_primary" in result.roles_used

    def test_map_unknown_role_to_worker(self, minimal_registry: Path):
        """Test that unknown roles default to worker_general."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        task_ir = {
            "task_type": "unknown",
            "agents": [{"role": "unknown_agent"}],
            "plan": {"steps": []},
        }

        result = dispatcher.dispatch(task_ir)
        assert "worker_general" in result.roles_used
        assert len(result.warnings) > 0

    def test_model_hint_override(self, minimal_registry: Path):
        """Test that model_hint overrides role mapping."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        task_ir = {
            "task_type": "code",
            "agents": [{"role": "coder", "model_hint": "worker_general"}],
            "plan": {"steps": []},
        }

        result = dispatcher.dispatch(task_ir)
        assert "worker_general" in result.roles_used


class TestSpeculativeDecoding:
    """Tests for speculative decoding configuration."""

    def test_draft_role_auto_included(self, minimal_registry: Path):
        """Test that draft role is automatically included."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        task_ir = {
            "task_type": "code",
            "agents": [{"role": "coder"}],
            "plan": {"steps": []},
        }

        result = dispatcher.dispatch(task_ir)

        # Should include both coder_primary and draft_qwen25_coder
        assert "coder_primary" in result.roles_used
        assert "draft_qwen25_coder" in result.roles_used


class TestStepExecution:
    """Tests for step execution generation."""

    def test_step_has_command(self, minimal_registry: Path, sample_task_ir: dict):
        """Test that steps have commands generated."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        result = dispatcher.dispatch(sample_task_ir)

        for step in result.steps:
            assert step.command != ""
            assert isinstance(step.command, str)

    def test_step_preserves_dependencies(
        self, minimal_registry: Path, sample_task_ir: dict
    ):
        """Test that step dependencies are preserved."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        result = dispatcher.dispatch(sample_task_ir)

        # S2 should depend on S1
        s2 = next(s for s in result.steps if s.step_id == "S2")
        assert "S1" in s2.depends_on

    def test_step_has_role_config(self, minimal_registry: Path, sample_task_ir: dict):
        """Test that steps have role configuration."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        result = dispatcher.dispatch(sample_task_ir)

        for step in result.steps:
            assert step.role_config is not None
            assert step.role_config.name in ["coder_primary", "worker_general"]


class TestFileDispatch:
    """Tests for file-based dispatch."""

    def test_dispatch_from_file(
        self, minimal_registry: Path, sample_task_ir: dict, tmp_path: Path
    ):
        """Test dispatching from a JSON file."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        # Write task IR to file
        task_file = tmp_path / "task.json"
        with task_file.open("w") as f:
            json.dump(sample_task_ir, f)

        result = dispatcher.dispatch_from_file(task_file)

        assert result.task_id == "test-task-001"
        assert len(result.steps) == 2

    def test_dispatch_from_nonexistent_file(self, minimal_registry: Path, tmp_path: Path):
        """Test error when file doesn't exist."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        with pytest.raises(DispatchError, match="not found"):
            dispatcher.dispatch_from_file(tmp_path / "nonexistent.json")

    def test_dispatch_from_invalid_json(self, minimal_registry: Path, tmp_path: Path):
        """Test error when file contains invalid JSON."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json {{{")

        with pytest.raises(DispatchError, match="Invalid JSON"):
            dispatcher.dispatch_from_file(invalid_file)


class TestRouting:
    """Tests for automatic routing."""

    def test_route_code_task(self, minimal_registry: Path):
        """Test routing a code task without explicit agents."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        task_ir = {
            "task_type": "code",
            "priority": "interactive",
            "objective": "Write code",
            "constraints": [],
            "inputs": [],
            "plan": {"steps": []},
            "escalation": {},
        }

        result = dispatcher.dispatch(task_ir)

        # Should route to coder via routing hints
        assert "coder_primary" in result.roles_used

    def test_route_general_task(self, minimal_registry: Path):
        """Test routing a general task."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        task_ir = {
            "task_type": "general",
            "priority": "batch",
            "objective": "General task",
            "constraints": [],
            "inputs": [],
            "plan": {"steps": []},
            "escalation": {},
        }

        result = dispatcher.dispatch(task_ir)
        assert "worker_general" in result.roles_used


class TestErrorHandling:
    """Tests for error handling."""

    def test_unknown_role_warns(self, minimal_registry: Path):
        """Test that unknown roles generate warnings."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        task_ir = {
            "task_type": "code",
            "agents": [{"role": "totally_unknown_role"}],  # Not in ROLE_MAPPING
            "plan": {"steps": []},
        }

        result = dispatcher.dispatch(task_ir)

        # Should have warning about the unknown role defaulting to worker_general
        assert len(result.warnings) > 0
        assert "totally_unknown_role" in result.warnings[0]

    def test_step_with_no_matching_role(self, minimal_registry: Path):
        """Test steps with actors that don't match any role."""
        registry = RegistryLoader(minimal_registry)
        dispatcher = Dispatcher(registry=registry)

        task_ir = {
            "task_type": "code",
            "agents": [{"role": "coder"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "unmapped_actor",  # Not in roles_used or ROLE_MAPPING
                        "action": "Do something",
                    }
                ]
            },
        }

        result = dispatcher.dispatch(task_ir)

        # Step should still be created, but may have no role_config
        assert len(result.steps) == 1
