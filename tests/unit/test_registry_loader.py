"""Unit tests for registry_loader module."""

from pathlib import Path

import pytest
import yaml

from src.registry_loader import (
    RegistryError,
    RegistryLoader,
)


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
            "test_role": {
                "tier": "C",
                "description": "Test role",
                "model": {
                    "name": "test-model",
                    "path": "test-model.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 1.0,
                },
                "acceleration": {
                    "type": "none",
                },
                "performance": {
                    "baseline_tps": 10.0,
                },
                "memory": {
                    "residency": "hot",
                },
            },
        },
        "routing_hints": [
            {"if": "task_type == 'code'", "use": ["test_role"]},
        ],
        "command_templates": {
            "baseline": "llama-cli -m {model_path} -t {threads}",
        },
    }

    # Create dummy model file
    (tmp_path / "test-model.gguf").touch()

    registry_path = tmp_path / "registry.yaml"
    with registry_path.open("w") as f:
        yaml.dump(registry, f)

    return registry_path


@pytest.fixture
def speculative_registry(tmp_path: Path) -> Path:
    """Create a registry with speculative decoding configuration."""
    registry = {
        "runtime_defaults": {
            "model_base_path": str(tmp_path),
            "threads": 96,
            "context_length": 8192,
        },
        "roles": {
            "target": {
                "tier": "B",
                "description": "Target model",
                "model": {
                    "name": "target-model",
                    "path": "target.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 10.0,
                },
                "acceleration": {
                    "type": "speculative_decoding",
                    "draft_role": "draft",
                    "k": 16,
                },
                "performance": {"baseline_tps": 5.0, "optimized_tps": 30.0},
                "memory": {"residency": "hot"},
            },
            "draft": {
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
        },
        "command_templates": {
            "speculative_decoding": "llama-spec -m {model_path} -md {draft_path} --draft-max {k}",
            "baseline": "llama-cli -m {model_path}",
        },
    }

    (tmp_path / "target.gguf").touch()
    (tmp_path / "draft.gguf").touch()

    registry_path = tmp_path / "registry.yaml"
    with registry_path.open("w") as f:
        yaml.dump(registry, f)

    return registry_path


class TestRegistryLoaderBasic:
    """Basic loading tests."""

    def test_load_minimal_registry(self, minimal_registry: Path):
        """Test loading a minimal valid registry."""
        loader = RegistryLoader(minimal_registry, validate_paths=True)

        assert len(loader.roles) == 1
        assert "test_role" in loader.roles
        assert loader.missing_models == []

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Test loading a nonexistent file raises error."""
        with pytest.raises(RegistryError, match="Registry not found"):
            RegistryLoader(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path):
        """Test loading invalid YAML raises error."""
        registry_path = tmp_path / "invalid.yaml"
        registry_path.write_text("not: valid: yaml: {{")

        with pytest.raises(RegistryError, match="Invalid YAML"):
            RegistryLoader(registry_path)

    def test_load_empty_file(self, tmp_path: Path):
        """Test loading empty file raises error."""
        registry_path = tmp_path / "empty.yaml"
        registry_path.write_text("")

        with pytest.raises(RegistryError, match="Empty registry"):
            RegistryLoader(registry_path)

    def test_load_no_roles(self, tmp_path: Path):
        """Test loading registry without roles raises error."""
        registry_path = tmp_path / "no_roles.yaml"
        registry_path.write_text("runtime_defaults:\n  threads: 96\n")

        with pytest.raises(RegistryError, match="No roles defined"):
            RegistryLoader(registry_path)


class TestRoleAccess:
    """Tests for role access methods."""

    def test_get_role(self, minimal_registry: Path):
        """Test getting a specific role."""
        loader = RegistryLoader(minimal_registry)
        role = loader.get_role("test_role")

        assert role.name == "test_role"
        assert role.tier == "C"
        assert role.model.name == "test-model"

    def test_get_nonexistent_role(self, minimal_registry: Path):
        """Test getting nonexistent role raises KeyError."""
        loader = RegistryLoader(minimal_registry)

        with pytest.raises(KeyError, match="Role not found"):
            loader.get_role("nonexistent")

    def test_get_roles_by_tier(self, minimal_registry: Path):
        """Test filtering roles by tier."""
        loader = RegistryLoader(minimal_registry)

        tier_c = loader.get_roles_by_tier("C")
        tier_a = loader.get_roles_by_tier("A")

        assert len(tier_c) == 1
        assert tier_c[0].name == "test_role"
        assert len(tier_a) == 0


class TestSpeculativeDecoding:
    """Tests for speculative decoding configuration."""

    def test_get_draft_for_role(self, speculative_registry: Path):
        """Test getting draft model for speculative decoding."""
        loader = RegistryLoader(speculative_registry)
        draft = loader.get_draft_for_role("target")

        assert draft is not None
        assert draft.name == "draft"
        assert draft.model.name == "draft-model"

    def test_get_draft_for_non_speculative(self, minimal_registry: Path):
        """Test getting draft for non-speculative role returns None."""
        loader = RegistryLoader(minimal_registry)
        draft = loader.get_draft_for_role("test_role")

        assert draft is None


class TestCommandGeneration:
    """Tests for command generation."""

    def test_generate_baseline_command(self, minimal_registry: Path):
        """Test generating baseline command."""
        loader = RegistryLoader(minimal_registry)
        cmd = loader.generate_command("test_role", prompt="Hello", n_tokens=32)

        assert "llama-cli" in cmd
        assert "-m" in cmd
        assert "test-model.gguf" in cmd
        assert "-t 96" in cmd
        assert "'Hello'" in cmd
        assert "-n 32" in cmd

    def test_generate_speculative_command(self, speculative_registry: Path):
        """Test generating speculative decoding command."""
        loader = RegistryLoader(speculative_registry)
        cmd = loader.generate_command("target", prompt="Code", n_tokens=64)

        assert "llama-spec" in cmd
        assert "target.gguf" in cmd
        assert "draft.gguf" in cmd
        assert "--draft-max 16" in cmd

    def test_generate_command_with_file(self, minimal_registry: Path):
        """Test generating command with prompt file."""
        loader = RegistryLoader(minimal_registry)
        cmd = loader.generate_command("test_role", prompt_file="/tmp/prompt.txt")

        assert "-f /tmp/prompt.txt" in cmd
        assert "-p" not in cmd.split("-f")[0]  # No inline prompt


class TestRouting:
    """Tests for task routing."""

    def test_route_code_task(self, minimal_registry: Path):
        """Test routing a code task."""
        loader = RegistryLoader(minimal_registry)

        task_ir = {
            "task_type": "code",
            "priority": "interactive",
            "objective": "Write a function",
            "constraints": [],
            "inputs": [],
            "escalation": {},
        }

        roles = loader.route_task(task_ir)
        assert roles == ["test_role"]

    def test_route_unknown_task(self, minimal_registry: Path):
        """Test routing unknown task defaults to frontdoor."""
        loader = RegistryLoader(minimal_registry)

        task_ir = {
            "task_type": "unknown",
            "priority": "batch",
            "objective": "Something",
            "constraints": [],
            "inputs": [],
            "escalation": {},
        }

        roles = loader.route_task(task_ir)
        assert roles == ["frontdoor"]


class TestValidation:
    """Tests for path validation."""

    def test_missing_model_detected(self, tmp_path: Path):
        """Test that missing model files are detected."""
        registry = {
            "runtime_defaults": {"model_base_path": str(tmp_path)},
            "roles": {
                "missing": {
                    "tier": "C",
                    "description": "Missing model",
                    "model": {
                        "name": "missing",
                        "path": "does-not-exist.gguf",
                        "quant": "Q4_K_M",
                        "size_gb": 1.0,
                    },
                    "acceleration": {"type": "none"},
                    "performance": {},
                    "memory": {"residency": "hot"},
                },
            },
        }

        registry_path = tmp_path / "registry.yaml"
        with registry_path.open("w") as f:
            yaml.dump(registry, f)

        loader = RegistryLoader(registry_path, validate_paths=True)

        assert len(loader.missing_models) == 1
        assert "missing" in loader.missing_models[0]

    def test_skip_validation(self, tmp_path: Path):
        """Test skipping path validation."""
        registry = {
            "runtime_defaults": {"model_base_path": str(tmp_path)},
            "roles": {
                "missing": {
                    "tier": "C",
                    "description": "Missing model",
                    "model": {
                        "name": "missing",
                        "path": "does-not-exist.gguf",
                        "quant": "Q4_K_M",
                        "size_gb": 1.0,
                    },
                    "acceleration": {"type": "none"},
                    "performance": {},
                    "memory": {"residency": "hot"},
                },
            },
        }

        registry_path = tmp_path / "registry.yaml"
        with registry_path.open("w") as f:
            yaml.dump(registry, f)

        loader = RegistryLoader(registry_path, validate_paths=False)

        assert len(loader.missing_models) == 0


class TestProductionRegistry:
    """Tests against the production registry."""

    def test_load_production_registry(self):
        """Test loading the actual production registry."""
        loader = RegistryLoader(validate_paths=True)

        # Should have all expected tiers
        assert len(loader.get_roles_by_tier("A")) >= 1
        assert len(loader.get_roles_by_tier("B")) >= 1
        assert len(loader.get_roles_by_tier("C")) >= 1
        assert len(loader.get_roles_by_tier("D")) >= 1

        # Key roles should exist
        assert "frontdoor" in loader.roles
        assert "coder_primary" in loader.roles

    def test_production_coder_uses_moe(self):
        """Test coder_primary uses MoE expert reduction."""
        loader = RegistryLoader(validate_paths=True)

        role = loader.get_role("coder_primary")
        assert role.acceleration.type == "moe_expert_reduction"
        assert role.acceleration.experts == 4

    def test_production_command_generation(self):
        """Test command generation for production roles."""
        loader = RegistryLoader(validate_paths=True)

        # MoE reduction for coder_primary
        cmd = loader.generate_command("coder_primary", prompt="test")
        assert "--override-kv" in cmd
        assert "expert_used_count" in cmd

        # MoE reduction for frontdoor
        cmd = loader.generate_command("frontdoor", prompt="test")
        assert "--override-kv" in cmd

        # Prompt lookup - now uses llama-lookup binary with --draft-max
        cmd = loader.generate_command("worker_general", prompt="test")
        assert "llama-lookup" in cmd
        assert "--draft-max" in cmd
