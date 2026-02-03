"""Unit tests for validate_ir module."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the validator script
VALIDATOR_PATH = Path(__file__).resolve().parent.parent.parent / "orchestration" / "validate_ir.py"
SCHEMA_DIR = VALIDATOR_PATH.parent


@pytest.fixture
def valid_task_ir() -> dict:
    """Return a valid TaskIR document."""
    return {
        "task_id": "test-001",
        "created_at": "2025-12-16T12:00:00Z",
        "task_type": "code",
        "priority": "interactive",
        "objective": "Test task",
        "inputs": [],
        "constraints": [],
        "assumptions": [],
        "agents": [{"tier": "B", "role": "coder"}],
        "plan": {
            "steps": [
                {
                    "id": "S1",
                    "actor": "coder",
                    "action": "Do something",
                    "outputs": ["output.py"],
                }
            ]
        },
        "gates": ["format"],
        "definition_of_done": ["Task completed"],
        "escalation": {"max_level": "B1", "on_second_failure": True},
    }


@pytest.fixture
def valid_architecture_ir() -> dict:
    """Return a valid ArchitectureIR document."""
    return {
        "name": "test-project",
        "version": "1.0.0",
        "goals": ["Build a test system"],
        "non_goals": [],
        "global_invariants": [],
        "repo_layout": {
            "folders": [{"path": "src/", "owner_role": "coder", "purpose": "Source code"}]
        },
        "modules": [
            {
                "id": "core",
                "name": "Core Module",
                "responsibilities": ["Handle core logic"],
                "public_api": [],
                "dependencies": {"allows": [], "forbids": []},
                "files": [{"path": "src/core.py", "purpose": "Core implementation"}],
            }
        ],
        "contracts": [],
        "cross_cutting": {
            "logging": [],
            "errors": {"strategy": "exceptions"},
            "config": [],
            "security": [],
        },
        "acceptance": {
            "tests": [],
            "benchmarks": [],
            "definition_of_done": ["All tests pass"],
        },
    }


def run_validator(kind: str, input_json: str) -> tuple[int, str, str]:
    """Run the validator script and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), kind, "-"],
        input=input_json,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


class TestTaskIRValidation:
    """Tests for TaskIR validation."""

    def test_valid_task_ir(self, valid_task_ir):
        """Test that valid TaskIR passes validation."""
        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 0
        assert "is valid" in stdout

    def test_missing_required_field(self, valid_task_ir):
        """Test that missing required field fails validation."""
        del valid_task_ir["task_id"]

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2
        assert "validation error" in stdout.lower()

    def test_invalid_task_type(self, valid_task_ir):
        """Test that invalid task_type fails validation."""
        valid_task_ir["task_type"] = "invalid"

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2
        assert "validation error" in stdout.lower()

    def test_invalid_priority(self, valid_task_ir):
        """Test that invalid priority fails validation."""
        valid_task_ir["priority"] = "urgent"  # not in enum

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2

    def test_invalid_agent_tier(self, valid_task_ir):
        """Test that invalid agent tier fails validation."""
        valid_task_ir["agents"][0]["tier"] = "X"

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2

    def test_invalid_agent_role(self, valid_task_ir):
        """Test that invalid agent role fails validation."""
        valid_task_ir["agents"][0]["role"] = "invalid_role"

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2

    def test_invalid_step_id_pattern(self, valid_task_ir):
        """Test that step ID must match pattern ^S[0-9]+$."""
        valid_task_ir["plan"]["steps"][0]["id"] = "step1"  # Should be S1

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2

    def test_invalid_gate(self, valid_task_ir):
        """Test that invalid gate name fails validation."""
        valid_task_ir["gates"] = ["invalid_gate"]

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2

    def test_empty_definition_of_done(self, valid_task_ir):
        """Test that empty definition_of_done fails validation."""
        valid_task_ir["definition_of_done"] = []

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2

    def test_invalid_escalation_level(self, valid_task_ir):
        """Test that invalid escalation level fails validation."""
        valid_task_ir["escalation"]["max_level"] = "B4"

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2

    def test_additional_properties_rejected(self, valid_task_ir):
        """Test that additional properties are rejected."""
        valid_task_ir["extra_field"] = "not allowed"

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))

        assert code == 2

    def test_context_field_optional(self, valid_task_ir):
        """Test that context field is optional."""
        # Context not present - should still be valid
        assert "context" not in valid_task_ir

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))
        assert code == 0

    def test_context_field_valid(self, valid_task_ir):
        """Test that valid context field passes."""
        valid_task_ir["context"] = {
            "conversation_id": "conv-123",
            "prior_task_ids": ["task-001"],
            "relevant_files": ["src/main.py"],
        }

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))
        assert code == 0

    def test_step_depends_on(self, valid_task_ir):
        """Test that depends_on with valid step IDs passes."""
        valid_task_ir["plan"]["steps"].append(
            {
                "id": "S2",
                "actor": "worker",
                "action": "Follow up",
                "outputs": ["result.txt"],
                "depends_on": ["S1"],
            }
        )

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))
        assert code == 0

    def test_step_parallel_group(self, valid_task_ir):
        """Test that parallel_group field works."""
        valid_task_ir["plan"]["steps"][0]["parallel_group"] = "group_a"

        code, stdout, stderr = run_validator("task", json.dumps(valid_task_ir))
        assert code == 0


class TestArchitectureIRValidation:
    """Tests for ArchitectureIR validation."""

    def test_valid_architecture_ir(self, valid_architecture_ir):
        """Test that valid ArchitectureIR passes validation."""
        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))

        assert code == 0
        assert "is valid" in stdout

    def test_missing_required_field(self, valid_architecture_ir):
        """Test that missing required field fails validation."""
        del valid_architecture_ir["name"]

        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))

        assert code == 2

    def test_empty_goals(self, valid_architecture_ir):
        """Test that empty goals fails validation."""
        valid_architecture_ir["goals"] = []

        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))

        assert code == 2

    def test_invalid_error_strategy(self, valid_architecture_ir):
        """Test that invalid error strategy fails validation."""
        valid_architecture_ir["cross_cutting"]["errors"]["strategy"] = "panic"

        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))

        assert code == 2

    def test_valid_adr(self, valid_architecture_ir):
        """Test that valid ADR passes."""
        valid_architecture_ir["decisions"] = [
            {
                "id": "ADR-001",
                "title": "Use Python",
                "status": "accepted",
                "rationale": "Team expertise",
            }
        ]

        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))
        assert code == 0

    def test_invalid_adr_id_pattern(self, valid_architecture_ir):
        """Test that invalid ADR ID pattern fails."""
        valid_architecture_ir["decisions"] = [
            {
                "id": "decision-1",  # Should be ADR-001
                "title": "Test",
                "status": "accepted",
                "rationale": "Test",
            }
        ]

        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))
        assert code == 2

    def test_invalid_adr_status(self, valid_architecture_ir):
        """Test that invalid ADR status fails."""
        valid_architecture_ir["decisions"] = [
            {
                "id": "ADR-001",
                "title": "Test",
                "status": "rejected",  # Not in enum
                "rationale": "Test",
            }
        ]

        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))
        assert code == 2

    def test_module_id_pattern(self, valid_architecture_ir):
        """Test that module ID must match pattern."""
        valid_architecture_ir["modules"][0]["id"] = "Core Module"  # Has space, uppercase

        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))
        assert code == 2

    def test_contract_types(self, valid_architecture_ir):
        """Test that contract types are validated."""
        valid_architecture_ir["contracts"] = [
            {
                "id": "api",
                "type": "openapi",
                "path": "api/openapi.yaml",
                "scope": "public",
            }
        ]

        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))
        assert code == 0

    def test_invalid_contract_type(self, valid_architecture_ir):
        """Test that invalid contract type fails."""
        valid_architecture_ir["contracts"] = [
            {
                "id": "api",
                "type": "swagger",  # Not in enum
                "path": "api.yaml",
                "scope": "public",
            }
        ]

        code, stdout, stderr = run_validator("arch", json.dumps(valid_architecture_ir))
        assert code == 2


class TestValidatorCLI:
    """Tests for CLI behavior."""

    def test_usage_message(self):
        """Test usage message on no arguments."""
        result = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Usage" in result.stdout

    def test_invalid_kind(self):
        """Test error on invalid kind argument."""
        result = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "invalid", "-"],
            input="{}",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "unknown kind" in result.stdout.lower()

    def test_invalid_json_input(self):
        """Test error on invalid JSON input."""
        code, stdout, stderr = run_validator("task", "not json")

        assert code == 2
        assert "Invalid JSON" in stdout

    def test_file_not_found(self):
        """Test error on nonexistent file."""
        result = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "task", "/nonexistent/file.json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 3
        assert "not found" in result.stdout.lower()


class TestExampleFile:
    """Test the example TaskIR file."""

    def test_example_task_ir_valid(self):
        """Test that the example TaskIR file is valid."""
        example_path = SCHEMA_DIR / "examples" / "task_ir_example.json"

        if not example_path.exists():
            pytest.skip("Example file not found")

        result = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "task", str(example_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "is valid" in result.stdout
