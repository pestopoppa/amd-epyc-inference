"""Extended unit tests for model_server module.

Focuses on uncovered paths in LlamaCppBackend:
- _build_command() for different acceleration types
- _parse_speed() for all 3 regex patterns
- _parse_tokens() with and without matches
- _extract_error() with various error patterns
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from src.model_server import (
    InferenceRequest,
    LlamaCppBackend,
    ModelServer,
    ModelServerError,
)
from src.registry_loader import (
    AccelerationConfig,
    MemoryConfig,
    ModelConfig,
    PerformanceMetrics,
    RegistryLoader,
    RoleConfig,
)


@pytest.fixture
def minimal_registry(tmp_path: Path) -> RegistryLoader:
    """Create a minimal registry for testing."""
    registry_data = {
        "runtime_defaults": {
            "model_base_path": str(tmp_path),
            "threads": 96,
            "context_length": 8192,
        },
        "roles": {
            "baseline_role": {
                "tier": "C",
                "description": "Baseline test",
                "model": {
                    "name": "test-model",
                    "path": "test-model.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 1.0,
                },
                "acceleration": {"type": "baseline"},
                "performance": {"baseline_tps": 10.0},
                "memory": {"residency": "hot"},
            },
            "spec_role": {
                "tier": "C",
                "description": "Spec decode test",
                "model": {
                    "name": "test-target",
                    "path": "test-target.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 5.0,
                },
                "acceleration": {
                    "type": "speculative_decoding",
                    "k": 24,
                },
                "performance": {"baseline_tps": 10.0},
                "memory": {"residency": "hot"},
                "draft_model": {
                    "name": "test-draft",
                    "path": "test-draft.gguf",
                    "quant": "Q8_0",
                },
            },
            "moe_role": {
                "tier": "B",
                "description": "MoE test",
                "model": {
                    "name": "test-moe",
                    "path": "test-moe.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 10.0,
                },
                "acceleration": {
                    "type": "moe_expert_reduction",
                    "experts": 4,
                    "override_key": "qwen3moe.expert_used_count",
                },
                "performance": {"baseline_tps": 5.0},
                "memory": {"residency": "warm"},
            },
            "lookup_role": {
                "tier": "C",
                "description": "Prompt lookup test",
                "model": {
                    "name": "test-lookup",
                    "path": "test-lookup.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 3.0,
                },
                "acceleration": {
                    "type": "prompt_lookup",
                    "k": 16,
                },
                "performance": {"baseline_tps": 8.0},
                "memory": {"residency": "hot"},
            },
        },
        "command_templates": {},
    }

    # Create model files
    (tmp_path / "test-model.gguf").touch()
    (tmp_path / "test-target.gguf").touch()
    (tmp_path / "test-draft.gguf").touch()
    (tmp_path / "test-moe.gguf").touch()
    (tmp_path / "test-lookup.gguf").touch()

    registry_path = tmp_path / "registry.yaml"
    with registry_path.open("w") as f:
        yaml.dump(registry_data, f)

    return RegistryLoader(str(registry_path))


class TestLlamaCppBackendBuildCommand:
    """Tests for _build_command() with different acceleration types."""

    def test_build_command_baseline(self, minimal_registry):
        """Test command building for baseline (no acceleration)."""
        backend = LlamaCppBackend(minimal_registry)
        role_config = minimal_registry.get_role("baseline_role")
        request = InferenceRequest(role="baseline_role", prompt="Hello", n_tokens=128)

        with patch("src.config.get_config") as mock_config:
            mock_config.return_value.paths.llama_cpp_bin = Path("/usr/bin")
            cmd = backend._build_command(role_config, request)

        assert "llama-completion" in cmd
        assert "-m" in cmd
        assert "test-model.gguf" in cmd
        assert "-n 128" in cmd
        assert "-t 96" in cmd
        assert '--temp 0.0' in cmd
        assert '-p "Hello"' in cmd
        # Should NOT have spec/moe/lookup flags
        assert "-md" not in cmd
        assert "--draft-max" not in cmd
        assert "--override-kv" not in cmd

    def test_build_command_speculative_decoding(self, minimal_registry):
        """Test command building for speculative decoding."""
        backend = LlamaCppBackend(minimal_registry)
        role_config = minimal_registry.get_role("spec_role")
        request = InferenceRequest(role="spec_role", prompt="Code this", n_tokens=256)

        # Create a mock draft role config
        draft_config = minimal_registry.get_role("baseline_role")  # Reuse as draft

        with patch("src.config.get_config") as mock_config:
            mock_config.return_value.paths.llama_cpp_bin = Path("/usr/bin")
            with patch.object(minimal_registry, "get_draft_for_role", return_value=draft_config):
                cmd = backend._build_command(role_config, request)

        assert "llama-speculative" in cmd  # Different binary
        assert "-md" in cmd  # Draft model
        assert "test-model.gguf" in cmd  # Draft path
        assert "--draft-max 24" in cmd
        assert "-n 256" in cmd

    def test_build_command_moe_expert_reduction(self, minimal_registry):
        """Test command building for MoE expert reduction."""
        backend = LlamaCppBackend(minimal_registry)
        role_config = minimal_registry.get_role("moe_role")
        request = InferenceRequest(role="moe_role", prompt="Explain", n_tokens=512)

        with patch("src.config.get_config") as mock_config:
            mock_config.return_value.paths.llama_cpp_bin = Path("/usr/bin")
            cmd = backend._build_command(role_config, request)

        assert "llama-completion" in cmd  # Uses completion binary
        assert "--override-kv qwen3moe.expert_used_count=int:4" in cmd
        assert "-n 512" in cmd
        # Should NOT have spec flags
        assert "-md" not in cmd

    def test_build_command_prompt_lookup(self, minimal_registry):
        """Test command building for prompt lookup."""
        backend = LlamaCppBackend(minimal_registry)
        role_config = minimal_registry.get_role("lookup_role")
        request = InferenceRequest(role="lookup_role", prompt="Summarize", n_tokens=128)

        with patch("src.config.get_config") as mock_config:
            mock_config.return_value.paths.llama_cpp_bin = Path("/usr/bin")
            cmd = backend._build_command(role_config, request)

        assert "llama-lookup" in cmd  # Different binary
        assert "--draft-max 16" in cmd
        assert "-n 128" in cmd


class TestLlamaCppBackendParsing:
    """Tests for output parsing methods."""

    def test_parse_speed_tokens_per_second_format(self, minimal_registry):
        """Test parsing speed from 'X.XX tokens per second' format."""
        backend = LlamaCppBackend(minimal_registry)
        output = """
        eval time = 2345.67 ms / 100 tokens (23.45 tokens per second)
        """

        speed = backend._parse_speed(output)
        assert speed == 23.45

    def test_parse_speed_decoded_format(self, minimal_registry):
        """Test parsing speed from 'decoded X tokens in Y.YYs, Z.ZZ t/s' format."""
        backend = LlamaCppBackend(minimal_registry)
        output = """
        decoded 128 tokens in 3.87s, 33.07 t/s
        """

        speed = backend._parse_speed(output)
        assert speed == 33.07

    def test_parse_speed_eval_time_format(self, minimal_registry):
        """Test parsing speed from eval time with ms > 0."""
        backend = LlamaCppBackend(minimal_registry)
        output = """
        eval time = 1500.00 ms / 50 tokens, 33.33 tokens per second
        """

        speed = backend._parse_speed(output)
        # Should match first pattern (tokens per second)
        assert speed == 33.33

    def test_parse_speed_eval_time_calculation(self, minimal_registry):
        """Test speed calculation from eval time when only timing info available."""
        backend = LlamaCppBackend(minimal_registry)
        # Output with timing but no explicit t/s
        output = """
        llama_perf_context_print: eval time = 2000.00 ms / 100 runs (20.00 ms per token, 50.00 tokens per second)
        """

        speed = backend._parse_speed(output)
        # Should extract the explicit "tokens per second"
        assert speed == 50.00

    def test_parse_speed_no_match(self, minimal_registry):
        """Test parsing speed returns 0.0 when no pattern matches."""
        backend = LlamaCppBackend(minimal_registry)
        output = "No speed information here"

        speed = backend._parse_speed(output)
        assert speed == 0.0

    def test_parse_tokens_decoded_format(self, minimal_registry):
        """Test parsing token count from 'decoded X tokens' format."""
        backend = LlamaCppBackend(minimal_registry)
        output = "decoded 128 tokens in 3.87s"

        tokens = backend._parse_tokens(output, requested=64)
        assert tokens == 128  # Actual count, not requested

    def test_parse_tokens_eval_format(self, minimal_registry):
        """Test parsing token count from 'eval: X tokens' format."""
        backend = LlamaCppBackend(minimal_registry)
        output = "eval: 256 tokens in 5.2s"

        tokens = backend._parse_tokens(output, requested=128)
        assert tokens == 256

    def test_parse_tokens_no_match_fallback(self, minimal_registry):
        """Test parsing tokens falls back to requested count."""
        backend = LlamaCppBackend(minimal_registry)
        output = "No token count here"

        tokens = backend._parse_tokens(output, requested=512)
        assert tokens == 512  # Fallback

    def test_extract_error_with_error_keyword(self, minimal_registry):
        """Test extracting error message with 'error:' pattern."""
        backend = LlamaCppBackend(minimal_registry)
        stderr = """
        Loading model...
        error: failed to load model from test.gguf
        """

        error_msg = backend._extract_error(stderr)
        assert "error:" in error_msg.lower()
        assert "failed to load" in error_msg.lower()

    def test_extract_error_with_failed_keyword(self, minimal_registry):
        """Test extracting error message with 'failed' pattern."""
        backend = LlamaCppBackend(minimal_registry)
        stderr = """
        Initialization complete
        Failed to allocate memory for context
        """

        error_msg = backend._extract_error(stderr)
        assert "failed" in error_msg.lower()

    def test_extract_error_with_invalid_keyword(self, minimal_registry):
        """Test extracting error message with 'invalid' pattern."""
        backend = LlamaCppBackend(minimal_registry)
        stderr = "invalid model format"

        error_msg = backend._extract_error(stderr)
        assert "invalid" in error_msg.lower()

    def test_extract_error_fallback_last_line(self, minimal_registry):
        """Test extracting error falls back to last non-empty line."""
        backend = LlamaCppBackend(minimal_registry)
        stderr = """
        Line 1
        Line 2
        Last line with info

        """

        error_msg = backend._extract_error(stderr)
        assert error_msg == "Last line with info"

    def test_extract_error_empty_stderr(self, minimal_registry):
        """Test extracting error from empty stderr."""
        backend = LlamaCppBackend(minimal_registry)
        stderr = ""

        error_msg = backend._extract_error(stderr)
        assert error_msg == "Unknown error"


class TestModelServerInferError:
    """Tests for ModelServer.infer() error handling."""

    def test_infer_invalid_role(self, minimal_registry):
        """Test inference with invalid role raises error."""
        server = ModelServer(registry=minimal_registry)
        request = InferenceRequest(role="nonexistent_role", prompt="Hello")

        with pytest.raises(ModelServerError, match="Role not found"):
            server.infer(request)

    def test_infer_updates_status_on_error(self, minimal_registry):
        """Test inference updates model status on error."""
        backend = LlamaCppBackend(minimal_registry)
        server = ModelServer(registry=minimal_registry, backend=backend)

        # Load the role first
        server.load("baseline_role")

        request = InferenceRequest(role="baseline_role", prompt="Hello")

        # Mock backend to return error
        with patch.object(backend, "infer") as mock_infer:
            from src.model_server import InferenceResult, ModelState

            mock_infer.return_value = InferenceResult(
                role="baseline_role",
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=1.0,
                success=False,
                error_message="Simulated error",
            )

            result = server.infer(request)

        assert result.success is False
        status = server.get_status("baseline_role")
        assert status.state == ModelState.ERROR
        assert status.error_message == "Simulated error"
