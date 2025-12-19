#!/usr/bin/env python3
"""
Command Executor for LLM Inference

Builds and executes llama.cpp commands for:
- Baseline inference (llama-completion)
- Speculative decoding (llama-speculative)
- MoE expert reduction (llama-completion with override)
- Prompt lookup (llama-lookup)

This module is shared with the orchestrator project.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from .registry import ModelRegistry, load_registry
except ImportError:
    from registry import ModelRegistry, load_registry


# Default inference parameters (fallbacks if registry unavailable)
DEFAULT_THREADS = 96
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TIMEOUT = 180


def get_binary_paths(registry: Optional["ModelRegistry"] = None) -> dict[str, str]:
    """Get binary paths from registry (single source of truth).

    Falls back to hardcoded paths only if registry is unavailable.
    """
    fallback = {
        "base_dir": "/mnt/raid0/llm/llama.cpp/build/bin",
        "completion": "llama-completion",
        "speculative": "llama-speculative",
        "lookup": "llama-lookup",
        "cli": "llama-cli",
    }

    if registry is None:
        try:
            registry = load_registry()
        except Exception:
            pass

    if registry and hasattr(registry, "data"):
        binaries = registry.data.get("runtime_defaults", {}).get("binaries", {})
        if binaries:
            return binaries

    return fallback


def get_binary(name: str, registry: Optional["ModelRegistry"] = None) -> str:
    """Get full path to a specific binary.

    Args:
        name: Binary name ('completion', 'speculative', 'lookup', 'cli')
        registry: Optional registry instance

    Returns:
        Full absolute path to the binary
    """
    paths = get_binary_paths(registry)
    base_dir = paths.get("base_dir", "/mnt/raid0/llm/llama.cpp/build/bin")
    binary_name = paths.get(name, name)
    return os.path.join(base_dir, binary_name)


def validate_binaries(registry: Optional["ModelRegistry"] = None) -> dict[str, str]:
    """Validate all required binaries exist.

    Raises:
        FileNotFoundError: If any binary is missing, with clear error message.

    Returns:
        Dict mapping binary name to full path (for logging/debugging).
    """
    required = ["completion", "speculative", "lookup"]
    paths = {}
    missing = []

    for name in required:
        path = get_binary(name, registry)
        paths[name] = path
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")

    if missing:
        raise FileNotFoundError(
            f"Missing llama.cpp binaries (check registry runtime_defaults.binaries):\n"
            + "\n".join(missing)
            + f"\n\nRegistry location: /mnt/raid0/llm/claude/orchestration/model_registry.yaml"
        )

    return paths


@dataclass
class InferenceResult:
    """Result of an inference run."""

    raw_output: str
    exit_code: int
    command: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class Config:
    """Benchmark configuration."""

    name: str  # e.g., 'baseline', 'spec_k8', 'moe4', 'lookup_n3'
    config_type: str  # 'baseline', 'spec', 'moe', 'lookup'

    # Speculative decoding
    spec_k: Optional[int] = None
    draft_model_path: Optional[str] = None

    # MoE expert reduction
    moe_experts: Optional[int] = None
    moe_override_key: Optional[str] = None

    # Prompt lookup
    lookup_ngram: Optional[int] = None

    # Speed-test optimization: if True, only measure speed (quality inherited from baseline)
    speed_test_only: bool = False
    inherits_quality_from: Optional[str] = None  # Config name to copy scores from

    @classmethod
    def baseline(cls) -> "Config":
        return cls(name="baseline", config_type="baseline")

    @classmethod
    def spec(cls, k: int, draft_path: str, draft_name: str = "") -> "Config":
        # Include draft model name in config name to distinguish between drafts
        if draft_name:
            name = f"spec_{draft_name}_k{k}"
        else:
            # Fallback: extract name from path
            draft_stem = Path(draft_path).stem if draft_path else "unknown"
            name = f"spec_{draft_stem}_k{k}"
        return cls(
            name=name,
            config_type="spec",
            spec_k=k,
            draft_model_path=draft_path,
        )

    @classmethod
    def moe(cls, experts: int, override_key: str) -> "Config":
        return cls(
            name=f"moe{experts}",
            config_type="moe",
            moe_experts=experts,
            moe_override_key=override_key,
        )

    @classmethod
    def lookup(cls, ngram: int) -> "Config":
        return cls(
            name=f"lookup_n{ngram}",
            config_type="lookup",
            lookup_ngram=ngram,
        )

    @classmethod
    def compound_moe_lookup(cls, experts: int, override_key: str, ngram: int) -> "Config":
        return cls(
            name=f"moe{experts}_lookup_n{ngram}",
            config_type="moe_lookup",
            moe_experts=experts,
            moe_override_key=override_key,
            lookup_ngram=ngram,
        )

class Executor:
    """Executes llama.cpp inference commands."""

    def __init__(self, registry: Optional[ModelRegistry] = None, validate: bool = True):
        self.registry = registry or load_registry()
        # Validate binaries exist on startup - fail fast with clear error
        if validate:
            self._binary_paths = validate_binaries(self.registry)
        else:
            self._binary_paths = None

    def get_configs_for_architecture(
        self,
        architecture: str,
        role: str,
        registry: Optional[ModelRegistry] = None,
    ) -> list[Config]:
        """Get applicable configs for a model architecture.

        Args:
            architecture: Model architecture ('dense', 'moe', 'qwen3moe', 'ssm_moe_hybrid', etc.)
            role: The model role for checking constraints and drafts.
            registry: Optional registry instance.

        Returns:
            List of Config objects to test.
        """
        reg = registry or self.registry
        configs = [Config.baseline()]

        # Get forbidden optimizations
        forbidden = reg.get_forbidden_configs(role)

        # Architecture-specific configs
        if architecture in ("moe", "qwen3moe", "qwen3vlmoe", "mixtral", "deepseek2"):
            override_key = reg.get_moe_override_key(role) or "qwen3moe.expert_used_count"
            for experts in [2, 4, 6, 8]:
                configs.append(Config.moe(experts, override_key))

            # MoE + lookup compound config (no spec decode - no MoE-compatible drafts exist)
            if "prompt_lookup" not in forbidden:
                configs.append(Config.compound_moe_lookup(4, override_key, 4))

        elif architecture in ("ssm_moe_hybrid", "qwen3next"):
            # SSM models - MoE reduction ONLY, no speculation
            override_key = reg.get_moe_override_key(role) or "qwen3moe.expert_used_count"
            for experts in [2, 4, 6, 8]:
                configs.append(Config.moe(experts, override_key))

        else:
            # Dense models - try speculative decoding with ALL compatible drafts
            # NOTE: Spec decode configs are speed_test_only because quality is identical
            # to baseline (same target model). Only speed differs.
            if "speculative_decoding" not in forbidden:
                drafts = reg.get_drafts_for_model(role)
                for draft_role in drafts:
                    draft_path = reg.get_model_path(draft_role)
                    if draft_path and os.path.exists(draft_path):
                        for k in [4, 8, 16, 24]:
                            cfg = Config.spec(k, draft_path, draft_role)
                            cfg.speed_test_only = True
                            cfg.inherits_quality_from = "baseline"
                            configs.append(cfg)
                        # Test ALL compatible drafts, not just the first one

            # Skip lookup for draft models (Tier D) - they're already fast
            # and lookup requires substantial prompts to be effective
            role_config = reg.get_role_config(role)
            is_draft = role_config and role_config.get("tier") == "D"

            if "prompt_lookup" not in forbidden and not is_draft:
                for ngram in [3, 4, 5]:
                    configs.append(Config.lookup(ngram))

        return configs

    def build_command(
        self,
        model_path: str,
        config: Config,
        prompt_file: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        threads: int = DEFAULT_THREADS,
    ) -> list[str]:
        """Build the llama.cpp command for a configuration.

        Args:
            model_path: Path to the target model.
            config: The benchmark configuration.
            prompt_file: Path to file containing the prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            threads: Number of threads.

        Returns:
            Command as list of strings.
        """
        # Select binary based on config type (reads from registry)
        if config.config_type == "spec" or config.config_type == "moe_spec":
            binary = get_binary("speculative", self.registry)
        elif config.config_type == "lookup" or config.config_type == "moe_lookup":
            binary = get_binary("lookup", self.registry)
        else:
            binary = get_binary("completion", self.registry)

        completion_binary = get_binary("completion", self.registry)

        # Base command with env wrapper
        cmd = [
            "env", "OMP_NUM_THREADS=1",
            "numactl", "--interleave=all",
            binary,
            "-m", model_path,
            "-t", str(threads),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-f", prompt_file,
        ]

        # --no-conversation only works with llama-completion (prevents interactive hangs)
        if binary == completion_binary:
            cmd.append("--no-conversation")

        # Add config-specific flags
        if config.config_type == "spec":
            cmd.extend([
                "-md", config.draft_model_path,
                "--draft-max", str(config.spec_k),
            ])
        elif config.config_type == "moe":
            cmd.extend([
                "--override-kv", f"{config.moe_override_key}=int:{config.moe_experts}",
            ])
        elif config.config_type == "lookup":
            cmd.extend([
                "--draft-max", str(config.lookup_ngram or 16),
            ])
        elif config.config_type == "moe_spec":
            cmd.extend([
                "-md", config.draft_model_path,
                "--draft-max", str(config.spec_k),
                "--override-kv", f"{config.moe_override_key}=int:{config.moe_experts}",
            ])
        elif config.config_type == "moe_lookup":
            cmd.extend([
                "--draft-max", str(config.lookup_ngram or 16),
                "--override-kv", f"{config.moe_override_key}=int:{config.moe_experts}",
            ])

        return cmd

    def run_inference(
        self,
        model_path: str,
        config: Config,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        threads: int = DEFAULT_THREADS,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> InferenceResult:
        """Run inference with the given configuration.

        Args:
            model_path: Path to the target model.
            config: The benchmark configuration.
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            threads: Number of threads.
            timeout: Timeout in seconds.

        Returns:
            InferenceResult with output and status.
        """
        # Write prompt to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="/mnt/raid0/llm/tmp"
        ) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            cmd = self.build_command(
                model_path, config, prompt_file, max_tokens, temperature, threads
            )
            cmd_str = " ".join(cmd)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return InferenceResult(
                    raw_output=result.stdout + result.stderr,
                    exit_code=result.returncode,
                    command=cmd_str,
                )
            except subprocess.TimeoutExpired:
                return InferenceResult(
                    raw_output="",
                    exit_code=-1,
                    command=cmd_str,
                    timed_out=True,
                )

        finally:
            # Clean up temp file
            if os.path.exists(prompt_file):
                os.unlink(prompt_file)


def build_command(
    model_path: str,
    config: Config,
    prompt_file: str,
    registry: Optional[ModelRegistry] = None,
) -> list[str]:
    """Convenience function to build a command."""
    executor = Executor(registry)
    return executor.build_command(model_path, config, prompt_file)


def run_inference(
    model_path: str,
    config: Config,
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    registry: Optional[ModelRegistry] = None,
) -> InferenceResult:
    """Convenience function to run inference."""
    executor = Executor(registry)
    return executor.run_inference(model_path, config, prompt, timeout=timeout)


if __name__ == "__main__":
    # Test the module
    import sys

    print("=== Executor Test ===\n")

    registry = load_registry()
    executor = Executor(registry)

    # Test config generation for different architectures
    test_roles = ["coder_primary", "worker_math", "ingest_long_context"]

    for role in test_roles:
        arch = registry.get_architecture(role)
        path = registry.get_model_path(role)
        print(f"Role: {role}")
        print(f"  Architecture: {arch}")
        print(f"  Path: {path}")

        configs = executor.get_configs_for_architecture(arch, role)
        print(f"  Configs ({len(configs)}):")
        for cfg in configs:
            print(f"    - {cfg.name}")
        print()

    # Test command building (dry run)
    print("Sample command for baseline:")
    cfg = Config.baseline()
    cmd = executor.build_command(
        "/mnt/raid0/llm/models/test.gguf",
        cfg,
        "/tmp/prompt.txt",
    )
    print(f"  {' '.join(cmd)}")
