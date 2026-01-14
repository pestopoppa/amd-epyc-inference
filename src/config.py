"""Centralized configuration for the orchestration system.

This module consolidates all configuration into a hierarchical structure
with environment variable support via pydantic-settings.

Usage:
    from src.config import get_config, OrchestratorConfig

    # Get config (loads from environment)
    config = get_config()

    # Access nested config
    print(config.llm.output_cap)
    print(config.escalation.max_retries)

Environment Variables:
    All settings can be overridden via environment variables with
    ORCHESTRATOR_ prefix and double underscore for nesting:

    ORCHESTRATOR_MOCK_MODE=1
    ORCHESTRATOR_LLM__OUTPUT_CAP=4096
    ORCHESTRATOR_ESCALATION__MAX_RETRIES=3
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# Try to use pydantic-settings if available, fall back to basic dataclass
try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    PYDANTIC_SETTINGS_AVAILABLE = False


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean from environment variable."""
    val = os.environ.get(name, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    """Parse integer from environment variable."""
    val = os.environ.get(name, "")
    if val.isdigit():
        return int(val)
    return default


def _env_float(name: str, default: float) -> float:
    """Parse float from environment variable."""
    val = os.environ.get(name, "")
    try:
        return float(val)
    except ValueError:
        return default


# ============================================================================
# Configuration Dataclasses (Pydantic-free fallback)
# ============================================================================


@dataclass
class LLMConfig:
    """Configuration for LLM primitives."""

    output_cap: int = 8192
    """Maximum characters per sub-LM output."""

    batch_parallelism: int = 4
    """Maximum parallel calls in llm_batch."""

    call_timeout: int = 120
    """Timeout per call in seconds."""

    mock_response_prefix: str = "[MOCK]"
    """Prefix for mock responses."""

    max_recursion_depth: int = 5
    """Maximum nesting depth for sub-LM calls."""

    default_prompt_rate: float = 0.50
    """Default cost rate per 1M prompt tokens."""

    default_completion_rate: float = 1.50
    """Default cost rate per 1M completion tokens."""


@dataclass
class EscalationConfigData:
    """Configuration for escalation policy."""

    max_retries: int = 2
    """Maximum retries before escalation."""

    max_escalations: int = 2
    """Maximum escalations per task."""

    optional_gates: frozenset[str] = field(
        default_factory=lambda: frozenset({"typecheck", "integration", "shellcheck"})
    )
    """Gates that can be skipped on timeout."""


@dataclass
class REPLConfigData:
    """Configuration for REPL environment."""

    max_output_len: int = 10000
    """Maximum output length per execution."""

    timeout_seconds: int = 30
    """Execution timeout."""

    forbidden_modules: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "os", "sys", "subprocess", "shutil", "pathlib",
            "socket", "http", "urllib", "ftplib", "smtplib",
            "pickle", "marshal", "shelve", "dbm",
            "ctypes", "multiprocessing", "threading",
            "importlib", "builtins", "__builtins__",
            "code", "codeop", "compile", "exec", "eval",
        })
    )
    """Modules blocked from import."""

    forbidden_builtins: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "__import__", "eval", "exec", "compile",
            "open", "input", "breakpoint",
            "globals", "locals", "vars",
            "getattr", "setattr", "delattr", "hasattr",
            "type", "object", "__build_class__",
            "memoryview", "bytearray",
        })
    )
    """Builtins blocked from use."""


@dataclass
class ServerConfigData:
    """Configuration for backend servers."""

    default_url: str = "http://localhost:8080"
    """Default server URL."""

    timeout: int = 300
    """Request timeout in seconds."""

    num_slots: int = 4
    """Number of parallel slots."""

    connect_timeout: int = 5
    """Connection timeout."""

    retry_count: int = 3
    """Number of retries on failure."""

    retry_backoff: float = 0.5
    """Backoff factor for retries."""


@dataclass
class MonitorConfigData:
    """Configuration for generation monitoring."""

    entropy_threshold: float = 2.5
    """Entropy threshold for early abort."""

    repetition_window: int = 50
    """Window size for repetition detection."""

    repetition_threshold: float = 0.3
    """Threshold for repetition ratio."""

    min_tokens_before_abort: int = 20
    """Minimum tokens before allowing abort."""


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    models_dir: Path = field(default_factory=lambda: Path("/mnt/raid0/llm/models"))
    """Directory for GGUF models."""

    cache_dir: Path = field(default_factory=lambda: Path("/mnt/raid0/llm/cache"))
    """Cache directory."""

    tmp_dir: Path = field(default_factory=lambda: Path("/mnt/raid0/llm/tmp"))
    """Temporary files directory."""

    registry_path: Path = field(
        default_factory=lambda: Path("/mnt/raid0/llm/claude/orchestration/model_registry.yaml")
    )
    """Path to model registry YAML."""


@dataclass
class FeaturesConfig:
    """Configuration for feature flags."""

    memrl: bool = False
    """Enable Memory-based RL (Q-scoring, learned routing)."""

    tools: bool = False
    """Enable tool registry for REPL."""

    scripts: bool = False
    """Enable script registry (requires tools)."""

    streaming: bool = False
    """Enable SSE streaming endpoints."""

    openai_compat: bool = False
    """Enable OpenAI-compatible API."""

    repl: bool = True
    """Enable REPL execution."""

    caching: bool = True
    """Enable response caching."""


@dataclass
class OrchestratorConfigData:
    """Root configuration for the orchestrator system.

    This dataclass provides the complete configuration hierarchy.
    For production use with environment variables, use get_config().
    """

    mock_mode: bool = True
    """Use mock responses instead of real inference."""

    debug: bool = False
    """Enable debug logging."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    """LLM primitives configuration."""

    escalation: EscalationConfigData = field(default_factory=EscalationConfigData)
    """Escalation policy configuration."""

    repl: REPLConfigData = field(default_factory=REPLConfigData)
    """REPL environment configuration."""

    server: ServerConfigData = field(default_factory=ServerConfigData)
    """Backend server configuration."""

    monitor: MonitorConfigData = field(default_factory=MonitorConfigData)
    """Generation monitor configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    """File paths configuration."""

    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    """Feature flags configuration."""


# ============================================================================
# Pydantic Settings (if available)
# ============================================================================

if PYDANTIC_SETTINGS_AVAILABLE:

    class LLMSettings(BaseSettings):
        """LLM configuration with env support."""

        output_cap: int = 8192
        batch_parallelism: int = 4
        call_timeout: int = 120
        mock_response_prefix: str = "[MOCK]"
        max_recursion_depth: int = 5
        default_prompt_rate: float = 0.50
        default_completion_rate: float = 1.50

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_LLM_",
            extra="ignore",
        )

    class EscalationSettings(BaseSettings):
        """Escalation configuration with env support."""

        max_retries: int = 2
        max_escalations: int = 2

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_ESCALATION_",
            extra="ignore",
        )

    class REPLSettings(BaseSettings):
        """REPL configuration with env support."""

        max_output_len: int = 10000
        timeout_seconds: int = 30

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_REPL_",
            extra="ignore",
        )

    class ServerSettings(BaseSettings):
        """Server configuration with env support."""

        default_url: str = "http://localhost:8080"
        timeout: int = 300
        num_slots: int = 4
        connect_timeout: int = 5
        retry_count: int = 3
        retry_backoff: float = 0.5

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_SERVER_",
            extra="ignore",
        )

    class FeaturesSettings(BaseSettings):
        """Feature flags with env support."""

        memrl: bool = False
        tools: bool = False
        scripts: bool = False
        streaming: bool = False
        openai_compat: bool = False
        repl: bool = True
        caching: bool = True

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_",
            extra="ignore",
        )

    class OrchestratorSettings(BaseSettings):
        """Root configuration with env support."""

        mock_mode: bool = True
        debug: bool = False

        llm: LLMSettings = Field(default_factory=LLMSettings)
        escalation: EscalationSettings = Field(default_factory=EscalationSettings)
        repl: REPLSettings = Field(default_factory=REPLSettings)
        server: ServerSettings = Field(default_factory=ServerSettings)
        features: FeaturesSettings = Field(default_factory=FeaturesSettings)

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_",
            extra="ignore",
        )


# ============================================================================
# Configuration Loading
# ============================================================================


def _load_from_env() -> OrchestratorConfigData:
    """Load configuration from environment variables (fallback method)."""
    return OrchestratorConfigData(
        mock_mode=_env_bool("ORCHESTRATOR_MOCK_MODE", True),
        debug=_env_bool("ORCHESTRATOR_DEBUG", False),
        llm=LLMConfig(
            output_cap=_env_int("ORCHESTRATOR_LLM_OUTPUT_CAP", 8192),
            batch_parallelism=_env_int("ORCHESTRATOR_LLM_BATCH_PARALLELISM", 4),
            call_timeout=_env_int("ORCHESTRATOR_LLM_CALL_TIMEOUT", 120),
            max_recursion_depth=_env_int("ORCHESTRATOR_LLM_MAX_RECURSION_DEPTH", 5),
        ),
        escalation=EscalationConfigData(
            max_retries=_env_int("ORCHESTRATOR_ESCALATION_MAX_RETRIES", 2),
            max_escalations=_env_int("ORCHESTRATOR_ESCALATION_MAX_ESCALATIONS", 2),
        ),
        repl=REPLConfigData(
            max_output_len=_env_int("ORCHESTRATOR_REPL_MAX_OUTPUT_LEN", 10000),
            timeout_seconds=_env_int("ORCHESTRATOR_REPL_TIMEOUT_SECONDS", 30),
        ),
        server=ServerConfigData(
            default_url=os.environ.get("ORCHESTRATOR_SERVER_DEFAULT_URL", "http://localhost:8080"),
            timeout=_env_int("ORCHESTRATOR_SERVER_TIMEOUT", 300),
            num_slots=_env_int("ORCHESTRATOR_SERVER_NUM_SLOTS", 4),
        ),
        features=FeaturesConfig(
            memrl=_env_bool("ORCHESTRATOR_MEMRL", False),
            tools=_env_bool("ORCHESTRATOR_TOOLS", False),
            scripts=_env_bool("ORCHESTRATOR_SCRIPTS", False),
            streaming=_env_bool("ORCHESTRATOR_STREAMING", False),
            openai_compat=_env_bool("ORCHESTRATOR_OPENAI_COMPAT", False),
            repl=_env_bool("ORCHESTRATOR_REPL", True),
            caching=_env_bool("ORCHESTRATOR_CACHING", True),
        ),
    )


@lru_cache(maxsize=1)
def get_config() -> OrchestratorConfigData:
    """Get the global configuration.

    Loads configuration from environment variables. Uses pydantic-settings
    if available, otherwise falls back to manual env parsing.

    Returns:
        OrchestratorConfigData with all settings.
    """
    if PYDANTIC_SETTINGS_AVAILABLE:
        settings = OrchestratorSettings()
        # Convert to dataclass for consistency
        return OrchestratorConfigData(
            mock_mode=settings.mock_mode,
            debug=settings.debug,
            llm=LLMConfig(
                output_cap=settings.llm.output_cap,
                batch_parallelism=settings.llm.batch_parallelism,
                call_timeout=settings.llm.call_timeout,
                mock_response_prefix=settings.llm.mock_response_prefix,
                max_recursion_depth=settings.llm.max_recursion_depth,
                default_prompt_rate=settings.llm.default_prompt_rate,
                default_completion_rate=settings.llm.default_completion_rate,
            ),
            escalation=EscalationConfigData(
                max_retries=settings.escalation.max_retries,
                max_escalations=settings.escalation.max_escalations,
            ),
            repl=REPLConfigData(
                max_output_len=settings.repl.max_output_len,
                timeout_seconds=settings.repl.timeout_seconds,
            ),
            server=ServerConfigData(
                default_url=settings.server.default_url,
                timeout=settings.server.timeout,
                num_slots=settings.server.num_slots,
                connect_timeout=settings.server.connect_timeout,
                retry_count=settings.server.retry_count,
                retry_backoff=settings.server.retry_backoff,
            ),
            features=FeaturesConfig(
                memrl=settings.features.memrl,
                tools=settings.features.tools,
                scripts=settings.features.scripts,
                streaming=settings.features.streaming,
                openai_compat=settings.features.openai_compat,
                repl=settings.features.repl,
                caching=settings.features.caching,
            ),
        )
    else:
        return _load_from_env()


def reset_config() -> None:
    """Reset the cached configuration.

    Call this if environment variables change during runtime.
    """
    get_config.cache_clear()


# Backwards compatibility aliases
Config = OrchestratorConfigData
