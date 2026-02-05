"""Centralized configuration for the orchestration system.

This module consolidates all configuration into a hierarchical structure
with environment variable support via pydantic-settings.

Usage:
    from src.config import get_config

    # Get config (loads from environment)
    config = get_config()

    # Access nested config
    print(config.llm.output_cap)
    print(config.server_urls.frontdoor)
    print(config.timeouts.for_role("architect_general"))

Environment Variables:
    All settings can be overridden via environment variables with
    ORCHESTRATOR_ prefix and section name:

    ORCHESTRATOR_MOCK_MODE=1
    ORCHESTRATOR_LLM_OUTPUT_CAP=4096
    ORCHESTRATOR_ESCALATION_MAX_RETRIES=3
    ORCHESTRATOR_SERVER_URLS_FRONTDOOR=http://custom:9999
    ORCHESTRATOR_TIMEOUTS_DEFAULT_REQUEST=60
    ORCHESTRATOR_PATHS_PROJECT_ROOT=/custom/path
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


# Try to use pydantic-settings if available, fall back to basic dataclass
try:
    from pydantic import Field as PydanticField
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
        return float(val) if val else default
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    """Parse string from environment variable."""
    return os.environ.get(name, default)


# Registry timeout cache (avoids repeated YAML parsing)
_REGISTRY_TIMEOUTS_CACHE: dict[str, int | float] | None = None


def _load_registry_timeouts() -> dict[str, int | float]:
    """Load all timeouts from the registry (single source of truth).

    Returns a flat dict with keys like:
        "roles.architect_general", "server.request", "services.ocr_pdf"
    """
    global _REGISTRY_TIMEOUTS_CACHE
    if _REGISTRY_TIMEOUTS_CACHE is not None:
        return _REGISTRY_TIMEOUTS_CACHE

    try:
        from src.registry_loader import RegistryLoader

        registry = RegistryLoader(validate_paths=False, allow_missing=True)
        raw_timeouts = registry._runtime_defaults.get("timeouts", {})

        # Flatten nested structure to "category.key" format
        flat: dict[str, int | float] = {"default": raw_timeouts.get("default", 600)}
        # All timeout categories defined in model_registry.yaml
        categories = [
            "roles", "server", "services", "pools", "benchmark",
            "repl", "tools", "external", "health", "scripts", "backends",
        ]
        for category in categories:
            cat_data = raw_timeouts.get(category, {})
            for key, value in cat_data.items():
                flat[f"{category}.{key}"] = value

        _REGISTRY_TIMEOUTS_CACHE = flat
        return flat
    except Exception as e:
        logger.debug("Registry timeouts unavailable, using hardcoded fallbacks: %s", e)
        _REGISTRY_TIMEOUTS_CACHE = {}
        return {}


def _registry_timeout(category: str, key: str, fallback: int | float) -> int | float:
    """Get timeout from registry, falling back to hardcoded default.

    Args:
        category: Timeout category (roles, server, services, pools, benchmark).
        key: Timeout key within category.
        fallback: Hardcoded fallback if registry unavailable.

    Returns:
        Timeout value from registry, or fallback.
    """
    timeouts = _load_registry_timeouts()
    full_key = f"{category}.{key}"
    if full_key in timeouts:
        return timeouts[full_key]
    # Try just the key for backward compat
    if key in timeouts:
        return timeouts[key]
    # Use hardcoded fallback (registry default only used for unknown keys without fallback)
    return fallback


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class LLMConfig:
    """Configuration for LLM primitives."""

    output_cap: int = 8192
    """Maximum characters per sub-LM output."""

    batch_parallelism: int = 4
    """Maximum parallel calls in llm_batch."""

    call_timeout: int = 600  # Increased from 300 - architect calls can take ~300s
    """Timeout per call in seconds (matches LlamaServerBackend)."""

    mock_response_prefix: str = "[MOCK]"
    """Prefix for mock responses."""

    max_recursion_depth: int = 5
    """Maximum nesting depth for sub-LM calls."""

    default_prompt_rate: float = 0.50
    """Default cost rate per 1M prompt tokens."""

    default_completion_rate: float = 1.50
    """Default cost rate per 1M completion tokens."""

    qwen_stop_token: str = "<|im_end|>"
    """Qwen chat-template stop token to prevent runaway generation."""


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
        default_factory=lambda: frozenset(
            {
                "os",
                "sys",
                "subprocess",
                "shutil",
                "pathlib",
                "socket",
                "http",
                "urllib",
                "ftplib",
                "smtplib",
                "pickle",
                "marshal",
                "shelve",
                "dbm",
                "ctypes",
                "multiprocessing",
                "threading",
                "importlib",
                "builtins",
                "__builtins__",
                "code",
                "codeop",
                "compile",
                "exec",
                "eval",
            }
        )
    )
    """Modules blocked from import."""

    forbidden_builtins: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "__import__",
                "eval",
                "exec",
                "compile",
                "open",
                "input",
                "breakpoint",
                "globals",
                "locals",
                "vars",
                "getattr",
                "setattr",
                "delattr",
                "hasattr",
                "type",
                "object",
                "__build_class__",
                "memoryview",
                "bytearray",
            }
        )
    )
    """Builtins blocked from use."""


@dataclass
class ServerConfigData:
    """Configuration for backend servers."""

    default_url: str = "http://localhost:8080"
    """Default server URL."""

    timeout: int = 600
    """Request timeout in seconds (increased for architect models)."""

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
    """Configuration for generation monitoring.

    Base defaults used by MonitorConfig(). Per-tier and per-task overrides
    are in tier_overrides and task_overrides, consumed by for_tier()/for_task().
    """

    entropy_threshold: float = 4.0
    """Sustained entropy above this triggers abort."""

    entropy_spike_threshold: float = 2.0
    """Single-token entropy jump threshold."""

    repetition_threshold: float = 0.3
    """Threshold for repeated n-gram ratio (0-1)."""

    min_tokens_before_abort: int = 50
    """Minimum tokens before allowing abort."""

    perplexity_window: int = 20
    """Rolling window size for perplexity trend."""

    max_length_multiplier: float = 2.0
    """Abort if >N x median task length."""

    entropy_sustained_count: int = 10
    """Tokens of high entropy before abort."""

    ngram_size: int = 3
    """N-gram size for repetition detection."""

    combined_threshold: float = 0.7
    """Weighted score for combined signals."""

    tier_overrides: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "worker": {
                "entropy_threshold": 4.5,
                "entropy_spike_threshold": 2.5,
                "min_tokens_before_abort": 50,
            },
            "coder": {
                "entropy_threshold": 5.0,
                "entropy_spike_threshold": 3.0,
                "min_tokens_before_abort": 100,
                "repetition_threshold": 0.2,
            },
            "architect": {
                "entropy_threshold": 6.0,
                "entropy_spike_threshold": 4.0,
                "min_tokens_before_abort": 200,
                "repetition_threshold": 0.4,
            },
            "ingest": {
                "entropy_threshold": 5.5,
                "entropy_spike_threshold": 3.5,
                "min_tokens_before_abort": 100,
            },
        }
    )
    """Per-tier threshold overrides. Keys are tier names, values are dicts of field→value."""

    task_overrides: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "code": {"min_tokens_before_abort": 100, "repetition_threshold": 0.2, "ngram_size": 4},
            "reasoning": {
                "entropy_threshold": 4.5,
                "min_tokens_before_abort": 30,
                "perplexity_window": 15,
            },
        }
    )
    """Per-task threshold overrides. Keys are task types, values are dicts of field→value."""


def _get_default_llm_root() -> str:
    """Get LLM root from environment or default."""
    return os.environ.get("ORCHESTRATOR_PATHS_LLM_ROOT", "/mnt/raid0/llm")


def _get_default_project_root() -> str:
    """Get project root from environment or default."""
    llm_root = _get_default_llm_root()
    return os.environ.get("ORCHESTRATOR_PATHS_PROJECT_ROOT", f"{llm_root}/claude")


@dataclass
class PathsConfig:
    """Configuration for file paths.

    All paths can be overridden via ORCHESTRATOR_PATHS_* environment variables.
    Default values assume /mnt/raid0/llm layout but can be reconfigured.
    """

    # Base paths (configure these to relocate everything)
    llm_root: Path = field(default_factory=lambda: Path(_get_default_llm_root()))
    """Root directory for all LLM-related files."""

    project_root: Path = field(default_factory=lambda: Path(_get_default_project_root()))
    """Project root directory (claude repo)."""

    # Derived paths - these use llm_root/project_root as base
    models_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_MODELS_DIR", f"{_get_default_llm_root()}/models")
        )
    )
    """Directory for GGUF models."""

    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_CACHE_DIR", f"{_get_default_llm_root()}/cache")
        )
    )
    """Cache directory."""

    tmp_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_TMP_DIR", f"{_get_default_llm_root()}/tmp")
        )
    )
    """Temporary files directory."""

    registry_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_REGISTRY_PATH",
                f"{_get_default_project_root()}/orchestration/model_registry.yaml",
            )
        )
    )
    """Path to model registry YAML."""

    tool_registry_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_TOOL_REGISTRY_PATH",
                f"{_get_default_project_root()}/orchestration/tool_registry.yaml",
            )
        )
    )
    """Path to tool registry YAML."""

    script_registry_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_SCRIPT_REGISTRY_DIR",
                f"{_get_default_project_root()}/orchestration/script_registry",
            )
        )
    )
    """Directory for script registry."""

    sessions_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_SESSIONS_DIR",
                f"{_get_default_project_root()}/orchestration/repl_memory/sessions",
            )
        )
    )
    """Session storage directory."""

    artifacts_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_ARTIFACTS_DIR", f"{_get_default_llm_root()}/tmp/claude/artifacts"
            )
        )
    )
    """Artifacts directory for context manager."""

    llama_cpp_bin: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN", f"{_get_default_llm_root()}/llama.cpp/build/bin"
            )
        )
    )
    """llama.cpp binary directory."""

    model_base: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_MODEL_BASE", f"{_get_default_llm_root()}/lmstudio/models"
            )
        )
    )
    """Base directory for LM Studio models."""

    log_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", f"{_get_default_project_root()}/logs")
        )
    )
    """Log files directory."""

    raid_prefix: str = field(
        default_factory=lambda: os.environ.get("ORCHESTRATOR_PATHS_RAID_PREFIX", "/mnt/raid0/")
    )
    """Required prefix for all data paths (security). Set to empty string to disable check."""


@dataclass
class ServerURLsConfig:
    """Server URL mapping for all orchestrator roles.

    Each field maps an orchestrator role to a llama-server URL.
    """

    # Tier A - Front Door / Orchestrator
    frontdoor: str = "http://localhost:8080"
    coder_primary: str = "http://localhost:8080"

    # Tier B - Specialists (code)
    coder: str = "http://localhost:8081"
    coder_escalation: str = "http://localhost:8081"

    # Tier C - Workers
    worker: str = "http://localhost:8082"
    worker_general: str = "http://localhost:8082"
    worker_explore: str = "http://localhost:8082"
    worker_math: str = "http://localhost:8082"
    worker_vision: str = "http://localhost:8086"
    vision_escalation: str = "http://localhost:8087"
    worker_code: str = "http://localhost:8092"
    worker_fast: str = "http://localhost:8102"
    worker_summarize: str = "http://localhost:8081"

    # Tier B - Architects
    architect_general: str = "http://localhost:8083"
    architect_coding: str = "http://localhost:8084"
    ingest_long_context: str = "http://localhost:8085"

    # Services
    api_url: str = "http://localhost:8000"
    ocr_server: str = "http://localhost:9001"
    vision_api: str = "http://localhost:8000/v1/vision/analyze"

    def as_dict(self) -> dict[str, str]:
        """Return role->URL mapping as dict (for LLMPrimitives compatibility).

        Excludes service URLs (api_url, ocr_server, vision_api).
        """
        d = asdict(self)
        # Remove non-role entries
        for key in ("api_url", "ocr_server", "vision_api"):
            d.pop(key, None)
        return d


@dataclass
class TimeoutsConfig:
    """All timeout values in seconds.

    Source of truth: orchestration/model_registry.yaml (runtime_defaults.timeouts)
    Hardcoded defaults here are fallbacks only - registry values take precedence.
    """

    # Role-specific request timeouts (read from registry.timeouts.roles.*)
    worker_explore: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_explore", 30))
    )
    worker_math: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_math", 30))
    )
    worker_vision: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_vision", 30))
    )
    worker_summarize: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_summarize", 120))
    )
    frontdoor: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "frontdoor", 60))
    )
    coder_primary: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "coder_primary", 60))
    )
    coder_escalation: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "coder_escalation", 120))
    )
    vision_escalation: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "vision_escalation", 60))
    )
    ingest_long_context: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "ingest_long_context", 120))
    )
    architect_general: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "architect_general", 600))
    )
    architect_coding: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "architect_coding", 600))
    )
    default_request: int = field(
        default_factory=lambda: int(_registry_timeout("", "default", 600))
    )
    """Fallback for unknown roles."""

    # Backend timeouts (read from registry.timeouts.server.*)
    server_request: int = field(
        default_factory=lambda: int(_registry_timeout("server", "request", 600))
    )
    server_connect: int = field(
        default_factory=lambda: int(_registry_timeout("server", "connect", 5))
    )

    # Service timeouts (read from registry.timeouts.services.*)
    ocr_single_page: float = field(
        default_factory=lambda: float(_registry_timeout("services", "ocr_single_page", 120.0))
    )
    ocr_pdf: float = field(
        default_factory=lambda: float(_registry_timeout("services", "ocr_pdf", 600.0))
    )
    health_check: float = field(
        default_factory=lambda: float(_registry_timeout("server", "health_check", 5.0))
    )
    vision_inference: int = field(
        default_factory=lambda: int(_registry_timeout("services", "vision_inference", 120))
    )
    vision_figure: float = field(
        default_factory=lambda: float(_registry_timeout("services", "vision_figure", 60.0))
    )
    ffmpeg_version: int = field(
        default_factory=lambda: int(_registry_timeout("services", "ffmpeg_version", 5))
    )
    ffmpeg_probe: int = field(
        default_factory=lambda: int(_registry_timeout("services", "ffmpeg_probe", 30))
    )
    ffmpeg_extract: int = field(
        default_factory=lambda: int(_registry_timeout("services", "ffmpeg_extract", 600))
    )
    exiftool: int = field(
        default_factory=lambda: int(_registry_timeout("services", "exiftool", 30))
    )
    gradio_client: float = field(
        default_factory=lambda: float(_registry_timeout("services", "gradio_client", 300.0))
    )

    def for_role(self, role: str) -> int:
        """Get timeout for a specific role, falling back to default."""
        _role_map = {
            "worker_explore": self.worker_explore,
            "worker_math": self.worker_math,
            "worker_vision": self.worker_vision,
            "worker_summarize": self.worker_summarize,
            "frontdoor": self.frontdoor,
            "coder_primary": self.coder_primary,
            "coder_escalation": self.coder_escalation,
            "vision_escalation": self.vision_escalation,
            "ingest_long_context": self.ingest_long_context,
            "architect_general": self.architect_general,
            "architect_coding": self.architect_coding,
        }
        return _role_map.get(str(role), self.default_request)

    def role_timeouts_dict(self) -> dict[str, int]:
        """Return role->timeout dict (for backward compat with ROLE_TIMEOUTS)."""
        return {
            "worker_explore": self.worker_explore,
            "worker_math": self.worker_math,
            "worker_vision": self.worker_vision,
            "worker_summarize": self.worker_summarize,
            "frontdoor": self.frontdoor,
            "coder_primary": self.coder_primary,
            "coder_escalation": self.coder_escalation,
            "vision_escalation": self.vision_escalation,
            "ingest_long_context": self.ingest_long_context,
            "architect_general": self.architect_general,
            "architect_coding": self.architect_coding,
        }


@dataclass
class VisionConfig:
    """Vision pipeline configuration."""

    base_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_VISION_DIR", f"{_get_default_llm_root()}/vision")
        )
    )
    llama_mtmd_cli: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LLAMA_MTMD",
                f"{_get_default_llm_root()}/llama.cpp/build/bin/llama-mtmd-cli",
            )
        )
    )
    vl_model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_VL_MODEL",
                f"{_get_default_llm_root()}/lmstudio/models/lmstudio-community/"
                "Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            )
        )
    )
    vl_mmproj_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_VL_MMPROJ",
                f"{_get_default_llm_root()}/lmstudio/models/lmstudio-community/"
                "Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf",
            )
        )
    )
    vl_server_port: int = 8086
    vl_escalation_server_port: int = 8087

    # Processing limits
    max_image_size_mb: int = 20
    max_image_dimension: int = 4096
    default_batch_size: int = 100
    max_concurrent_workers: int = 4
    default_video_fps: float = 1.0
    default_vl_max_tokens: int = 512
    default_vl_threads: int = 8

    # Thumbnail settings
    thumb_size: tuple[int, int] = (256, 256)
    thumb_quality: int = 85
    temp_jpeg_quality: int = 95

    # Face detection
    face_min_confidence: float = 0.9
    face_embedding_dim: int = 512
    face_identification_threshold: float = 0.6

    # Model names
    arcface_model_name: str = "buffalo_l"
    clip_model_name: str = "ViT-B/32"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    onnx_providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    supported_image_extensions: list[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png", "heic", "webp", "bmp", "tiff"]
    )


@dataclass
class ChatPipelineConfig:
    """Chat pipeline configuration thresholds."""

    # Three-stage summarization
    summarization_threshold_tokens: int = 5000
    """~20K chars triggers Stage 1+2."""
    multi_doc_discount: float = 0.7
    """Lower threshold for multiple documents."""
    compression_enabled: bool = False
    """Stage 0 compression (disabled due to LLMLingua-2 quality issues)."""
    compression_min_chars: int = 30000
    compression_target_ratio: float = 0.5
    stage1_context_limit: int = 20000

    # Long context exploration
    long_context_enabled: bool = True
    long_context_threshold_chars: int = 20000
    """~5K tokens triggers exploration mode."""
    long_context_max_turns: int = 8

    # Quality detection thresholds
    repetition_unique_ratio: float = 0.5
    garbled_short_line_ratio: float = 0.6
    min_answer_length: int = 50

    # Review Q-value thresholds
    review_low_q_threshold: float = 0.6
    review_skip_q_threshold: float = 0.6

    # Plan review phase transitions
    plan_review_phase_a_min: int = 50
    plan_review_phase_b_mean_q: float = 0.7
    plan_review_phase_b_min_q: float = 0.5
    plan_review_phase_c_min_q: float = 0.7
    plan_review_phase_c_min_total: int = 100
    plan_review_phase_c_skip_rate: float = 0.90
    """Fraction of reviews skipped in Phase C (spot-check)."""


@dataclass
class DelegationConfig:
    """Configuration for proactive delegation."""

    max_iterations: int = 3
    max_total_iterations: int = 10
    max_concurrent_analysis: int = 4
    max_review_tokens: int = 128
    max_taskir_tokens: int = 256
    max_plan_review_tokens: int = 128


@dataclass
class ServicesConfig:
    """Configuration for services (OCR, PDF, archives, drafts)."""

    lightonocr_model: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LIGHTONOCR_MODEL",
                f"{_get_default_llm_root()}/models/LightOnOCR-2-1B-bbox-Q4_K_M.gguf",
            )
        )
    )
    lightonocr_mmproj: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LIGHTONOCR_MMPROJ",
                f"{_get_default_llm_root()}/models/LightOnOCR-2-1B-bbox-mmproj-F16.gguf",
            )
        )
    )
    lightonocr_max_tokens: int = 2048

    draft_cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_DRAFT_CACHE", f"{_get_default_llm_root()}/cache/drafts"
            )
        )
    )
    draft_cache_ttl_hours: float = 24.0

    archive_extract_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_ARCHIVE_EXTRACT",
                f"{_get_default_project_root()}/tmp/archives",
            )
        )
    )
    max_archive_size: int = 500 * 1024 * 1024  # 500 MB
    max_extracted_size: int = 1024 * 1024 * 1024  # 1 GB
    max_archive_files: int = 1000

    pdf_router_temp_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_PDF_ROUTER_TEMP", f"{_get_default_llm_root()}/tmp/pdf_router"
            )
        )
    )


@dataclass
class ApiConfig:
    """Configuration for API middleware (CORS, rate limiting)."""

    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]
    )
    """Allowed CORS origins. Must be explicit when allow_credentials is True."""

    cors_allow_credentials: bool = True
    """Whether to allow credentials in CORS requests."""

    rate_limit_rpm: int = 60
    """Requests per minute per client IP."""

    rate_limit_burst: int = 10
    """Maximum burst size above the sustained rate."""


@dataclass
class ExternalAPIConfig:
    """Configuration for a single external API backend."""

    api_key: str = ""
    """API key (loaded from environment)."""

    base_url: str = ""
    """Base URL for the API."""

    default_model: str = ""
    """Default model name to use."""

    timeout: int = 120
    """Request timeout in seconds."""

    max_retries: int = 3
    """Maximum retries on transient failures."""


@dataclass
class ExternalBackendsConfig:
    """Configuration for external API backends (Anthropic, OpenAI, etc.).

    API keys are loaded from environment variables:
      - ANTHROPIC_API_KEY
      - OPENAI_API_KEY

    Usage:
        config = get_config()
        if config.external_backends.anthropic.api_key:
            backend = AnthropicBackend(config.external_backends.anthropic)
    """

    anthropic: ExternalAPIConfig = field(
        default_factory=lambda: ExternalAPIConfig(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            base_url="https://api.anthropic.com",
            default_model="claude-3-5-sonnet-20241022",
            timeout=120,
            max_retries=3,
        )
    )
    """Anthropic API configuration."""

    openai: ExternalAPIConfig = field(
        default_factory=lambda: ExternalAPIConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url="https://api.openai.com/v1",
            default_model="gpt-4o",
            timeout=120,
            max_retries=3,
        )
    )
    """OpenAI API configuration."""

    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic.api_key)

    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai.api_key)


@dataclass
class WorkerPoolPathsConfig:
    """Paths for worker pool management."""

    llama_server_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LLAMA_SERVER",
                f"{_get_default_llm_root()}/llama.cpp/build/bin/llama-server",
            )
        )
    )
    log_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", f"{_get_default_project_root()}/logs")
        )
    )
    model_base: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_MODEL_BASE", f"{_get_default_llm_root()}/lmstudio/models"
            )
        )
    )


# NOTE: FeaturesConfig is kept for backward compatibility but features
# should be managed via src.features.Features (its own lifecycle/singleton).
@dataclass
class FeaturesConfig:
    """Configuration for feature flags (DEPRECATED: use src.features instead)."""

    memrl: bool = False
    tools: bool = False
    scripts: bool = False
    streaming: bool = False
    openai_compat: bool = False
    repl: bool = True
    caching: bool = True


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

    # Existing sections
    llm: LLMConfig = field(default_factory=LLMConfig)
    escalation: EscalationConfigData = field(default_factory=EscalationConfigData)
    repl: REPLConfigData = field(default_factory=REPLConfigData)
    server: ServerConfigData = field(default_factory=ServerConfigData)
    monitor: MonitorConfigData = field(default_factory=MonitorConfigData)
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    # New sections (Phase 3)
    server_urls: ServerURLsConfig = field(default_factory=ServerURLsConfig)
    timeouts: TimeoutsConfig = field(default_factory=TimeoutsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    chat: ChatPipelineConfig = field(default_factory=ChatPipelineConfig)
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    worker_pool: WorkerPoolPathsConfig = field(default_factory=WorkerPoolPathsConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    external_backends: ExternalBackendsConfig = field(default_factory=ExternalBackendsConfig)


# ============================================================================
# Pydantic Settings (if available)
# ============================================================================

if PYDANTIC_SETTINGS_AVAILABLE:

    class LLMSettings(BaseSettings):
        output_cap: int = 8192
        batch_parallelism: int = 4
        call_timeout: int = 600  # Increased from 300 - architect calls can take ~300s
        mock_response_prefix: str = "[MOCK]"
        max_recursion_depth: int = 5
        default_prompt_rate: float = 0.50
        default_completion_rate: float = 1.50
        qwen_stop_token: str = "<|im_end|>"

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_LLM_",
            extra="ignore",
        )

    class EscalationSettings(BaseSettings):
        max_retries: int = 2
        max_escalations: int = 2

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_ESCALATION_",
            extra="ignore",
        )

    class REPLSettings(BaseSettings):
        max_output_len: int = 10000
        timeout_seconds: int = 30

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_REPL_",
            extra="ignore",
        )

    class ServerSettings(BaseSettings):
        default_url: str = "http://localhost:8080"
        timeout: int = 600
        num_slots: int = 4
        connect_timeout: int = 5
        retry_count: int = 3
        retry_backoff: float = 0.5

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_SERVER_",
            extra="ignore",
        )

    class ServerURLsSettings(BaseSettings):
        frontdoor: str = "http://localhost:8080"
        coder_primary: str = "http://localhost:8080"
        coder: str = "http://localhost:8081"
        coder_escalation: str = "http://localhost:8081"
        worker: str = "http://localhost:8082"
        worker_general: str = "http://localhost:8082"
        worker_explore: str = "http://localhost:8082"
        worker_math: str = "http://localhost:8082"
        worker_vision: str = "http://localhost:8086"
        vision_escalation: str = "http://localhost:8087"
        worker_code: str = "http://localhost:8092"
        worker_fast: str = "http://localhost:8102"
        worker_summarize: str = "http://localhost:8081"
        architect_general: str = "http://localhost:8083"
        architect_coding: str = "http://localhost:8084"
        ingest_long_context: str = "http://localhost:8085"
        api_url: str = "http://localhost:8000"
        ocr_server: str = "http://localhost:9001"
        vision_api: str = "http://localhost:8000/v1/vision/analyze"

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_SERVER_URLS_",
            extra="ignore",
        )

    class TimeoutsSettings(BaseSettings):
        worker_explore: int = 30
        worker_math: int = 30
        worker_vision: int = 30
        worker_summarize: int = 120
        frontdoor: int = 60
        coder_primary: int = 60
        coder_escalation: int = 120
        vision_escalation: int = 60
        ingest_long_context: int = 120
        architect_general: int = 600
        architect_coding: int = 600
        default_request: int = 600
        server_request: int = 600
        server_connect: int = 5
        ocr_single_page: float = 120.0
        ocr_pdf: float = 600.0
        health_check: float = 5.0
        vision_inference: int = 120
        vision_figure: float = 60.0
        ffmpeg_version: int = 5
        ffmpeg_probe: int = 30
        ffmpeg_extract: int = 600
        exiftool: int = 30
        gradio_client: float = 300.0

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_TIMEOUTS_",
            extra="ignore",
        )

    class ChatPipelineSettings(BaseSettings):
        summarization_threshold_tokens: int = 5000
        multi_doc_discount: float = 0.7
        compression_enabled: bool = False
        compression_min_chars: int = 30000
        compression_target_ratio: float = 0.5
        stage1_context_limit: int = 20000
        long_context_enabled: bool = True
        long_context_threshold_chars: int = 20000
        long_context_max_turns: int = 8
        repetition_unique_ratio: float = 0.5
        garbled_short_line_ratio: float = 0.6
        min_answer_length: int = 50
        review_low_q_threshold: float = 0.6
        review_skip_q_threshold: float = 0.6
        plan_review_phase_a_min: int = 50
        plan_review_phase_b_mean_q: float = 0.7
        plan_review_phase_b_min_q: float = 0.5
        plan_review_phase_c_min_q: float = 0.7
        plan_review_phase_c_min_total: int = 100
        plan_review_phase_c_skip_rate: float = 0.90

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_CHAT_",
            extra="ignore",
        )

    class MonitorSettings(BaseSettings):
        entropy_threshold: float = 4.0
        entropy_spike_threshold: float = 2.0
        repetition_threshold: float = 0.3
        min_tokens_before_abort: int = 50
        perplexity_window: int = 20
        max_length_multiplier: float = 2.0
        entropy_sustained_count: int = 10
        ngram_size: int = 3
        combined_threshold: float = 0.7

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_MONITOR_",
            extra="ignore",
        )

    class FeaturesSettings(BaseSettings):
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
        mock_mode: bool = True
        debug: bool = False

        llm: LLMSettings = PydanticField(default_factory=LLMSettings)
        escalation: EscalationSettings = PydanticField(default_factory=EscalationSettings)
        repl: REPLSettings = PydanticField(default_factory=REPLSettings)
        server: ServerSettings = PydanticField(default_factory=ServerSettings)
        monitor: MonitorSettings = PydanticField(default_factory=MonitorSettings)
        features: FeaturesSettings = PydanticField(default_factory=FeaturesSettings)
        server_urls: ServerURLsSettings = PydanticField(default_factory=ServerURLsSettings)
        timeouts: TimeoutsSettings = PydanticField(default_factory=TimeoutsSettings)
        chat: ChatPipelineSettings = PydanticField(default_factory=ChatPipelineSettings)

        model_config = SettingsConfigDict(
            env_prefix="ORCHESTRATOR_",
            extra="ignore",
        )


# ============================================================================
# Configuration Loading
# ============================================================================


def _load_from_env() -> OrchestratorConfigData:
    """Load configuration from environment variables (fallback method)."""
    P = "ORCHESTRATOR_"
    return OrchestratorConfigData(
        mock_mode=_env_bool(f"{P}MOCK_MODE", True),
        debug=_env_bool(f"{P}DEBUG", False),
        llm=LLMConfig(
            output_cap=_env_int(f"{P}LLM_OUTPUT_CAP", 8192),
            batch_parallelism=_env_int(f"{P}LLM_BATCH_PARALLELISM", 4),
            call_timeout=_env_int(
                f"{P}LLM_CALL_TIMEOUT", int(_registry_timeout("server", "request", 600))
            ),
            max_recursion_depth=_env_int(f"{P}LLM_MAX_RECURSION_DEPTH", 5),
            default_prompt_rate=_env_float(f"{P}LLM_DEFAULT_PROMPT_RATE", 0.50),
            default_completion_rate=_env_float(f"{P}LLM_DEFAULT_COMPLETION_RATE", 1.50),
            qwen_stop_token=_env_str(f"{P}LLM_QWEN_STOP_TOKEN", "<|im_end|>"),
        ),
        escalation=EscalationConfigData(
            max_retries=_env_int(f"{P}ESCALATION_MAX_RETRIES", 2),
            max_escalations=_env_int(f"{P}ESCALATION_MAX_ESCALATIONS", 2),
        ),
        repl=REPLConfigData(
            max_output_len=_env_int(f"{P}REPL_MAX_OUTPUT_LEN", 10000),
            timeout_seconds=_env_int(f"{P}REPL_TIMEOUT_SECONDS", 30),
        ),
        server=ServerConfigData(
            default_url=_env_str(f"{P}SERVER_DEFAULT_URL", "http://localhost:8080"),
            timeout=_env_int(
                f"{P}SERVER_TIMEOUT", int(_registry_timeout("server", "request", 600))
            ),
            num_slots=_env_int(f"{P}SERVER_NUM_SLOTS", 4),
            connect_timeout=_env_int(f"{P}SERVER_CONNECT_TIMEOUT", 5),
            retry_count=_env_int(f"{P}SERVER_RETRY_COUNT", 3),
            retry_backoff=_env_float(f"{P}SERVER_RETRY_BACKOFF", 0.5),
        ),
        monitor=MonitorConfigData(
            entropy_threshold=_env_float(f"{P}MONITOR_ENTROPY_THRESHOLD", 4.0),
            entropy_spike_threshold=_env_float(f"{P}MONITOR_ENTROPY_SPIKE_THRESHOLD", 2.0),
            repetition_threshold=_env_float(f"{P}MONITOR_REPETITION_THRESHOLD", 0.3),
            min_tokens_before_abort=_env_int(f"{P}MONITOR_MIN_TOKENS_BEFORE_ABORT", 50),
            perplexity_window=_env_int(f"{P}MONITOR_PERPLEXITY_WINDOW", 20),
            max_length_multiplier=_env_float(f"{P}MONITOR_MAX_LENGTH_MULTIPLIER", 2.0),
            entropy_sustained_count=_env_int(f"{P}MONITOR_ENTROPY_SUSTAINED_COUNT", 10),
            ngram_size=_env_int(f"{P}MONITOR_NGRAM_SIZE", 3),
            combined_threshold=_env_float(f"{P}MONITOR_COMBINED_THRESHOLD", 0.7),
        ),
        server_urls=ServerURLsConfig(
            frontdoor=_env_str(f"{P}SERVER_URLS_FRONTDOOR", "http://localhost:8080"),
            coder_primary=_env_str(f"{P}SERVER_URLS_CODER_PRIMARY", "http://localhost:8080"),
            coder=_env_str(f"{P}SERVER_URLS_CODER", "http://localhost:8081"),
            coder_escalation=_env_str(f"{P}SERVER_URLS_CODER_ESCALATION", "http://localhost:8081"),
            worker=_env_str(f"{P}SERVER_URLS_WORKER", "http://localhost:8082"),
            worker_general=_env_str(f"{P}SERVER_URLS_WORKER_GENERAL", "http://localhost:8082"),
            worker_explore=_env_str(f"{P}SERVER_URLS_WORKER_EXPLORE", "http://localhost:8082"),
            worker_math=_env_str(f"{P}SERVER_URLS_WORKER_MATH", "http://localhost:8082"),
            worker_vision=_env_str(f"{P}SERVER_URLS_WORKER_VISION", "http://localhost:8086"),
            vision_escalation=_env_str(
                f"{P}SERVER_URLS_VISION_ESCALATION", "http://localhost:8087"
            ),
            worker_code=_env_str(f"{P}SERVER_URLS_WORKER_CODE", "http://localhost:8092"),
            worker_fast=_env_str(f"{P}SERVER_URLS_WORKER_FAST", "http://localhost:8102"),
            worker_summarize=_env_str(f"{P}SERVER_URLS_WORKER_SUMMARIZE", "http://localhost:8081"),
            architect_general=_env_str(
                f"{P}SERVER_URLS_ARCHITECT_GENERAL", "http://localhost:8083"
            ),
            architect_coding=_env_str(f"{P}SERVER_URLS_ARCHITECT_CODING", "http://localhost:8084"),
            ingest_long_context=_env_str(
                f"{P}SERVER_URLS_INGEST_LONG_CONTEXT", "http://localhost:8085"
            ),
            api_url=_env_str(f"{P}SERVER_URLS_API_URL", "http://localhost:8000"),
            ocr_server=_env_str(f"{P}SERVER_URLS_OCR_SERVER", "http://localhost:9001"),
            vision_api=_env_str(
                f"{P}SERVER_URLS_VISION_API", "http://localhost:8000/v1/vision/analyze"
            ),
        ),
        timeouts=TimeoutsConfig(
            # All defaults come from registry; env vars can override
            worker_explore=_env_int(
                f"{P}TIMEOUTS_WORKER_EXPLORE",
                int(_registry_timeout("roles", "worker_explore", 30)),
            ),
            worker_math=_env_int(
                f"{P}TIMEOUTS_WORKER_MATH",
                int(_registry_timeout("roles", "worker_math", 30)),
            ),
            worker_vision=_env_int(
                f"{P}TIMEOUTS_WORKER_VISION",
                int(_registry_timeout("roles", "worker_vision", 30)),
            ),
            worker_summarize=_env_int(
                f"{P}TIMEOUTS_WORKER_SUMMARIZE",
                int(_registry_timeout("roles", "worker_summarize", 120)),
            ),
            frontdoor=_env_int(
                f"{P}TIMEOUTS_FRONTDOOR",
                int(_registry_timeout("roles", "frontdoor", 60)),
            ),
            coder_primary=_env_int(
                f"{P}TIMEOUTS_CODER_PRIMARY",
                int(_registry_timeout("roles", "coder_primary", 60)),
            ),
            coder_escalation=_env_int(
                f"{P}TIMEOUTS_CODER_ESCALATION",
                int(_registry_timeout("roles", "coder_escalation", 120)),
            ),
            vision_escalation=_env_int(
                f"{P}TIMEOUTS_VISION_ESCALATION",
                int(_registry_timeout("roles", "vision_escalation", 60)),
            ),
            ingest_long_context=_env_int(
                f"{P}TIMEOUTS_INGEST_LONG_CONTEXT",
                int(_registry_timeout("roles", "ingest_long_context", 120)),
            ),
            architect_general=_env_int(
                f"{P}TIMEOUTS_ARCHITECT_GENERAL",
                int(_registry_timeout("roles", "architect_general", 600)),
            ),
            architect_coding=_env_int(
                f"{P}TIMEOUTS_ARCHITECT_CODING",
                int(_registry_timeout("roles", "architect_coding", 600)),
            ),
            default_request=_env_int(
                f"{P}TIMEOUTS_DEFAULT_REQUEST",
                int(_registry_timeout("roles", "default_request", 120)),
            ),
            server_request=_env_int(
                f"{P}TIMEOUTS_SERVER_REQUEST",
                int(_registry_timeout("server", "request", 600)),
            ),
            server_connect=_env_int(
                f"{P}TIMEOUTS_SERVER_CONNECT",
                int(_registry_timeout("server", "connect", 5)),
            ),
        ),
        chat=ChatPipelineConfig(
            summarization_threshold_tokens=_env_int(
                f"{P}CHAT_SUMMARIZATION_THRESHOLD_TOKENS", 5000
            ),
            multi_doc_discount=_env_float(f"{P}CHAT_MULTI_DOC_DISCOUNT", 0.7),
            compression_enabled=_env_bool(f"{P}CHAT_COMPRESSION_ENABLED", False),
            compression_min_chars=_env_int(f"{P}CHAT_COMPRESSION_MIN_CHARS", 30000),
            compression_target_ratio=_env_float(f"{P}CHAT_COMPRESSION_TARGET_RATIO", 0.5),
            stage1_context_limit=_env_int(f"{P}CHAT_STAGE1_CONTEXT_LIMIT", 20000),
            long_context_enabled=_env_bool(f"{P}CHAT_LONG_CONTEXT_ENABLED", True),
            long_context_threshold_chars=_env_int(f"{P}CHAT_LONG_CONTEXT_THRESHOLD_CHARS", 20000),
            long_context_max_turns=_env_int(f"{P}CHAT_LONG_CONTEXT_MAX_TURNS", 8),
        ),
        # paths, features, vision, delegation, services, worker_pool
        # use plain defaults (env var override via pydantic-settings only)
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
                qwen_stop_token=settings.llm.qwen_stop_token,
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
            monitor=MonitorConfigData(
                entropy_threshold=settings.monitor.entropy_threshold,
                entropy_spike_threshold=settings.monitor.entropy_spike_threshold,
                repetition_threshold=settings.monitor.repetition_threshold,
                min_tokens_before_abort=settings.monitor.min_tokens_before_abort,
                perplexity_window=settings.monitor.perplexity_window,
                max_length_multiplier=settings.monitor.max_length_multiplier,
                entropy_sustained_count=settings.monitor.entropy_sustained_count,
                ngram_size=settings.monitor.ngram_size,
                combined_threshold=settings.monitor.combined_threshold,
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
            server_urls=ServerURLsConfig(
                frontdoor=settings.server_urls.frontdoor,
                coder_primary=settings.server_urls.coder_primary,
                coder=settings.server_urls.coder,
                coder_escalation=settings.server_urls.coder_escalation,
                worker=settings.server_urls.worker,
                worker_general=settings.server_urls.worker_general,
                worker_explore=settings.server_urls.worker_explore,
                worker_math=settings.server_urls.worker_math,
                worker_vision=settings.server_urls.worker_vision,
                vision_escalation=settings.server_urls.vision_escalation,
                worker_code=settings.server_urls.worker_code,
                worker_fast=settings.server_urls.worker_fast,
                worker_summarize=settings.server_urls.worker_summarize,
                architect_general=settings.server_urls.architect_general,
                architect_coding=settings.server_urls.architect_coding,
                ingest_long_context=settings.server_urls.ingest_long_context,
                api_url=settings.server_urls.api_url,
                ocr_server=settings.server_urls.ocr_server,
                vision_api=settings.server_urls.vision_api,
            ),
            timeouts=TimeoutsConfig(
                worker_explore=settings.timeouts.worker_explore,
                worker_math=settings.timeouts.worker_math,
                worker_vision=settings.timeouts.worker_vision,
                worker_summarize=settings.timeouts.worker_summarize,
                frontdoor=settings.timeouts.frontdoor,
                coder_primary=settings.timeouts.coder_primary,
                coder_escalation=settings.timeouts.coder_escalation,
                vision_escalation=settings.timeouts.vision_escalation,
                ingest_long_context=settings.timeouts.ingest_long_context,
                architect_general=settings.timeouts.architect_general,
                architect_coding=settings.timeouts.architect_coding,
                default_request=settings.timeouts.default_request,
                server_request=settings.timeouts.server_request,
                server_connect=settings.timeouts.server_connect,
                ocr_single_page=settings.timeouts.ocr_single_page,
                ocr_pdf=settings.timeouts.ocr_pdf,
                health_check=settings.timeouts.health_check,
                vision_inference=settings.timeouts.vision_inference,
                vision_figure=settings.timeouts.vision_figure,
                ffmpeg_version=settings.timeouts.ffmpeg_version,
                ffmpeg_probe=settings.timeouts.ffmpeg_probe,
                ffmpeg_extract=settings.timeouts.ffmpeg_extract,
                exiftool=settings.timeouts.exiftool,
                gradio_client=settings.timeouts.gradio_client,
            ),
            chat=ChatPipelineConfig(
                summarization_threshold_tokens=settings.chat.summarization_threshold_tokens,
                multi_doc_discount=settings.chat.multi_doc_discount,
                compression_enabled=settings.chat.compression_enabled,
                compression_min_chars=settings.chat.compression_min_chars,
                compression_target_ratio=settings.chat.compression_target_ratio,
                stage1_context_limit=settings.chat.stage1_context_limit,
                long_context_enabled=settings.chat.long_context_enabled,
                long_context_threshold_chars=settings.chat.long_context_threshold_chars,
                long_context_max_turns=settings.chat.long_context_max_turns,
                repetition_unique_ratio=settings.chat.repetition_unique_ratio,
                garbled_short_line_ratio=settings.chat.garbled_short_line_ratio,
                min_answer_length=settings.chat.min_answer_length,
                review_low_q_threshold=settings.chat.review_low_q_threshold,
                review_skip_q_threshold=settings.chat.review_skip_q_threshold,
                plan_review_phase_a_min=settings.chat.plan_review_phase_a_min,
                plan_review_phase_b_mean_q=settings.chat.plan_review_phase_b_mean_q,
                plan_review_phase_b_min_q=settings.chat.plan_review_phase_b_min_q,
                plan_review_phase_c_min_q=settings.chat.plan_review_phase_c_min_q,
                plan_review_phase_c_min_total=settings.chat.plan_review_phase_c_min_total,
                plan_review_phase_c_skip_rate=settings.chat.plan_review_phase_c_skip_rate,
            ),
            # These sections use plain defaults (path serialization is complex)
            paths=PathsConfig(),
            vision=VisionConfig(),
            delegation=DelegationConfig(),
            services=ServicesConfig(),
            worker_pool=WorkerPoolPathsConfig(),
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
