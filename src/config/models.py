"""Dataclass model definitions for orchestrator configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .validation import _registry_runtime_value, _registry_timeout

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

    depth_role_overrides: str = field(
        default_factory=lambda: str(
            _registry_runtime_value(("llm", "depth_role_overrides"), "1:worker_general")
        )
    )
    """Optional depth->role override map (CSV or JSON), e.g. "1:worker_general,2:worker_math"."""

    depth_override_max_depth: int = field(
        default_factory=lambda: int(_registry_runtime_value(("llm", "depth_override_max_depth"), 3))
    )
    """Maximum nested depth eligible for override routing."""


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
    worker_coder: str = "http://localhost:8102"
    worker_code: str = "http://localhost:8102"
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
        default_factory=lambda: int(_registry_timeout("roles", "worker_explore", 60))
    )
    worker_math: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_math", 60))
    )
    worker_vision: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_vision", 60))
    )
    worker_summarize: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_summarize", 120))
    )
    worker_general: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_general", 60))
    )
    worker_coder: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_fast", 30))
    )
    worker_code: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_coder", 30))
    )
    worker_fast: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_fast", 30))
    )
    frontdoor: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "frontdoor", 90))
    )
    coder_escalation: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "coder_escalation", 120))
    )
    vision_escalation: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "vision_escalation", 120))
    )
    ingest_long_context: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "ingest_long_context", 300))
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
            "worker_general": self.worker_general,
            "worker_coder": self.worker_coder,
            "worker_code": self.worker_code,
            "worker_fast": self.worker_fast,
            "frontdoor": self.frontdoor,
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
            "worker_general": self.worker_general,
            "worker_coder": self.worker_coder,
            "worker_code": self.worker_code,
            "worker_fast": self.worker_fast,
            "frontdoor": self.frontdoor,
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

    # Session compaction tuning (C1 virtual memory pattern)
    session_compaction_keep_recent_ratio: float = 0.20
    """Fraction of context to keep verbatim after compaction (default 20%, min 3000 chars)."""
    session_compaction_recompaction_interval: int = 0
    """Re-trigger compaction every N turns after first compaction. 0 = disabled."""
    session_compaction_min_turns: int = 5
    """Minimum turns before compaction can run (default 5 to avoid very-early churn)."""

    # Try-cheap-first: speculative pre-filter using 7B worker before specialist.
    # Phase A = try all, Phase B = MemRL-guided, Phase C = fully learned.
    try_cheap_first_enabled: bool = True
    try_cheap_first_phase: str = "A"
    """A = try all non-forced, B = Q-value guided, C = fully learned."""
    try_cheap_first_role: str = "worker_explore"
    """Role used for cheap attempts (fastest HOT model)."""
    try_cheap_first_max_tokens: int = 1024
    """Token budget for cheap attempt (keep short to minimize waste)."""
    try_cheap_first_quality_threshold: float = 0.6
    """Minimum quality score to accept cheap answer."""
    try_cheap_first_q_threshold: float = 0.65
    """Min Q-value for cheap-first in Phase B/C (skip cheap if below)."""


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
class MemRLRetrievalConfigData:
    """Configuration for MemRL retrieval/risk/prior tuning knobs."""

    semantic_k: int = 20
    min_similarity: float = 0.3
    min_q_value: float = 0.3
    q_weight: float = 0.7
    cost_lambda: float = 0.15
    top_n: int = 5
    confidence_threshold: float = 0.6
    confidence_estimator: str = "median"
    confidence_trim_ratio: float = 0.2
    confidence_min_neighbors: int = 3
    calibrated_confidence_threshold: float | None = None
    conformal_margin: float = 0.0
    risk_control_enabled: bool = False
    risk_budget_id: str = "default"
    risk_gate_min_samples: int = 3
    risk_abstain_target_role: str = "architect_general"
    risk_gate_rollout_ratio: float = 1.0
    risk_gate_kill_switch: bool = False
    risk_budget_guardrail_min_events: int = 50
    risk_budget_guardrail_max_abstain_rate: float = 0.60
    prior_strength: float = 0.15
    warm_probability_hit: float = 0.8
    warm_probability_miss: float = 0.2
    warm_cost_fallback_s: float = 1.0
    cold_cost_fallback_s: float = 3.0


@dataclass
class ThinkHarderConfigData:
    """Configuration for think-harder regulation knobs."""

    min_expected_roi: float = 0.02
    min_samples: int = 5
    cooldown_turns: int = 2
    ema_alpha: float = 0.25
    min_marginal_utility: float = 0.0
    token_budget_min: int = 2048
    token_budget_max: int = 4096
    token_budget_fallback: int = 4096
    temperature_min: float = 0.30
    temperature_max: float = 0.50
    cot_roi_threshold: float = 0.35
    token_penalty_per_4k: float = 0.15
    ema_alpha_min: float = 0.05
    ema_alpha_max: float = 1.0


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
    max_archive_size: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_SIZE", str(500 * 1024 * 1024)))
    )
    max_extracted_size: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_EXTRACTED_SIZE", str(1024 * 1024 * 1024)))
    )
    max_archive_files: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_FILES", "1000"))
    )
    max_archive_single_file: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_SINGLE_FILE", str(100 * 1024 * 1024)))
    )
    max_archive_compression_ratio: float = field(
        default_factory=lambda: float(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_COMPRESSION_RATIO", "100.0"))
    )
    max_archive_recursion_depth: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_RECURSION_DEPTH", "2"))
    )

    pdf_min_entropy: float = field(
        default_factory=lambda: float(os.environ.get("ORCHESTRATOR_SERVICES_PDF_MIN_ENTROPY", "3.5"))
    )
    pdf_max_garbage_ratio: float = field(
        default_factory=lambda: float(os.environ.get("ORCHESTRATOR_SERVICES_PDF_MAX_GARBAGE_RATIO", "0.15"))
    )
    pdf_min_word_length_avg: float = field(
        default_factory=lambda: float(os.environ.get("ORCHESTRATOR_SERVICES_PDF_MIN_WORD_LENGTH_AVG", "2.5"))
    )
    pdf_min_text_length: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_PDF_MIN_TEXT_LENGTH", "100"))
    )
    pdftotext_timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_PDFTOTEXT_TIMEOUT_SECONDS", "30"))
    )

    pdf_router_temp_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_PDF_ROUTER_TEMP", f"{_get_default_llm_root()}/tmp/pdf_router"
            )
        )
    )

    llm_cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LLM_CACHE",
                f"{_get_default_project_root()}/cache/llm_responses",
            )
        )
    )
    """Content-addressable LLM response cache directory."""

    llm_cache_ttl_hours: float = 168.0
    """Cache TTL in hours (default 1 week)."""

    llm_cache_max_entries: int = 10_000
    """Maximum cache entries before LRU eviction."""


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

    rate_limit_rpm: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_API_RATE_LIMIT_RPM", "60"))
    )
    """Requests per minute per client IP."""

    rate_limit_burst: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_API_RATE_LIMIT_BURST", "10"))
    )
    """Maximum burst size above the sustained rate."""

    rate_limit_cleanup_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("ORCHESTRATOR_API_RATE_LIMIT_CLEANUP_INTERVAL_SECONDS", "300.0")
        )
    )
    """Cleanup cadence for stale rate-limit buckets."""

    rate_limit_stale_bucket_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("ORCHESTRATOR_API_RATE_LIMIT_STALE_BUCKET_TTL_SECONDS", "600.0")
        )
    )
    """Idle TTL before a client bucket is pruned."""


@dataclass
class SessionPersistenceConfigData:
    """Configuration for session checkpoint/summary cadence."""

    checkpoint_turn_interval: int = 5
    checkpoint_idle_minutes: int = 30
    summary_idle_hours: int = 2
    checkpoint_globals_warn_mb: int = 50
    checkpoint_globals_hard_mb: int = 100


@dataclass
class SessionLifecycleConfigData:
    """Configuration for session status transitions."""

    active_to_idle_hours: float = 1.0
    idle_to_stale_days: float = 7.0


@dataclass
class HealthTrackerConfigData:
    """Configuration for backend health tracker circuit breaker."""

    default_failure_threshold: int = 3
    default_cooldown_s: float = 30.0
    max_cooldown_s: float = 300.0


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
    memrl_retrieval: MemRLRetrievalConfigData = field(default_factory=MemRLRetrievalConfigData)
    think_harder: ThinkHarderConfigData = field(default_factory=ThinkHarderConfigData)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    worker_pool: WorkerPoolPathsConfig = field(default_factory=WorkerPoolPathsConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    session_persistence: SessionPersistenceConfigData = field(
        default_factory=SessionPersistenceConfigData
    )
    session_lifecycle: SessionLifecycleConfigData = field(default_factory=SessionLifecycleConfigData)
    health_tracker: HealthTrackerConfigData = field(default_factory=HealthTrackerConfigData)
    external_backends: ExternalBackendsConfig = field(default_factory=ExternalBackendsConfig)

