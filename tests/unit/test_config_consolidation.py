"""Tests for Phase 3 configuration consolidation.

Verifies that all new config sections have correct defaults matching
the hardcoded values they replace across the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Tests that assert /mnt/raid0/ paths only apply on the production machine
_on_raid = pytest.mark.skipif(
    not Path("/mnt/raid0").exists(),
    reason="Path assertions require /mnt/raid0 (production machine only)",
)

from src.config import (
    ChatPipelineConfig,
    DelegationConfig,
    LLMConfig,
    MonitorConfigData,
    OrchestratorConfigData,
    PathsConfig,
    ServerURLsConfig,
    ServicesConfig,
    TimeoutsConfig,
    VisionConfig,
    WorkerPoolPathsConfig,
    get_config,
    reset_config,
)


@pytest.fixture(autouse=True)
def _clean_config():
    """Reset config cache before and after each test."""
    reset_config()
    yield
    reset_config()


# ── Module import ─────────────────────────────────────────────────────────


class TestConfigImports:
    """Verify all new sections are importable."""

    def test_config_module_imports(self):
        import src.config

        assert hasattr(src.config, "get_config")
        assert hasattr(src.config, "reset_config")

    def test_all_sections_importable(self):
        """Every new config section must be importable."""
        from src.config import (
            ServerURLsConfig,
            TimeoutsConfig,
            VisionConfig,
            ChatPipelineConfig,
            DelegationConfig,
            ServicesConfig,
            WorkerPoolPathsConfig,
        )

        for cls in [
            ServerURLsConfig,
            TimeoutsConfig,
            VisionConfig,
            ChatPipelineConfig,
            DelegationConfig,
            ServicesConfig,
            WorkerPoolPathsConfig,
        ]:
            assert cls is not None

    def test_config_is_leaf_module(self):
        """config.py must NOT import from src/ (prevents circular imports).

        Exception: Lazy imports inside try/except blocks are allowed for registry loading.
        """
        import src.config
        import inspect

        source = inspect.getsource(src.config)
        # Allow 'from src.' only inside docstrings/comments — not as actual imports
        # Look for actual import statements
        import ast

        # Allowed lazy imports (inside try/except, used for registry loading)
        allowed_imports = {"src.registry_loader"}

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("src.") and node.module not in allowed_imports:
                        assert False, (
                            f"config.py must be a leaf module: found 'from {node.module}'"
                        )


# ── get_config() and reset_config() ──────────────────────────────────────


class TestGetConfig:
    """Test config singleton and reset behavior."""

    def test_get_config_returns_dataclass(self):
        cfg = get_config()
        assert isinstance(cfg, OrchestratorConfigData)

    def test_get_config_is_cached(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_cache(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_get_config_all_sections_present(self):
        cfg = get_config()
        assert isinstance(cfg.server_urls, ServerURLsConfig)
        assert isinstance(cfg.timeouts, TimeoutsConfig)
        assert isinstance(cfg.vision, VisionConfig)
        assert isinstance(cfg.chat, ChatPipelineConfig)
        assert isinstance(cfg.delegation, DelegationConfig)
        assert isinstance(cfg.services, ServicesConfig)
        assert isinstance(cfg.worker_pool, WorkerPoolPathsConfig)
        assert isinstance(cfg.monitor, MonitorConfigData)
        assert isinstance(cfg.paths, PathsConfig)
        assert isinstance(cfg.llm, LLMConfig)


# ── ServerURLsConfig as single source of truth ─────────────────────────


class TestServerURLsDefaults:
    """Verify ServerURLsConfig is the single source of truth for server URLs."""

    def test_as_dict_contains_all_role_urls(self):
        """as_dict() must contain all expected orchestrator role URLs."""
        cfg = ServerURLsConfig()
        config_dict = cfg.as_dict()
        # All roles that LLMPrimitives consumers depend on
        expected_roles = {
            "frontdoor",
            "coder_primary",
            "coder",
            "coder_escalation",
            "worker",
            "worker_general",
            "worker_explore",
            "worker_math",
            "worker_vision",
            "vision_escalation",
            "worker_code",
            "worker_fast",
            "architect_general",
            "architect_coding",
            "ingest_long_context",
        }
        for role in expected_roles:
            assert role in config_dict, f"Missing role: {role}"
            assert config_dict[role].startswith("http://"), (
                f"URL for {role} must be HTTP: got {config_dict[role]}"
            )

    def test_as_dict_excludes_services(self):
        """as_dict() must NOT contain api_url, ocr_server, vision_api."""
        cfg = ServerURLsConfig()
        d = cfg.as_dict()
        assert "api_url" not in d
        assert "ocr_server" not in d
        assert "vision_api" not in d

    def test_service_urls_present(self):
        """Service URLs must be accessible on the config object."""
        cfg = ServerURLsConfig()
        assert cfg.api_url == "http://localhost:8000"
        assert cfg.ocr_server == "http://localhost:9001"
        assert cfg.vision_api == "http://localhost:8000/v1/vision/analyze"

    def test_specific_role_urls(self):
        """Spot-check specific role->URL mappings."""
        cfg = ServerURLsConfig()
        assert cfg.frontdoor == "http://localhost:8080"
        assert cfg.coder_escalation == "http://localhost:8081"
        assert cfg.worker_explore == "http://localhost:8082"
        assert cfg.worker_vision == "http://localhost:8086"
        assert cfg.vision_escalation == "http://localhost:8087"
        assert cfg.architect_general == "http://localhost:8083"
        assert cfg.architect_coding == "http://localhost:8084"
        assert cfg.ingest_long_context == "http://localhost:8085"
        assert cfg.worker_fast == "http://localhost:8102"
        assert cfg.worker_summarize == "http://localhost:8081"


# ── TimeoutsConfig defaults match ROLE_TIMEOUTS in chat_utils.py ─────────


class TestTimeoutsDefaults:
    """Verify TimeoutsConfig defaults match chat_utils.py ROLE_TIMEOUTS."""

    def test_for_role_matches_role_timeouts(self):
        """for_role() must return same values as ROLE_TIMEOUTS dict."""
        from src.api.routes.chat_utils import ROLE_TIMEOUTS, DEFAULT_TIMEOUT_S

        cfg = TimeoutsConfig()
        for role, expected_timeout in ROLE_TIMEOUTS.items():
            assert cfg.for_role(role) == expected_timeout, (
                f"Timeout mismatch for role {role}: "
                f"config={cfg.for_role(role)}, expected={expected_timeout}"
            )
        # Unknown role should return default
        assert cfg.for_role("nonexistent") == DEFAULT_TIMEOUT_S

    def test_role_timeouts_dict_matches(self):
        """role_timeouts_dict() must match ROLE_TIMEOUTS exactly."""
        from src.api.routes.chat_utils import ROLE_TIMEOUTS

        cfg = TimeoutsConfig()
        config_dict = cfg.role_timeouts_dict()
        for role, timeout in ROLE_TIMEOUTS.items():
            assert config_dict[role] == timeout

    def test_default_request_matches_default_timeout_s(self):
        from src.api.routes.chat_utils import DEFAULT_TIMEOUT_S

        cfg = TimeoutsConfig()
        assert cfg.default_request == DEFAULT_TIMEOUT_S

    def test_specific_timeout_values(self):
        """Spot-check critical timeout values."""
        cfg = TimeoutsConfig()
        # All roles: uniform 600s for seeding fairness
        assert cfg.worker_explore == 600
        assert cfg.worker_math == 600
        assert cfg.worker_vision == 600
        assert cfg.worker_summarize == 600
        assert cfg.frontdoor == 600
        assert cfg.coder_primary == 600
        assert cfg.coder_escalation == 600
        assert cfg.architect_general == 600
        assert cfg.architect_coding == 600
        # Backend: unified 600s timeout
        assert cfg.server_request == 600
        assert cfg.server_connect == 5


# ── MonitorConfigData defaults match generation_monitor.py ───────────────


class TestMonitorDefaults:
    """Verify MonitorConfigData defaults match generation_monitor.py MonitorConfig."""

    def test_entropy_threshold(self):
        cfg = MonitorConfigData()
        assert cfg.entropy_threshold == 4.0

    def test_entropy_spike_threshold(self):
        cfg = MonitorConfigData()
        assert cfg.entropy_spike_threshold == 2.0

    def test_min_tokens_before_abort(self):
        cfg = MonitorConfigData()
        assert cfg.min_tokens_before_abort == 50

    def test_perplexity_window(self):
        cfg = MonitorConfigData()
        assert cfg.perplexity_window == 20

    def test_repetition_threshold(self):
        cfg = MonitorConfigData()
        assert cfg.repetition_threshold == 0.3

    def test_ngram_size(self):
        cfg = MonitorConfigData()
        assert cfg.ngram_size == 3

    def test_combined_threshold(self):
        cfg = MonitorConfigData()
        assert cfg.combined_threshold == 0.7


# ── LLMConfig defaults match LLMPrimitivesConfig ─────────────────────────


class TestLLMConfigDefaults:
    """Verify LLMConfig defaults match llm_primitives.py LLMPrimitivesConfig."""

    def test_matches_llm_primitives_config(self):
        from src.llm_primitives import LLMPrimitivesConfig

        cfg = LLMConfig()
        ref = LLMPrimitivesConfig()

        assert cfg.output_cap == ref.output_cap
        assert cfg.batch_parallelism == ref.batch_parallelism
        assert cfg.call_timeout == ref.call_timeout
        assert cfg.mock_response_prefix == ref.mock_response_prefix
        assert cfg.max_recursion_depth == ref.max_recursion_depth
        assert cfg.default_prompt_rate == ref.default_prompt_rate
        assert cfg.default_completion_rate == ref.default_completion_rate

    def test_qwen_stop_token(self):
        from src.api.routes.chat_utils import QWEN_STOP

        cfg = LLMConfig()
        assert cfg.qwen_stop_token == QWEN_STOP


# ── ChatPipelineConfig defaults match chat_utils.py constants ────────────


class TestChatPipelineDefaults:
    """Verify ChatPipelineConfig defaults match THREE_STAGE_CONFIG etc."""

    def test_three_stage_config_values(self):
        from src.api.routes.chat_utils import THREE_STAGE_CONFIG

        cfg = ChatPipelineConfig()
        assert cfg.summarization_threshold_tokens == THREE_STAGE_CONFIG["threshold_tokens"]
        assert cfg.multi_doc_discount == THREE_STAGE_CONFIG["multi_doc_discount"]
        assert cfg.compression_enabled == THREE_STAGE_CONFIG["compression"]["enabled"]
        assert cfg.compression_min_chars == THREE_STAGE_CONFIG["compression"]["min_chars"]
        assert cfg.compression_target_ratio == THREE_STAGE_CONFIG["compression"]["target_ratio"]
        assert cfg.stage1_context_limit == THREE_STAGE_CONFIG["compression"]["stage1_context_limit"]

    def test_long_context_config_values(self):
        from src.api.routes.chat_utils import LONG_CONTEXT_CONFIG

        cfg = ChatPipelineConfig()
        assert cfg.long_context_enabled == LONG_CONTEXT_CONFIG["enabled"]
        assert cfg.long_context_threshold_chars == LONG_CONTEXT_CONFIG["threshold_chars"]
        assert cfg.long_context_max_turns == LONG_CONTEXT_CONFIG["max_turns"]

    def test_quality_detection_defaults(self):
        cfg = ChatPipelineConfig()
        assert cfg.repetition_unique_ratio == 0.5
        assert cfg.garbled_short_line_ratio == 0.6
        assert cfg.min_answer_length == 50

    def test_review_q_thresholds(self):
        cfg = ChatPipelineConfig()
        assert cfg.review_low_q_threshold == 0.6
        assert cfg.review_skip_q_threshold == 0.6

    def test_plan_review_phase_defaults(self):
        cfg = ChatPipelineConfig()
        assert cfg.plan_review_phase_a_min == 50
        assert cfg.plan_review_phase_b_mean_q == 0.7
        assert cfg.plan_review_phase_b_min_q == 0.5
        assert cfg.plan_review_phase_c_min_q == 0.7
        assert cfg.plan_review_phase_c_min_total == 100
        assert cfg.plan_review_phase_c_skip_rate == 0.90


# ── VisionConfig defaults match src/vision/config.py ─────────────────────


class TestVisionDefaults:
    """Verify VisionConfig defaults match vision/config.py constants."""

    def test_processing_limits(self):
        cfg = VisionConfig()
        assert cfg.max_image_size_mb == 20
        assert cfg.max_image_dimension == 4096
        assert cfg.default_batch_size == 100
        assert cfg.max_concurrent_workers == 4
        assert cfg.default_video_fps == 1.0
        assert cfg.default_vl_max_tokens == 512
        assert cfg.default_vl_threads == 8

    def test_thumbnail_settings(self):
        cfg = VisionConfig()
        assert cfg.thumb_size == (256, 256)
        assert cfg.thumb_quality == 85

    def test_face_detection(self):
        cfg = VisionConfig()
        assert cfg.face_min_confidence == 0.9
        assert cfg.face_embedding_dim == 512
        assert cfg.face_identification_threshold == 0.6

    def test_model_names(self):
        cfg = VisionConfig()
        assert cfg.arcface_model_name == "buffalo_l"
        assert cfg.clip_model_name == "ViT-B/32"
        assert cfg.sentence_transformer_model == "all-MiniLM-L6-v2"

    @_on_raid
    def test_paths_on_raid(self):
        cfg = VisionConfig()
        assert str(cfg.base_dir).startswith("/mnt/raid0/")
        assert str(cfg.llama_mtmd_cli).startswith("/mnt/raid0/")
        assert str(cfg.vl_model_path).startswith("/mnt/raid0/")

    def test_server_ports(self):
        cfg = VisionConfig()
        assert cfg.vl_server_port == 8086
        assert cfg.vl_escalation_server_port == 8087


# ── DelegationConfig defaults ────────────────────────────────────────────


class TestDelegationDefaults:
    def test_iteration_limits(self):
        cfg = DelegationConfig()
        assert cfg.max_iterations == 3
        assert cfg.max_total_iterations == 10
        assert cfg.max_concurrent_analysis == 4

    def test_token_limits(self):
        cfg = DelegationConfig()
        assert cfg.max_review_tokens == 128
        assert cfg.max_taskir_tokens == 256
        assert cfg.max_plan_review_tokens == 128


# ── ServicesConfig defaults ──────────────────────────────────────────────


class TestServicesDefaults:
    @_on_raid
    def test_ocr_model_paths_on_raid(self):
        cfg = ServicesConfig()
        assert str(cfg.lightonocr_model).startswith("/mnt/raid0/")
        assert str(cfg.lightonocr_mmproj).startswith("/mnt/raid0/")

    def test_ocr_max_tokens(self):
        cfg = ServicesConfig()
        assert cfg.lightonocr_max_tokens == 2048

    @_on_raid
    def test_draft_cache(self):
        cfg = ServicesConfig()
        assert str(cfg.draft_cache_dir).startswith("/mnt/raid0/")
        assert cfg.draft_cache_ttl_hours == 24.0

    def test_archive_limits(self):
        cfg = ServicesConfig()
        assert cfg.max_archive_size == 500 * 1024 * 1024
        assert cfg.max_extracted_size == 1024 * 1024 * 1024
        assert cfg.max_archive_files == 1000


# ── PathsConfig defaults ─────────────────────────────────────────────────


class TestPathsDefaults:
    @_on_raid
    def test_all_paths_on_raid(self):
        cfg = PathsConfig()
        for field_name in [
            "models_dir",
            "cache_dir",
            "tmp_dir",
            "registry_path",
            "tool_registry_path",
            "script_registry_dir",
            "project_root",
            "sessions_dir",
            "artifacts_dir",
            "llama_cpp_bin",
            "model_base",
            "log_dir",
        ]:
            path = getattr(cfg, field_name)
            assert str(path).startswith("/mnt/raid0/"), (
                f"PathsConfig.{field_name} = {path} is NOT on /mnt/raid0/"
            )

    @_on_raid
    def test_raid_prefix(self):
        cfg = PathsConfig()
        assert cfg.raid_prefix == "/mnt/raid0/"

    @_on_raid
    def test_specific_paths(self):
        cfg = PathsConfig()
        assert cfg.project_root == Path("/mnt/raid0/llm/claude")
        assert cfg.models_dir == Path("/mnt/raid0/llm/models")
        assert cfg.sessions_dir == Path("/mnt/raid0/llm/claude/orchestration/repl_memory/sessions")


# ── WorkerPoolPathsConfig defaults ───────────────────────────────────────


class TestWorkerPoolPathsDefaults:
    @_on_raid
    def test_paths_on_raid(self):
        cfg = WorkerPoolPathsConfig()
        assert str(cfg.llama_server_path).startswith("/mnt/raid0/")
        assert str(cfg.log_dir).startswith("/mnt/raid0/")
        assert str(cfg.model_base).startswith("/mnt/raid0/")


# ── Environment variable overrides ───────────────────────────────────────


class TestEnvVarOverrides:
    """Test that environment variables properly override config defaults."""

    def test_mock_mode_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_MOCK_MODE": "0"}):
            reset_config()
            cfg = get_config()
            assert cfg.mock_mode is False

    def test_llm_output_cap_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_LLM_OUTPUT_CAP": "4096"}):
            reset_config()
            cfg = get_config()
            assert cfg.llm.output_cap == 4096

    def test_llm_call_timeout_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_LLM_CALL_TIMEOUT": "600"}):
            reset_config()
            cfg = get_config()
            assert cfg.llm.call_timeout == 600

    def test_timeout_role_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_TIMEOUTS_ARCHITECT_GENERAL": "600"}):
            reset_config()
            cfg = get_config()
            assert cfg.timeouts.architect_general == 600
            assert cfg.timeouts.for_role("architect_general") == 600

    def test_server_url_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_SERVER_URLS_FRONTDOOR": "http://custom:9999"}):
            reset_config()
            cfg = get_config()
            assert cfg.server_urls.frontdoor == "http://custom:9999"

    def test_monitor_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_MONITOR_ENTROPY_THRESHOLD": "5.0"}):
            reset_config()
            cfg = get_config()
            assert cfg.monitor.entropy_threshold == 5.0

    def test_chat_pipeline_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_CHAT_LONG_CONTEXT_MAX_TURNS": "16"}):
            reset_config()
            cfg = get_config()
            assert cfg.chat.long_context_max_turns == 16

    def test_escalation_max_retries_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_ESCALATION_MAX_RETRIES": "5"}):
            reset_config()
            cfg = get_config()
            assert cfg.escalation.max_retries == 5


# ── End-to-end wiring verification ────────────────────────────────────────
# These tests verify that consumer modules ACTUALLY read from get_config().


class TestWiringChatUtils:
    """Verify chat_utils.py reads from config at module level."""

    def test_role_timeouts_from_config(self):
        """ROLE_TIMEOUTS dict must match config timeouts."""
        from src.api.routes.chat_utils import ROLE_TIMEOUTS

        cfg = get_config()
        for role, timeout in ROLE_TIMEOUTS.items():
            assert cfg.timeouts.for_role(role) == timeout

    def test_default_timeout_s_from_config(self):
        from src.api.routes.chat_utils import DEFAULT_TIMEOUT_S

        assert DEFAULT_TIMEOUT_S == get_config().timeouts.default_request

    def test_three_stage_config_sourced(self):
        from src.api.routes.chat_utils import THREE_STAGE_CONFIG

        cfg = get_config().chat
        assert THREE_STAGE_CONFIG["threshold_tokens"] == cfg.summarization_threshold_tokens

    def test_qwen_stop_from_config(self):
        from src.api.routes.chat_utils import QWEN_STOP

        assert QWEN_STOP == get_config().llm.qwen_stop_token


class TestWiringChatReview:
    """Verify chat_review.py reads from config."""

    def test_compute_plan_review_phase_uses_config(self):
        """_compute_plan_review_phase must use ChatPipelineConfig thresholds."""
        from src.api.routes.chat_review import _compute_plan_review_phase

        # Phase A: less than 50 reviews
        assert _compute_plan_review_phase({"total_reviews": 10}) == "A"
        # Phase A: exactly at threshold
        assert _compute_plan_review_phase({"total_reviews": 49}) == "A"
        # Phase A: >= 50 but no Q values
        assert _compute_plan_review_phase({"total_reviews": 50}) == "A"

    def test_detect_output_quality_uses_config_thresholds(self):
        """_detect_output_quality_issue uses repetition/garbled thresholds from config."""
        from src.api.routes.chat_review import _detect_output_quality_issue

        # A string with high repetition should be detected
        repeated = " ".join(["the same thing"] * 50)
        result = _detect_output_quality_issue(repeated)
        assert result is not None and "high_repetition" in result


class TestWiringChatVision:
    """Verify chat_vision.py reads URLs from config."""

    def test_vision_module_imports_config(self):
        """chat_vision.py must import get_config."""
        import src.api.routes.chat_vision as cv
        import inspect

        source = inspect.getsource(cv)
        assert "get_config" in source

    def test_execute_vision_tool_uses_config_ocr_url(self):
        """_execute_vision_tool references config for OCR URL."""
        import src.api.routes.chat_vision as cv
        import inspect

        source = inspect.getsource(cv._execute_vision_tool)
        assert "get_config" in source


class TestWiringRegistryLoader:
    """Verify registry_loader.py sources defaults from config."""

    @_on_raid
    def test_default_model_base_from_config(self):
        """RegistryLoader._model_base_path should match config.paths.model_base."""
        cfg = get_config()
        # The default should match — we can't instantiate RegistryLoader without
        # a valid YAML file, so we check the config value directly
        assert cfg.paths.model_base == Path("/mnt/raid0/llm/lmstudio/models")

    @_on_raid
    def test_default_registry_path_from_config(self):
        cfg = get_config()
        assert cfg.paths.registry_path == Path(
            "/mnt/raid0/llm/claude/orchestration/model_registry.yaml"
        )


class TestWiringBuiltinTools:
    """Verify builtin_tools.py uses config for security prefix."""

    @_on_raid
    def test_raid_prefix_in_config(self):
        cfg = get_config()
        assert cfg.paths.raid_prefix == "/mnt/raid0/"

    def test_write_json_security_check_uses_config(self):
        """write_json tool should reference config for raid_prefix."""
        import src.builtin_tools as bt
        import inspect

        source = inspect.getsource(bt)
        assert "get_config" in source


class TestWiringBackends:
    """Verify backend modules source defaults from config."""

    def test_server_config_defaults_from_config(self):
        """ServerConfig fields should match config.server defaults."""
        from src.backends.llama_server import ServerConfig

        sc = ServerConfig()
        cfg = get_config().server
        assert sc.base_url == cfg.default_url
        assert sc.timeout == cfg.timeout
        assert sc.num_slots == cfg.num_slots

    def test_monitor_config_defaults_from_config(self):
        """MonitorConfig fields should match config.monitor defaults."""
        from src.generation_monitor import MonitorConfig

        mc = MonitorConfig()
        cfg = get_config().monitor
        assert mc.entropy_threshold == cfg.entropy_threshold
        assert mc.repetition_threshold == cfg.repetition_threshold
        assert mc.ngram_size == cfg.ngram_size

    def test_escalation_config_from_config(self):
        """EscalationConfig should match config.escalation."""
        from src.escalation import EscalationConfig

        ec = EscalationConfig()
        cfg = get_config().escalation
        assert ec.max_retries == cfg.max_retries
        assert ec.max_escalations == cfg.max_escalations


class TestWiringServices:
    """Verify service modules source defaults from config."""

    def test_document_client_urls_from_config(self):
        """document_client.py DEFAULT_OCR_URL must match config."""
        from src.services.document_client import DEFAULT_OCR_URL

        assert DEFAULT_OCR_URL == get_config().server_urls.ocr_server

    def test_vision_config_values_from_config(self):
        """vision/config.py constants must match VisionConfig."""
        from src.vision.config import MAX_IMAGE_SIZE_MB, MAX_IMAGE_DIMENSION

        cfg = get_config().vision
        assert MAX_IMAGE_SIZE_MB == cfg.max_image_size_mb
        assert MAX_IMAGE_DIMENSION == cfg.max_image_dimension
