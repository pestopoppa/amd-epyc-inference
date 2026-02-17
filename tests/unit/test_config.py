"""Characterization tests for src.config module.

Tests the public API: env var helpers, config dataclass defaults,
ServerURLsConfig.as_dict(), MonitorConfigData tier_overrides,
TimeoutsConfig defaults, and get_config() singleton behavior.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.config import (
    EscalationConfigData,
    LLMConfig,
    MonitorConfigData,
    PathsConfig,
    REPLConfigData,
    ServerConfigData,
    ServerURLsConfig,
    TimeoutsConfig,
    _env_bool,
    _env_float,
    _env_int,
    get_config,
)


# ============================================================================
# _env_bool
# ============================================================================


class TestEnvBool:
    """Tests for _env_bool() helper."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "Yes", "ON"])
    def test_truthy_values(self, value: str) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": value}):
            assert _env_bool("TEST_BOOL") is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "No", "OFF"])
    def test_falsy_values(self, value: str) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": value}):
            assert _env_bool("TEST_BOOL") is False

    def test_missing_returns_default_false(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_BOOL_MISSING", None)
            assert _env_bool("TEST_BOOL_MISSING") is False

    def test_missing_returns_default_true(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_BOOL_MISSING2", None)
            assert _env_bool("TEST_BOOL_MISSING2", default=True) is True

    def test_unrecognized_value_returns_default(self) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": "maybe"}):
            assert _env_bool("TEST_BOOL", default=True) is True
        with patch.dict(os.environ, {"TEST_BOOL": "maybe"}):
            assert _env_bool("TEST_BOOL", default=False) is False


# ============================================================================
# _env_int
# ============================================================================


class TestEnvInt:
    """Tests for _env_int() helper."""

    def test_valid_int(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert _env_int("TEST_INT", 0) == 42

    def test_invalid_int_returns_default(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            assert _env_int("TEST_INT", 99) == 99

    def test_missing_returns_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_INT_MISSING", None)
            assert _env_int("TEST_INT_MISSING", 7) == 7

    def test_negative_returns_default(self) -> None:
        # isdigit() returns False for negative numbers
        with patch.dict(os.environ, {"TEST_INT": "-5"}):
            assert _env_int("TEST_INT", 10) == 10

    def test_zero(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "0"}):
            assert _env_int("TEST_INT", 99) == 0


# ============================================================================
# _env_float
# ============================================================================


class TestEnvFloat:
    """Tests for _env_float() helper."""

    def test_valid_float(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            assert _env_float("TEST_FLOAT", 0.0) == pytest.approx(3.14)

    def test_valid_int_as_float(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "42"}):
            assert _env_float("TEST_FLOAT", 0.0) == pytest.approx(42.0)

    def test_invalid_float_returns_default(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "abc"}):
            assert _env_float("TEST_FLOAT", 1.5) == pytest.approx(1.5)

    def test_missing_returns_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_FLOAT_MISSING", None)
            assert _env_float("TEST_FLOAT_MISSING", 2.5) == pytest.approx(2.5)

    def test_empty_string_returns_default(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": ""}):
            assert _env_float("TEST_FLOAT", 9.9) == pytest.approx(9.9)


# ============================================================================
# LLMConfig defaults
# ============================================================================


class TestLLMConfig:
    """Tests for LLMConfig dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.output_cap == 8192
        assert cfg.batch_parallelism == 4
        assert cfg.call_timeout == 600
        assert cfg.mock_response_prefix == "[MOCK]"
        assert cfg.max_recursion_depth == 5
        assert cfg.default_prompt_rate == pytest.approx(0.50)
        assert cfg.default_completion_rate == pytest.approx(1.50)
        assert cfg.qwen_stop_token == "<|im_end|>"


# ============================================================================
# EscalationConfigData defaults
# ============================================================================


class TestEscalationConfigData:
    """Tests for EscalationConfigData dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = EscalationConfigData()
        assert cfg.max_retries == 2
        assert cfg.max_escalations == 2

    def test_optional_gates_content(self) -> None:
        cfg = EscalationConfigData()
        assert "typecheck" in cfg.optional_gates
        assert "integration" in cfg.optional_gates
        assert "shellcheck" in cfg.optional_gates
        assert isinstance(cfg.optional_gates, frozenset)


# ============================================================================
# REPLConfigData forbidden modules and builtins
# ============================================================================


class TestREPLConfigData:
    """Tests for REPLConfigData dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = REPLConfigData()
        assert cfg.max_output_len == 10000
        assert cfg.timeout_seconds == 30

    def test_forbidden_modules_contains_dangerous(self) -> None:
        cfg = REPLConfigData()
        dangerous = {"os", "sys", "subprocess", "shutil", "socket", "pickle", "ctypes"}
        assert dangerous.issubset(cfg.forbidden_modules)
        assert isinstance(cfg.forbidden_modules, frozenset)

    def test_forbidden_builtins_contains_dangerous(self) -> None:
        cfg = REPLConfigData()
        dangerous = {"__import__", "eval", "exec", "compile", "open"}
        assert dangerous.issubset(cfg.forbidden_builtins)
        assert isinstance(cfg.forbidden_builtins, frozenset)


# ============================================================================
# ServerConfigData defaults
# ============================================================================


class TestServerConfigData:
    """Tests for ServerConfigData dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = ServerConfigData()
        assert cfg.default_url == "http://localhost:8080"
        assert cfg.timeout == 600
        assert cfg.num_slots == 4
        assert cfg.connect_timeout == 5
        assert cfg.retry_count == 3
        assert cfg.retry_backoff == pytest.approx(0.5)


# ============================================================================
# PathsConfig default paths
# ============================================================================


class TestPathsConfig:
    """Tests for PathsConfig dataclass defaults."""

    def test_default_llm_root(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_PATHS_LLM_ROOT", None)
            os.environ.pop("ORCHESTRATOR_PATHS_PROJECT_ROOT", None)
            cfg = PathsConfig()
            assert str(cfg.llm_root) == "/mnt/raid0/llm"

    def test_default_project_root(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_PATHS_LLM_ROOT", None)
            os.environ.pop("ORCHESTRATOR_PATHS_PROJECT_ROOT", None)
            cfg = PathsConfig()
            assert str(cfg.project_root) == "/mnt/raid0/llm/claude"

    def test_raid_prefix_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_PATHS_RAID_PREFIX", None)
            cfg = PathsConfig()
            assert cfg.raid_prefix == "/mnt/raid0/"


# ============================================================================
# ServerURLsConfig.as_dict()
# ============================================================================


class TestServerURLsConfig:
    """Tests for ServerURLsConfig and as_dict()."""

    def test_as_dict_excludes_service_urls(self) -> None:
        cfg = ServerURLsConfig()
        d = cfg.as_dict()
        assert "api_url" not in d
        assert "ocr_server" not in d
        assert "vision_api" not in d

    def test_as_dict_includes_role_urls(self) -> None:
        cfg = ServerURLsConfig()
        d = cfg.as_dict()
        expected_keys = {
            "frontdoor",
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
            "worker_summarize",
            "architect_general",
            "architect_coding",
            "ingest_long_context",
        }
        assert expected_keys == set(d.keys())

    def test_as_dict_values_are_strings(self) -> None:
        cfg = ServerURLsConfig()
        d = cfg.as_dict()
        for v in d.values():
            assert isinstance(v, str)

    def test_default_frontdoor_url(self) -> None:
        cfg = ServerURLsConfig()
        assert cfg.frontdoor == "http://localhost:8080"

    def test_default_architect_urls(self) -> None:
        cfg = ServerURLsConfig()
        assert cfg.architect_general == "http://localhost:8083"
        assert cfg.architect_coding == "http://localhost:8084"


# ============================================================================
# MonitorConfigData tier_overrides structure
# ============================================================================


class TestMonitorConfigData:
    """Tests for MonitorConfigData dataclass defaults and tier_overrides."""

    def test_defaults(self) -> None:
        cfg = MonitorConfigData()
        assert cfg.entropy_threshold == pytest.approx(4.0)
        assert cfg.entropy_spike_threshold == pytest.approx(2.0)
        assert cfg.repetition_threshold == pytest.approx(0.3)
        assert cfg.min_tokens_before_abort == 50
        assert cfg.perplexity_window == 20
        assert cfg.ngram_size == 3

    def test_tier_overrides_has_expected_tiers(self) -> None:
        cfg = MonitorConfigData()
        assert set(cfg.tier_overrides.keys()) == {"worker", "coder", "architect", "ingest"}

    def test_tier_overrides_architect_has_higher_thresholds(self) -> None:
        cfg = MonitorConfigData()
        arch = cfg.tier_overrides["architect"]
        assert arch["entropy_threshold"] > cfg.entropy_threshold
        assert arch["min_tokens_before_abort"] > cfg.min_tokens_before_abort

    def test_task_overrides_has_expected_tasks(self) -> None:
        cfg = MonitorConfigData()
        assert "code" in cfg.task_overrides
        assert "reasoning" in cfg.task_overrides


# ============================================================================
# TimeoutsConfig defaults
# ============================================================================


class TestTimeoutsConfig:
    """Tests for TimeoutsConfig dataclass defaults and for_role()."""

    def test_for_role_known(self) -> None:
        cfg = TimeoutsConfig()
        # architect_general should be >= 600 (registry or fallback)
        result = cfg.for_role("architect_general")
        assert isinstance(result, int)
        assert result > 0

    def test_for_role_unknown_returns_default(self) -> None:
        cfg = TimeoutsConfig()
        result = cfg.for_role("nonexistent_role")
        assert result == cfg.default_request

    def test_role_timeouts_dict_keys(self) -> None:
        cfg = TimeoutsConfig()
        d = cfg.role_timeouts_dict()
        assert "frontdoor" in d
        assert "architect_general" in d
        assert "worker_fast" in d
        # Should not contain service timeouts
        assert "ocr_single_page" not in d

    def test_service_timeouts_are_positive(self) -> None:
        cfg = TimeoutsConfig()
        assert cfg.ocr_single_page > 0
        assert cfg.ocr_pdf > 0
        assert cfg.health_check > 0


# ============================================================================
# get_config() singleton behavior
# ============================================================================


class TestGetConfig:
    """Tests for get_config() cached singleton."""

    def test_returns_orchestrator_config_data(self) -> None:
        get_config.cache_clear()
        cfg = get_config()
        from src.config import OrchestratorConfigData

        assert isinstance(cfg, OrchestratorConfigData)

    def test_returns_same_instance(self) -> None:
        get_config.cache_clear()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_cache_clear_yields_new_instance(self) -> None:
        get_config.cache_clear()
        cfg1 = get_config()
        get_config.cache_clear()
        cfg2 = get_config()
        # After clearing cache, a new object is created
        assert cfg1 is not cfg2

    def test_mock_mode_default_true(self) -> None:
        get_config.cache_clear()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_MOCK_MODE", None)
            get_config.cache_clear()
            cfg = get_config()
            assert cfg.mock_mode is True

    def test_has_nested_sections(self) -> None:
        get_config.cache_clear()
        cfg = get_config()
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.escalation, EscalationConfigData)
        assert isinstance(cfg.repl, REPLConfigData)
        assert isinstance(cfg.server, ServerConfigData)
        assert isinstance(cfg.server_urls, ServerURLsConfig)
        assert isinstance(cfg.timeouts, TimeoutsConfig)
        assert isinstance(cfg.monitor, MonitorConfigData)
        assert isinstance(cfg.paths, PathsConfig)
