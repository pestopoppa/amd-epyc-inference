"""Characterization tests for src/features.py.

Tests the Features dataclass, validation logic, singleton accessor,
environment variable parsing, and get_features() factory function.
"""

from __future__ import annotations

import pytest

from src.features import (
    Features,
    _env_bool,
    features,
    get_features,
    reset_features,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Reset the features singleton before and after every test."""
    reset_features()
    yield
    reset_features()


# ---------------------------------------------------------------------------
# 1. Features defaults (dataclass field defaults)
# ---------------------------------------------------------------------------
class TestFeaturesDefaults:
    def test_dataclass_defaults(self):
        """Features() with no args should produce the field-level defaults."""
        f = Features()
        # Explicitly False by default
        assert f.memrl is False
        assert f.scripts is False
        assert f.streaming is False
        assert f.openai_compat is False
        assert f.react_mode is False
        assert f.output_formalizer is False
        assert f.deferred_tool_results is False
        assert f.restricted_python is False
        assert f.specialist_routing is False
        assert f.plan_review is False
        assert f.architect_delegation is False
        assert f.parallel_execution is False
        assert f.personas is False
        assert f.staged_rewards is False
        assert f.skillbank is False
        assert f.input_formalizer is False
        assert f.unified_streaming is False
        assert f.side_effect_tracking is False
        assert f.structured_tool_output is False
        assert f.model_fallback is False
        assert f.content_cache is False
        assert f.session_compaction is False
        assert f.resume_tokens is False
        assert f.approval_gates is False
        assert f.binding_routing is False

        # Explicitly True by default
        assert f.tools is True
        assert f.repl is True
        assert f.caching is True
        assert f.structured_delimiters is True
        assert f.semantic_classifiers is True
        assert f.generation_monitor is True
        assert f.mock_mode is True


# ---------------------------------------------------------------------------
# 2-6. Features.validate() dependency checks
# ---------------------------------------------------------------------------
class TestFeaturesValidate:
    def test_scripts_without_tools_error(self):
        """scripts=True without tools=True is a dependency violation."""
        f = Features(scripts=True, tools=False)
        errors = f.validate()
        assert any("scripts" in e and "tools" in e for e in errors)

    def test_specialist_routing_without_memrl_error(self):
        """specialist_routing requires memrl."""
        f = Features(specialist_routing=True, memrl=False)
        errors = f.validate()
        assert any("specialist_routing" in e and "memrl" in e for e in errors)

    def test_plan_review_without_memrl_error(self):
        """plan_review requires memrl."""
        f = Features(plan_review=True, memrl=False)
        errors = f.validate()
        assert any("plan_review" in e and "memrl" in e for e in errors)

    def test_approval_gates_without_resume_tokens_error(self):
        """approval_gates requires resume_tokens."""
        f = Features(approval_gates=True, resume_tokens=False)
        errors = f.validate()
        assert any("approval_gates" in e and "resume_tokens" in e for e in errors)

    def test_approval_gates_without_side_effect_tracking_error(self):
        """approval_gates also requires side_effect_tracking."""
        f = Features(approval_gates=True, side_effect_tracking=False)
        errors = f.validate()
        assert any("approval_gates" in e and "side_effect_tracking" in e for e in errors)

    def test_valid_config_no_errors(self):
        """A fully-satisfied dependency set should produce zero errors."""
        f = Features(
            memrl=True,
            tools=True,
            scripts=True,
            specialist_routing=True,
            plan_review=True,
            architect_delegation=True,
            parallel_execution=True,
            personas=True,
            staged_rewards=True,
            skillbank=True,
            resume_tokens=True,
            side_effect_tracking=True,
            approval_gates=True,
            restricted_python=False,  # skip library check
        )
        errors = f.validate()
        assert errors == []

    def test_architect_delegation_without_memrl_error(self):
        """architect_delegation requires memrl."""
        f = Features(architect_delegation=True, memrl=False)
        errors = f.validate()
        assert any("architect_delegation" in e and "memrl" in e for e in errors)

    def test_parallel_execution_without_architect_delegation_error(self):
        """parallel_execution requires architect_delegation."""
        f = Features(parallel_execution=True, architect_delegation=False)
        errors = f.validate()
        assert any("parallel_execution" in e and "architect_delegation" in e for e in errors)

    def test_personas_without_memrl_error(self):
        """personas requires memrl."""
        f = Features(personas=True, memrl=False)
        errors = f.validate()
        assert any("personas" in e and "memrl" in e for e in errors)

    def test_staged_rewards_without_memrl_error(self):
        """staged_rewards requires memrl."""
        f = Features(staged_rewards=True, memrl=False)
        errors = f.validate()
        assert any("staged_rewards" in e and "memrl" in e for e in errors)

    def test_skillbank_without_memrl_error(self):
        """skillbank requires memrl."""
        f = Features(skillbank=True, memrl=False)
        errors = f.validate()
        assert any("skillbank" in e and "memrl" in e for e in errors)


# ---------------------------------------------------------------------------
# 7. Features.summary()
# ---------------------------------------------------------------------------
class TestFeaturesSummary:
    def test_summary_returns_dict_with_all_keys(self):
        """summary() must return a dict covering every documented flag."""
        f = Features()
        s = f.summary()
        expected_keys = {
            "memrl", "tools", "scripts", "streaming", "openai_compat",
            "repl", "caching", "structured_delimiters", "react_mode",
            "output_formalizer", "parallel_tools", "deferred_tool_results",
            "escalation_compression", "script_interception",
            "credential_redaction", "cascading_tool_policy",
            "restricted_python", "specialist_routing",
            "plan_review", "architect_delegation", "parallel_execution",
            "personas", "staged_rewards", "input_formalizer",
            "generation_monitor", "semantic_classifiers", "unified_streaming",
            "side_effect_tracking", "structured_tool_output", "model_fallback",
            "content_cache", "session_compaction", "resume_tokens",
            "approval_gates", "binding_routing", "skillbank", "mock_mode",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_values_match_fields(self):
        """summary() values must agree with the instance attributes."""
        f = Features(memrl=True, tools=False, streaming=True)
        s = f.summary()
        assert s["memrl"] is True
        assert s["tools"] is False
        assert s["streaming"] is True


# ---------------------------------------------------------------------------
# 8. Features.enabled_features()
# ---------------------------------------------------------------------------
class TestEnabledFeatures:
    def test_enabled_features_returns_only_true(self):
        """enabled_features() should list exactly the flags that are True."""
        f = Features(memrl=True, tools=True, scripts=False, streaming=False,
                     repl=False, caching=False, structured_delimiters=False,
                     semantic_classifiers=False, generation_monitor=False,
                     mock_mode=False)
        enabled = f.enabled_features()
        assert "memrl" in enabled
        assert "tools" in enabled
        assert "scripts" not in enabled
        assert "streaming" not in enabled

    def test_all_false_returns_empty(self):
        """If every flag is False, enabled_features() should be empty."""
        kwargs = {name: False for name in Features().summary()}
        f = Features(**kwargs)
        assert f.enabled_features() == []


# ---------------------------------------------------------------------------
# 9-10. get_features() test-mode vs production defaults
# ---------------------------------------------------------------------------
class TestGetFeatures:
    def test_test_mode_defaults(self, monkeypatch):
        """get_features() in test mode: mock_mode=True, repl=True, tools=False."""
        # Clear any env vars that might interfere
        for key in list(monkeypatch._patches if hasattr(monkeypatch, '_patches') else []):
            pass  # monkeypatch handles cleanup automatically
        f = get_features(production=False)
        assert f.mock_mode is True
        assert f.repl is True
        assert f.tools is False
        assert f.memrl is False
        assert f.streaming is False

    def test_production_defaults(self):
        """get_features(production=True): mock_mode=False, tools=True, memrl=True."""
        f = get_features(production=True)
        assert f.mock_mode is False
        assert f.tools is True
        assert f.memrl is True
        assert f.streaming is True
        assert f.scripts is True
        assert f.openai_compat is True
        assert f.repl is True

    def test_override_applies(self):
        """Explicit override dict takes precedence over defaults."""
        f = get_features(override={"memrl": True})
        assert f.memrl is True

    def test_override_can_disable_production_default(self):
        """Override can turn off a production-default-on flag."""
        f = get_features(production=True, override={"tools": False})
        assert f.tools is False


# ---------------------------------------------------------------------------
# 12. _env_bool reads from ORCHESTRATOR_ prefix
# ---------------------------------------------------------------------------
class TestEnvBool:
    @pytest.mark.parametrize("val,expected", [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ])
    def test_truthy_falsy_values(self, monkeypatch, val, expected):
        """_env_bool should parse various truthy/falsy strings."""
        monkeypatch.setenv("ORCHESTRATOR_TEST_FLAG", val)
        assert _env_bool("TEST_FLAG", default=not expected) is expected

    def test_missing_env_uses_default(self, monkeypatch):
        """Missing env var returns the default."""
        monkeypatch.delenv("ORCHESTRATOR_MISSING", raising=False)
        assert _env_bool("MISSING", default=True) is True
        assert _env_bool("MISSING", default=False) is False

    def test_unrecognized_value_uses_default(self, monkeypatch):
        """Unrecognized string falls back to default."""
        monkeypatch.setenv("ORCHESTRATOR_WEIRD", "maybe")
        assert _env_bool("WEIRD", default=True) is True
        assert _env_bool("WEIRD", default=False) is False

    def test_prefix_is_applied(self, monkeypatch):
        """_env_bool('MEMRL') reads ORCHESTRATOR_MEMRL."""
        monkeypatch.setenv("ORCHESTRATOR_MEMRL", "1")
        assert _env_bool("MEMRL") is True


# ---------------------------------------------------------------------------
# 13-14. Singleton: features() and reset_features()
# ---------------------------------------------------------------------------
class TestSingleton:
    def test_features_returns_same_instance(self):
        """features() should return the same object on repeated calls."""
        a = features()
        b = features()
        assert a is b

    def test_reset_features_clears_singleton(self):
        """After reset_features(), a new instance should be created."""
        a = features()
        reset_features()
        b = features()
        assert a is not b


# ---------------------------------------------------------------------------
# 15. Environment variable override of feature flags via get_features()
# ---------------------------------------------------------------------------
class TestEnvVarOverride:
    def test_env_var_overrides_test_default(self, monkeypatch):
        """Setting ORCHESTRATOR_MEMRL=1 should override the test default (False)."""
        monkeypatch.setenv("ORCHESTRATOR_MEMRL", "1")
        f = get_features(production=False)
        assert f.memrl is True

    def test_env_var_overrides_production_default(self, monkeypatch):
        """Setting ORCHESTRATOR_TOOLS=0 should override the production default (True)."""
        monkeypatch.setenv("ORCHESTRATOR_TOOLS", "0")
        f = get_features(production=True)
        assert f.tools is False

    def test_singleton_reads_env_at_creation(self, monkeypatch):
        """The singleton should reflect env vars present at first call."""
        monkeypatch.setenv("ORCHESTRATOR_STREAMING", "1")
        reset_features()
        f = features()
        assert f.streaming is True
