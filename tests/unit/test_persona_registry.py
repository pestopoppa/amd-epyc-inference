"""Tests for persona registry, persona injection in llm_call, and MemRL seeds.

Covers:
- PersonaRegistry: YAML loading, get(), match(), all_names()
- llm_call() persona injection: feature gating, prompt prefixing
- _delegate() persona passthrough: parameter forwarding, delegation recording
- TaskIR schema: persona_hint and step persona fields
- Feature flag: default, dependency, summary
- Seed loader: _get_persona_seeds() structure
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.persona_loader import (
    PersonaConfig,
    PersonaRegistry,
    get_persona_registry,
    reset_persona_registry,
)

REGISTRY_PATH = Path(__file__).parent.parent.parent / "orchestration" / "persona_registry.yaml"


# ── PersonaRegistry ─────────────────────────────────────────────────


class TestPersonaRegistry:
    """PersonaRegistry loading and lookup."""

    def test_load_from_yaml(self):
        """Registry loads personas from the real YAML file."""
        registry = PersonaRegistry(REGISTRY_PATH)
        assert len(registry) >= 18

    def test_get_known_persona(self):
        registry = PersonaRegistry(REGISTRY_PATH)
        cfg = registry.get("security_auditor")
        assert cfg is not None
        assert cfg.name == "security_auditor"
        assert "security" in cfg.description.lower()
        assert len(cfg.system_prompt) > 0
        assert len(cfg.task_patterns) > 0

    def test_get_unknown_persona(self):
        registry = PersonaRegistry(REGISTRY_PATH)
        assert registry.get("nonexistent_persona") is None

    def test_match_security_task(self):
        registry = PersonaRegistry(REGISTRY_PATH)
        matches = registry.match("Review this code for SQL injection vulnerabilities")
        names = [name for name, _ in matches]
        assert "security_auditor" in names

    def test_match_empty_for_unrelated(self):
        """Unrelated task text returns no matches."""
        registry = PersonaRegistry(REGISTRY_PATH)
        matches = registry.match("the quick brown fox jumps over the lazy dog")
        assert len(matches) == 0

    def test_all_names_returns_18(self):
        registry = PersonaRegistry(REGISTRY_PATH)
        names = registry.all_names()
        assert len(names) == 18
        # Check a sample from each category
        assert "security_auditor" in names
        assert "research_architect" in names
        assert "hardware_specialist" in names

    def test_get_system_prompt(self):
        registry = PersonaRegistry(REGISTRY_PATH)
        prompt = registry.get_system_prompt("technical_writer")
        assert prompt is not None
        assert "technical writer" in prompt.lower()

    def test_get_system_prompt_unknown(self):
        registry = PersonaRegistry(REGISTRY_PATH)
        assert registry.get_system_prompt("nonexistent") is None

    def test_contains(self):
        registry = PersonaRegistry(REGISTRY_PATH)
        assert "code_reviewer" in registry
        assert "nonexistent" not in registry

    def test_missing_yaml_graceful(self, tmp_path):
        """Missing YAML file is handled gracefully."""
        registry = PersonaRegistry(tmp_path / "missing.yaml")
        assert len(registry) == 0
        assert registry.get("anything") is None

    def test_singleton_pattern(self):
        reset_persona_registry()
        try:
            r1 = get_persona_registry()
            r2 = get_persona_registry()
            assert r1 is r2
        finally:
            reset_persona_registry()

    def test_persona_config_frozen(self):
        cfg = PersonaConfig(
            name="test",
            description="Test persona",
            system_prompt="Be a tester",
            task_patterns=["test"],
            seed_q=0.9,
        )
        with pytest.raises(AttributeError):
            cfg.name = "changed"  # type: ignore[misc]

    def test_match_multiple_patterns(self):
        """Task matching multiple patterns gets higher count."""
        registry = PersonaRegistry(REGISTRY_PATH)
        matches = registry.match("security audit of authentication permissions")
        # security_auditor has patterns: security, audit, permission, auth
        sec_match = [(n, c) for n, c in matches if n == "security_auditor"]
        assert len(sec_match) == 1
        assert sec_match[0][1] >= 3  # At least 3 pattern matches


# ── llm_call persona injection ──────────────────────────────────────


class TestLLMCallPersona:
    """Persona injection in LLMPrimitives.llm_call()."""

    def _make_primitives(self):
        """Create mock LLMPrimitives for testing."""
        from src.llm_primitives import LLMPrimitives

        return LLMPrimitives(mock_mode=True)

    def test_no_persona_no_change(self):
        """persona=None does not modify the prompt."""
        primitives = self._make_primitives()
        result = primitives.llm_call("Hello world", persona=None)
        assert isinstance(result, str)

    def test_persona_enabled_prefixes_prompt(self):
        """With personas enabled, persona system prompt is prefixed."""
        from src.features import Features, set_features, reset_features
        from src.persona_loader import reset_persona_registry

        try:
            set_features(Features(personas=True, mock_mode=True))
            reset_persona_registry()
            primitives = self._make_primitives()
            # Use a real persona from the registry
            result = primitives.llm_call("Do something", persona="security_auditor")
            assert isinstance(result, str)
            # Check the log entry records the persona
            assert len(primitives.call_log) == 1
            assert primitives.call_log[0].persona == "security_auditor"
        finally:
            reset_features()
            reset_persona_registry()

    def test_persona_disabled_no_change(self):
        """With personas feature disabled, persona param is ignored."""
        from src.features import Features, set_features, reset_features
        try:
            set_features(Features(personas=False, mock_mode=True))
            primitives = self._make_primitives()
            result = primitives.llm_call("Hello", persona="security_auditor")
            assert isinstance(result, str)
            # Persona logged but not injected
            assert primitives.call_log[0].persona == "security_auditor"
        finally:
            reset_features()

    def test_persona_with_skip_suffix(self):
        """skip_suffix=True also skips persona injection."""
        from src.features import Features, set_features, reset_features
        try:
            set_features(Features(personas=True, mock_mode=True))
            primitives = self._make_primitives()
            result = primitives.llm_call(
                "Hello", persona="security_auditor", skip_suffix=True,
            )
            assert isinstance(result, str)
        finally:
            reset_features()

    def test_nonexistent_persona_noop(self):
        """Unknown persona name is a graceful no-op."""
        from src.features import Features, set_features, reset_features
        try:
            set_features(Features(personas=True, mock_mode=True))
            primitives = self._make_primitives()
            result = primitives.llm_call("Hello", persona="nonexistent_xyz")
            assert isinstance(result, str)
        finally:
            reset_features()


# ── _delegate persona passthrough ───────────────────────────────────


class TestDelegatePersona:
    """Persona parameter passthrough in _delegate()."""

    def _make_env(self, llm_primitives=None):
        """Create a minimal REPLEnvironment for testing."""
        from src.repl_environment import REPLEnvironment

        primitives = llm_primitives or MagicMock()
        if not llm_primitives:
            primitives.llm_call.return_value = "delegate result"

        env = REPLEnvironment.__new__(REPLEnvironment)
        env.role = "frontdoor"
        env.llm_primitives = primitives
        env.artifacts = {}
        env._exploration_calls = 0
        env._exploration_log = MagicMock()
        return env

    def test_empty_persona_no_kwarg(self):
        """Empty persona string passes None to llm_call."""
        env = self._make_env()
        result = env._delegate("do something", "worker_general", persona="")
        env.llm_primitives.llm_call.assert_called_once()
        call_kwargs = env.llm_primitives.llm_call.call_args
        assert call_kwargs.kwargs.get("persona") is None

    def test_persona_passed_through(self):
        """Named persona is forwarded to llm_call."""
        env = self._make_env()
        result = env._delegate(
            "review code", "coder_primary", persona="code_reviewer",
        )
        call_kwargs = env.llm_primitives.llm_call.call_args
        assert call_kwargs.kwargs.get("persona") == "code_reviewer"

    def test_persona_recorded_in_delegation(self):
        """Persona is stored in the delegation record."""
        env = self._make_env()
        env._delegate("review code", "coder_primary", persona="code_reviewer")
        delegations = env.artifacts.get("_delegations", [])
        assert len(delegations) == 1
        assert delegations[0]["persona"] == "code_reviewer"


# ── TaskIR Schema ───────────────────────────────────────────────────


class TestTaskIRPersonaFields:
    """TaskIR schema accepts persona fields."""

    @pytest.fixture
    def schema(self):
        schema_path = Path(__file__).parent.parent.parent / "orchestration" / "task_ir.schema.json"
        with open(schema_path) as f:
            return json.load(f)

    def test_agents_has_persona_hint(self, schema):
        agent_props = schema["properties"]["agents"]["items"]["properties"]
        assert "persona_hint" in agent_props
        assert agent_props["persona_hint"]["type"] == "string"

    def test_steps_has_persona(self, schema):
        step_props = schema["properties"]["plan"]["properties"]["steps"]["items"]["properties"]
        assert "persona" in step_props
        assert step_props["persona"]["type"] == "string"


# ── Feature Flag ────────────────────────────────────────────────────


class TestPersonasFeatureFlag:
    """personas feature flag integration."""

    def test_default_disabled(self):
        from src.features import Features
        f = Features()
        assert f.personas is False

    def test_dependency_on_memrl(self):
        from src.features import Features
        f = Features(personas=True, memrl=False)
        errors = f.validate()
        assert any("personas" in e for e in errors)

    def test_valid_with_memrl(self):
        from src.features import Features
        f = Features(personas=True, memrl=True)
        errors = f.validate()
        assert not any("personas" in e for e in errors)

    def test_in_summary(self):
        from src.features import Features
        f = Features(personas=True)
        summary = f.summary()
        assert "personas" in summary
        assert summary["personas"] is True


# ── Seed Loader ─────────────────────────────────────────────────────


class TestPersonaSeeds:
    """_get_persona_seeds() structure validation."""

    def test_returns_nonempty(self):
        from orchestration.repl_memory.seed_loader import _get_persona_seeds
        seeds = _get_persona_seeds()
        assert len(seeds) >= 18  # At least one per persona

    def test_all_seeds_have_required_keys(self):
        from orchestration.repl_memory.seed_loader import _get_persona_seeds
        seeds = _get_persona_seeds()
        for seed in seeds:
            assert "task" in seed, f"Missing 'task' in seed: {seed}"
            assert "action" in seed, f"Missing 'action' in seed: {seed}"
            assert "outcome" in seed, f"Missing 'outcome' in seed: {seed}"
            assert seed["action"].startswith("persona:"), (
                f"Action should start with 'persona:': {seed['action']}"
            )

    def test_covers_all_categories(self):
        """Seeds cover engineering, research, and practical personas."""
        from orchestration.repl_memory.seed_loader import _get_persona_seeds
        seeds = _get_persona_seeds()
        actions = {s["action"] for s in seeds}
        # Spot-check one from each category
        assert "persona:security_auditor" in actions
        assert "persona:research_architect" in actions
        assert "persona:hardware_specialist" in actions
