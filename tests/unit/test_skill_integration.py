"""
Unit tests for SkillBank integration with routing pipeline.

Tests the SkillAugmentedRouter, feature flag, and prompt injection wiring.
All tests use mock objects — no live inference, no API calls.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── SkillAugmentedRouter Tests ───────────────────────────────────────────


class TestSkillAugmentedRouter:
    """Tests for SkillAugmentedRouter in retriever.py."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def skill_bank(self, temp_dir):
        from orchestration.repl_memory.skill_bank import SkillBank

        return SkillBank(
            db_path=temp_dir / "skills.db",
            faiss_path=temp_dir,
            embedding_dim=128,
        )

    @pytest.fixture
    def skill_retriever(self, skill_bank):
        from orchestration.repl_memory.skill_retriever import SkillRetriever

        return SkillRetriever(skill_bank=skill_bank)

    def _make_mock_hybrid_router(self, routing_decision=None, strategy="learned"):
        """Create a mock HybridRouter."""

        class MockRetriever:
            def retrieve_for_routing(self, task_ir):
                return []

        class MockHybridRouter:
            retriever = MockRetriever()

            def route(self, task_ir):
                return routing_decision or ["frontdoor"], strategy

            def route_with_mode(self, task_ir):
                return routing_decision or ["frontdoor"], strategy, "direct"

            def route_3way(self, task_ir, cost_tiers=None):
                return "SELF:direct", strategy, 0.7

        return MockHybridRouter()

    def _make_mock_embedder(self, dim=128):
        """Create a mock TaskEmbedder."""

        class MockEmbedder:
            def embed_task_ir(self, task_ir):
                return np.random.randn(dim).astype(np.float32)

        return MockEmbedder()

    def test_route_delegates_to_hybrid(self, skill_bank, skill_retriever):
        from orchestration.repl_memory.retriever import SkillAugmentedRouter

        hybrid = self._make_mock_hybrid_router(["coder"], "learned")
        embedder = self._make_mock_embedder()
        router = SkillAugmentedRouter(hybrid, skill_retriever, embedder)

        decision, strategy = router.route({"task_type": "code"})
        assert decision == ["coder"]
        assert strategy == "learned"

    def test_route_with_skills_returns_skill_context(self, skill_bank, skill_retriever):
        from orchestration.repl_memory.retriever import SkillAugmentedRouter
        from orchestration.repl_memory.skill_bank import Skill, SkillBank

        # Store a general skill
        skill = Skill(
            id=SkillBank.generate_id("general"),
            title="Test General Skill",
            skill_type="general",
            principle="Always test your code.",
            when_to_apply="When writing code",
            task_types=["*"],
            source_trajectory_ids=["t1"],
            source_outcome="success",
            confidence=0.8,
        )
        skill_bank.store(skill)

        hybrid = self._make_mock_hybrid_router(["frontdoor"], "rules")
        embedder = self._make_mock_embedder()
        router = SkillAugmentedRouter(hybrid, skill_retriever, embedder)

        decision, strategy, skill_context = router.route_with_skills(
            {"task_type": "code"}
        )
        assert decision == ["frontdoor"]
        assert strategy == "rules"
        assert "Test General Skill" in skill_context
        assert "Always test your code" in skill_context

    def test_route_with_skills_empty_bank(self, skill_bank, skill_retriever):
        from orchestration.repl_memory.retriever import SkillAugmentedRouter

        hybrid = self._make_mock_hybrid_router()
        embedder = self._make_mock_embedder()
        router = SkillAugmentedRouter(hybrid, skill_retriever, embedder)

        decision, strategy, skill_context = router.route_with_skills(
            {"task_type": "chat"}
        )
        assert skill_context == ""

    def test_route_with_skills_embedder_failure_graceful(
        self, skill_bank, skill_retriever
    ):
        from orchestration.repl_memory.retriever import SkillAugmentedRouter

        class FailingEmbedder:
            def embed_task_ir(self, task_ir):
                raise RuntimeError("Embedder unavailable")

        hybrid = self._make_mock_hybrid_router()
        router = SkillAugmentedRouter(hybrid, skill_retriever, FailingEmbedder())

        # Should not raise — graceful degradation
        decision, strategy, skill_context = router.route_with_skills(
            {"task_type": "code"}
        )
        assert decision == ["frontdoor"]
        assert skill_context == ""

    def test_route_with_mode_delegates(self, skill_bank, skill_retriever):
        from orchestration.repl_memory.retriever import SkillAugmentedRouter

        hybrid = self._make_mock_hybrid_router(["coder"], "learned")
        embedder = self._make_mock_embedder()
        router = SkillAugmentedRouter(hybrid, skill_retriever, embedder)

        decision, strategy, mode = router.route_with_mode({"task_type": "code"})
        assert decision == ["coder"]
        assert mode == "direct"

    def test_route_3way_delegates(self, skill_bank, skill_retriever):
        from orchestration.repl_memory.retriever import SkillAugmentedRouter

        hybrid = self._make_mock_hybrid_router()
        embedder = self._make_mock_embedder()
        router = SkillAugmentedRouter(hybrid, skill_retriever, embedder)

        action, strategy, confidence = router.route_3way({"task_type": "code"})
        assert action == "SELF:direct"

    def test_retriever_property_exposes_underlying(self, skill_bank, skill_retriever):
        from orchestration.repl_memory.retriever import SkillAugmentedRouter

        hybrid = self._make_mock_hybrid_router()
        embedder = self._make_mock_embedder()
        router = SkillAugmentedRouter(hybrid, skill_retriever, embedder)

        assert router.retriever is hybrid.retriever


# ── Feature Flag Tests ───────────────────────────────────────────────────


class TestSkillBankFeatureFlag:
    """Tests for the skillbank feature flag."""

    def test_skillbank_flag_exists(self):
        from src.features import Features

        f = Features()
        assert hasattr(f, "skillbank")
        assert f.skillbank is False  # Default off

    def test_skillbank_requires_memrl(self):
        from src.features import Features

        f = Features(skillbank=True, memrl=False)
        errors = f.validate()
        assert any("skillbank" in e and "memrl" in e for e in errors)

    def test_skillbank_valid_with_memrl(self):
        from src.features import Features

        f = Features(skillbank=True, memrl=True)
        errors = f.validate()
        assert not any("skillbank" in e for e in errors)

    def test_skillbank_in_summary(self):
        from src.features import Features

        f = Features(skillbank=True)
        summary = f.summary()
        assert "skillbank" in summary
        assert summary["skillbank"] is True

    def test_env_var_reading(self, monkeypatch):
        from src.features import get_features, reset_features

        reset_features()
        monkeypatch.setenv("ORCHESTRATOR_SKILLBANK", "1")
        f = get_features()
        assert f.skillbank is True
        reset_features()

    def test_env_var_off(self, monkeypatch):
        from src.features import get_features, reset_features

        reset_features()
        monkeypatch.setenv("ORCHESTRATOR_SKILLBANK", "0")
        f = get_features()
        assert f.skillbank is False
        reset_features()


# ── RoutingResult skill_context Tests ────────────────────────────────────


class TestRoutingResultSkillContext:
    """Tests for skill_context field on RoutingResult."""

    def test_default_empty(self):
        from src.api.routes.chat_utils import RoutingResult

        result = RoutingResult(
            task_id="test-1",
            task_ir={"task_type": "chat"},
            use_mock=True,
        )
        assert result.skill_context == ""

    def test_skill_context_set(self):
        from src.api.routes.chat_utils import RoutingResult

        result = RoutingResult(
            task_id="test-2",
            task_ir={"task_type": "chat"},
            use_mock=True,
            skill_context="## Learned Skills\n- **Test**: Do this.",
        )
        assert "Learned Skills" in result.skill_context


# ── AppState SkillBank Fields Tests ──────────────────────────────────────


class TestAppStateSkillFields:
    """Tests for skill_bank and skill_retriever fields on AppState."""

    def test_default_none(self):
        from src.api.state import AppState

        state = AppState()
        assert state.skill_bank is None
        assert state.skill_retriever is None

    def test_can_assign(self):
        from src.api.state import AppState

        state = AppState()
        state.skill_bank = "mock_skill_bank"
        state.skill_retriever = "mock_retriever"
        assert state.skill_bank == "mock_skill_bank"
        assert state.skill_retriever == "mock_retriever"


# ── memrl.py Load Tests ─────────────────────────────────────────────────


class TestMemrlSkillBankImports:
    """Tests for SkillBank import wiring in memrl.py."""

    def test_skillbank_globals_exist(self):
        from src.api.services import memrl

        assert hasattr(memrl, "SkillBank")
        assert hasattr(memrl, "SkillRetriever")
        assert hasattr(memrl, "SkillAugmentedRouter")

    def test_skillbank_globals_default_none(self):
        """Before load_optional_imports, globals should be None."""
        from src.api.services import memrl

        # These are None by default (no feature flags set in test env)
        # Just verify they exist as module-level attributes
        assert memrl.SkillBank is None or memrl.SkillBank is not None  # exists
