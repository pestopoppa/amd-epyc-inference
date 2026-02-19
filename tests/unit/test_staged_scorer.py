"""Tests for staged reward shaping (PARL-inspired explore → exploit annealing).

Covers:
- StagedConfig defaults
- StagedQScorer: lambda annealing schedule, exploration bonus, staged reward
- count_by_combo(): action-only, action+task_type, empty
- Feature flag: default disabled, dependency on memrl
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from orchestration.repl_memory.staged_scorer import StagedConfig, StagedQScorer
from src.features import Features


# ── Helpers ──────────────────────────────────────────────────────────────


class FakeStore:
    """Minimal store stub with count_by_combo() for unit tests."""

    def __init__(self, counts: dict[tuple[str, str], int] | None = None):
        self._counts = counts or {}

    def count_by_combo(self, action: str, task_type: str | None = None) -> int:
        return self._counts.get((action, task_type or ""), 0)


# ── StagedQScorer ────────────────────────────────────────────────────────


class TestStagedQScorer:
    """Staged reward annealing tests."""

    def test_initial_lambda(self):
        """Fresh scorer has lambda=initial_lambda."""
        scorer = StagedQScorer()
        assert scorer.current_lambda == pytest.approx(0.3)
        assert scorer.global_step == 0

    def test_lambda_anneals(self):
        """After 25 steps (half of 50), lambda ≈ 0.15."""
        scorer = StagedQScorer()
        scorer._global_step = 25
        assert scorer.current_lambda == pytest.approx(0.15)

    def test_lambda_at_anneal_end(self):
        """After exactly anneal_steps, lambda = min_lambda."""
        scorer = StagedQScorer()
        scorer._global_step = 50
        assert scorer.current_lambda == pytest.approx(0.0)

    def test_lambda_beyond_anneal(self):
        """After 100 steps (past anneal_steps), lambda stays at min_lambda."""
        scorer = StagedQScorer()
        scorer._global_step = 100
        assert scorer.current_lambda == pytest.approx(0.0)

    def test_custom_config(self):
        """Custom config overrides defaults."""
        cfg = StagedConfig(initial_lambda=0.5, anneal_steps=100, min_lambda=0.05)
        scorer = StagedQScorer(config=cfg)
        assert scorer.current_lambda == pytest.approx(0.5)
        scorer._global_step = 50
        assert scorer.current_lambda == pytest.approx(0.25)
        scorer._global_step = 100
        assert scorer.current_lambda == pytest.approx(0.05)

    def test_exploration_bonus_high_for_new_combo(self):
        """N=0 → exploration bonus = 1.0."""
        store = FakeStore()
        scorer = StagedQScorer(config=StagedConfig(initial_lambda=1.0, anneal_steps=1000))
        # With lambda=1.0, staged = 1.0 * bonus + 0.0 * base = bonus
        result = scorer.compute_staged_reward(0.0, "action_x", "code", store)
        assert result == pytest.approx(1.0)  # 1/sqrt(0+1) = 1.0

    def test_exploration_bonus_decays(self):
        """N=3 → exploration bonus = 1/sqrt(4) = 0.5."""
        store = FakeStore({("action_x", "code"): 3})
        scorer = StagedQScorer(config=StagedConfig(initial_lambda=1.0, anneal_steps=1000))
        result = scorer.compute_staged_reward(0.0, "action_x", "code", store)
        assert result == pytest.approx(0.5)  # 1/sqrt(3+1) = 0.5

    def test_staged_reward_early(self):
        """At step 0 with lambda=0.3, reward shifts toward exploration bonus.

        base_reward=0.8, N=0 → bonus=1.0
        staged = 0.3 * 1.0 + 0.7 * 0.8 = 0.3 + 0.56 = 0.86
        """
        store = FakeStore()
        scorer = StagedQScorer()
        result = scorer.compute_staged_reward(0.8, "act", "code", store)
        assert result == pytest.approx(0.86)
        assert scorer.global_step == 1  # Step incremented

    def test_staged_reward_late(self):
        """After annealing, lambda=0, reward = base_reward unchanged."""
        store = FakeStore()
        scorer = StagedQScorer()
        scorer._global_step = 50  # Fully annealed
        result = scorer.compute_staged_reward(0.8, "act", "code", store)
        assert result == pytest.approx(0.8)
        assert scorer.global_step == 51

    def test_staged_reward_clamped(self):
        """Result clamped to [-1, 1]."""
        store = FakeStore()
        scorer = StagedQScorer(config=StagedConfig(initial_lambda=1.0, anneal_steps=1000))
        # base=1.0, bonus=1.0, staged = 1.0*1.0 + 0.0*1.0 = 1.0 (at boundary)
        result = scorer.compute_staged_reward(1.0, "act", "code", store)
        assert result <= 1.0
        # Negative base still clamped
        result2 = scorer.compute_staged_reward(-1.5, "act", "code", store)
        assert result2 >= -1.0

    def test_reset(self):
        """reset() sets step counter back to 0."""
        scorer = StagedQScorer()
        scorer._global_step = 42
        scorer.reset()
        assert scorer.global_step == 0
        assert scorer.current_lambda == pytest.approx(0.3)

    def test_step_increments_per_call(self):
        """Each compute_staged_reward call increments step by 1."""
        store = FakeStore()
        scorer = StagedQScorer()
        for i in range(5):
            assert scorer.global_step == i
            scorer.compute_staged_reward(0.5, "act", "code", store)
        assert scorer.global_step == 5


# ── count_by_combo on real SQLite ────────────────────────────────────────


class TestCountByCombo:
    """Tests for EpisodicStore.count_by_combo() with real SQLite."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a minimal EpisodicStore with test data."""
        from orchestration.repl_memory.episodic_store import EpisodicStore

        store = EpisodicStore(
            db_path=tmp_path,
            embedding_dim=4,
            use_faiss=False,
        )
        return store

    def _insert_memory(self, store, action: str, task_type: str):
        """Insert a memory row directly via SQL for test isolation."""
        import uuid
        from datetime import datetime, timezone

        import numpy as np

        now = datetime.now(timezone.utc).isoformat()
        memory_id = str(uuid.uuid4())
        context = json.dumps({"task_type": task_type})

        # Add a dummy embedding
        embedding = np.random.randn(4).astype(np.float32)
        embedding_idx = store._embedding_store.add(memory_id, embedding)

        with sqlite3.connect(store.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO memories "
                "(id, embedding_idx, action, action_type, context, outcome, q_value, created_at, updated_at) "
                "VALUES (?, ?, ?, 'routing', ?, 'success', 0.5, ?, ?)",
                (memory_id, embedding_idx, action, context, now, now),
            )
            conn.commit()

    def test_count_by_action(self, store):
        """Filters by action string."""
        self._insert_memory(store, "frontdoor:direct", "code")
        self._insert_memory(store, "frontdoor:direct", "chat")
        self._insert_memory(store, "coder_escalation", "code")

        assert store.count_by_combo("frontdoor:direct") == 2
        assert store.count_by_combo("coder_escalation") == 1
        assert store.count_by_combo("nonexistent") == 0

    def test_count_by_action_and_type(self, store):
        """Filters by both action and task_type in context JSON."""
        self._insert_memory(store, "frontdoor:direct", "code")
        self._insert_memory(store, "frontdoor:direct", "code")
        self._insert_memory(store, "frontdoor:direct", "chat")

        assert store.count_by_combo("frontdoor:direct", "code") == 2
        assert store.count_by_combo("frontdoor:direct", "chat") == 1
        assert store.count_by_combo("frontdoor:direct", "ingest") == 0

    def test_count_empty(self, store):
        """Returns 0 for unseen combos on empty store."""
        assert store.count_by_combo("anything") == 0
        assert store.count_by_combo("anything", "code") == 0


# ── Feature flag ─────────────────────────────────────────────────────────


class TestFeatureFlag:
    """staged_rewards feature flag tests."""

    def test_default_disabled(self):
        """staged_rewards is False by default."""
        features = Features()
        assert features.staged_rewards is False

    def test_dependency_on_memrl(self):
        """staged_rewards without memrl produces validation error."""
        features = Features(staged_rewards=True, memrl=False)
        errors = features.validate()
        assert any("staged_rewards" in e for e in errors)

    def test_valid_with_memrl(self):
        """staged_rewards + memrl passes validation."""
        features = Features(staged_rewards=True, memrl=True)
        errors = features.validate()
        assert not any("staged_rewards" in e for e in errors)

    def test_in_summary(self):
        """staged_rewards appears in summary dict."""
        features = Features(staged_rewards=True)
        summary = features.summary()
        assert "staged_rewards" in summary
        assert summary["staged_rewards"] is True
