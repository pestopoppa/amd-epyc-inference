"""Tests for WarmStartProtocol and RoleConfig."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.episodic_store import EpisodicStore
from orchestration.repl_memory.q_scorer import ScoringConfig
from orchestration.repl_memory.replay.warm_start import (
    RoleConfig,
    WarmStartProtocol,
)

def _make_store_in(base_dir: Path, name: str = "default") -> EpisodicStore:
    store_dir = base_dir / name
    store_dir.mkdir(parents=True, exist_ok=True)
    return EpisodicStore(db_path=store_dir, embedding_dim=1024, use_faiss=True)


def _random_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(1024).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-8
    return emb


def _make_store(tmp_path: Path, name: str = "default") -> EpisodicStore:
    """Wrapper for backward compat — delegates to _make_store_in."""
    return _make_store_in(tmp_path, name)


def _store_routing_memory(
    store: EpisodicStore,
    action: str = "coder_escalation",
    model_id: str = "model-A",
    q_value: float = 0.8,
    seed: int = 42,
) -> str:
    return store.store(
        embedding=_random_embedding(seed),
        action=action,
        action_type="routing",
        context={"task_type": "code", "role": action},
        outcome="success",
        initial_q=q_value,
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# RoleConfig tests
# ---------------------------------------------------------------------------

class TestRoleConfig:
    def test_default_for_role(self):
        rc = RoleConfig.default_for_role("coder_escalation", "qwen2.5-coder-32b")
        assert rc.role == "coder_escalation"
        assert rc.model_id == "qwen2.5-coder-32b"
        assert rc.retrieval_config.semantic_k == 20
        assert rc.scoring_config.learning_rate == 0.1


# ---------------------------------------------------------------------------
# WarmStartProtocol.detect_model_swap tests
# ---------------------------------------------------------------------------

class TestDetectModelSwap:
    def test_no_memories_no_swap(self, tmp_path):
        store = _make_store_in(tmp_path, "detect_empty")
        assert WarmStartProtocol.detect_model_swap("coder", "model-B", store) is False

    def test_same_model_no_swap(self, tmp_path):
        store = _make_store_in(tmp_path, "detect_same")
        _store_routing_memory(store, "coder", "model-A", seed=1)
        _store_routing_memory(store, "coder", "model-A", seed=2)
        assert WarmStartProtocol.detect_model_swap("coder", "model-A", store) is False

    def test_different_model_swap_detected(self, tmp_path):
        store = _make_store_in(tmp_path, "detect_swap")
        # 3 memories from model-A
        for i in range(3):
            _store_routing_memory(store, "coder", "model-A", seed=i)
        # Now check with model-B → majority (3/3) are from model-A → swap
        assert WarmStartProtocol.detect_model_swap("coder", "model-B", store) is True

    def test_mixed_models_majority_rules(self, tmp_path):
        store = _make_store_in(tmp_path, "detect_mixed")
        # 2 from model-A, 3 from model-B
        for i in range(2):
            _store_routing_memory(store, "coder", "model-A", seed=i)
        for i in range(3):
            _store_routing_memory(store, "coder", "model-B", seed=10 + i)
        # Check with model-C: 5 memories, all different → swap
        assert WarmStartProtocol.detect_model_swap("coder", "model-C", store) is True
        # Check with model-B: only 2 from model-A are different → 2/5 < 50% → no swap
        assert WarmStartProtocol.detect_model_swap("coder", "model-B", store) is False

    def test_null_model_ids_excluded(self, tmp_path):
        store = _make_store_in(tmp_path, "detect_null")
        # Store without model_id (legacy)
        store.store(
            embedding=_random_embedding(1),
            action="coder", action_type="routing",
            context={}, outcome="success", initial_q=0.8,
            model_id=None,
        )
        assert WarmStartProtocol.detect_model_swap("coder", "model-B", store) is False


# ---------------------------------------------------------------------------
# WarmStartProtocol.execute_warm_start tests
# ---------------------------------------------------------------------------

class TestExecuteWarmStart:
    def test_reset_q_values(self, tmp_path):
        store = _make_store_in(tmp_path, "warmstart_reset")
        mid1 = _store_routing_memory(store, "coder", "model-A", q_value=0.9, seed=1)
        mid2 = _store_routing_memory(store, "coder", "model-A", q_value=0.8, seed=2)

        stats = WarmStartProtocol.execute_warm_start("coder", "model-B", store)

        assert stats.memories_reset == 2
        assert stats.model_id_old == "model-A"
        assert stats.model_id_new == "model-B"
        assert stats.warmup_tasks_remaining == 50

        # Verify Q-values are reset
        m1 = store.get_by_id(mid1)
        m2 = store.get_by_id(mid2)
        assert m1.q_value == pytest.approx(0.5)
        assert m2.q_value == pytest.approx(0.5)
        # model_id should be updated
        assert m1.model_id == "model-B"
        assert m2.model_id == "model-B"

    def test_does_not_reset_same_model(self, tmp_path):
        store = _make_store_in(tmp_path, "warmstart_same")
        mid = _store_routing_memory(store, "coder", "model-B", q_value=0.9, seed=1)

        stats = WarmStartProtocol.execute_warm_start("coder", "model-B", store)
        assert stats.memories_reset == 0

        m = store.get_by_id(mid)
        assert m.q_value == pytest.approx(0.9)

    def test_resets_null_model_id(self, tmp_path):
        """Memories with model_id=NULL get reset and assigned new model_id."""
        store = _make_store_in(tmp_path, "warmstart_null")
        mid = store.store(
            embedding=_random_embedding(1),
            action="coder", action_type="routing",
            context={}, outcome="success", initial_q=0.8,
            model_id=None,
        )

        stats = WarmStartProtocol.execute_warm_start("coder", "model-B", store)
        assert stats.memories_reset == 1

        m = store.get_by_id(mid)
        assert m.q_value == pytest.approx(0.5)
        assert m.model_id == "model-B"

    def test_only_affects_target_role(self, tmp_path):
        store = _make_store_in(tmp_path, "warmstart_role")
        mid_coder = _store_routing_memory(store, "coder", "model-A", q_value=0.9, seed=1)
        mid_ingest = _store_routing_memory(store, "ingest", "model-A", q_value=0.9, seed=2)

        WarmStartProtocol.execute_warm_start("coder", "model-B", store)

        assert store.get_by_id(mid_coder).q_value == pytest.approx(0.5)
        assert store.get_by_id(mid_ingest).q_value == pytest.approx(0.9)  # Untouched


# ---------------------------------------------------------------------------
# WarmStartProtocol.is_warmup_active tests
# ---------------------------------------------------------------------------

class TestIsWarmupActive:
    def test_active_with_no_scored(self, tmp_path):
        store = _make_store_in(tmp_path, "warmup_active")
        _store_routing_memory(store, "coder", "model-B", seed=1)
        assert WarmStartProtocol.is_warmup_active("coder", store, "model-B") is True

    def test_inactive_when_no_model_id(self, tmp_path):
        store = _make_store_in(tmp_path, "warmup_no_model")
        assert WarmStartProtocol.is_warmup_active("coder", store, None) is False

    def test_inactive_after_enough_scoring(self, tmp_path):
        store = _make_store_in(tmp_path, "warmup_complete")
        import sqlite3

        # Store and mark 50 memories as scored (update_count > 0)
        for i in range(50):
            mid = _store_routing_memory(store, "coder", "model-B", seed=i)
            with sqlite3.connect(store.sqlite_path) as conn:
                conn.execute(
                    "UPDATE memories SET update_count = 1 WHERE id = ?", (mid,)
                )
                conn.commit()

        assert WarmStartProtocol.is_warmup_active("coder", store, "model-B") is False


# ---------------------------------------------------------------------------
# Warmup scoring config tests
# ---------------------------------------------------------------------------

class TestWarmupScoringConfig:
    def test_doubled_learning_rate(self):
        base = ScoringConfig(learning_rate=0.1)
        warmup = WarmStartProtocol.get_warmup_scoring_config(base)
        assert warmup.learning_rate == pytest.approx(0.2)
        # Other fields preserved
        assert warmup.success_reward == base.success_reward
        assert warmup.failure_reward == base.failure_reward
        assert warmup.cost_penalty_lambda == base.cost_penalty_lambda
