"""Tests for runtime risk-gate behavior in HybridRouter/TwoPhaseRetriever."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.retriever import (
    HybridRouter,
    RetrievalConfig,
    RetrievalResult,
    RuleBasedRouter,
    TwoPhaseRetriever,
)


def _result(action: str, q_conf: float, q_value: float = 0.8) -> RetrievalResult:
    mem = SimpleNamespace(
        action=action,
        q_value=q_value,
        context={"elapsed_seconds": 2.0},
        update_count=5,
    )
    return RetrievalResult(
        memory=mem,
        similarity=0.9,
        q_value=q_value,
        combined_score=q_value,
        q_confidence=q_conf,
        selection_score=q_value,
        p_warm=0.2,
        warm_cost_s=1.0,
        cold_cost_s=3.0,
        expected_cost_s=2.6,
    )


def _retriever(config: RetrievalConfig) -> TwoPhaseRetriever:
    store = MagicMock()
    embedder = MagicMock()
    embedder.embed_task_ir.return_value = [0.0] * 8
    return TwoPhaseRetriever(store=store, embedder=embedder, config=config)


def test_evaluate_risk_gate_disabled():
    retriever = _retriever(RetrievalConfig(risk_control_enabled=False))
    gate = retriever.evaluate_risk_gate([_result("coder_escalation", q_conf=0.1)])
    assert gate["enforced"] is False
    assert gate["action"] == "not_enforced"


def test_evaluate_risk_gate_abstains_when_confidence_below_threshold():
    retriever = _retriever(
        RetrievalConfig(
            risk_control_enabled=True,
            calibrated_confidence_threshold=0.7,
            conformal_margin=0.05,
            risk_gate_min_samples=1,
        )
    )
    gate = retriever.evaluate_risk_gate([_result("coder_escalation", q_conf=0.6)])
    assert gate["enforced"] is True
    assert gate["passed"] is False
    assert gate["action"] == "abstain_escalate"


def test_hybrid_router_risk_abstain_escalates_to_target_role():
    retriever = _retriever(
        RetrievalConfig(
            risk_control_enabled=True,
            calibrated_confidence_threshold=0.8,
            risk_gate_min_samples=1,
            risk_abstain_target_role="architect_general",
        )
    )
    retriever.retrieve_for_routing = MagicMock(return_value=[_result("coder_escalation", q_conf=0.4)])

    rule_router = RuleBasedRouter(routing_hints=[])
    hybrid = HybridRouter(retriever=retriever, rule_based_router=rule_router)
    routing, strategy = hybrid.route({"task_type": "chat", "objective": "hard task"})

    assert strategy == "risk_abstain_escalate"
    assert routing == ["architect_general"]
    assert hybrid.last_decision_meta.get("risk_gate_action") == "abstain_escalate"


def test_hybrid_router_allows_learned_when_risk_gate_passes():
    retriever = _retriever(
        RetrievalConfig(
            risk_control_enabled=True,
            calibrated_confidence_threshold=0.6,
            risk_gate_min_samples=1,
        )
    )
    retriever.retrieve_for_routing = MagicMock(return_value=[_result("coder_escalation", q_conf=0.9)])

    rule_router = RuleBasedRouter(routing_hints=[])
    hybrid = HybridRouter(retriever=retriever, rule_based_router=rule_router)
    routing, strategy = hybrid.route({"task_type": "chat", "objective": "simple task"})

    assert strategy in {"learned", "rules"}
    assert strategy != "risk_abstain_escalate"
    assert hybrid.last_decision_meta.get("risk_gate_action") == "accept"
    assert routing


def test_evaluate_risk_gate_respects_rollout_sampling():
    retriever = _retriever(
        RetrievalConfig(
            risk_control_enabled=True,
            risk_gate_rollout_ratio=0.0,
            risk_gate_min_samples=1,
        )
    )
    gate = retriever.evaluate_risk_gate([_result("coder_escalation", q_conf=0.1)], route_key="abc")
    assert gate["enforced"] is False
    assert gate["reason"] == "rollout_sampling_excluded"


def test_evaluate_risk_gate_guardrail_disables_strict_gate():
    retriever = _retriever(
        RetrievalConfig(
            risk_control_enabled=True,
            risk_gate_rollout_ratio=1.0,
            risk_gate_min_samples=1,
            risk_budget_guardrail_min_events=1,
            risk_budget_guardrail_max_abstain_rate=0.0,
        )
    )
    # First evaluation abstains and records budget stats.
    retriever.evaluate_risk_gate([_result("coder_escalation", q_conf=0.1)], route_key="k1")
    # Subsequent evaluations should be blocked by guardrail.
    gate2 = retriever.evaluate_risk_gate([_result("coder_escalation", q_conf=0.9)], route_key="k2")
    assert gate2["enforced"] is False
    assert gate2["reason"] == "budget_guardrail_abstain_rate"


def test_hybrid_router_prior_blend_can_flip_to_prior_favored_action():
    retriever = _retriever(
        RetrievalConfig(
            risk_control_enabled=False,
            confidence_threshold=0.5,
            prior_strength=0.25,
        )
    )
    r_coder = _result("coder_escalation", q_conf=0.9, q_value=0.8)
    r_frontdoor = _result("frontdoor", q_conf=0.9, q_value=0.75)
    r_coder.selection_score = 0.80
    r_frontdoor.selection_score = 0.75
    retriever.retrieve_for_routing = MagicMock(return_value=[r_coder, r_frontdoor])
    retriever.should_use_learned = MagicMock(return_value=True)

    rule_router = RuleBasedRouter(routing_hints=[])
    hybrid = HybridRouter(retriever=retriever, rule_based_router=rule_router)
    routing, strategy = hybrid.route(
        {"task_type": "chat", "objective": "test"},
        priors={"frontdoor": 1.0, "coder_escalation": 0.0},
    )

    assert strategy == "learned"
    assert routing[0] == "frontdoor"
    assert hybrid.last_decision_meta.get("prior_term_topk")
