# Chapter 28: Calibration and Risk Control

This chapter documents calibration-aware routing controls added to the MemRL decision loop.

## Scope

The routing stack now supports:

1. Robust confidence estimation from neighbor Q-values.
2. Calibrated confidence thresholds.
3. Conformal-style safety margin for abstain/escalate behavior.
4. Replay-time calibration metrics for ongoing validation.

## Runtime Controls

`RetrievalConfig` controls:

- `confidence_threshold`
- `calibrated_confidence_threshold`
- `conformal_margin`
- `risk_control_enabled`
- `risk_budget_id`
- `risk_gate_min_samples`
- `risk_abstain_target_role`
- `risk_gate_rollout_ratio`
- `risk_gate_kill_switch`
- `risk_budget_guardrail_min_events`
- `risk_budget_guardrail_max_abstain_rate`
- `confidence_estimator` (`median` or `trimmed_mean`)
- `confidence_trim_ratio`
- `confidence_min_neighbors`

Effective routing threshold:

`effective_threshold = (calibrated or base threshold) + conformal_margin`

When confidence is below this threshold under strict gate enforcement, hybrid routing
emits `risk_abstain_escalate` and routes to `risk_abstain_target_role`.
Gate provenance is logged with:

- `risk_gate_action`
- `risk_gate_reason`
- `risk_budget_id`

Rollout/guardrail controls:

- deterministic rollout sampling by route key (`risk_gate_rollout_ratio`)
- emergency kill switch (`risk_gate_kill_switch`)
- budget guardrail to auto-disable strict gating if abstain rate exceeds configured budget

## Metrics

Replay now emits:

- `ece_global`
- `brier_global`
- `conformal_coverage`
- `conformal_risk`

These are computed in:

- `orchestration/repl_memory/replay/engine.py`
- `orchestration/repl_memory/replay/metrics.py`

## Operational Workflow

1. Run replay on recent trajectories with baseline config.
2. Run replay with candidate calibration/risk settings.
3. Compare quality/cost/calibration metrics.
4. Promote only if risk/coverage targets and utility KPIs pass.

## Related Modules

- `orchestration/repl_memory/retriever.py`
- `orchestration/repl_memory/replay/engine.py`
- `scripts/benchmark/seed_specialist_routing.py`
- `src/pipeline_monitor/claude_debugger.py`

---

*Previous: [Chapter 27: SkillBank & Experience Distillation](27-skillbank-experience-distillation.md)*
