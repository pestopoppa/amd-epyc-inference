"""Reward computation, escalation chain detection, and reward injection.

Imports only seeding_types — no other project modules.
"""

from __future__ import annotations

from typing import Any

from seeding_types import (
    ComparativeResult,
    ESCALATION_REWARD,
    ROLE_COST_TIER,
    RoleResult,
)

__all__ = [
    "DEFAULT_BASELINE_TPS",
    "compute_comparative_rewards",
    "detect_escalation_chains",
]

# Default per-role optimized tokens/second from production benchmarks.
# Update these when swapping models in the orchestrator stack.
DEFAULT_BASELINE_TPS: dict[str, float] = {
    "frontdoor": 18.3,
    "coder_primary": 18.3,
    "coder_escalation": 39.44,
    "architect_general": 6.75,
    "architect_coding": 10.3,
    "ingest_long_context": 6.29,
    "worker_explore": 27.88,
    "worker_math": 48.5,
    "worker_vision": 15.28,
    "vision_escalation": 27.6,
}


def compute_comparative_rewards(
    role_results: dict[str, RoleResult],
    baseline_key: str = "frontdoor:direct",
    cost_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute comparative rewards relative to the baseline.

    Reward scheme (xRouter-style, correctness-gated cost penalty):
      specialist correct & baseline wrong -> +1.0 (clear specialist win)
      specialist wrong & baseline right   -> -0.5 (specialist worse)
      both correct -> 0.5 - lambda * max(0, cost_ratio - 1.0)  (cost-aware)
      both wrong   -> -0.3 (neither helps)
    """
    cost_config = cost_config or {}
    lam = cost_config.get("lambda", 0.15)
    baseline_tps = cost_config.get("baseline_tps_by_role", DEFAULT_BASELINE_TPS)

    rewards: dict[str, float] = {}
    baseline = role_results.get(baseline_key)
    if baseline is None:
        for key, result in role_results.items():
            rewards[key] = 1.0 if result.passed else 0.0
        return rewards

    baseline_passed = baseline.passed

    for key, result in role_results.items():
        if key == baseline_key:
            rewards[key] = 1.0 if result.passed else 0.0
        elif result.passed and not baseline_passed:
            rewards[key] = 1.0
        elif not result.passed and baseline_passed:
            rewards[key] = -0.5
        elif result.passed and baseline_passed:
            base = 0.5
            role_tps = baseline_tps.get(result.role, 0)
            gen_elapsed = result.generation_ms / 1000.0 if result.generation_ms > 0 else 0
            actual_elapsed = gen_elapsed if gen_elapsed > 0 else result.elapsed_seconds
            if (role_tps > 0 and result.tokens_generated > 0
                    and actual_elapsed > 0):
                expected = result.tokens_generated / role_tps
                cost_ratio = actual_elapsed / expected
                cost_penalty = lam * max(0.0, cost_ratio - 1.0)
                rewards[key] = max(0.1, base - cost_penalty)
            else:
                rewards[key] = 0.3
        else:
            rewards[key] = -0.3

    return rewards


def detect_escalation_chains(
    role_results: dict[str, RoleResult],
) -> list[dict[str, Any]]:
    """Detect cases where a cheap model fails but a more expensive one passes.

    Returns list of escalation chain dicts:
      {"from_role": "worker_explore", "from_mode": "direct",
       "to_role": "coder_escalation", "to_mode": "direct",
       "action": "escalate:worker_explore->coder_escalation",
       "reward": 0.8}
    """
    chains: list[dict[str, Any]] = []
    entries = []
    for key, rr in role_results.items():
        role, mode = key.split(":", 1)
        tier = ROLE_COST_TIER.get(role, 99)
        entries.append((tier, role, mode, rr))

    entries.sort(key=lambda x: x[0])

    # For each failed cheap role, find the cheapest passing expensive role
    for i, (tier_i, role_i, mode_i, rr_i) in enumerate(entries):
        if rr_i.passed or rr_i.error:
            continue  # Only look at failures (not errors)
        for j in range(i + 1, len(entries)):
            tier_j, role_j, mode_j, rr_j = entries[j]
            if tier_j <= tier_i:
                continue
            if rr_j.passed:
                chains.append({
                    "from_role": role_i,
                    "from_mode": mode_i,
                    "to_role": role_j,
                    "to_mode": mode_j,
                    "action": f"escalate:{role_i}->{role_j}",
                    "reward": ESCALATION_REWARD,
                })
                break  # Only the cheapest passing escalation target

    return chains


def _inject_escalation_chains_http(
    comp: ComparativeResult,
    chains: list[dict[str, Any]],
    url: str,
    client: "Any",
) -> int:
    """Inject escalation chain rewards via HTTP API.

    Returns number of rewards successfully injected.
    """
    injected = 0
    for chain in chains:
        try:
            resp = client.post(
                f"{url}/chat/reward",
                json={
                    "task_description": comp.prompt[:200],
                    "action": chain["action"],
                    "reward": chain["reward"],
                    "context": {
                        "task_type": comp.suite,
                        "source": "escalation_chain",
                        "question_id": comp.question_id,
                        "action_type": "escalation",
                        "from_role": chain["from_role"],
                        "to_role": chain["to_role"],
                    },
                },
                timeout=10,
            )
            if resp.status_code == 200:
                injected += 1
        except Exception:
            continue
    return injected


def _inject_rewards_http(
    comp: ComparativeResult,
    url: str,
    client: "Any",
) -> int:
    """Inject comparative rewards for one question via HTTP API.

    Returns number of rewards successfully injected.
    """
    injected = 0
    for action_key, reward in comp.rewards.items():
        try:
            resp = client.post(
                f"{url}/chat/reward",
                json={
                    "task_description": comp.prompt[:200],
                    "action": action_key,
                    "reward": reward,
                    "context": {
                        "task_type": comp.suite,
                        "source": "comparative_seeding",
                        "question_id": comp.question_id,
                        "comparative": True,
                    },
                },
                timeout=10,
            )
            if resp.status_code == 200:
                injected += 1
        except Exception:
            continue
    return injected
