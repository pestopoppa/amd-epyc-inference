"""Unit tests for delegation telemetry and attribution."""

import sys

sys.path.insert(0, "scripts/benchmark")


def test_score_delegation_chain_prefers_canonical_success():
    from seeding_rewards import score_delegation_chain
    from seeding_types import RoleResult, ACTION_WORKER

    results = {
        "frontdoor:repl": RoleResult(
            role="frontdoor",
            mode="repl",
            answer="ok",
            passed=True,
            elapsed_seconds=1.0,
            delegation_events=[{"from_role": "frontdoor", "to_role": "worker_general"}],
            delegation_success=False,
        )
    }

    rewards = score_delegation_chain(results)
    assert rewards[ACTION_WORKER] == 0.0


def test_score_delegation_chain_fallbacks_to_passed():
    from seeding_rewards import score_delegation_chain
    from seeding_types import RoleResult, ACTION_WORKER

    results = {
        "frontdoor:repl": RoleResult(
            role="frontdoor",
            mode="repl",
            answer="ok",
            passed=True,
            elapsed_seconds=1.0,
            tools_called=["delegate_to_worker"],
        )
    }

    rewards = score_delegation_chain(results)
    assert rewards[ACTION_WORKER] == 1.0
