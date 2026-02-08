#!/usr/bin/env python3
"""Tests for seeding_rewards — pure reward computation functions."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "benchmark"))

from seeding_types import (
    ACTION_ARCHITECT,
    ACTION_SELF_DIRECT,
    ACTION_SELF_REPL,
    ACTION_WORKER,
    ESCALATION_REWARD,
    RoleResult,
)
from seeding_rewards import (
    compute_comparative_rewards,
    compute_tool_value,
    detect_escalation_chains,
    score_delegation_chain,
    success_reward,
    compute_3way_rewards,
)


def _rr(role: str, mode: str, passed: bool, **kw) -> RoleResult:
    """Helper to build minimal RoleResult."""
    return RoleResult(
        role=role, mode=mode, answer="ans" if passed else "",
        passed=passed, elapsed_seconds=1.0, **kw,
    )


# ---------------------------------------------------------------------------
# success_reward
# ---------------------------------------------------------------------------

class TestSuccessReward:
    def test_pass_returns_one(self):
        assert success_reward(True) == 1.0

    def test_fail_returns_zero(self):
        assert success_reward(False) == 0.0


# ---------------------------------------------------------------------------
# compute_comparative_rewards
# ---------------------------------------------------------------------------

class TestComparativeRewards:
    def test_baseline_gets_binary(self):
        results = {
            "frontdoor:direct": _rr("frontdoor", "direct", True),
            "coder:direct": _rr("coder", "direct", True),
        }
        rewards = compute_comparative_rewards(results, baseline_key="frontdoor:direct")
        assert rewards["frontdoor:direct"] == 1.0

    def test_specialist_correct_baseline_wrong(self):
        results = {
            "frontdoor:direct": _rr("frontdoor", "direct", False),
            "coder:direct": _rr("coder", "direct", True),
        }
        rewards = compute_comparative_rewards(results, baseline_key="frontdoor:direct")
        assert rewards["coder:direct"] == 1.0

    def test_specialist_wrong_baseline_correct(self):
        results = {
            "frontdoor:direct": _rr("frontdoor", "direct", True),
            "coder:direct": _rr("coder", "direct", False),
        }
        rewards = compute_comparative_rewards(results, baseline_key="frontdoor:direct")
        assert rewards["coder:direct"] == -0.5

    def test_both_wrong(self):
        results = {
            "frontdoor:direct": _rr("frontdoor", "direct", False),
            "coder:direct": _rr("coder", "direct", False),
        }
        rewards = compute_comparative_rewards(results, baseline_key="frontdoor:direct")
        assert rewards["coder:direct"] == -0.3

    def test_missing_baseline_binary_fallback(self):
        results = {"coder:direct": _rr("coder", "direct", True)}
        rewards = compute_comparative_rewards(results, baseline_key="nonexistent")
        assert rewards["coder:direct"] == 1.0

    def test_keys_match_input(self):
        results = {
            "frontdoor:direct": _rr("frontdoor", "direct", True),
            "arch:delegated": _rr("arch", "delegated", True),
        }
        rewards = compute_comparative_rewards(results, baseline_key="frontdoor:direct")
        assert set(rewards.keys()) == set(results.keys())


# ---------------------------------------------------------------------------
# detect_escalation_chains
# ---------------------------------------------------------------------------

class TestEscalationChains:
    def test_cheap_fail_expensive_pass(self):
        results = {
            "worker_explore:direct": _rr("worker_explore", "direct", False),
            "coder_escalation:direct": _rr("coder_escalation", "direct", True),
        }
        chains = detect_escalation_chains(results)
        assert len(chains) == 1
        assert chains[0]["from_role"] == "worker_explore"
        assert chains[0]["to_role"] == "coder_escalation"
        assert chains[0]["reward"] == ESCALATION_REWARD

    def test_no_chain_when_both_pass(self):
        results = {
            "worker_explore:direct": _rr("worker_explore", "direct", True),
            "coder_escalation:direct": _rr("coder_escalation", "direct", True),
        }
        chains = detect_escalation_chains(results)
        assert len(chains) == 0

    def test_no_chain_when_both_fail(self):
        results = {
            "worker_explore:direct": _rr("worker_explore", "direct", False),
            "coder_escalation:direct": _rr("coder_escalation", "direct", False),
        }
        chains = detect_escalation_chains(results)
        assert len(chains) == 0

    def test_skip_error_entries(self):
        results = {
            "worker_explore:direct": _rr("worker_explore", "direct", False, error="timeout"),
            "coder_escalation:direct": _rr("coder_escalation", "direct", True),
        }
        chains = detect_escalation_chains(results)
        # Error entries are skipped (rr.error is truthy)
        assert len(chains) == 0


# ---------------------------------------------------------------------------
# compute_tool_value
# ---------------------------------------------------------------------------

class TestComputeToolValue:
    def test_tools_helped(self):
        tv = compute_tool_value(direct_passed=False, repl_passed=True)
        assert tv["tools_helped"] is True
        assert tv["tools_neutral"] is False
        assert tv["tools_hurt"] is False
        assert tv["tool_advantage"] == 1

    def test_tools_hurt(self):
        tv = compute_tool_value(direct_passed=True, repl_passed=False)
        assert tv["tools_helped"] is False
        assert tv["tools_hurt"] is True
        assert tv["tool_advantage"] == -1

    def test_tools_neutral_both_pass(self):
        tv = compute_tool_value(direct_passed=True, repl_passed=True)
        assert tv["tools_neutral"] is True
        assert tv["tool_advantage"] == 0

    def test_tools_neutral_both_fail(self):
        tv = compute_tool_value(direct_passed=False, repl_passed=False)
        assert tv["tools_neutral"] is True
        assert tv["tool_advantage"] == 0


# ---------------------------------------------------------------------------
# score_delegation_chain
# ---------------------------------------------------------------------------

class TestScoreDelegationChain:
    def test_no_delegation_no_worker_reward(self):
        results = {
            "frontdoor:direct": _rr("frontdoor", "direct", True),
            "frontdoor:repl": _rr("frontdoor", "repl", True),
        }
        rewards = score_delegation_chain(results)
        assert ACTION_WORKER not in rewards

    def test_delegation_via_events(self):
        rr = _rr("frontdoor", "repl", True, delegation_events=[{"worker": "explore"}])
        results = {"frontdoor:repl": rr}
        rewards = score_delegation_chain(results)
        assert ACTION_WORKER in rewards
        assert rewards[ACTION_WORKER] == 1.0

    def test_delegation_via_role_history(self):
        rr = _rr("architect_general", "delegated", True,
                  role_history=["architect_general", "worker_explore"])
        results = {"architect_general:delegated": rr}
        rewards = score_delegation_chain(results)
        assert ACTION_WORKER in rewards

    def test_infra_error_skipped(self):
        rr = _rr("frontdoor", "repl", False,
                  error_type="infrastructure",
                  delegation_events=[{"worker": "x"}])
        results = {"frontdoor:repl": rr}
        rewards = score_delegation_chain(results)
        assert ACTION_WORKER not in rewards


# ---------------------------------------------------------------------------
# compute_3way_rewards
# ---------------------------------------------------------------------------

class TestCompute3wayRewards:
    def test_all_pass(self):
        results = {
            "frontdoor:direct": _rr("frontdoor", "direct", True),
            "frontdoor:repl": _rr("frontdoor", "repl", True),
            "architect_general:delegated": _rr("architect_general", "delegated", True),
        }
        rewards = compute_3way_rewards(results)
        assert rewards[ACTION_SELF_DIRECT] == 1.0
        assert rewards[ACTION_SELF_REPL] == 1.0
        assert rewards[ACTION_ARCHITECT] == 1.0

    def test_all_fail(self):
        results = {
            "frontdoor:direct": _rr("frontdoor", "direct", False),
            "frontdoor:repl": _rr("frontdoor", "repl", False),
            "architect_coding:delegated": _rr("architect_coding", "delegated", False),
        }
        rewards = compute_3way_rewards(results)
        assert rewards[ACTION_SELF_DIRECT] == 0.0
        assert rewards[ACTION_SELF_REPL] == 0.0
        assert rewards[ACTION_ARCHITECT] == 0.0

    def test_best_architect_wins(self):
        results = {
            "architect_general:delegated": _rr("architect_general", "delegated", False),
            "architect_coding:delegated": _rr("architect_coding", "delegated", True),
        }
        rewards = compute_3way_rewards(results)
        assert rewards[ACTION_ARCHITECT] == 1.0
