#!/usr/bin/env python3
"""Comparative Specialist Seeding for MemRL.

Runs each debug question through MULTIPLE specialists, scores deterministically,
and injects comparative rewards to bootstrap Q-values for specialist routing.

This solves the cold-start problem: without specialist experience, all Q-values
are 0.5 and routing never activates. Seeding generates ground truth per task type.

Comparative reward scheme:
  specialist correct & frontdoor wrong → +1.0 (specialist clearly better)
  specialist wrong & frontdoor right   → -0.5 (specialist worse)
  both correct                         → +0.3 (speed parity)
  both wrong                           → -0.3 (neither helps)

Usage:
    # Dry run (score without reward injection)
    python scripts/benchmark/seed_specialist_routing.py --dry-run --suites thinking coder

    # Full seeding run
    ORCHESTRATOR_SPECIALIST_ROUTING=1 \\
      python scripts/benchmark/seed_specialist_routing.py --suites all --sample-size 10

    # Specific roles only
    python scripts/benchmark/seed_specialist_routing.py --roles frontdoor coder_primary architect_general
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_ORCHESTRATOR_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120
DEFAULT_SUITES = [
    "thinking", "general", "math", "agentic",
    "coder", "instruction_precision",
]
DEFAULT_ROLES = ["frontdoor", "coder_primary", "coder_escalation", "architect_general"]


@dataclass
class RoleResult:
    """Result of running a question through a specific role."""

    role: str
    answer: str
    passed: bool
    elapsed_seconds: float
    error: str | None = None


@dataclass
class ComparativeResult:
    """Comparative result across roles for a single question."""

    suite: str
    question_id: str
    prompt: str
    expected: str
    role_results: dict[str, RoleResult] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)


def call_orchestrator_forced(
    prompt: str,
    force_role: str,
    url: str = DEFAULT_ORCHESTRATOR_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call orchestrator with forced role routing.

    Args:
        prompt: The question to send.
        force_role: Role to force (bypasses routing).
        url: Orchestrator API URL.
        timeout: Request timeout in seconds.

    Returns:
        Response dict.
    """
    import httpx

    try:
        response = httpx.post(
            f"{url}/chat",
            json={
                "prompt": prompt,
                "real_mode": True,
                "force_role": force_role,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": "", "error": str(e)}


def score_answer_deterministic(
    answer: str,
    expected: str,
    scoring_method: str = "exact_match",
    scoring_config: dict[str, Any] | None = None,
) -> bool:
    """Score an answer deterministically.

    Args:
        answer: Model's answer.
        expected: Expected answer.
        scoring_method: Scoring method name (e.g. multiple_choice, exact_match).
        scoring_config: Optional config dict for the scorer.

    Returns:
        True if answer is correct.
    """
    from benchmark.debug_scorer import score_answer

    return score_answer(answer, expected, scoring_method, scoring_config or {})


def compute_comparative_rewards(
    role_results: dict[str, RoleResult],
    baseline_role: str = "frontdoor",
) -> dict[str, float]:
    """Compute comparative rewards relative to the baseline (frontdoor).

    Reward scheme:
      specialist correct & baseline wrong → +1.0
      specialist wrong & baseline right   → -0.5
      both correct                        → +0.3
      both wrong                          → -0.3

    Args:
        role_results: Results per role.
        baseline_role: The baseline to compare against.

    Returns:
        Dict of role -> reward.
    """
    rewards = {}
    baseline = role_results.get(baseline_role)
    if baseline is None:
        # No baseline — use absolute scoring
        for role, result in role_results.items():
            rewards[role] = 1.0 if result.passed else -0.5
        return rewards

    baseline_passed = baseline.passed

    for role, result in role_results.items():
        if role == baseline_role:
            # Baseline always gets absolute reward
            rewards[role] = 1.0 if result.passed else -0.5
        elif result.passed and not baseline_passed:
            # Specialist correct, baseline wrong — strong positive
            rewards[role] = 1.0
        elif not result.passed and baseline_passed:
            # Specialist wrong, baseline correct — negative
            rewards[role] = -0.5
        elif result.passed and baseline_passed:
            # Both correct — mild positive (speed parity)
            rewards[role] = 0.3
        else:
            # Both wrong — mild negative
            rewards[role] = -0.3

    return rewards


def run_seeding(
    suites: list[str],
    roles: list[str],
    sample_per_suite: int,
    seed: int,
    url: str,
    timeout: int,
    dry_run: bool = False,
) -> list[ComparativeResult]:
    """Run comparative seeding across all suites and roles.

    Args:
        suites: Suites to run.
        roles: Roles to compare.
        sample_per_suite: Questions per suite.
        seed: RNG seed.
        url: Orchestrator URL.
        timeout: Request timeout.
        dry_run: If True, score but don't inject rewards.

    Returns:
        List of ComparativeResult.
    """
    import yaml

    DEBUG_PROMPTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "debug"

    rng = random.Random(seed)
    all_results: list[ComparativeResult] = []

    # Load prompts from debug suite (deterministic scoring)
    all_prompts: list[dict[str, Any]] = []
    suite_names = DEFAULT_SUITES if suites == ["all"] else suites

    for suite_name in suite_names:
        yaml_path = DEBUG_PROMPTS_DIR / f"{suite_name}.yaml"
        if not yaml_path.exists():
            logger.warning(f"No debug suite for '{suite_name}' at {yaml_path}")
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        questions = data.get("questions", [])
        if not questions:
            logger.warning(f"Empty debug suite: {suite_name}")
            continue

        if len(questions) > sample_per_suite:
            questions = rng.sample(questions, sample_per_suite)

        for q in questions:
            all_prompts.append({
                "suite": suite_name,
                "id": q["id"],
                "prompt": q["prompt"].strip(),
                "expected": q.get("expected", ""),
                "scoring_method": q.get("scoring_method", "exact_match"),
                "scoring_config": q.get("scoring_config", {}),
            })

    logger.info(f"Loaded {len(all_prompts)} questions across {len(suite_names)} suites")
    logger.info(f"Testing {len(roles)} roles: {', '.join(roles)}")
    logger.info(f"Total API calls: {len(all_prompts) * len(roles)}")

    # Architect models are slower — give them more time
    SLOW_ROLES = {"architect_general", "architect_coding"}
    SLOW_ROLE_TIMEOUT = max(timeout, 300)

    for i, prompt_info in enumerate(all_prompts):
        suite = prompt_info["suite"]
        qid = prompt_info["id"]
        prompt = prompt_info["prompt"]
        expected = prompt_info["expected"]
        scoring_method = prompt_info["scoring_method"]
        scoring_config = prompt_info["scoring_config"]

        logger.info(f"[{i+1}/{len(all_prompts)}] {suite}/{qid}")

        role_results: dict[str, RoleResult] = {}

        for role in roles:
            role_timeout = SLOW_ROLE_TIMEOUT if role in SLOW_ROLES else timeout
            q_start = time.perf_counter()
            response = call_orchestrator_forced(prompt, role, url, role_timeout)
            q_elapsed = time.perf_counter() - q_start

            answer = response.get("answer", "")
            error = response.get("error")

            if error:
                passed = False
            else:
                passed = score_answer_deterministic(answer, expected, scoring_method, scoring_config)

            role_results[role] = RoleResult(
                role=role,
                answer=answer[:500] if answer else "",
                passed=passed,
                elapsed_seconds=q_elapsed,
                error=error,
            )

            status = "PASS" if passed else ("ERROR" if error else "FAIL")
            logger.info(f"  {role:25s} → {status} ({q_elapsed:.1f}s)")

        # Compute comparative rewards
        rewards = compute_comparative_rewards(role_results)

        comp = ComparativeResult(
            suite=suite,
            question_id=qid,
            prompt=prompt[:200],
            expected=expected[:200],
            role_results=role_results,
            rewards=rewards,
        )
        all_results.append(comp)

        # Log comparative rewards
        for role, reward in sorted(rewards.items()):
            logger.info(f"    reward[{role}] = {reward:+.1f}")

    # Inject rewards into MemRL
    if not dry_run:
        injected = _inject_comparative_rewards(all_results, url)
        logger.info(f"\nInjected {injected} comparative rewards into MemRL")
    else:
        logger.info("\n[DRY RUN] Skipping reward injection")

    return all_results


def _inject_comparative_rewards(
    results: list[ComparativeResult],
    url: str,
) -> int:
    """Inject comparative rewards into MemRL.

    Args:
        results: Comparative results.
        url: Orchestrator URL.

    Returns:
        Number of rewards injected.
    """
    injected = 0

    # Try direct Python API
    try:
        from src.api.services.memrl import store_external_reward
        from src.api.state import get_state
        state = get_state()

        for comp in results:
            for role, reward in comp.rewards.items():
                action = f"{role}:direct"
                context = {
                    "task_type": comp.suite,
                    "source": "comparative_seeding",
                    "question_id": comp.question_id,
                    "comparative": True,
                }
                success = store_external_reward(
                    state=state,
                    task_description=comp.prompt[:200],
                    action=action,
                    reward=reward,
                    context=context,
                )
                if success:
                    injected += 1
        return injected
    except Exception as e:
        logger.warning(f"Direct API injection failed: {e}")

    # Fallback: HTTP API
    try:
        import httpx

        for comp in results:
            for role, reward in comp.rewards.items():
                action = f"{role}:direct"
                try:
                    resp = httpx.post(
                        f"{url}/memrl/reward",
                        json={
                            "task_description": comp.prompt[:200],
                            "action": action,
                            "reward": reward,
                            "context": {
                                "task_type": comp.suite,
                                "source": "comparative_seeding",
                                "question_id": comp.question_id,
                            },
                        },
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        injected += 1
                except Exception:
                    continue
    except ImportError:
        logger.error("httpx not available for HTTP injection")

    return injected


def print_seeding_summary(results: list[ComparativeResult], roles: list[str]) -> None:
    """Print summary of seeding results."""
    # Per-role accuracy
    role_stats: dict[str, dict[str, int]] = {r: {"pass": 0, "fail": 0, "error": 0} for r in roles}
    suite_stats: dict[str, dict[str, dict[str, int]]] = {}

    for comp in results:
        suite = comp.suite
        if suite not in suite_stats:
            suite_stats[suite] = {r: {"pass": 0, "fail": 0} for r in roles}

        for role, rr in comp.role_results.items():
            if role not in role_stats:
                continue
            if rr.error:
                role_stats[role]["error"] += 1
            elif rr.passed:
                role_stats[role]["pass"] += 1
                suite_stats.get(suite, {}).get(role, {})["pass"] = \
                    suite_stats.get(suite, {}).get(role, {}).get("pass", 0) + 1
            else:
                role_stats[role]["fail"] += 1
                suite_stats.get(suite, {}).get(role, {})["fail"] = \
                    suite_stats.get(suite, {}).get(role, {}).get("fail", 0) + 1

    print(f"\n{'='*70}")
    print("COMPARATIVE SEEDING SUMMARY")
    print(f"{'='*70}")
    print(f"Questions: {len(results)}")
    print(f"Roles tested: {', '.join(roles)}")

    print(f"\n{'Role':25s} {'Pass':>6s} {'Fail':>6s} {'Error':>6s} {'Accuracy':>10s}")
    print("-" * 55)
    for role in roles:
        s = role_stats[role]
        total = s["pass"] + s["fail"]
        acc = s["pass"] / total * 100 if total > 0 else 0
        print(f"{role:25s} {s['pass']:6d} {s['fail']:6d} {s['error']:6d} {acc:9.1f}%")

    # Per-suite breakdown
    if len(suite_stats) > 1:
        print(f"\nPer-suite breakdown (accuracy %):")
        header = f"{'Suite':20s}" + "".join(f" {r:>15s}" for r in roles)
        print(header)
        print("-" * len(header))
        for suite in sorted(suite_stats.keys()):
            row = f"{suite:20s}"
            for role in roles:
                rs = suite_stats[suite].get(role, {"pass": 0, "fail": 0})
                total = rs["pass"] + rs["fail"]
                acc = rs["pass"] / total * 100 if total > 0 else 0
                row += f" {acc:14.1f}%"
            print(row)

    # Reward distribution
    reward_totals: dict[str, float] = {r: 0.0 for r in roles}
    for comp in results:
        for role, reward in comp.rewards.items():
            if role in reward_totals:
                reward_totals[role] += reward

    print(f"\nCumulative rewards:")
    for role in roles:
        print(f"  {role:25s} {reward_totals[role]:+.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Comparative Specialist Seeding for MemRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suites", nargs="+", default=DEFAULT_SUITES,
        help=f"Suites to seed (default: {' '.join(DEFAULT_SUITES)})",
    )
    parser.add_argument(
        "--roles", nargs="+", default=DEFAULT_ROLES,
        help=f"Roles to compare (default: {' '.join(DEFAULT_ROLES)})",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="Questions per suite (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--url", default=DEFAULT_ORCHESTRATOR_URL,
        help=f"Orchestrator URL (default: {DEFAULT_ORCHESTRATOR_URL})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Request timeout (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score only, don't inject rewards",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file (default: auto-generated)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Comparative Specialist Seeding")
    logger.info(f"  Suites: {args.suites}")
    logger.info(f"  Roles: {args.roles}")
    logger.info(f"  Sample size: {args.sample_size}/suite")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info("=" * 60)

    results = run_seeding(
        suites=args.suites,
        roles=args.roles,
        sample_per_suite=args.sample_size,
        seed=args.seed,
        url=args.url,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )

    print_seeding_summary(results, args.roles)

    # Save results
    output_path = args.output
    if output_path is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = str(
            PROJECT_ROOT / "benchmarks" / "results" / "orchestrator"
            / f"seeding_{ts}.json"
        )

    output_data = {
        "config": {
            "suites": args.suites,
            "roles": args.roles,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "dry_run": args.dry_run,
        },
        "results": [asdict(r) for r in results],
        "timestamp": datetime.utcnow().isoformat(),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
