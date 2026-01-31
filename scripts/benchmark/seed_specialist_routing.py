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

    # Full seeding run (all roles × all modes)
    ORCHESTRATOR_SPECIALIST_ROUTING=1 \\
      python scripts/benchmark/seed_specialist_routing.py --suites all --sample-size 10

    # Specific roles only
    python scripts/benchmark/seed_specialist_routing.py --roles frontdoor coder_primary architect_general

    # Direct mode only (legacy behavior)
    python scripts/benchmark/seed_specialist_routing.py --modes direct

    # Test specific modes
    python scripts/benchmark/seed_specialist_routing.py --modes direct react repl
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

# Lazy import — resolved after sys.path setup above
from src.llm_primitives import LLMPrimitives

# Default configuration
DEFAULT_ORCHESTRATOR_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120
DEFAULT_SUITES = [
    "thinking", "general", "math", "agentic",
    "coder", "instruction_precision", "vl",
]
DEFAULT_ROLES = [
    "frontdoor", "coder_primary", "coder_escalation",
    "architect_general", "architect_coding",
    "worker_vision", "vision_escalation",
]
DEFAULT_MODES = ["direct", "react", "repl"]

# Architect roles delegate tool work to faster specialists.
# They support direct (no tools) and delegated (multi-loop delegation) modes.
ARCHITECT_ROLES = {"architect_general", "architect_coding"}
ARCHITECT_MODES = {"direct", "delegated"}

# Vision roles have model-specific mode constraints.
# worker_vision (Qwen2.5-VL-7B) supports tool calls → direct + react.
# vision_escalation (Qwen3-VL-30B-A3B) is 0% agentic → direct only.
VISION_ROLES = {"worker_vision", "vision_escalation"}
VISION_MODES: dict[str, set[str]] = {
    "worker_vision": {"direct", "react"},
    "vision_escalation": {"direct"},
}

# Legacy alias for backwards compatibility
DIRECT_ONLY_ROLES = ARCHITECT_ROLES


@dataclass
class RoleResult:
    """Result of running a question through a specific role+mode."""

    role: str
    mode: str
    answer: str
    passed: bool
    elapsed_seconds: float
    error: str | None = None
    tokens_generated: int = 0
    tools_used: int = 0
    tools_called: list[str] = field(default_factory=list)
    routed_to: str = ""
    role_history: list[str] = field(default_factory=list)
    routing_strategy: str = ""
    turns: int = 0
    tokens_used: int = 0
    formalization_applied: bool = False
    cache_stats: dict[str, Any] | None = None
    # Clean timing data from llama.cpp (excludes prompt eval overhead)
    predicted_tps: float = 0.0
    generation_ms: float = 0.0
    prompt_eval_ms: float = 0.0
    http_overhead_ms: float = 0.0


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
    force_mode: str = "direct",
    url: str = DEFAULT_ORCHESTRATOR_URL,
    timeout: int = DEFAULT_TIMEOUT,
    image_path: str = "",
    cache_prompt: bool | None = None,
) -> dict[str, Any]:
    """Call orchestrator with forced role and mode routing.

    Args:
        prompt: The question to send.
        force_role: Role to force (bypasses routing).
        force_mode: Execution mode to force ('direct', 'react', or 'repl').
        url: Orchestrator API URL.
        timeout: Request timeout in seconds.
        image_path: Optional path to image for VL questions.
        cache_prompt: Override cache_prompt (None=default, False=disable).

    Returns:
        Response dict.
    """
    import httpx

    payload: dict[str, Any] = {
        "prompt": prompt,
        "real_mode": True,
        "force_role": force_role,
        "force_mode": force_mode,
    }
    if image_path:
        payload["image_path"] = image_path
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt

    try:
        response = httpx.post(
            f"{url}/chat",
            json=payload,
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


# Default per-role optimized tokens/second from production benchmarks.
# Mirrors ScoringConfig.baseline_tps_by_role in q_scorer.py.
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

    Keys are 'role:mode' (e.g. 'frontdoor:direct', 'coder_primary:react').

    Reward scheme (xRouter-style, correctness-gated cost penalty):
      specialist correct & baseline wrong → +1.0 (clear specialist win)
      specialist wrong & baseline right   → -0.5 (specialist worse)
      both correct → 0.5 - lambda * max(0, cost_ratio - 1.0)  (cost-aware)
      both wrong   → -0.3 (neither helps)
      incorrect (no baseline) → 0.0 (hard gate)

    Cost normalization:
      cost_ratio = actual_elapsed / expected_elapsed
      expected_elapsed = tokens_generated / baseline_tps[role]
      Penalty only applies when cost_ratio > 1.0 (slower than expected).

    Args:
        role_results: Results per role:mode key.
        baseline_key: The baseline key to compare against.
        cost_config: Optional dict with 'lambda' (float) and
            'baseline_tps_by_role' (dict[str, float]).

    Returns:
        Dict of role:mode -> reward.
    """
    cost_config = cost_config or {}
    lam = cost_config.get("lambda", 0.15)
    baseline_tps = cost_config.get("baseline_tps_by_role", DEFAULT_BASELINE_TPS)

    rewards = {}
    baseline = role_results.get(baseline_key)
    if baseline is None:
        # No baseline — use absolute scoring (correct=1.0, wrong=0.0)
        for key, result in role_results.items():
            rewards[key] = 1.0 if result.passed else 0.0
        return rewards

    baseline_passed = baseline.passed

    for key, result in role_results.items():
        if key == baseline_key:
            # Baseline gets absolute reward
            rewards[key] = 1.0 if result.passed else 0.0
        elif result.passed and not baseline_passed:
            # Specialist correct, baseline wrong — strong positive
            rewards[key] = 1.0
        elif not result.passed and baseline_passed:
            # Specialist wrong, baseline correct — negative
            rewards[key] = -0.5
        elif result.passed and baseline_passed:
            # Both correct — cost-aware reward (xRouter-style)
            # Prefer generation_ms (excludes prompt eval) for clean cost measurement
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
                # No cost data — fall back to flat reward
                rewards[key] = 0.3
        else:
            # Both wrong — mild negative
            rewards[key] = -0.3

    return rewards


def _build_role_mode_combos(
    roles: list[str],
    modes: list[str],
) -> list[tuple[str, str]]:
    """Build (role, mode) combinations respecting role-specific constraints.

    Architect roles get ARCHITECT_MODES (direct + delegated).
    Vision roles get per-model modes from VISION_MODES.
    All other roles get every mode in the modes list.

    Args:
        roles: List of role names.
        modes: List of mode names.

    Returns:
        List of (role, mode) tuples.
    """
    combos = []
    for role in roles:
        if role in ARCHITECT_ROLES:
            for mode in ARCHITECT_MODES:
                combos.append((role, mode))
        elif role in VISION_ROLES:
            for mode in VISION_MODES.get(role, {"direct"}):
                combos.append((role, mode))
        else:
            for mode in modes:
                combos.append((role, mode))
    return combos


def _deduplicate_roles(
    roles: list[str],
    server_urls: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Remove roles that share the same backend URL.

    When two roles (e.g. frontdoor and coder_primary) map to the same
    llama-server endpoint, testing both produces identical results because
    skip_suffix=True in all seeding paths eliminates prompt differences.
    This function detects URL collisions and returns a deduplicated list,
    plus an alias map so rewards can be cloned to the skipped roles.

    Args:
        roles: Ordered list of role names.
        server_urls: URL map (defaults to LLMPrimitives.DEFAULT_SERVER_URLS).

    Returns:
        (unique_roles, alias_map) where alias_map maps each skipped role
        to the canonical role that will be tested in its place.
    """
    urls = server_urls or LLMPrimitives.DEFAULT_SERVER_URLS
    seen: dict[str, str] = {}  # url -> first role using it
    unique: list[str] = []
    aliases: dict[str, str] = {}  # skipped_role -> canonical_role

    for role in roles:
        url = urls.get(role, "")
        if url and url in seen:
            aliases[role] = seen[url]
        else:
            if url:
                seen[url] = role
            unique.append(role)

    return unique, aliases


def _modes_for_role(role: str, modes: list[str]) -> list[str]:
    """Return the effective mode list for a role."""
    if role in ARCHITECT_ROLES:
        return sorted(ARCHITECT_MODES)
    if role in VISION_ROLES:
        return sorted(VISION_MODES.get(role, {"direct"}))
    return list(modes)


def _check_server_health(url: str, timeout: int = 5) -> bool:
    """Check if the orchestrator server is healthy.

    Args:
        url: Orchestrator base URL.
        timeout: Health check timeout.

    Returns:
        True if server is reachable and healthy.
    """
    import httpx

    try:
        resp = httpx.get(f"{url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def run_seeding(
    suites: list[str],
    roles: list[str],
    modes: list[str],
    sample_per_suite: int,
    seed: int,
    url: str,
    timeout: int,
    dry_run: bool = False,
    skip_cache: bool = False,
    cooldown: float = 0.0,
    no_dedup: bool = False,
) -> list[ComparativeResult]:
    """Run comparative seeding across all suites, roles, and modes.

    Args:
        suites: Suites to run.
        roles: Roles to compare.
        modes: Execution modes to test per role.
        sample_per_suite: Questions per suite.
        seed: RNG seed.
        url: Orchestrator URL.
        timeout: Request timeout.
        dry_run: If True, score but don't inject rewards.
        skip_cache: If True, pass cache_prompt=False to disable KV caching.
        cooldown: Seconds to wait between requests.
        no_dedup: If True, disable URL-based role deduplication.

    Returns:
        List of ComparativeResult.
    """
    import yaml

    DEBUG_PROMPTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "debug"

    rng = random.Random(seed)
    all_results: list[ComparativeResult] = []

    # Deduplicate roles sharing the same backend URL (e.g. frontdoor
    # and coder_primary both hit localhost:8080 with identical prompts).
    alias_map: dict[str, str] = {}
    if not no_dedup:
        unique_roles, alias_map = _deduplicate_roles(roles)
        if alias_map:
            for alias, canonical in sorted(alias_map.items()):
                canon_url = LLMPrimitives.DEFAULT_SERVER_URLS.get(canonical, "?")
                logger.info(f"Dedup: {alias} → {canonical} (same backend {canon_url})")
            logger.info(f"Roles after dedup: {unique_roles} (aliases: {alias_map})")
        roles_to_test = unique_roles
    else:
        roles_to_test = list(roles)

    # Build (role, mode) combos — architects get direct only
    combos = _build_role_mode_combos(roles_to_test, modes)
    combo_keys = [f"{r}:{m}" for r, m in combos]

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
                "image_path": q.get("image_path", ""),
            })

    vl_count = sum(1 for p in all_prompts if p.get("image_path"))
    text_count = len(all_prompts) - vl_count
    # VL questions test vision roles + frontdoor; text questions test text roles
    vl_combos = [(r, m) for r, m in combos if r in VISION_ROLES or r == "frontdoor"]
    text_combos = [(r, m) for r, m in combos if r not in VISION_ROLES]
    total_calls = vl_count * len(vl_combos) + text_count * len(text_combos)

    logger.info(f"Loaded {len(all_prompts)} questions across {len(suite_names)} suites ({vl_count} VL, {text_count} text)")
    logger.info(f"Testing {len(combos)} role×mode combos: {', '.join(combo_keys)}")
    logger.info(f"Total API calls: {total_calls} (VL: {vl_count}×{len(vl_combos)}, text: {text_count}×{len(text_combos)})")

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
        image_path = prompt_info.get("image_path", "")

        # Smart combo filtering: VL questions → vision roles + frontdoor baseline;
        # text questions → text roles only (skip vision).
        is_vl = bool(image_path)
        if is_vl:
            active_combos = [
                (r, m) for r, m in combos
                if r in VISION_ROLES or r == "frontdoor"
            ]
        else:
            active_combos = [
                (r, m) for r, m in combos
                if r not in VISION_ROLES
            ]

        logger.info(f"[{i+1}/{len(all_prompts)}] {suite}/{qid} ({'VL' if is_vl else 'text'}, {len(active_combos)} combos)")

        role_results: dict[str, RoleResult] = {}

        cache_prompt_val = False if skip_cache else None

        for combo_idx, (role, mode) in enumerate(active_combos):
            # Health check before each call to detect server death early
            if combo_idx > 0 and not _check_server_health(url):
                logger.error(f"  Server unhealthy before {role}:{mode} — skipping remaining combos for {qid}")
                break

            key = f"{role}:{mode}"
            role_timeout = SLOW_ROLE_TIMEOUT if role in SLOW_ROLES else timeout
            q_start = time.perf_counter()
            response = call_orchestrator_forced(
                prompt, role, mode, url, role_timeout,
                image_path=image_path, cache_prompt=cache_prompt_val,
            )
            q_elapsed = time.perf_counter() - q_start

            # Cooldown between requests to reduce memory pressure
            if cooldown > 0 and combo_idx < len(active_combos) - 1:
                time.sleep(cooldown)

            answer = response.get("answer", "")
            error = response.get("error")
            tokens_generated = response.get("tokens_generated", 0)
            tools_used = response.get("tools_used", 0)
            tools_called = response.get("tools_called", [])
            routed_to = response.get("routed_to", "")
            role_history = response.get("role_history", [])
            routing_strategy = response.get("routing_strategy", "")
            turns = response.get("turns", 0)
            tokens_used = response.get("tokens_used", 0)
            formalization_applied = response.get("formalization_applied", False)
            cache_stats = response.get("cache_stats")
            predicted_tps = response.get("predicted_tps", 0.0)
            generation_ms = response.get("generation_ms", 0.0)
            prompt_eval_ms = response.get("prompt_eval_ms", 0.0)
            http_overhead_ms = response.get("http_overhead_ms", 0.0)

            if error:
                passed = False
            else:
                passed = score_answer_deterministic(answer, expected, scoring_method, scoring_config)

            role_results[key] = RoleResult(
                role=role,
                mode=mode,
                answer=answer[:500] if answer else "",
                passed=passed,
                elapsed_seconds=q_elapsed,
                error=error,
                tokens_generated=tokens_generated,
                tools_used=tools_used,
                tools_called=tools_called,
                routed_to=routed_to,
                role_history=role_history,
                routing_strategy=routing_strategy,
                turns=turns,
                tokens_used=tokens_used,
                formalization_applied=formalization_applied,
                cache_stats=cache_stats,
                predicted_tps=predicted_tps,
                generation_ms=generation_ms,
                prompt_eval_ms=prompt_eval_ms,
                http_overhead_ms=http_overhead_ms,
            )

            status = "PASS" if passed else ("ERROR" if error else "FAIL")
            # Use clean predicted_tps from llama.cpp if available, else fall back to polluted
            polluted_tps = tokens_generated / q_elapsed if q_elapsed > 0 and tokens_generated > 0 else 0
            display_tps = predicted_tps if predicted_tps > 0 else polluted_tps
            # Build detail parts — always show all fields
            parts = []
            parts.append(f"{tokens_generated} tok" if tokens_generated > 0 else "0 tok")
            parts.append(f"{display_tps:.1f} t/s" if display_tps > 0 else "0 t/s")
            if predicted_tps > 0:
                parts.append(f"gen={generation_ms:.0f}ms")
                parts.append(f"prompt={prompt_eval_ms:.0f}ms")
                if http_overhead_ms > 0:
                    parts.append(f"overhead={http_overhead_ms:.0f}ms")
            tools_str = ",".join(tools_called) if tools_called else str(tools_used)
            parts.append(f"tools=[{tools_str}]")
            parts.append(f"turns={turns}")
            parts.append(f"route={routed_to or role}")
            chain_str = "→".join(role_history) if role_history else role
            parts.append(f"chain={chain_str}")
            parts.append(f"strat={routing_strategy or '-'}")
            parts.append(f"formal={'Y' if formalization_applied else 'N'}")
            hit = cache_stats.get("hit", False) if cache_stats else None
            parts.append(f"cache={'HIT' if hit else 'MISS' if hit is not None else '-'}")
            parts.append(f"ctx={tokens_used}")
            logger.info(f"  {key:30s} → {status} ({q_elapsed:.1f}s, {', '.join(parts)})")

        # Compute comparative rewards (baseline is frontdoor:direct)
        rewards = compute_comparative_rewards(role_results, baseline_key="frontdoor:direct")

        # Clone rewards and results to aliased (deduplicated) roles
        for alias, canonical in alias_map.items():
            for mode in _modes_for_role(alias, modes):
                canonical_key = f"{canonical}:{mode}"
                alias_key = f"{alias}:{mode}"
                if canonical_key in rewards:
                    rewards[alias_key] = rewards[canonical_key]
                if canonical_key in role_results:
                    role_results[alias_key] = role_results[canonical_key]

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
        for key, reward in sorted(rewards.items()):
            alias_tag = ""
            role_part = key.split(":")[0]
            if role_part in alias_map:
                alias_tag = f" (={alias_map[role_part]})"
            logger.info(f"    reward[{key}] = {reward:+.1f}{alias_tag}")

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
            for action_key, reward in comp.rewards.items():
                # action_key is already 'role:mode' (e.g. 'frontdoor:direct')
                context = {
                    "task_type": comp.suite,
                    "source": "comparative_seeding",
                    "question_id": comp.question_id,
                    "comparative": True,
                }
                success = store_external_reward(
                    state=state,
                    task_description=comp.prompt[:200],
                    action=action_key,
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
            for action_key, reward in comp.rewards.items():
                try:
                    resp = httpx.post(
                        f"{url}/memrl/reward",
                        json={
                            "task_description": comp.prompt[:200],
                            "action": action_key,
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


def print_seeding_summary(
    results: list[ComparativeResult],
    roles: list[str],
    modes: list[str],
    alias_map: dict[str, str] | None = None,
) -> None:
    """Print summary of seeding results."""
    alias_map = alias_map or {}

    # Build combo keys for display (include aliased roles)
    combos = _build_role_mode_combos(roles, modes)
    combo_keys = [f"{r}:{m}" for r, m in combos]

    # Per-combo accuracy + throughput stats
    key_stats: dict[str, dict[str, Any]] = {
        k: {"pass": 0, "fail": 0, "error": 0,
            "total_tokens": 0, "total_elapsed": 0.0, "total_tools": 0,
            "total_turns": 0, "samples": 0,
            "total_generation_ms": 0.0, "total_prompt_eval_ms": 0.0,
            "predicted_tps_sum": 0.0, "predicted_tps_count": 0}
        for k in combo_keys
    }
    suite_stats: dict[str, dict[str, dict[str, int]]] = {}

    for comp in results:
        suite = comp.suite
        if suite not in suite_stats:
            suite_stats[suite] = {k: {"pass": 0, "fail": 0} for k in combo_keys}

        for key, rr in comp.role_results.items():
            if key not in key_stats:
                continue
            key_stats[key]["samples"] += 1
            key_stats[key]["total_tokens"] += rr.tokens_generated
            key_stats[key]["total_elapsed"] += rr.elapsed_seconds
            key_stats[key]["total_tools"] += rr.tools_used
            key_stats[key]["total_turns"] += rr.turns
            key_stats[key]["total_generation_ms"] += rr.generation_ms
            key_stats[key]["total_prompt_eval_ms"] += rr.prompt_eval_ms
            if rr.predicted_tps > 0:
                key_stats[key]["predicted_tps_sum"] += rr.predicted_tps
                key_stats[key]["predicted_tps_count"] += 1
            if rr.error:
                key_stats[key]["error"] += 1
            elif rr.passed:
                key_stats[key]["pass"] += 1
                suite_stats.get(suite, {}).get(key, {})["pass"] = \
                    suite_stats.get(suite, {}).get(key, {}).get("pass", 0) + 1
            else:
                key_stats[key]["fail"] += 1
                suite_stats.get(suite, {}).get(key, {})["fail"] = \
                    suite_stats.get(suite, {}).get(key, {}).get("fail", 0) + 1

    print(f"\n{'='*100}")
    print("COMPARATIVE SEEDING SUMMARY (role×mode)")
    print(f"{'='*100}")
    print(f"Questions: {len(results)}")
    print(f"Combos tested: {', '.join(combo_keys)}")
    if alias_map:
        dedup_strs = [f"{a} → {c}" for a, c in sorted(alias_map.items())]
        print(f"Deduplicated: {', '.join(dedup_strs)} (same backend, rewards cloned)")

    print(f"\n{'Role:Mode':30s} {'Pass':>5s} {'Fail':>5s} {'Err':>4s} {'Acc%':>6s} {'AvgTok':>7s} {'Avg t/s':>8s} {'AvgGen':>8s} {'AvgTools':>8s} {'AvgTurns':>8s}")
    print("-" * 110)
    for key in combo_keys:
        s = key_stats[key]
        total = s["pass"] + s["fail"]
        n = s["samples"] or 1
        acc = s["pass"] / total * 100 if total > 0 else 0
        avg_tok = s["total_tokens"] / n
        # Prefer clean predicted_tps average; fall back to polluted tokens/elapsed
        if s["predicted_tps_count"] > 0:
            avg_tps = s["predicted_tps_sum"] / s["predicted_tps_count"]
        else:
            avg_tps = s["total_tokens"] / s["total_elapsed"] if s["total_elapsed"] > 0 else 0
        avg_gen_ms = s["total_generation_ms"] / n
        avg_tools = s["total_tools"] / n
        avg_turns = s["total_turns"] / n
        gen_str = f"{avg_gen_ms:.0f}ms" if avg_gen_ms > 0 else "-"
        role_part = key.split(":")[0]
        alias_tag = f" (={alias_map[role_part]})" if role_part in alias_map else ""
        print(
            f"{key:30s} {s['pass']:5d} {s['fail']:5d} {s['error']:4d} "
            f"{acc:5.1f}% {avg_tok:7.0f} {avg_tps:7.1f} {gen_str:>8s} {avg_tools:8.1f} {avg_turns:8.1f}"
            f"{alias_tag}"
        )

    # Per-suite breakdown
    if len(suite_stats) > 1:
        print(f"\nPer-suite breakdown (accuracy %):")
        header = f"{'Suite':20s}" + "".join(f" {k:>20s}" for k in combo_keys)
        print(header)
        print("-" * len(header))
        for suite in sorted(suite_stats.keys()):
            row = f"{suite:20s}"
            for key in combo_keys:
                rs = suite_stats[suite].get(key, {"pass": 0, "fail": 0})
                total = rs["pass"] + rs["fail"]
                acc = rs["pass"] / total * 100 if total > 0 else 0
                row += f" {acc:19.1f}%"
            print(row)

    # Reward distribution
    reward_totals: dict[str, float] = {k: 0.0 for k in combo_keys}
    for comp in results:
        for key, reward in comp.rewards.items():
            if key in reward_totals:
                reward_totals[key] += reward

    print(f"\nCumulative rewards:")
    for key in combo_keys:
        print(f"  {key:30s} {reward_totals[key]:+.1f}")


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
        "--modes", nargs="+", default=DEFAULT_MODES,
        help=f"Execution modes to test (default: {' '.join(DEFAULT_MODES)}). "
        "Architect roles always use direct only.",
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
    parser.add_argument(
        "--skip-cache", action="store_true",
        help="Disable KV cache reuse (cache_prompt=False). Eliminates cache "
        "management overhead for unique prompts.",
    )
    parser.add_argument(
        "--cooldown", type=float, default=0.0,
        help="Seconds to wait between requests (default: 0). Reduces memory "
        "pressure on servers between consecutive inference calls.",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Disable URL-based role deduplication. By default, roles sharing "
        "the same backend URL (e.g. frontdoor and coder_primary on :8080) "
        "are tested once and rewards cloned. Use this flag to force testing "
        "every role independently.",
    )

    args = parser.parse_args()

    # Compute alias_map for logging and summary (even before run_seeding)
    alias_map: dict[str, str] = {}
    if not args.no_dedup:
        _, alias_map = _deduplicate_roles(args.roles)

    combos = _build_role_mode_combos(args.roles, args.modes)
    logger.info("=" * 60)
    logger.info("Comparative Specialist Seeding (role×mode)")
    logger.info(f"  Suites: {args.suites}")
    logger.info(f"  Roles: {args.roles}")
    logger.info(f"  Modes: {args.modes}")
    logger.info(f"  Combos: {len(combos)} ({', '.join(f'{r}:{m}' for r, m in combos)})")
    logger.info(f"  Sample size: {args.sample_size}/suite")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Skip cache: {args.skip_cache}")
    logger.info(f"  Cooldown: {args.cooldown}s")
    logger.info(f"  Dedup: {'OFF' if args.no_dedup else 'ON'}"
                + (f" (aliases: {alias_map})" if alias_map else ""))
    logger.info("=" * 60)

    results = run_seeding(
        suites=args.suites,
        roles=args.roles,
        modes=args.modes,
        sample_per_suite=args.sample_size,
        seed=args.seed,
        url=args.url,
        timeout=args.timeout,
        dry_run=args.dry_run,
        skip_cache=args.skip_cache,
        cooldown=args.cooldown,
        no_dedup=args.no_dedup,
    )

    print_seeding_summary(results, args.roles, args.modes, alias_map=alias_map)

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
            "modes": args.modes,
            "combos": [f"{r}:{m}" for r, m in combos],
            "sample_size": args.sample_size,
            "seed": args.seed,
            "dry_run": args.dry_run,
            "dedup": not args.no_dedup,
            "alias_map": alias_map,
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
