#!/usr/bin/env python3
"""Comparative Specialist Routing Evaluation for MemRL.

THE canonical evaluation script for orchestrator routing optimization.
Runs each question through MULTIPLE role×mode combos, scores deterministically,
and injects comparative rewards so MemRL learns optimal routing.

Comparative reward scheme (xRouter-style, correctness-gated cost penalty):
  specialist correct & frontdoor wrong → +1.0 (specialist clearly better)
  specialist wrong & frontdoor right   → -0.5 (specialist worse)
  both correct                         → 0.5 - λ*max(0, cost_ratio-1) (cost-aware)
  both wrong                           → -0.3 (neither helps)

Usage:
    # THE command. Launch and leave running for days.
    python scripts/benchmark/seed_specialist_routing.py \\
      --continuous --suites all --sample-size 10 --cooldown 2.0 --preflight

    # Resume after restart
    python scripts/benchmark/seed_specialist_routing.py \\
      --continuous --resume seeding_20260201_143022

    # Quick stats from all sessions
    python scripts/benchmark/seed_specialist_routing.py --stats

    # One-shot batch (original behavior)
    python scripts/benchmark/seed_specialist_routing.py --suites thinking coder --sample-size 5

    # Dry run (no reward injection)
    python scripts/benchmark/seed_specialist_routing.py --dry-run --suites thinking --sample-size 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bootstrap: needed before seeding_types can be imported. Same value as seeding_types.PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Re-exports from extracted modules (test backwards compatibility) ──
# Tests import these symbols from this file; keep them accessible here.

from seeding_types import (  # noqa: E402, F401
    ARCHITECT_MODES,
    ARCHITECT_ROLES,
    ComparativeResult,
    DEBUG_PROMPTS_DIR,
    DEFAULT_MODES,
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_ROLES,
    DEFAULT_SUITES,
    DEFAULT_TIMEOUT,
    ESCALATION_REWARD,
    EVAL_DIR,
    HEAVY_PORTS,
    HealthCheckError,
    MODEL_PORTS,
    ROLE_COST_TIER,
    ROLE_PORT,
    RoleResult,
    SEEN_FILE,
    VISION_MODES,
    VISION_ROLES,
    state,
)
from seeding_rewards import (  # noqa: E402, F401
    DEFAULT_BASELINE_TPS,
    _inject_escalation_chains_http,
    _inject_rewards_http,
    compute_comparative_rewards,
    detect_escalation_chains,
)
from seeding_infra import (  # noqa: E402, F401
    MAX_RECOVERY_ATTEMPTS,
    _attempt_recovery,
    _check_server_health,
    _wait_for_heavy_models_idle,
    run_preflight,
)


# ── Signal handlers ──────────────────────────────────────────────────


def _handle_sigint(sig, frame):
    if state.shutdown:
        state.close_poll_client()
        sys.exit(1)
    state.shutdown = True
    print("\n[SIGINT] Finishing current question, then stopping...")


def _handle_sigterm(sig, frame):
    state.shutdown = True
    state.close_poll_client()


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigterm)


# ── Checkpoint management ─────────────────────────────────────────────


def _ensure_eval_dir():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_checkpoint(session_id: str) -> list[ComparativeResult]:
    """Load completed results from a session's JSONL checkpoint."""
    path = EVAL_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Reconstruct ComparativeResult from serialized dict
                role_results = {}
                for k, v in data.get("role_results", {}).items():
                    role_results[k] = RoleResult(**v)
                results.append(ComparativeResult(
                    suite=data["suite"],
                    question_id=data["question_id"],
                    prompt=data.get("prompt", ""),
                    expected=data.get("expected", ""),
                    dataset_source=data.get("dataset_source", "yaml"),
                    prompt_hash=data.get("prompt_hash", ""),
                    timestamp=data.get("timestamp", ""),
                    role_results=role_results,
                    rewards=data.get("rewards", {}),
                    rewards_injected=data.get("rewards_injected", 0),
                ))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return results


def append_checkpoint(session_id: str, result: ComparativeResult):
    """Append one result to the session's JSONL file (atomic-ish)."""
    _ensure_eval_dir()
    path = EVAL_DIR / f"{session_id}.jsonl"
    line = json.dumps(asdict(result), ensure_ascii=False)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_seen_questions() -> set[str]:
    """Load all prompt_ids ever evaluated across all sessions."""
    seen: set[str] = set()
    if not EVAL_DIR.exists():
        return seen

    for path in EVAL_DIR.glob("*.jsonl"):
        if path.name == "seen_questions.jsonl":
            # Read dedicated seen file
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pid = data.get("prompt_id", "")
                        if pid:
                            seen.add(pid)
                    except json.JSONDecodeError:
                        continue
        else:
            # Read from checkpoint files
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pid = data.get("question_id", "")
                        if pid:
                            seen.add(pid)
                    except json.JSONDecodeError:
                        continue

    return seen


def record_seen(prompt_id: str, suite: str, session_id: str):
    """Append to the global seen questions file."""
    _ensure_eval_dir()
    entry = {
        "prompt_id": prompt_id,
        "suite": suite,
        "session": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(SEEN_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Question sampling ─────────────────────────────────────────────────


def _load_from_dataset_adapter(
    suite_name: str, sample_count: int, seed: int,
) -> list[dict]:
    """Sample questions from HF dataset adapters."""
    try:
        from dataset_adapters import get_adapter, ADAPTER_SUITES
    except ImportError:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from dataset_adapters import get_adapter, ADAPTER_SUITES
        except ImportError:
            return []

    if suite_name not in ADAPTER_SUITES:
        return []

    adapter = get_adapter(suite_name)
    if adapter is None:
        return []

    prompts = adapter.sample(n=sample_count, seed=seed)
    if prompts:
        logger.info(f"  [{suite_name}] Sampled {len(prompts)} from "
                     f"{adapter.total_available} HF dataset questions (seed={seed})")
    return prompts


def _load_from_yaml(
    suite_name: str, sample_count: int, seed: int,
) -> list[dict]:
    """Fall back to static YAML debug prompts."""
    try:
        import yaml
    except ImportError:
        return []

    yaml_path = DEBUG_PROMPTS_DIR / f"{suite_name}.yaml"
    if not yaml_path.exists():
        return []

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    questions = data.get("questions", [])
    if not questions:
        return []

    rng = random.Random(seed)
    n = min(sample_count, len(questions))
    sampled = rng.sample(questions, n)
    logger.info(f"  [{suite_name}] Sampled {n}/{len(questions)} from YAML (seed={seed})")

    result = []
    for q in sampled:
        result.append({
            "id": q["id"],
            "suite": suite_name,
            "prompt": q["prompt"].strip(),
            "context": "",
            "expected": q.get("expected", ""),
            "image_path": q.get("image_path", ""),
            "tier": q.get("tier", 1),
            "scoring_method": q.get("scoring_method", "exact_match"),
            "scoring_config": q.get("scoring_config", {}),
            "dataset_source": "yaml",
        })
    return result


def sample_unseen_questions(
    suites: list[str],
    sample_per_suite: int,
    seen: set[str],
    seed: int,
) -> list[dict]:
    """Sample questions not in the seen set.

    Tries HF dataset adapters first, falls back to YAML.
    Oversamples by 3x to compensate for dedup filtering.
    """
    suite_names = DEFAULT_SUITES if suites == ["all"] else suites
    all_prompts: list[dict] = []

    for suite_name in suite_names:
        oversample = sample_per_suite * 3

        prompts = _load_from_dataset_adapter(suite_name, oversample, seed)
        if not prompts:
            prompts = _load_from_yaml(suite_name, oversample, seed)

        # Filter out seen questions
        fresh = [p for p in prompts if p["id"] not in seen]
        if len(fresh) < len(prompts):
            filtered = len(prompts) - len(fresh)
            logger.info(f"  [{suite_name}] Filtered {filtered} previously seen questions")

        all_prompts.extend(fresh[:sample_per_suite])

    return all_prompts


# ── Core functions ────────────────────────────────────────────────────


def _erase_slots(port: int) -> None:
    """Force-cancel in-progress inference on a llama-server port.

    After a timeout the server may still be grinding on the old request.
    Erasing slots prevents cascading timeouts on subsequent requests.
    """
    import httpx

    try:
        resp = httpx.get(f"http://localhost:{port}/slots", timeout=5)
        if resp.status_code != 200:
            return
        for slot in resp.json():
            if slot.get("is_processing"):
                slot_id = slot.get("id", 0)
                httpx.post(
                    f"http://localhost:{port}/slots/{slot_id}?action=erase",
                    timeout=5,
                )
                logger.info(f"  → erased slot {slot_id} on port {port}")
    except Exception:
        pass  # best-effort cleanup


def call_orchestrator_forced(
    prompt: str,
    force_role: str,
    force_mode: str = "direct",
    url: str = DEFAULT_ORCHESTRATOR_URL,
    timeout: int = DEFAULT_TIMEOUT,
    image_path: str = "",
    cache_prompt: bool | None = None,
    client: "httpx.Client | None" = None,
) -> dict[str, Any]:
    """Call orchestrator with forced role and mode routing."""
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
        if client is not None:
            response = client.post(f"{url}/chat", json=payload)
        else:
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
    """Score an answer deterministically."""
    from benchmark.debug_scorer import score_answer

    return score_answer(answer, expected, scoring_method, scoring_config or {})


# ── Combo building ───────────────────────────────────────────────────


def _build_role_mode_combos(
    roles: list[str],
    modes: list[str],
) -> list[tuple[str, str]]:
    """Build (role, mode) combinations with two invariants:

    1. MODE-FIRST: Cycle through modes before roles, so consecutive calls
       hit different backend servers (natural cooldown for each server).
    2. HEAVY SEPARATION: Heavy model combos (architects, ingest) are never
       adjacent. Light combos are interleaved between them so light work
       runs while heavy servers cool down.

    The idle-wait in evaluate_question() enforces that heavy models are
    actually idle before any request, but good ordering reduces idle-wait
    time by doing useful light work in the gaps.
    """
    all_modes = list(modes)
    for m in sorted(ARCHITECT_MODES):
        if m not in all_modes:
            all_modes.append(m)

    light: list[tuple[str, str]] = []
    heavy: list[tuple[str, str]] = []

    for mode in all_modes:
        for role in roles:
            port = ROLE_PORT.get(role, 0)
            is_heavy = port in HEAVY_PORTS
            if role in ARCHITECT_ROLES:
                if mode in ARCHITECT_MODES:
                    heavy.append((role, mode))
            elif role in VISION_ROLES:
                if mode in VISION_MODES.get(role, {"direct"}):
                    (heavy if is_heavy else light).append((role, mode))
            else:
                if mode in modes:
                    (heavy if is_heavy else light).append((role, mode))

    # Interleave: spread heavy combos evenly across the light sequence.
    # With N light and M heavy: place one heavy after every N//M light combos.
    if not heavy:
        return light
    if not light:
        return heavy

    result: list[tuple[str, str]] = []
    gap = max(1, len(light) // len(heavy))
    heavy_iter = iter(heavy)
    next_heavy = next(heavy_iter, None)

    for i, combo in enumerate(light):
        result.append(combo)
        if next_heavy is not None and (i + 1) % gap == 0:
            result.append(next_heavy)
            next_heavy = next(heavy_iter, None)

    # Append remaining heavy combos at the end
    if next_heavy is not None:
        result.append(next_heavy)
    for h in heavy_iter:
        result.append(h)

    return result


def _deduplicate_roles(
    roles: list[str],
    server_urls: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Remove roles that share the same backend URL."""
    from src.config import get_config as _get_cfg
    urls = server_urls or _get_cfg().server_urls.as_dict()
    seen: dict[str, str] = {}
    unique: list[str] = []
    aliases: dict[str, str] = {}

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


# ── Main evaluation loop ──────────────────────────────────────────────


def evaluate_question(
    prompt_info: dict,
    combos: list[tuple[str, str]],
    alias_map: dict[str, str],
    modes: list[str],
    url: str,
    timeout: int,
    client: "httpx.Client",
    skip_cache: bool = False,
    cooldown: float = 0.0,
    dry_run: bool = False,
    escalation_chains: bool = False,
) -> ComparativeResult | None:
    """Evaluate one question across all role×mode combos.

    Returns ComparativeResult or None if shutdown requested.
    """
    suite = prompt_info["suite"]
    qid = prompt_info["id"]
    prompt = prompt_info["prompt"]
    expected = prompt_info.get("expected", "")
    scoring_method = prompt_info.get("scoring_method", "exact_match")
    scoring_config = prompt_info.get("scoring_config", {})
    image_path = prompt_info.get("image_path", "")
    dataset_source = prompt_info.get("dataset_source", "yaml")

    # Smart combo filtering: VL → vision + frontdoor; text → non-vision
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

    role_results: dict[str, RoleResult] = {}
    cache_prompt_val = False if skip_cache else None

    SLOW_ROLES = {"architect_general", "architect_coding"}
    SLOW_ROLE_TIMEOUT = max(timeout, 300)

    for combo_idx, (role, mode) in enumerate(active_combos):
        if state.shutdown:
            return None

        # Before hitting a heavy model, confirm all heavy ports are idle.
        target_port = ROLE_PORT.get(role, 0)
        if target_port in HEAVY_PORTS:
            _wait_for_heavy_models_idle()

        key = f"{role}:{mode}"
        if target_port in HEAVY_PORTS:
            logger.info(f"  → {key} (heavy model, expect 30-120s)...")
        role_timeout = SLOW_ROLE_TIMEOUT if role in SLOW_ROLES else timeout
        q_start = time.perf_counter()
        response = call_orchestrator_forced(
            prompt, role, mode, url, role_timeout,
            image_path=image_path, cache_prompt=cache_prompt_val,
            client=client,
        )
        q_elapsed = time.perf_counter() - q_start

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
            # After a timeout/error on a heavy port, erase its slots so the
            # server isn't still grinding when the next combo arrives.
            if target_port in HEAVY_PORTS and tokens_generated == 0:
                _erase_slots(target_port)
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
        display_tps = predicted_tps if predicted_tps > 0 else 0

        # Main line: role:mode → status (time, speed, tokens)
        parts = [f"  {key:30s} → {status} ({q_elapsed:.1f}s"]
        if display_tps > 0:
            parts.append(f", {display_tps:.1f} t/s")
        parts.append(f", {tokens_generated} tok)")
        logger.info("".join(parts))

        # Detail line: tools, chain, delegation
        details = []
        if tools_used > 0:
            tool_names = ", ".join(tools_called) if tools_called else "?"
            details.append(f"tools({tools_used}): {tool_names}")
        if role_history and len(role_history) > 1:
            details.append(f"chain: {' → '.join(role_history)}")
        if formalization_applied:
            details.append("formalized")
        if generation_ms > 0:
            details.append(f"gen={generation_ms:.0f}ms")
        if prompt_eval_ms > 0:
            details.append(f"prompt_eval={prompt_eval_ms:.0f}ms")
        if details:
            logger.info(f"  {'':30s}   {'  '.join(details)}")

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

    # Inject rewards immediately (per-question, not batched)
    rewards_injected = 0
    if not dry_run:
        # Use HTTP API (works whether running in-process or externally)
        comp_for_inject = ComparativeResult(
            suite=suite, question_id=qid, prompt=prompt[:200],
            expected=expected[:200], rewards=rewards,
        )
        rewards_injected = _inject_rewards_http(comp_for_inject, url, client)

    # Escalation chains: detect cheap-fail → expensive-pass patterns
    escalation_data: list[dict[str, Any]] = []
    if escalation_chains and not dry_run:
        escalation_data = detect_escalation_chains(role_results)
        if escalation_data:
            comp_for_esc = ComparativeResult(
                suite=suite, question_id=qid, prompt=prompt[:200],
                expected=expected[:200], rewards=rewards,
            )
            esc_injected = _inject_escalation_chains_http(
                comp_for_esc, escalation_data, url, client,
            )
            rewards_injected += esc_injected
            for chain in escalation_data:
                logger.info(
                    f"    escalation: {chain['from_role']}:{chain['from_mode']} → "
                    f"{chain['to_role']}:{chain['to_mode']} "
                    f"reward={chain['reward']:+.2f}"
                )

    # Log rewards
    for key, reward in sorted(rewards.items()):
        alias_tag = ""
        role_part = key.split(":")[0]
        if role_part in alias_map:
            alias_tag = f" (={alias_map[role_part]})"
        logger.info(f"    reward[{key}] = {reward:+.2f}{alias_tag}")

    return ComparativeResult(
        suite=suite,
        question_id=qid,
        prompt=prompt[:200],
        expected=expected[:200],
        dataset_source=dataset_source,
        prompt_hash=_prompt_hash(prompt),
        timestamp=datetime.now(timezone.utc).isoformat(),
        role_results=role_results,
        rewards=rewards,
        rewards_injected=rewards_injected,
    )


def run_batch(
    suites: list[str],
    roles: list[str],
    modes: list[str],
    sample_per_suite: int,
    seed: int,
    url: str,
    timeout: int,
    session_id: str,
    dry_run: bool = False,
    skip_cache: bool = False,
    cooldown: float = 0.0,
    no_dedup: bool = False,
    escalation_chains: bool = False,
) -> list[ComparativeResult]:
    """Run one evaluation batch: sample, evaluate per-question, checkpoint."""

    # Deduplicate roles
    alias_map: dict[str, str] = {}
    if not no_dedup:
        unique_roles, alias_map = _deduplicate_roles(roles)
        if alias_map:
            for alias, canonical in sorted(alias_map.items()):
                from src.config import get_config as _get_cfg2
                canon_url = _get_cfg2().server_urls.as_dict().get(canonical, "?")
                logger.info(f"Dedup: {alias} → {canonical} (same backend {canon_url})")
        roles_to_test = unique_roles
    else:
        roles_to_test = list(roles)

    combos = _build_role_mode_combos(roles_to_test, modes)
    combo_keys = [f"{r}:{m}" for r, m in combos]

    # Health check (continuous loop pre-checks this, but one-shot mode needs it)
    if not _check_server_health(url):
        raise HealthCheckError(f"API unreachable: {url}")

    # Load existing checkpoint + seen set
    completed = load_checkpoint(session_id)
    completed_ids = {r.question_id for r in completed}
    seen = load_seen_questions()
    logger.info(f"Checkpoint: {len(completed)} completed, {len(seen)} total seen")

    # Sample unseen questions
    questions = sample_unseen_questions(suites, sample_per_suite, seen, seed)
    questions = [q for q in questions if q["id"] not in completed_ids]

    if not questions:
        logger.info("No unseen questions available. Try a different seed or suite.")
        return completed

    vl_count = sum(1 for p in questions if p.get("image_path"))
    text_count = len(questions) - vl_count

    logger.info(f"\n{'='*60}")
    logger.info(f"Session: {session_id}")
    logger.info(f"Batch: {len(questions)} questions ({text_count} text, {vl_count} VL)")
    logger.info(f"Combos: {len(combos)} ({', '.join(combo_keys)})")
    heavy_count = sum(1 for r, m in combos if ROLE_PORT.get(r, 0) in HEAVY_PORTS)
    if heavy_count:
        logger.info(f"  Heavy combos per question: {heavy_count} (30-120s each)")
    logger.info(f"Seed: {seed}  Rewards: {'off' if dry_run else 'on'}")
    logger.info(f"{'='*60}\n")

    SLOW_ROLE_TIMEOUT = max(timeout, 300)
    import httpx as _httpx
    _client = _httpx.Client(timeout=SLOW_ROLE_TIMEOUT)

    new_results: list[ComparativeResult] = []
    consecutive_zero_success = 0

    try:
        for i, prompt_info in enumerate(questions):
            if state.shutdown:
                logger.info(f"\n[Stopped after {i} questions]")
                break

            qid = prompt_info["id"]
            suite = prompt_info["suite"]
            is_vl = bool(prompt_info.get("image_path"))
            logger.info(f"[{i+1}/{len(questions)}] {suite}/{qid} ({'VL' if is_vl else 'text'})")

            result = evaluate_question(
                prompt_info, combos, alias_map, modes,
                url, timeout, _client,
                skip_cache=skip_cache, cooldown=cooldown, dry_run=dry_run,
                escalation_chains=escalation_chains,
            )

            if result is None:
                break  # Shutdown

            # Checkpoint immediately
            append_checkpoint(session_id, result)
            record_seen(result.question_id, result.suite, session_id)
            new_results.append(result)

            # Track consecutive failures for abort
            any_success = any(rr.error is None for rr in result.role_results.values())
            if any_success:
                consecutive_zero_success = 0
            else:
                consecutive_zero_success += 1
                if consecutive_zero_success >= 3:
                    logger.error(
                        f"Aborting: {consecutive_zero_success} consecutive questions "
                        f"with zero successful combos — server appears dead"
                    )
                    break
    finally:
        _client.close()

    all_results = completed + new_results
    return all_results


# ── Stats / summary ───────────────────────────────────────────────────


def print_batch_summary(
    results: list[ComparativeResult],
    roles: list[str],
    modes: list[str],
    alias_map: dict[str, str] | None = None,
) -> None:
    """Print summary of results."""
    alias_map = alias_map or {}
    combos = _build_role_mode_combos(roles, modes)
    combo_keys = [f"{r}:{m}" for r, m in combos]

    key_stats: dict[str, dict[str, Any]] = {
        k: {"pass": 0, "fail": 0, "error": 0,
            "total_tokens": 0, "total_elapsed": 0.0,
            "samples": 0, "predicted_tps_sum": 0.0, "predicted_tps_count": 0,
            "total_reward": 0.0}
        for k in combo_keys
    }

    for comp in results:
        for key, rr in comp.role_results.items():
            if key not in key_stats:
                continue
            key_stats[key]["samples"] += 1
            key_stats[key]["total_tokens"] += rr.tokens_generated
            key_stats[key]["total_elapsed"] += rr.elapsed_seconds
            if rr.predicted_tps > 0:
                key_stats[key]["predicted_tps_sum"] += rr.predicted_tps
                key_stats[key]["predicted_tps_count"] += 1
            if rr.error:
                key_stats[key]["error"] += 1
            elif rr.passed:
                key_stats[key]["pass"] += 1
            else:
                key_stats[key]["fail"] += 1
        for key, reward in comp.rewards.items():
            if key in key_stats:
                key_stats[key]["total_reward"] += reward

    print(f"\n{'='*100}")
    print("COMPARATIVE EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"Questions: {len(results)}")
    if alias_map:
        dedup_strs = [f"{a} → {c}" for a, c in sorted(alias_map.items())]
        print(f"Deduplicated: {', '.join(dedup_strs)}")

    print(f"\n{'Role:Mode':30s} {'Pass':>5s} {'Fail':>5s} {'Err':>4s} {'Acc%':>6s} {'Avg t/s':>8s} {'Reward':>8s}")
    print("-" * 75)
    for key in combo_keys:
        s = key_stats[key]
        total = s["pass"] + s["fail"]
        acc = s["pass"] / total * 100 if total > 0 else 0
        if s["predicted_tps_count"] > 0:
            avg_tps = s["predicted_tps_sum"] / s["predicted_tps_count"]
        else:
            avg_tps = s["total_tokens"] / s["total_elapsed"] if s["total_elapsed"] > 0 else 0
        role_part = key.split(":")[0]
        alias_tag = f" (={alias_map[role_part]})" if role_part in alias_map else ""
        print(
            f"{key:30s} {s['pass']:5d} {s['fail']:5d} {s['error']:4d} "
            f"{acc:5.1f}% {avg_tps:7.1f} {s['total_reward']:+7.1f}"
            f"{alias_tag}"
        )

    # Total rewards injected
    total_injected = sum(r.rewards_injected for r in results)
    print(f"\nRewards injected: {total_injected}")


def print_stats():
    """Aggregate stats across all seeding sessions."""
    if not EVAL_DIR.exists():
        print("No evaluation data found.")
        return

    sessions: dict[str, list[ComparativeResult]] = {}
    for path in sorted(EVAL_DIR.glob("seeding_*.jsonl")):
        sid = path.stem
        results = load_checkpoint(sid)
        if results:
            sessions[sid] = results

    if not sessions:
        print("No seeding sessions found.")
        return

    print(f"\n{'='*60}")
    print("ALL SEEDING SESSIONS")
    print(f"{'='*60}")

    total_questions = 0
    all_combo_stats: dict[str, dict[str, int]] = {}

    for sid, results in sessions.items():
        total_questions += len(results)
        ts = results[0].timestamp[:10] if results and results[0].timestamp else "?"
        print(f"  {sid:45s} {len(results):4d} questions  {ts}")

        for comp in results:
            for key, rr in comp.role_results.items():
                if key not in all_combo_stats:
                    all_combo_stats[key] = {"pass": 0, "fail": 0, "error": 0, "total": 0}
                all_combo_stats[key]["total"] += 1
                if rr.error:
                    all_combo_stats[key]["error"] += 1
                elif rr.passed:
                    all_combo_stats[key]["pass"] += 1
                else:
                    all_combo_stats[key]["fail"] += 1

    print(f"\nTotal questions: {total_questions}")
    print(f"Sessions: {len(sessions)}")

    seen = load_seen_questions()
    print(f"Unique questions seen: {len(seen)}")

    if all_combo_stats:
        print(f"\nAggregate accuracy by role×mode:")
        print(f"  {'Role:Mode':30s} {'Pass':>5s} {'Fail':>5s} {'Err':>4s} {'Acc%':>6s} {'N':>5s} {'≥3?':>4s}")
        print("  " + "-" * 60)
        for key in sorted(all_combo_stats.keys()):
            s = all_combo_stats[key]
            total = s["pass"] + s["fail"]
            acc = s["pass"] / total * 100 if total > 0 else 0
            # ≥3 observations = MemRL confidence threshold met
            confident = "YES" if s["total"] >= 3 else "no"
            print(
                f"  {key:30s} {s['pass']:5d} {s['fail']:5d} {s['error']:4d} "
                f"{acc:5.1f}% {s['total']:5d} {confident:>4s}"
            )

    # Coverage: combos with ≥3 observations
    covered = sum(1 for s in all_combo_stats.values() if s["total"] >= 3)
    total_combos = len(all_combo_stats)
    print(f"\nMemRL coverage: {covered}/{total_combos} combos have ≥3 observations")


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Comparative Specialist Routing Evaluation for MemRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run continuously for days (THE command):
  %(prog)s --continuous --suites all --sample-size 10 --cooldown 2.0 --preflight

  # Resume after restart:
  %(prog)s --continuous --resume seeding_20260201_143022

  # Quick stats:
  %(prog)s --stats

  # One-shot batch:
  %(prog)s --suites thinking coder --sample-size 5
""",
    )
    parser.add_argument(
        "--suites", nargs="+", default=DEFAULT_SUITES,
        help=f"Suites to evaluate (default: {' '.join(DEFAULT_SUITES)})",
    )
    parser.add_argument(
        "--roles", nargs="+", default=DEFAULT_ROLES,
        help=f"Roles to compare (default: {' '.join(DEFAULT_ROLES)})",
    )
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        help=f"Execution modes to test (default: {' '.join(DEFAULT_MODES)}). "
        "Architect roles always use direct+delegated.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="Questions per suite per batch (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: timestamp)",
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
        help="Output JSON file (default: auto-generated). Only for one-shot mode.",
    )
    parser.add_argument(
        "--skip-cache", action="store_true",
        help="Disable KV cache reuse (cache_prompt=False).",
    )
    parser.add_argument(
        "--cooldown", type=float, default=0.0,
        help="Seconds between requests (default: 0). Reduces server memory pressure.",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Disable URL-based role deduplication.",
    )
    # Continuous mode
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run batches continuously until Ctrl+C. Checkpoints per question.",
    )
    parser.add_argument(
        "--continuous-interval", type=int, default=30,
        help="Seconds between continuous batches (default: 30)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resume a specific session ID (e.g. seeding_20260201_143022)",
    )
    # Preflight & stats
    parser.add_argument(
        "--preflight", action="store_true",
        help="Run health checks and smoke test before starting.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show aggregate stats from all sessions, then exit.",
    )
    parser.add_argument(
        "--no-escalation-chains", action="store_true",
        help="Disable escalation chain reward detection (enabled by default). "
        "Escalation chains detect when cheap models fail but expensive models "
        "pass on the same question, and inject escalation rewards into MemRL.",
    )

    args = parser.parse_args()

    # Stats mode — just print and exit
    if args.stats:
        print_stats()
        return

    # Preflight
    if args.preflight:
        if not run_preflight(args.url):
            sys.exit(1)

    # Session ID
    if args.resume:
        session_id = args.resume
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"seeding_{ts}"

    # Default seed
    base_seed = args.seed if args.seed is not None else int(time.time())

    # Compute alias_map for summary display
    alias_map: dict[str, str] = {}
    if not args.no_dedup:
        _, alias_map = _deduplicate_roles(args.roles)

    if args.continuous:
        # ── Continuous mode ──
        batch = 0
        consecutive_failures = 0
        logger.info(f"Starting continuous evaluation: session={session_id}")
        logger.info(f"  Ctrl+C to stop gracefully (finishes current question)")

        while not state.shutdown:
            # ── Health gate with auto-recovery ──
            if not _check_server_health(args.url):
                consecutive_failures += 1
                if consecutive_failures > MAX_RECOVERY_ATTEMPTS:
                    logger.error(
                        f"API unrecoverable after {MAX_RECOVERY_ATTEMPTS} attempts. Exiting."
                    )
                    break
                backoff = min(30 * (2 ** (consecutive_failures - 1)), 600)
                logger.warning(
                    f"API down (attempt {consecutive_failures}/{MAX_RECOVERY_ATTEMPTS}). "
                    f"Attempting recovery..."
                )
                recovered = _attempt_recovery(args.url)
                if recovered:
                    logger.info("  Recovery successful — resuming evaluation")
                    consecutive_failures = 0
                    continue
                logger.warning(f"  Recovery failed — sleeping {backoff}s before retry")
                for _ in range(backoff):
                    if state.shutdown:
                        break
                    time.sleep(1)
                continue
            consecutive_failures = 0

            batch += 1
            batch_seed = base_seed + batch
            logger.info(f"\n[Batch {batch}, seed={batch_seed}]")

            try:
                results = run_batch(
                    suites=args.suites,
                    roles=args.roles,
                    modes=args.modes,
                    sample_per_suite=args.sample_size,
                    seed=batch_seed,
                    url=args.url,
                    timeout=args.timeout,
                    session_id=session_id,
                    dry_run=args.dry_run,
                    skip_cache=args.skip_cache,
                    cooldown=args.cooldown,
                    no_dedup=args.no_dedup,
                    escalation_chains=not args.no_escalation_chains,
                )
            except HealthCheckError:
                # API died mid-batch — loop back to health gate
                logger.warning("API died during batch — will attempt recovery")
                continue

            if results:
                print_batch_summary(results, args.roles, args.modes, alias_map=alias_map)

            if state.shutdown:
                break

            logger.info(f"\n[Sleeping {args.continuous_interval}s before next batch...]")
            for _ in range(args.continuous_interval):
                if state.shutdown:
                    break
                time.sleep(1)

        logger.info(f"\nSession complete: {session_id}")
        logger.info(f"  Run --stats to see aggregate results")

    else:
        # ── One-shot mode (original behavior with HF datasets) ──
        results = run_batch(
            suites=args.suites,
            roles=args.roles,
            modes=args.modes,
            sample_per_suite=args.sample_size,
            seed=base_seed,
            url=args.url,
            timeout=args.timeout,
            session_id=session_id,
            dry_run=args.dry_run,
            skip_cache=args.skip_cache,
            cooldown=args.cooldown,
            no_dedup=args.no_dedup,
            escalation_chains=not args.no_escalation_chains,
        )

        if results:
            print_batch_summary(results, args.roles, args.modes, alias_map=alias_map)

        # Save JSON output (legacy format for backwards compat)
        output_path = args.output
        if output_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                PROJECT_ROOT / "benchmarks" / "results" / "orchestrator"
                / f"seeding_{ts}.json"
            )

        all_combos = _build_role_mode_combos(args.roles, args.modes)
        output_data = {
            "config": {
                "suites": args.suites,
                "roles": args.roles,
                "modes": args.modes,
                "combos": [f"{r}:{m}" for r, m in all_combos],
                "sample_size": args.sample_size,
                "seed": base_seed,
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
        logger.info(f"JSONL checkpoint: {EVAL_DIR / f'{session_id}.jsonl'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        state.close_poll_client()
