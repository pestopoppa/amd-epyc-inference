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
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
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

# ── Constants ─────────────────────────────────────────────────────────

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

EVAL_DIR = PROJECT_ROOT / "benchmarks" / "results" / "eval"
SEEN_FILE = EVAL_DIR / "seen_questions.jsonl"
DEBUG_PROMPTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "debug"

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

# Per-suite timeout overrides (seconds)
SUITE_TIMEOUTS: dict[str, int] = {
    "long_context": 600,
    "coder": 180,
    "vl": 180,
}

# Cost tiers for escalation chain detection (lower = cheaper).
# When a cheaper role fails and a more expensive role passes,
# that's a valuable escalation signal for MemRL.
ROLE_COST_TIER: dict[str, int] = {
    "worker_explore": 1,
    "worker_math": 1,
    "worker_vision": 1,
    "frontdoor": 2,
    "coder_primary": 2,
    "vision_escalation": 3,
    "coder_escalation": 3,
    "architect_general": 4,
    "architect_coding": 5,
}

# Escalation reward: lower than +1.0 (direct win) because escalation adds latency.
ESCALATION_REWARD = 0.8

# Graceful shutdown flag
_shutdown = False

# Module-level httpx.Client for connection-reusing polling/health checks.
# Bare httpx.get() creates a new TCP connection per call, which accumulates
# in TIME_WAIT and can exhaust socket descriptors over multi-day runs.
_poll_client: "httpx.Client | None" = None


def _get_poll_client() -> "httpx.Client":
    """Get or create the module-level httpx client for polling."""
    global _poll_client
    if _poll_client is None:
        import httpx
        _poll_client = httpx.Client(timeout=10)
    return _poll_client


def _close_poll_client() -> None:
    """Close the module-level polling client if open."""
    global _poll_client
    if _poll_client is not None:
        try:
            _poll_client.close()
        except Exception:
            pass
        _poll_client = None


def _handle_sigint(sig, frame):
    global _shutdown
    if _shutdown:
        _close_poll_client()
        sys.exit(1)
    _shutdown = True
    print("\n[SIGINT] Finishing current question, then stopping...")


def _handle_sigterm(sig, frame):
    global _shutdown
    _shutdown = True
    _close_poll_client()


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigterm)


# ── Exceptions ────────────────────────────────────────────────────────


class HealthCheckError(Exception):
    """Raised when the orchestrator API is unreachable."""

    pass


# ── Data structures ───────────────────────────────────────────────────


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
    dataset_source: str = "yaml"
    prompt_hash: str = ""
    timestamp: str = ""
    role_results: dict[str, RoleResult] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)
    rewards_injected: int = 0


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
      specialist correct & baseline wrong → +1.0 (clear specialist win)
      specialist wrong & baseline right   → -0.5 (specialist worse)
      both correct → 0.5 - lambda * max(0, cost_ratio - 1.0)  (cost-aware)
      both wrong   → -0.3 (neither helps)
    """
    cost_config = cost_config or {}
    lam = cost_config.get("lambda", 0.15)
    baseline_tps = cost_config.get("baseline_tps_by_role", DEFAULT_BASELINE_TPS)

    rewards = {}
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


def _check_server_health(url: str, timeout: int = 5) -> bool:
    """Check if the orchestrator server is healthy."""
    try:
        resp = _get_poll_client().get(f"{url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


# ── Backend idle enforcement ─────────────────────────────────────────

# Heavy model ports: MUST be confirmed idle before sending the next request.
# These models saturate memory bandwidth when generating — concurrent inference
# across any of them destroys throughput for ALL of them.
HEAVY_PORTS = {8083, 8084, 8085, 8087}

# Role → backend port mapping (mirrors get_config().server_urls)
ROLE_PORT: dict[str, int] = {
    "frontdoor": 8080,
    "coder_primary": 8080,
    "coder_escalation": 8081,
    "worker_explore": 8082,
    "worker_math": 8082,
    "worker_vision": 8086,
    "vision_escalation": 8087,
    "architect_general": 8083,
    "architect_coding": 8084,
    "ingest_long_context": 8085,
}


def _is_server_idle(port: int, timeout: int = 3) -> bool:
    """Check if all slots on a llama-server port are idle."""
    try:
        resp = _get_poll_client().get(f"http://localhost:{port}/slots", timeout=timeout)
        if resp.status_code != 200:
            return True  # Can't check — assume idle
        slots = resp.json()
        return not any(s.get("is_processing", False) for s in slots)
    except Exception:
        return True  # Server unreachable — assume idle


def _wait_for_heavy_models_idle(max_wait: int = 600) -> None:
    """Block until ALL heavy model servers are idle.

    Called before every combo to ensure no concurrent heavy inference.
    Light/fast workers (8080-8082, 8086) are allowed to overlap.
    """
    start = time.perf_counter()
    while True:
        all_idle = True
        busy_ports = []
        for port in HEAVY_PORTS:
            if not _is_server_idle(port):
                all_idle = False
                busy_ports.append(port)
        if all_idle:
            elapsed = time.perf_counter() - start
            if elapsed > 1.0:
                logger.info(f"  [idle-wait] Heavy models idle after {elapsed:.1f}s")
            return
        if time.perf_counter() - start > max_wait:
            logger.warning(
                f"  [idle-wait] Timeout after {max_wait}s, ports still busy: {busy_ports}"
            )
            return
        if _shutdown:
            return
        time.sleep(2)


# ── Preflight checks ──────────────────────────────────────────────────


STACK_SCRIPT = PROJECT_ROOT / "scripts" / "server" / "orchestrator_stack.py"


def _check_port(port: int) -> bool:
    """Check if a port is listening."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


# Model server ports (excluding API port 8000)
MODEL_PORTS = [8080, 8081, 8082, 8083, 8084, 8085, 8086, 8087, 8090]


def _kill_port(port: int) -> bool:
    """Kill the process listening on a port. Returns True if killed."""
    import subprocess

    result = subprocess.run(
        ["fuser", "-k", f"{port}/tcp"],
        capture_output=True,
        timeout=10,
    )
    time.sleep(1)
    return not _check_port(port)


def _launch_api_only() -> bool:
    """Launch just the orchestrator API (uvicorn on port 8000).

    Used when model servers are already running but the API is not.
    If port 8000 is already taken by a stale process, kills it first.
    """
    import subprocess

    # Kill stale API process if port 8000 is occupied
    if _check_port(8000):
        logger.warning("  Port 8000 already in use — killing stale process...")
        if not _kill_port(8000):
            logger.error("  Could not free port 8000")
            return False
        logger.info("  Port 8000 freed")

    logger.info("  Launching orchestrator API only (model servers already running)...")

    env = os.environ.copy()
    env["HF_HOME"] = "/mnt/raid0/llm/cache/huggingface"
    env["TMPDIR"] = "/mnt/raid0/llm/tmp"
    env["ORCHESTRATOR_CACHING"] = "1"
    env["ORCHESTRATOR_STREAMING"] = "1"
    env["ORCHESTRATOR_MOCK_MODE"] = "0"
    env["ORCHESTRATOR_REAL_MODE"] = "1"
    env["ORCHESTRATOR_SCRIPTS"] = "1"
    env["ORCHESTRATOR_REACT_MODE"] = "1"
    env["ORCHESTRATOR_MEMRL"] = "1"
    env["ORCHESTRATOR_TOOLS"] = "1"
    env["ORCHESTRATOR_GENERATION_MONITOR"] = "1"

    log_file = PROJECT_ROOT / "logs" / "orchestrator_autolaunch.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.api:app",
            "--host", "127.0.0.1",
            "--port", "8000",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )

    # Wait for API to become healthy — verify OUR process is still alive
    for attempt in range(24):  # Up to 2 minutes
        if proc.poll() is not None:
            logger.error(f"  API process exited (code={proc.returncode}). Check log: {log_file}")
            return False
        if _check_server_health(DEFAULT_ORCHESTRATOR_URL):
            logger.info(f"  API healthy (pid={proc.pid}) after {(attempt + 1) * 5}s")
            return True
        time.sleep(5)

    logger.error(f"  API did not start. Check log: {log_file}")
    proc.kill()
    return False


def _auto_launch_stack(hot_only: bool = True) -> bool:
    """Launch the full orchestrator stack and wait for it to become healthy.

    Only called when NO ports are in use (cold start).
    Returns True if the stack came up successfully.
    """
    import subprocess

    if not STACK_SCRIPT.exists():
        logger.error(f"  Stack script not found: {STACK_SCRIPT}")
        return False

    cmd = [sys.executable, str(STACK_SCRIPT), "start"]
    if hot_only:
        cmd.append("--hot-only")

    logger.info(f"  Auto-launching full stack: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"  Stack launch failed (exit {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-5:]:
                    logger.error(f"    {line}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("  Stack launch timed out after 600s")
        return False

    # Wait for API to become healthy
    logger.info("  Waiting for API to become healthy...")
    for attempt in range(60):  # Up to 5 minutes
        if _check_server_health(DEFAULT_ORCHESTRATOR_URL):
            logger.info(f"  API healthy after {(attempt + 1) * 5}s")
            return True
        time.sleep(5)

    logger.error("  API did not become healthy within 5 minutes")
    return False


MAX_RECOVERY_ATTEMPTS = 10


def _attempt_recovery(url: str) -> bool:
    """Attempt to recover a dead orchestrator API.

    Checks what's still running and takes the minimal action:
    - Model ports up → restart API only (kills stale process on :8000)
    - Nothing up → full stack launch
    """
    model_ports_up = sum(1 for p in MODEL_PORTS if _check_port(p))

    if model_ports_up > 0:
        logger.info(
            f"  Recovery: {model_ports_up} model port(s) still up — restarting API only"
        )
        return _launch_api_only()
    else:
        logger.info("  Recovery: no model ports up — launching full stack")
        return _auto_launch_stack()


def run_preflight(url: str) -> bool:
    """Run preflight health checks on orchestrator and backends.

    Auto-launches the orchestrator stack if the API is not reachable.
    Returns True if all checks pass.
    """
    logger.info("=" * 60)
    logger.info("PREFLIGHT CHECKS")
    logger.info("=" * 60)

    # 1. Orchestrator API health (auto-launch if down)
    api_healthy = _check_server_health(url)
    model_ports_up = sum(1 for p in MODEL_PORTS if _check_port(p))

    if api_healthy:
        logger.info(f"  API already running ({url})")
    elif model_ports_up > 0:
        # Model servers running but API is not — just start the API
        # _launch_api_only() handles killing stale processes on :8000
        logger.info(
            f"  API not reachable but {model_ports_up} model port(s) are up "
            f"— launching API only..."
        )
        if not _launch_api_only():
            logger.error("PREFLIGHT FAILED: Could not start orchestrator API")
            return False
    else:
        # Nothing running — launch full stack
        logger.info("  No stack running — launching full stack...")
        if not _auto_launch_stack():
            logger.error("PREFLIGHT FAILED: Could not start orchestrator stack")
            return False
    logger.info(f"  API health: OK ({url})")

    # 2. Backend health (check ports via /health on orchestrator)
    try:
        resp = _get_poll_client().get(f"{url}/health", timeout=10)
        if resp.status_code == 200:
            health_data = resp.json()
            backends = health_data.get("backends", {})
            if backends:
                for name, status in backends.items():
                    ok = status.get("healthy", False) if isinstance(status, dict) else status
                    tag = "OK" if ok else "DOWN"
                    logger.info(f"  Backend {name}: {tag}")
    except Exception:
        pass  # Health endpoint may not expose backends — continue

    # 3. Smoke test (60s timeout — if 2+2 takes longer, something is broken)
    logger.info("  Smoke test: 2+2...")
    try:
        resp = _get_poll_client().post(
            f"{url}/chat",
            json={"prompt": "What is 2+2? Answer with just the number.", "real_mode": True},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("answer", "")[:50]
            routed_to = data.get("routed_to", "unknown")
            logger.info(f"  Smoke test OK: routed_to={routed_to}, answer={answer}")
        else:
            logger.error(f"  Smoke test FAIL: HTTP {resp.status_code}")
            return False
    except Exception as e:
        if "timeout" in str(e).lower() or "Timeout" in type(e).__name__:
            logger.error("  Smoke test TIMEOUT (60s) — API may be misconfigured for real_mode")
            logger.error("  Try: kill API on :8000 and relaunch, or check orchestrator_autolaunch.log")
        else:
            logger.error(f"  Smoke test FAIL: {e}")
        return False

    logger.info("PREFLIGHT PASSED")
    logger.info("=" * 60)
    return True


# ── Escalation chain detection ────────────────────────────────────────


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
    client: "httpx.Client",
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


# ── Reward injection ──────────────────────────────────────────────────


def _inject_rewards_http(
    comp: ComparativeResult,
    url: str,
    client: "httpx.Client",
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
    suite_timeout = SUITE_TIMEOUTS.get(suite, timeout)

    for combo_idx, (role, mode) in enumerate(active_combos):
        if _shutdown:
            return None

        # Before hitting a heavy model, confirm all heavy ports are idle.
        # The HTTP call blocks until the full chain completes (architect →
        # workers → synthesis → response), so the heavy model is normally
        # idle by the time we get here. This is a safety check for edge
        # cases (slot not yet freed, background KV cache cleanup, etc.).
        # Light→light transitions skip this — no need to poll.
        target_port = ROLE_PORT.get(role, 0)
        if target_port in HEAVY_PORTS:
            _wait_for_heavy_models_idle()

        key = f"{role}:{mode}"
        role_timeout = SLOW_ROLE_TIMEOUT if role in SLOW_ROLES else max(timeout, suite_timeout)
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
    logger.info(f"Seed: {seed}  Rewards: {'off' if dry_run else 'on'}")
    logger.info(f"{'='*60}\n")

    SLOW_ROLE_TIMEOUT = max(timeout, 300)
    import httpx as _httpx
    _client = _httpx.Client(timeout=SLOW_ROLE_TIMEOUT)

    new_results: list[ComparativeResult] = []
    consecutive_zero_success = 0

    try:
        for i, prompt_info in enumerate(questions):
            if _shutdown:
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

        while not _shutdown:
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
                    if _shutdown:
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

            if _shutdown:
                break

            logger.info(f"\n[Sleeping {args.continuous_interval}s before next batch...]")
            for _ in range(args.continuous_interval):
                if _shutdown:
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
        _close_poll_client()
