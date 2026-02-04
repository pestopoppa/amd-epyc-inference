"""Constants, dataclasses, and shared state for the seeding evaluation suite.

This module has NO project imports — it sits at the bottom of the dependency graph.
Timeouts are read directly from the model registry YAML (single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ARCHITECT_MODES", "ARCHITECT_ROLES", "ComparativeResult",
    "DEBUG_PROMPTS_DIR", "DEFAULT_MODES", "DEFAULT_ORCHESTRATOR_URL",
    "DEFAULT_ROLES", "DEFAULT_SUITES", "DEFAULT_TIMEOUT",
    "ESCALATION_REWARD", "EVAL_DIR", "HEAVY_PORTS",
    "HealthCheckError", "MODEL_PORTS", "PROJECT_ROOT",
    "ROLE_COST_TIER", "ROLE_PORT", "RoleResult",
    "SEEN_FILE", "STACK_SCRIPT",
    "VISION_MODES", "VISION_ROLES", "state",
    # Phase 4: 3-way routing action keys
    "ACTION_SELF_DIRECT", "ACTION_SELF_REPL", "ACTION_ARCHITECT", "ACTION_WORKER",
    "THREE_WAY_ACTIONS", "THREE_WAY_COST_TIER",
]


# ── Path constants ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent

EVAL_DIR = PROJECT_ROOT / "benchmarks" / "results" / "eval"
SEEN_FILE = EVAL_DIR / "seen_questions.jsonl"
DEBUG_PROMPTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "debug"


# ── Registry timeout reader (no project imports) ──────────────────────

def _read_registry_timeout(category: str, key: str, fallback: int) -> int:
    """Read timeout from model_registry.yaml without project imports."""
    registry_path = PROJECT_ROOT / "orchestration" / "model_registry.yaml"
    try:
        with registry_path.open() as f:
            data = yaml.safe_load(f)
        timeouts = data.get("runtime_defaults", {}).get("timeouts", {})
        cat_data = timeouts.get(category, {})
        return cat_data.get(key, timeouts.get("default", fallback))
    except Exception:
        return fallback


# ── Orchestrator defaults ─────────────────────────────────────────────

DEFAULT_ORCHESTRATOR_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = _read_registry_timeout("benchmark", "seeding_default", 600)
DEFAULT_SUITES = [
    # Core capability suites
    "thinking", "general", "math", "agentic",
    "coder", "instruction_precision", "vl",
    "long_context", "tool_compliance",
    # Mode-advantage suites (tool-use vs direct)
    "mode_advantage", "mode_advantage_hard",
    # HuggingFace dataset adapters (hard benchmarks)
    "gpqa", "simpleqa", "hotpotqa",
    "livecodebench", "debugbench", "usaco",
]
DEFAULT_ROLES = [
    "frontdoor", "coder_primary", "coder_escalation",
    "architect_general", "architect_coding",
    "worker_vision", "vision_escalation",
]
# NOTE: React mode has been unified into REPL with structured_mode=True.
# "react" is no longer a separate mode - REPL is the universal superset.
DEFAULT_MODES = ["direct", "repl"]


# ── Role / mode constraints ──────────────────────────────────────────

ARCHITECT_ROLES = {"architect_general", "architect_coding"}
ARCHITECT_MODES = {"direct", "delegated"}

VISION_ROLES = {"worker_vision", "vision_escalation"}
VISION_MODES: dict[str, set[str]] = {
    "worker_vision": {"direct", "react"},
    "vision_escalation": {"direct"},
}

# ── Cost / escalation constants ──────────────────────────────────────

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

ESCALATION_REWARD = 0.8


# ── Phase 4: 3-way routing action keys ───────────────────────────────
# Simplified action vocabulary for faithful probability estimation.
# Q-values converge to P(success|action), cost applied at routing time.

ACTION_SELF_DIRECT = "SELF:direct"  # Frontdoor without tools
ACTION_SELF_REPL = "SELF:repl"      # Frontdoor with tools, no delegation
ACTION_ARCHITECT = "ARCHITECT"       # Architect with full delegation freedom
ACTION_WORKER = "WORKER"             # Worker models (scored via delegation chain)

THREE_WAY_ACTIONS = [ACTION_SELF_DIRECT, ACTION_SELF_REPL, ACTION_ARCHITECT, ACTION_WORKER]

# Cost tiers for 3-way routing (applied at decision time, not during learning)
THREE_WAY_COST_TIER: dict[str, int] = {
    ACTION_SELF_DIRECT: 2,  # Frontdoor, low cost
    ACTION_SELF_REPL: 2,    # Same model, just with tools
    ACTION_ARCHITECT: 4,    # Expensive architect models
    ACTION_WORKER: 1,       # Cheapest, small worker models
}


# ── Server topology ──────────────────────────────────────────────────

HEAVY_PORTS = {8083, 8084, 8085, 8087}

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

MODEL_PORTS = [8080, 8081, 8082, 8083, 8084, 8085, 8086, 8087, 8090]

STACK_SCRIPT = PROJECT_ROOT / "scripts" / "server" / "orchestrator_stack.py"


# ── Exceptions ────────────────────────────────────────────────────────


class HealthCheckError(Exception):
    """Raised when the orchestrator API is unreachable."""

    pass


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class RoleResult:
    """Result of running a question through a specific role+mode."""

    role: str
    mode: str
    answer: str
    passed: bool
    elapsed_seconds: float
    error: str | None = None
    error_type: str = "none"
    tokens_generated: int = 0
    tools_used: int = 0
    tools_called: list[str] = field(default_factory=list)
    delegation_events: list[dict] = field(default_factory=list)
    tools_success: bool | None = None
    delegation_success: bool | None = None
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


# ── Shared mutable state ─────────────────────────────────────────────


class _State:
    """Process-wide mutable state shared across all seeding modules.

    Replaces module-level globals (_shutdown, _poll_client) with an
    explicit singleton so signal handlers and infra code can coordinate.
    """

    def __init__(self) -> None:
        self.shutdown: bool = False
        self._poll_client: "Any" = None  # httpx.Client, lazily created

    def get_poll_client(self) -> "Any":
        """Get or create the connection-reusing httpx client for polling."""
        if self._poll_client is None:
            import httpx
            self._poll_client = httpx.Client(timeout=10)
        return self._poll_client

    def close_poll_client(self) -> None:
        """Close the polling client if open."""
        if self._poll_client is not None:
            try:
                self._poll_client.close()
            except Exception:
                pass
            self._poll_client = None


state = _State()
