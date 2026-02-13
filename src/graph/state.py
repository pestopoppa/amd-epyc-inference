"""Graph state, dependencies, and result types for orchestration graph.

TaskState is the mutable state threaded through the graph execution.
TaskDeps holds immutable references to infrastructure (LLM, REPL, MemRL).
TaskResult is the final output produced at graph termination.
GraphConfig controls retry/escalation limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

from src.escalation import ErrorCategory
from src.roles import Role

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives
    from src.repl_environment import REPLEnvironment
    from src.api.protocols import FailureGraphProtocol


# ---------------------------------------------------------------------------
# GraphConfig — controls retry/escalation limits
# ---------------------------------------------------------------------------


@dataclass
class GraphConfig:
    """Configuration for the orchestration graph.

    Defaults are intentionally sourced from the centralized config module
    so that environment variable overrides work transparently.
    """

    max_retries: int = 2
    max_escalations: int = 2
    max_turns: int = 10
    optional_gates: frozenset[str] = field(
        default_factory=lambda: frozenset({"typecheck", "integration", "shellcheck"})
    )
    no_escalate_categories: frozenset[ErrorCategory] = field(
        default_factory=lambda: frozenset({ErrorCategory.FORMAT, ErrorCategory.SCHEMA})
    )
    confidence_threshold: float = 0.7

    @classmethod
    def from_config(cls) -> GraphConfig:
        """Create from centralized config (src.config)."""
        from src.config import get_config

        esc = get_config().escalation
        return cls(
            max_retries=esc.max_retries,
            max_escalations=esc.max_escalations,
            optional_gates=esc.optional_gates,
        )


# ---------------------------------------------------------------------------
# TaskState — mutable state tracked across graph execution
# ---------------------------------------------------------------------------


@dataclass
class TaskState:
    """Mutable state that travels through the graph.

    Updated by each node's ``run()`` method. Persisted via snapshots
    when using SQLiteStatePersistence.
    """

    task_id: str = ""
    prompt: str = ""
    context: str = ""
    current_role: Role | str = Role.FRONTDOOR
    consecutive_failures: int = 0
    consecutive_nudges: int = 0
    escalation_count: int = 0
    role_history: list[str] = field(default_factory=list)
    escalation_prompt: str = ""
    last_error: str = ""
    last_output: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    turns: int = 0
    max_turns: int = 10

    # Delegation tracking
    delegation_events: list[dict] = field(default_factory=list)

    # Session compaction tracking
    compaction_count: int = 0

    # Resume token for crash recovery (Phase 3B)
    resume_token: str = ""

    # Pending approval for halt-and-resume (Phase 4A)
    pending_approval: Any = None

    # Think-harder escalation (same model, boosted config before model swap)
    think_harder_config: dict | None = None
    think_harder_attempted: bool = False
    think_harder_succeeded: bool | None = None

    # Tool requirement (from routing classification)
    tool_required: bool = False
    tool_hint: str | None = None

    # Grammar enforcement tracking
    grammar_enforced: bool = False

    # Cache affinity bonus applied during routing
    cache_affinity_bonus: float = 0.0

    def record_role(self, role: Role | str) -> None:
        """Append a role to history and update current_role."""
        self.current_role = role
        self.role_history.append(str(role))


# ---------------------------------------------------------------------------
# TaskResult — final output produced at graph termination
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Result produced when the graph reaches an End node."""

    answer: str
    success: bool
    role_history: list[str] = field(default_factory=list)
    tool_outputs: list[Any] = field(default_factory=list)
    tools_used: int = 0
    turns: int = 0
    delegation_events: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MemRLSuggestor Protocol — pluggable MemRL query interface
# ---------------------------------------------------------------------------


@runtime_checkable
class MemRLSuggestor(Protocol):
    """Protocol for querying learned escalation policy."""

    def query_escalation(
        self,
        role: str,
        error_category: str,
        error_message: str,
        failure_count: int,
    ) -> MemRLSuggestion | None: ...


@dataclass
class MemRLSuggestion:
    """Advisory suggestion from learned escalation policy."""

    action: str  # "retry", "escalate", "fail"
    target_role: str | None = None
    confidence: float = 0.0
    similar_cases: int = 0
    memory_id: str | None = None


# ---------------------------------------------------------------------------
# HypothesisGraphProtocol — for typing deps
# ---------------------------------------------------------------------------


@runtime_checkable
class HypothesisGraphProtocol(Protocol):
    """Interface for hypothesis confidence tracking."""

    def add_evidence(self, hypothesis_id: str, evidence: str, delta: float) -> None: ...
    def get_confidence(self, hypothesis_id: str) -> float: ...


# ---------------------------------------------------------------------------
# TaskDeps — immutable dependency container
# ---------------------------------------------------------------------------


@dataclass
class TaskDeps:
    """Immutable references injected into the graph at run time.

    These are not modified by graph execution — they provide the
    infrastructure that nodes need to do their work.
    """

    primitives: LLMPrimitives | None = None
    repl: REPLEnvironment | None = None
    failure_graph: FailureGraphProtocol | None = None
    hypothesis_graph: HypothesisGraphProtocol | None = None
    memrl_suggestor: MemRLSuggestor | None = None
    config: GraphConfig = field(default_factory=GraphConfig)
    progress_logger: Any = None  # ProgressLoggerProtocol
    session_store: Any = None  # SQLiteSessionStore
    approval_callback: Any = None  # ApprovalCallback protocol (Phase 4A)
