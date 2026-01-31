"""Protocol interfaces for AppState component types.

Replaces Any-typed fields on AppState with structural type contracts.
Existing implementation classes (QScorer, EpisodicStore, HybridRouter, etc.)
automatically satisfy these Protocols without inheriting from them.

Usage:
    # Static type checking only (TYPE_CHECKING guard in state.py)
    from src.api.protocols import QScorerProtocol
    q_scorer: QScorerProtocol | None = None

    # Runtime isinstance checks (optional, for defense-in-depth)
    if isinstance(obj, QScorerProtocol):
        obj.score_pending_tasks()
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class QScorerProtocol(Protocol):
    """Interface for Q-value scoring of completed tasks."""

    def _score_task(self, task_id: str, mode_context: str | None = None) -> Any: ...
    def score_external_result(self, **kwargs: Any) -> dict: ...
    def score_pending_tasks(self) -> dict: ...


@runtime_checkable
class EpisodicStoreProtocol(Protocol):
    """Interface for episodic memory storage."""

    def get_action_q_summary(self) -> Any: ...


@runtime_checkable
class HybridRouterProtocol(Protocol):
    """Interface for learned + rule-based routing."""

    retriever: Any

    def route(self, task_ir: dict) -> tuple[list, str]: ...
    def route_with_mode(self, task_ir: dict) -> tuple[list, str, str]: ...


@runtime_checkable
class ProgressLoggerProtocol(Protocol):
    """Interface for task progress logging."""

    def log(self, entry: Any) -> None: ...
    def log_task_started(self, task_id: str, **kwargs: Any) -> None: ...
    def log_task_completed(self, task_id: str, success: bool, details: str) -> None: ...
    def log_escalation(self, **kwargs: Any) -> None: ...
    def log_exploration(self, **kwargs: Any) -> None: ...
    def flush(self) -> None: ...


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Interface for REPL tool registry."""

    def load_permissions_from_registry(self, registry_path: str) -> None: ...


@runtime_checkable
class ScriptRegistryProtocol(Protocol):
    """Interface for prepared script registry."""

    def load_from_directory(self, script_dir: str) -> None: ...


@runtime_checkable
class RegistryLoaderProtocol(Protocol):
    """Interface for model registry loader."""

    routing_hints: dict


@runtime_checkable
class FailureGraphProtocol(Protocol):
    """Interface for failure tracking graph."""

    def get_failure_risk(self, model_id: str) -> float: ...
    def record_failure(self, model_id: str, task_type: str, **kwargs: Any) -> None: ...
