"""Application state container for the orchestrator API.

This module provides the global application state and thread-safe operations
for statistics tracking and MemRL component management.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives
    from src.gate_runner import GateRunner
    from src.failure_router import FailureRouter

logger = logging.getLogger(__name__)

# Type hints for optional imports (set at runtime by _load_optional_imports)
ProgressLogger: type | None = None
QScorer: type | None = None
EpisodicStore: type | None = None
HybridRouter: type | None = None
ToolRegistry: type | None = None
ScriptRegistry: type | None = None


@dataclass
class AppState:
    """Application state container.

    Thread-safe statistics tracking using locks for concurrent request handling.

    Attributes:
        llm_primitives: LLM abstraction layer for inference.
        gate_runner: Quality gate execution engine.
        failure_router: Escalation routing logic.
        progress_logger: MemRL progress logger (optional).
        q_scorer: Q-value scorer for MemRL (optional).
        episodic_store: Episodic memory store for MemRL (optional).
        hybrid_router: Learned + rule-based routing (optional).
        tool_registry: Tool permissions registry (optional).
        script_registry: Script registry (optional).
        registry: Model registry loader (optional).
    """

    llm_primitives: LLMPrimitives | None = None
    gate_runner: GateRunner | None = None
    failure_router: FailureRouter | None = None
    progress_logger: Any | None = None  # ProgressLogger type

    # Q-scorer components (for idle-time scoring)
    q_scorer: Any | None = None  # QScorer type
    episodic_store: Any | None = None  # EpisodicStore type

    # Hybrid routing (learned + rule-based)
    hybrid_router: Any | None = None  # HybridRouter type

    # Tool and script registries for REPL
    tool_registry: Any | None = None  # ToolRegistry type
    script_registry: Any | None = None  # ScriptRegistry type

    # Registry loader (for role defaults)
    registry: Any = None  # RegistryLoader, typed as Any to avoid import cycle

    # Stats tracking (protected by _stats_lock)
    total_requests: int = 0
    total_turns: int = 0
    mock_requests: int = 0
    real_requests: int = 0

    # Idle scoring control (protected by _stats_lock)
    active_requests: int = 0
    q_scorer_enabled: bool = True
    _q_scorer_task: Any = None  # asyncio.Task

    # Lazy initialization flag for MemRL components
    # These are only loaded when real inference or Q-scoring is needed
    _memrl_initialized: bool = False

    # Thread safety lock for statistics
    _stats_lock: threading.Lock = field(default_factory=threading.Lock)

    def increment_request(self, mock_mode: bool, turns: int) -> None:
        """Track a completed request (thread-safe).

        Args:
            mock_mode: Whether the request used mock mode.
            turns: Number of orchestration turns used.
        """
        with self._stats_lock:
            self.total_requests += 1
            self.total_turns += turns
            if mock_mode:
                self.mock_requests += 1
            else:
                self.real_requests += 1

    def increment_active(self) -> None:
        """Increment active request counter (thread-safe)."""
        with self._stats_lock:
            self.active_requests += 1

    def decrement_active(self) -> None:
        """Decrement active request counter (thread-safe)."""
        with self._stats_lock:
            self.active_requests -= 1

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics (thread-safe snapshot).

        Returns:
            Dictionary with statistics.
        """
        with self._stats_lock:
            return {
                "total_requests": self.total_requests,
                "total_turns": self.total_turns,
                "average_turns_per_request": (
                    self.total_turns / self.total_requests
                    if self.total_requests > 0
                    else 0.0
                ),
                "mock_requests": self.mock_requests,
                "real_requests": self.real_requests,
                "active_requests": self.active_requests,
            }

    def reset_stats(self) -> None:
        """Reset all statistics (thread-safe)."""
        with self._stats_lock:
            self.total_requests = 0
            self.total_turns = 0
            self.mock_requests = 0
            self.real_requests = 0


# Global application state singleton
_state: AppState | None = None


def get_state() -> AppState:
    """Get the global application state.

    Creates a new AppState if one doesn't exist.

    Returns:
        The global AppState instance.
    """
    global _state
    if _state is None:
        _state = AppState()
    return _state


def set_state(state: AppState) -> None:
    """Set the global application state.

    Primarily used for testing.

    Args:
        state: The AppState instance to use.
    """
    global _state
    _state = state


def reset_state() -> None:
    """Reset the global application state.

    Primarily used for testing.
    """
    global _state
    _state = None
