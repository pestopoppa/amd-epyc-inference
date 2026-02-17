"""Priority-ordered routing bindings (OpenClaw pattern).

Provides routing overrides that can come from multiple sources
(registry defaults, classifier heuristics, Q-values, user preference,
session state). Higher-priority bindings win.

Only active when features().binding_routing is True.

Priority levels (mapped from OpenClaw's peer→guild→channel→default):
    DEFAULT (0)    — model_registry.yaml task_type → role
    CLASSIFIER (10) — keyword heuristic from _classify_and_route()
    Q_VALUE (20)   — MemRL Q-value suggestion
    USER_PREF (30) — ChatRequest.preferred_role header
    SESSION (40)   — Session-specific override

Usage:
    from src.routing_bindings import BindingRouter, RoutingBinding, BindingPriority

    router = BindingRouter()
    router.add(RoutingBinding("code", "coder_escalation", BindingPriority.DEFAULT))
    router.add(RoutingBinding("code", "architect_coding", BindingPriority.USER_PREF))

    resolved = router.resolve("code")  # Returns "architect_coding" (higher priority)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

log = logging.getLogger(__name__)


class BindingPriority(IntEnum):
    """Priority levels for routing bindings. Higher wins."""

    DEFAULT = 0  # model_registry.yaml task_type → role
    CLASSIFIER = 10  # _classify_and_route() keyword heuristic
    Q_VALUE = 20  # MemRL Q-value suggestion
    USER_PREF = 30  # ChatRequest.preferred_role header
    SESSION = 40  # Session-specific override (during conversation)


@dataclass
class RoutingBinding:
    """A single routing override entry."""

    task_type: str  # e.g., "code", "ingest", "explore"
    role: str  # e.g., "coder_escalation", "architect_coding"
    priority: BindingPriority = BindingPriority.DEFAULT
    source: str = ""  # Human-readable source description
    active: bool = True


class BindingRouter:
    """Priority-ordered routing resolver.

    Maintains a set of bindings per task type. When resolving,
    returns the role from the highest-priority active binding.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, list[RoutingBinding]] = {}

    def add(self, binding: RoutingBinding) -> None:
        """Add a routing binding.

        Args:
            binding: The binding to add.
        """
        if binding.task_type not in self._bindings:
            self._bindings[binding.task_type] = []
        self._bindings[binding.task_type].append(binding)
        log.debug(
            "Binding added: %s → %s (priority=%d, source=%s)",
            binding.task_type, binding.role, binding.priority, binding.source,
        )

    def resolve(self, task_type: str) -> str | None:
        """Resolve the highest-priority role for a task type.

        Args:
            task_type: The task type to resolve.

        Returns:
            Role string from highest-priority active binding,
            or None if no bindings exist.
        """
        bindings = self._bindings.get(task_type, [])
        active = [b for b in bindings if b.active]
        if not active:
            return None

        winner = max(active, key=lambda b: b.priority)
        return winner.role

    def resolve_with_info(self, task_type: str) -> tuple[str | None, BindingPriority | None, str]:
        """Resolve with full binding info.

        Returns:
            (role, priority, source) tuple.
        """
        bindings = self._bindings.get(task_type, [])
        active = [b for b in bindings if b.active]
        if not active:
            return None, None, ""

        winner = max(active, key=lambda b: b.priority)
        return winner.role, winner.priority, winner.source

    def set_session_binding(self, task_type: str, role: str) -> None:
        """Set a session-level binding override (highest priority).

        Args:
            task_type: Task type to override.
            role: Role to route to.
        """
        # Remove existing session bindings for this task type
        if task_type in self._bindings:
            self._bindings[task_type] = [
                b for b in self._bindings[task_type]
                if b.priority != BindingPriority.SESSION
            ]

        self.add(RoutingBinding(
            task_type=task_type,
            role=role,
            priority=BindingPriority.SESSION,
            source="session_override",
        ))

    def clear_session_bindings(self) -> None:
        """Clear all session-level bindings (end of conversation)."""
        for task_type in self._bindings:
            self._bindings[task_type] = [
                b for b in self._bindings[task_type]
                if b.priority != BindingPriority.SESSION
            ]

    def list_bindings(self, task_type: str | None = None) -> list[dict]:
        """List all bindings, optionally filtered by task type.

        Args:
            task_type: Optional filter.

        Returns:
            List of binding info dicts.
        """
        result = []
        types = [task_type] if task_type else list(self._bindings.keys())
        for tt in types:
            for b in self._bindings.get(tt, []):
                result.append({
                    "task_type": b.task_type,
                    "role": b.role,
                    "priority": b.priority.name,
                    "priority_value": int(b.priority),
                    "source": b.source,
                    "active": b.active,
                })
        return result

    def clear(self) -> None:
        """Remove all bindings."""
        self._bindings.clear()

    def prior_distribution(self, task_type: str) -> dict[str, float]:
        """Return a soft prior distribution over roles for a task type.

        Converts binding priorities into normalized weights instead of a hard winner.
        """
        bindings = self._bindings.get(task_type, [])
        active = [b for b in bindings if b.active]
        if not active:
            return {}

        # Shift priority by +1 so DEFAULT contributes non-zero mass.
        masses: dict[str, float] = {}
        for b in active:
            masses[b.role] = masses.get(b.role, 0.0) + float(int(b.priority) + 1)
        total = sum(masses.values())
        if total <= 0:
            return {}
        return {role: mass / total for role, mass in masses.items()}
