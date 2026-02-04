"""Concurrency policy for orchestrator roles."""

from __future__ import annotations


# Small worker roles allowed to run concurrently.
_SMALL_WORKER_ROLES = frozenset(
    {
        "worker_fast",
        "worker_explore",
        "worker_math",
        "worker_general",
        "worker_vision",
    }
)


# Per-role max concurrency caps (small workers only).
_ROLE_MAX_CONCURRENCY = {
    "worker_fast": 4,
    "worker_explore": 2,
    "worker_math": 2,
    "worker_general": 2,
    "worker_vision": 2,
}


def is_small_worker_role(role: str) -> bool:
    """Return True if role is a small worker role allowed to run concurrently."""
    return role in _SMALL_WORKER_ROLES


def get_role_max_concurrency(role: str) -> int:
    """Return max concurrency for a role (defaults to 1 for large roles)."""
    return _ROLE_MAX_CONCURRENCY.get(role, 1)


def small_worker_roles() -> frozenset[str]:
    """Return the set of small worker roles."""
    return _SMALL_WORKER_ROLES
