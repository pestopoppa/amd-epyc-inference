"""Escalation helper utilities for orchestration graph execution."""

from __future__ import annotations


def detect_role_cycle(role_history: list[str]) -> bool:
    """Detect short-period role cycles that indicate escalation bouncing."""
    if len(role_history) < 4:
        return False
    if role_history[-1] == role_history[-3] and role_history[-2] == role_history[-4]:
        return True
    if len(role_history) >= 6:
        if (
            role_history[-1] == role_history[-4]
            and role_history[-2] == role_history[-5]
            and role_history[-3] == role_history[-6]
        ):
            return True
    return False

