"""Speculative prefill pre-warming for architect model servers.

When classify_task_complexity() returns COMPLEX at request ingestion, fires
a non-blocking n_predict=0, cache_prompt=true request to the architect server
with the system prompt prefix. This warms the KV cache before escalation
actually happens.

Saving: ~500 tokens of system prompt prefix at 1.2 t/s = 417ms per architect
escalation that hits the pre-warmed slot.

Risk: Medium. Pre-warm occupies a slot. We check /slots before pre-warming
to avoid evicting hot slots.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Architect server ports (from model routing strategy)
ARCHITECT_PORTS = {
    "architect_general": 8083,
    "architect_coding": 8084,
}

# System prompt prefix used across architect roles (warm this into KV cache)
ARCHITECT_SYSTEM_PREFIX = (
    "You are a senior software architect. Analyze the task, identify components, "
    "assess complexity, and provide a structured response. Your role is to reason "
    "about architecture, design patterns, and system-level concerns.\n\n"
)


class EscalationPrewarmer:
    """Speculatively pre-warms architect KV cache for complex tasks.

    Usage:
        prewarmer = EscalationPrewarmer()
        # At turn 1, fire and forget:
        asyncio.create_task(prewarmer.prewarm_if_complex(objective, complexity))
    """

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout
        self._prewarm_count = 0
        self._prewarm_hits = 0  # Incremented externally when prewarm slot is used
        self._prewarm_by_port: dict[int, int] = {}
        self._prewarm_hits_by_role: dict[str, int] = {}
        self._lock = threading.Lock()

    async def prewarm_if_complex(
        self,
        objective: str,
        complexity_level: str,
        target_port: int | None = None,
    ) -> bool:
        """Pre-warm architect slot if task is complex.

        Args:
            objective: Task description.
            complexity_level: Result of classify_task_complexity() (as string).
            target_port: Override architect port (default: architect_general).

        Returns:
            True if pre-warm was sent successfully.
        """
        if complexity_level not in ("COMPLEX",):
            return False

        port = target_port or ARCHITECT_PORTS.get("architect_general", 8083)

        # Check if slot is available before pre-warming
        slot_available = await self._check_slot_available(port)
        if not slot_available:
            logger.debug("Pre-warm skipped: no idle slot on port %d", port)
            return False

        # Send non-blocking prefill request
        return await self._send_prewarm(port, objective)

    async def _check_slot_available(self, port: int) -> bool:
        """Check /slots endpoint for an idle slot."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://localhost:{port}/slots",
                    timeout=self._timeout,
                )
                if resp.status_code != 200:
                    return False
                data = resp.json()
                # llama-server returns a list for -np>1, a single dict for -np=1
                slots = data if isinstance(data, list) else [data]
                # Check is_processing (modern llama-server) or state==0 (legacy)
                idle_slots = [
                    s for s in slots
                    if not s.get("is_processing", True) or s.get("state") == 0
                ]
                return len(idle_slots) > 0
        except Exception as e:
            logger.debug("Slot check failed for port %d: %s", port, e)
            return False

    async def _send_prewarm(self, port: int, objective: str) -> bool:
        """Send a n_predict=0, cache_prompt=true request to warm KV cache."""
        # Build the prefix that will be shared with the actual escalation
        prewarm_prompt = ARCHITECT_SYSTEM_PREFIX + objective[:2000]

        payload = {
            "prompt": prewarm_prompt,
            "n_predict": 0,  # No generation, just prefill
            "cache_prompt": True,  # Cache the KV
        }

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://localhost:{port}/completion",
                    json=payload,
                    timeout=self._timeout,
                )
                if resp.status_code == 200:
                    with self._lock:
                        self._prewarm_count += 1
                        self._prewarm_by_port[port] = self._prewarm_by_port.get(port, 0) + 1
                    logger.info(
                        "Pre-warmed architect slot on port %d (prompt: %d chars)",
                        port,
                        len(prewarm_prompt),
                    )
                    return True
                logger.debug("Pre-warm failed: HTTP %d", resp.status_code)
                return False
        except Exception as e:
            logger.debug("Pre-warm request failed for port %d: %s", port, e)
            return False

    def record_prewarm_hit(self, role: str) -> None:
        """Record that execution actually escalated to an architect role."""
        role_key = str(role or "").strip() or "unknown"
        with self._lock:
            self._prewarm_hits += 1
            self._prewarm_hits_by_role[role_key] = (
                self._prewarm_hits_by_role.get(role_key, 0) + 1
            )

    def get_stats(self) -> dict[str, Any]:
        """Get pre-warming statistics."""
        with self._lock:
            prewarm_count = self._prewarm_count
            prewarm_hits = self._prewarm_hits
            prewarm_by_port = dict(self._prewarm_by_port)
            prewarm_hits_by_role = dict(self._prewarm_hits_by_role)
        return {
            "prewarm_count": prewarm_count,
            "prewarm_hits": prewarm_hits,
            "prewarm_by_port": prewarm_by_port,
            "prewarm_hits_by_role": prewarm_hits_by_role,
            "hit_rate": (
                prewarm_hits / prewarm_count
                if prewarm_count > 0
                else 0.0
            ),
        }


_shared_prewarmer: EscalationPrewarmer | None = None
_shared_lock = threading.Lock()


def get_shared_prewarmer() -> EscalationPrewarmer:
    """Return the process-wide prewarmer used by orchestration helpers."""
    global _shared_prewarmer
    if _shared_prewarmer is None:
        with _shared_lock:
            if _shared_prewarmer is None:
                _shared_prewarmer = EscalationPrewarmer()
    return _shared_prewarmer
