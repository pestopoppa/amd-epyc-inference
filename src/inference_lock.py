"""Cross-process inference lock for CPU-exclusive heavy models.

Heavy models acquire an exclusive lock. Light roles (workers/embedders)
acquire a shared lock and may run concurrently only when no heavy lock
is held. This enforces CPU exclusivity across uvicorn workers.
"""

from __future__ import annotations

import fcntl
import logging
import time
from contextlib import contextmanager
from pathlib import Path

from src.config import get_config

log = logging.getLogger(__name__)


HEAVY_ROLES = {
    "frontdoor",
    "coder_escalation",
    "architect_general",
    "architect_coding",
    "ingest_long_context",
    "vision_escalation",
}

LIGHT_ROLES = {
    "worker_explore",
    "worker_math",
    "worker_fast",
    "worker_vision",
}

# Embedders use shared lock; identify by role or context where possible.


def _lock_path() -> Path:
    configured = get_config().paths.tmp_dir / "heavy_model.lock"
    try:
        configured.parent.mkdir(parents=True, exist_ok=True)
        return configured
    except OSError:
        import tempfile
        fallback = Path(tempfile.gettempdir()) / "heavy_model.lock"
        return fallback


def _is_heavy_role(role: str) -> bool:
    if role in HEAVY_ROLES:
        return True
    if role == "coder_primary":
        return True  # alias of frontdoor
    # Default to heavy for unknown roles (safer for CPU contention)
    return role not in LIGHT_ROLES


@contextmanager
def inference_lock(role: str, shared: bool | None = None):
    """Acquire inference lock for the given role.

    Heavy roles take an exclusive lock; light roles take a shared lock.
    """
    lock_file = _lock_path()

    if shared is None:
        shared = not _is_heavy_role(role)

    lock_type = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    mode = "shared" if shared else "exclusive"

    start = time.perf_counter()
    with open(lock_file, "a") as fh:
        fcntl.flock(fh.fileno(), lock_type)
        wait_s = time.perf_counter() - start
        if wait_s > 1.0:
            log.info("Inference lock acquired (%s, role=%s) after %.2fs", mode, role, wait_s)
        try:
            yield
        finally:
            held_s = time.perf_counter() - start
            if held_s > 30.0:
                log.warning(
                    "Inference lock held %.1fs (%s, role=%s)",
                    held_s,
                    mode,
                    role,
                )
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
