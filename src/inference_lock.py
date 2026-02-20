"""Cross-process inference lock for CPU-exclusive heavy models.

Heavy models acquire an exclusive lock. Light roles (workers/embedders)
acquire a shared lock and may run concurrently only when no heavy lock
is held. This enforces CPU exclusivity across uvicorn workers.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from src.config import get_config
from src.env_parsing import env_float as _env_float

log = logging.getLogger(__name__)

# Cache per-port slot erase strategy to avoid re-probing on every call.
_SLOT_ERASE_CAPABILITY: dict[int, str | None | bool] = {}


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


def _lock_timeout_s(shared: bool) -> float:
    # Defaults keep legacy behavior (blocking) from turning into hard failures
    # too aggressively, while still preventing truly unbounded hangs.
    default_timeout = 180.0
    key_specific = (
        "ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_SHARED_S"
        if shared
        else "ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_EXCLUSIVE_S"
    )
    if key_specific in os.environ:
        return _env_float(key_specific, default_timeout)
    return _env_float("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S", default_timeout)


def _lock_poll_s() -> float:
    # Retry cadence for LOCK_NB acquisition loop.
    return max(0.005, _env_float("ORCHESTRATOR_INFERENCE_LOCK_POLL_MS", 50.0) / 1000.0)


def _lock_log_every_s() -> float:
    # Emit periodic "still waiting" diagnostics while blocked.
    return max(1.0, _env_float("ORCHESTRATOR_INFERENCE_LOCK_LOG_EVERY_S", 15.0))


def _lock_trace_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_INFERENCE_LOCK_TRACE", "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _current_lock_owner_pids(lock_file: Path) -> list[str]:
    """Best-effort lock owners from /proc/locks for the target file inode."""
    try:
        inode = str(lock_file.stat().st_ino)
    except Exception:
        return []
    owners: set[str] = set()
    try:
        with open("/proc/locks", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 6:
                    continue
                pid = parts[4]
                dev_inode = parts[5]
                if not pid.isdigit():
                    continue
                if dev_inode.rsplit(":", 1)[-1] == inode:
                    owners.add(pid)
    except Exception:
        return []
    return sorted(owners)


def _erase_port_slots(port: int) -> None:
    """Fire-and-forget slot erase on a llama-server port.

    Lightweight version of the benchmark ``_erase_slots``: queries /slots,
    erases any processing slot, caches the working strategy per port.
    """
    try:
        import httpx
    except ImportError:
        log.debug("httpx not available; skipping slot erase on port %d", port)
        return

    cap = _SLOT_ERASE_CAPABILITY.get(port)
    if cap is False:
        return

    try:
        resp = httpx.get(f"http://localhost:{port}/slots", timeout=5)
        if resp.status_code != 200:
            return
        for slot in resp.json():
            if not slot.get("is_processing"):
                continue
            slot_id = slot.get("id", 0)
            strategies: list[str]
            if isinstance(cap, str):
                strategies = [cap]
            else:
                strategies = ["POST_QUERY", "GET_QUERY", "POST_JSON"]

            for strategy in strategies:
                try:
                    if strategy == "POST_QUERY":
                        r = httpx.post(
                            f"http://localhost:{port}/slots/{slot_id}?action=erase",
                            timeout=5,
                        )
                    elif strategy == "GET_QUERY":
                        r = httpx.get(
                            f"http://localhost:{port}/slots/{slot_id}?action=erase",
                            timeout=5,
                        )
                    elif strategy == "POST_JSON":
                        r = httpx.post(
                            f"http://localhost:{port}/slots/{slot_id}",
                            json={"action": "erase"},
                            timeout=5,
                        )
                    else:
                        continue
                    if r.status_code == 200:
                        _SLOT_ERASE_CAPABILITY[port] = strategy
                        log.info("Erased slot %d on port %d (strategy=%s)", slot_id, port, strategy)
                        break
                    if r.status_code in {404, 405, 501}:
                        continue
                except Exception:
                    continue
            else:
                # No strategy worked — if we had a cached one, reset it.
                if isinstance(cap, str):
                    _SLOT_ERASE_CAPABILITY[port] = None
    except Exception as e:
        log.debug("Slot erase failed on port %d: %s", port, e)


def _lock_holder_ports(lock_file: Path) -> list[int]:
    """Map current lock holder PIDs to llama-server ports via /proc/cmdline."""
    pids = _current_lock_owner_pids(lock_file)
    ports: list[int] = []
    for pid in pids:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="replace")
            # cmdline is NUL-separated
            args = cmdline.split("\x00")
            for i, arg in enumerate(args):
                if arg == "--port" and i + 1 < len(args):
                    port_str = args[i + 1].strip()
                    if port_str.isdigit():
                        ports.append(int(port_str))
                    break
        except Exception:
            continue
    return ports


def _acquire_lock_with_timeout(
    fd: int,
    lock_type: int,
    role: str,
    mode: str,
    lock_file: Path,
    timeout_s: float,
    poll_s: float,
    log_every_s: float,
    cancel_check: Callable[[], bool] | None = None,
    deadline_s: float | None = None,
    request_tag: str | None = None,
) -> float:
    """Acquire fcntl lock with periodic diagnostics and bounded wait.

    Returns:
        Wait seconds before acquisition.

    Raises:
        TimeoutError: If lock couldn't be acquired within timeout_s (>0).
        OSError: Unexpected flock/system errors.
    """
    start = time.perf_counter()
    deadline = None if timeout_s <= 0 else (start + timeout_s)
    last_log = start

    while True:
        if cancel_check is not None:
            try:
                if cancel_check():
                    raise TimeoutError(
                        "Inference lock timeout (cancelled) "
                        f"(role={role}, mode={mode}, lock={lock_file}, request={request_tag or 'n/a'})"
                    )
            except TimeoutError:
                raise
            except Exception:
                # Ignore cancellation-check failures; lock behavior should remain safe.
                pass
        if deadline_s is not None and time.perf_counter() >= deadline_s:
            raise TimeoutError(
                "Inference lock timeout (request deadline exceeded) "
                f"(role={role}, mode={mode}, lock={lock_file}, request={request_tag or 'n/a'})"
            )
        try:
            fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
            return time.perf_counter() - start
        except (BlockingIOError, OSError) as exc:
            if isinstance(exc, OSError) and exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise

            now = time.perf_counter()
            waited = now - start
            if deadline is not None and now >= deadline:
                raise TimeoutError(
                    "Inference lock timeout "
                    f"(role={role}, mode={mode}, waited={waited:.2f}s, lock={lock_file}, request={request_tag or 'n/a'})"
                ) from exc

            if now - last_log >= log_every_s:
                holder_meta = ""
                try:
                    pids = _current_lock_owner_pids(lock_file)
                    if not pids:
                        result = subprocess.run(
                            ["lsof", "-t", str(lock_file)],
                            capture_output=True,
                            text=True,
                            timeout=1.0,
                        )
                        pids = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
                    if pids:
                        holder_pids = pids[:6]
                        holder_meta = f", holders={','.join(holder_pids)}"
                        try:
                            ps_out = subprocess.run(
                                ["ps", "-o", "pid=,cmd=", "-p", ",".join(holder_pids)],
                                capture_output=True,
                                text=True,
                                timeout=1.0,
                            )
                            lines = [
                                " ".join(line.strip().split())
                                for line in ps_out.stdout.splitlines()
                                if line.strip()
                            ]
                            if lines:
                                holder_meta += f", holder_cmds={'; '.join(lines[:3])[:220]}"
                        except Exception:
                            pass
                except Exception:
                    holder_meta = ""
                log.warning(
                    "Inference lock wait ongoing (role=%s, mode=%s, waited=%.2fs, lock=%s, request=%s%s)",
                    role,
                    mode,
                    waited,
                    lock_file,
                    request_tag or "n/a",
                    holder_meta,
                )
                last_log = now

            time.sleep(poll_s)


def _lock_filename_for_role(role: str) -> str:
    """Resolve lock filename for role, with optional embedder isolation."""
    role_norm = role.strip().lower()
    if role_norm == "embedder" or role_norm.startswith("embedder_"):
        # Keep embedder activity out of the heavy-model lock domain by default.
        return os.environ.get(
            "ORCHESTRATOR_INFERENCE_LOCK_EMBEDDER_FILE",
            "embedder_model.lock",
        )
    return os.environ.get("ORCHESTRATOR_INFERENCE_LOCK_FILE", "heavy_model.lock")


def _lock_path(role: str) -> Path:
    filename = _lock_filename_for_role(role).strip() or "heavy_model.lock"
    configured = get_config().paths.tmp_dir / filename
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
    # Default to heavy for unknown roles (safer for CPU contention)
    return role not in LIGHT_ROLES


@contextmanager
def inference_lock(
    role: str,
    shared: bool | None = None,
    cancel_check: Callable[[], bool] | None = None,
    deadline_s: float | None = None,
    request_tag: str | None = None,
    port: int | None = None,
):
    """Acquire inference lock for the given role.

    Heavy roles take an exclusive lock; light roles take a shared lock.

    Args:
        port: llama-server port for this role. When provided, enables
            slot cleanup: on lock timeout (erase holder's slots) and on
            error inside the lock (erase our own slots).
    """
    lock_file = _lock_path(role)

    if shared is None:
        shared = not _is_heavy_role(role)

    lock_type = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    mode = "shared" if shared else "exclusive"
    timeout_s = _lock_timeout_s(shared)
    poll_s = _lock_poll_s()
    log_every_s = _lock_log_every_s()

    with open(lock_file, "a") as fh:
        try:
            wait_s = _acquire_lock_with_timeout(
                fh.fileno(),
                lock_type,
                role=role,
                mode=mode,
                lock_file=lock_file,
                timeout_s=timeout_s,
                poll_s=poll_s,
                log_every_s=log_every_s,
                cancel_check=cancel_check,
                deadline_s=deadline_s,
                request_tag=request_tag,
            )
        except TimeoutError:
            # Lock acquisition failed — the holder is still generating tokens.
            # Erase slots on the holder's port(s) to free resources.
            try:
                holder_ports = _lock_holder_ports(lock_file)
                for hp in holder_ports:
                    log.warning(
                        "Lock timeout: erasing slots on holder port %d (role=%s, request=%s)",
                        hp, role, request_tag or "n/a",
                    )
                    _erase_port_slots(hp)
            except Exception as erase_exc:
                log.debug("Slot erase on lock timeout failed: %s", erase_exc)
            raise

        if wait_s > 1.0:
            log.info("Inference lock acquired (%s, role=%s) after %.2fs", mode, role, wait_s)
        if _lock_trace_enabled():
            log.warning(
                "Inference lock acquire trace pid=%d role=%s mode=%s request=%s wait_s=%.3f lock=%s",
                os.getpid(),
                role,
                mode,
                request_tag or "n/a",
                wait_s,
                lock_file,
            )
        acquired_at = time.perf_counter()
        _inner_error = False
        try:
            yield
        except BaseException:
            _inner_error = True
            raise
        finally:
            held_s = time.perf_counter() - acquired_at
            if held_s > 30.0:
                log.warning(
                    "Inference lock held %.1fs (%s, role=%s)",
                    held_s,
                    mode,
                    role,
                )
            if _lock_trace_enabled():
                log.warning(
                    "Inference lock release trace pid=%d role=%s mode=%s request=%s held_s=%.3f lock=%s",
                    os.getpid(),
                    role,
                    mode,
                    request_tag or "n/a",
                    held_s,
                    lock_file,
                )
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            # If inference inside the lock failed (timeout/cancel/error),
            # erase our own slot so the backend stops generating tokens.
            if _inner_error and port is not None:
                try:
                    log.info(
                        "Post-lock cleanup: erasing slots on port %d (role=%s, request=%s)",
                        port, role, request_tag or "n/a",
                    )
                    _erase_port_slots(port)
                except Exception as erase_exc:
                    log.debug("Post-lock slot erase failed: %s", erase_exc)
