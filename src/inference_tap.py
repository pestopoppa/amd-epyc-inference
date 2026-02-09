"""Inference tap for prompt/response visibility with low-overhead modes.

Activated by setting INFERENCE_TAP_FILE env var to a file path.
When active, model calls write prompt/response to the tap file so that
``tail -f`` shows live activity.

Streaming policy is configurable via ``INFERENCE_TAP_STREAM_MODE``:

- ``safe`` (default): stream only non-heavy roles; heavy roles use stable
  non-streaming inference and write response after completion.
- ``force``: stream all roles (highest live fidelity, highest contention risk).
- ``off``: disable streaming path even when tap is active.

Performance impact is zero when disabled — is_active() is an O(1)
env-var check with no I/O.
"""

from __future__ import annotations

import os
import threading
import time as _time
from contextlib import contextmanager
from datetime import datetime


_ENV_KEY = "INFERENCE_TAP_FILE"
_ENV_STREAM_MODE = "INFERENCE_TAP_STREAM_MODE"

# Sentinel file written by the TUI so that API workers (separate processes)
# can discover the tap path without needing the env var.
_SENTINEL = "/mnt/raid0/llm/tmp/.inference_tap_active"

# Module-level lock for serialising writes across threads
_write_lock = threading.Lock()

# Cache sentinel reads — the file only changes when the TUI starts/stops,
# so 5-second staleness is fine and avoids per-request I/O.
_sentinel_cache: tuple[str, float] = ("", 0.0)

# Roles that have shown the strongest contention/timeout behavior under
# long-held locks. In "safe" tap mode these stay on non-stream inference.
_HEAVY_STREAM_ROLES: frozenset[str] = frozenset()


def _read_sentinel() -> str:
    """Read the sentinel file, caching the result for 5 seconds."""
    global _sentinel_cache
    now = _time.monotonic()
    if now - _sentinel_cache[1] < 5.0:
        return _sentinel_cache[0]
    try:
        with open(_SENTINEL) as f:
            val = f.read().strip()
    except (FileNotFoundError, OSError):
        val = ""
    _sentinel_cache = (val, now)
    return val


def is_active() -> bool:
    """Return True when the inference tap is enabled."""
    return bool(os.environ.get(_ENV_KEY) or _read_sentinel())


def _tap_path() -> str:
    """Return the configured tap file path."""
    return os.environ.get(_ENV_KEY, "") or _read_sentinel()


def stream_mode() -> str:
    """Return normalized tap stream mode: safe|force|off."""
    mode = (os.environ.get(_ENV_STREAM_MODE, "safe") or "safe").strip().lower()
    if mode in {"safe", "force", "off"}:
        return mode
    return "safe"


def should_stream_role(role: str) -> bool:
    """Whether tap should force streaming transport for this role."""
    mode = stream_mode()
    if mode == "off":
        return False
    if mode == "force":
        return True
    return role not in _HEAVY_STREAM_ROLES


class TapWriter:
    """Thread-safe writer that appends tap output to a file.

    Keeps a per-section file descriptor open and closes on context exit.
    This avoids open/close on every streamed chunk.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fh = None

    def _append(self, text: str) -> None:
        with _write_lock:
            if self._fh is None:
                self._fh = open(self._path, "a")
            self._fh.write(text)
            self._fh.flush()

    def write_header(self, role: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append(
            f"{'=' * 72}\n"
            f"[{ts}] ROLE={role}\n"
            f"{'-' * 72}\n"
            f"PROMPT:\n"
        )

    def write_prompt(self, prompt: str, max_chars: int = 2000) -> None:
        if len(prompt) > max_chars:
            text = prompt[:max_chars] + f"\n... [{len(prompt) - max_chars} chars truncated]"
        else:
            text = prompt
        self._append(text + "\n" + "-" * 72 + "\nRESPONSE:\n")

    def write_chunk(self, chunk: str) -> None:
        """Write a single streaming chunk (called per SSE event)."""
        self._append(chunk)

    def write_response(self, text: str) -> None:
        """Write a complete response when non-stream path is used."""
        if text:
            self._append(text)

    def write_timings(
        self,
        tokens: int,
        prompt_ms: float,
        gen_ms: float,
        tps: float,
    ) -> None:
        total_s = (prompt_ms + gen_ms) / 1000.0
        self._append(
            f"\n{'-' * 72}\n"
            f"TIMINGS: {tokens} tokens in {total_s:.2f}s "
            f"(prompt={prompt_ms:.0f}ms, gen={gen_ms:.0f}ms, {tps:.1f} t/s)\n"
            f"{'=' * 72}\n\n"
        )

    def close(self) -> None:
        with _write_lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None


class _NullWriter:
    """No-op writer when tap is disabled."""

    def write_header(self, role: str) -> None:
        pass

    def write_prompt(self, prompt: str, max_chars: int = 2000) -> None:
        pass

    def write_chunk(self, chunk: str) -> None:
        pass

    def write_response(self, text: str) -> None:
        pass

    def write_timings(
        self,
        tokens: int,
        prompt_ms: float,
        gen_ms: float,
        tps: float,
    ) -> None:
        pass

    def close(self) -> None:
        pass


@contextmanager
def tap_section(role: str, prompt: str):
    """Context manager that yields a TapWriter (or _NullWriter if inactive).

    Usage::

        with tap_section(role, prompt) as tap:
            result = backend.infer_stream_text(role_config, request,
                                                on_chunk=tap.write_chunk)
            tap.write_timings(result.tokens_generated, ...)
    """
    if not is_active():
        yield _NullWriter()
        return

    path = _tap_path()
    writer = TapWriter(path)
    writer.write_header(role)
    writer.write_prompt(prompt)
    try:
        yield writer
    finally:
        writer.close()
