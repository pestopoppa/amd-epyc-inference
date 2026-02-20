"""REPL tap file helpers for graph execution diagnostics."""

from __future__ import annotations

import threading

REPL_TAP_PATH = "/mnt/raid0/llm/tmp/repl_tap.log"
_repl_tap_lock = threading.Lock()


def tap_write_repl_exec(code: str, turn: int) -> None:
    """Write REPL execution input to the REPL tap file."""
    try:
        preview = code[:4000]
        if len(code) > 4000:
            preview += f"\n... [{len(code) - 4000} chars truncated]"
        text = (
            f"[turn {turn}] $ python3 <<'CODE'\n"
            f"{preview}\n"
            f"CODE\n"
        )
        with _repl_tap_lock:
            with open(REPL_TAP_PATH, "a") as f:
                f.write(text)
                f.flush()
    except Exception:
        pass


def tap_write_repl_result(
    output: str, error: str | None, is_final: bool, turn: int,
) -> None:
    """Write REPL execution result to the REPL tap file."""
    try:
        parts: list[str] = []
        if is_final:
            parts.append(f"[turn {turn}] FINAL")
        if output:
            out_preview = output[:4000]
            if len(output) > 4000:
                out_preview += f"\n... [{len(output) - 4000} chars truncated]"
            parts.append(out_preview)
        elif not error:
            parts.append(f"[turn {turn}] (no output)")
        if error:
            err_preview = error[:2000]
            if len(error) > 2000:
                err_preview += f"\n... [{len(error) - 2000} chars truncated]"
            parts.append(f"[turn {turn}] ERROR:\n{err_preview}")
        parts.append("")  # trailing newline
        text = "\n".join(parts)
        with _repl_tap_lock:
            with open(REPL_TAP_PATH, "a") as f:
                f.write(text)
                f.flush()
    except Exception:
        pass

