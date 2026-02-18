"""Delegation report artifact persistence and lazy retrieval helpers."""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

_REPORT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{6,128}$")
_HANDLE_RE = re.compile(r"\[REPORT_HANDLE\s+id=([A-Za-z0-9_.:-]+)")


def report_dir() -> Path:
    """Return report artifact directory."""
    default = "/mnt/raid0/llm/claude/tmp/delegation_reports"
    return Path(os.environ.get("ORCHESTRATOR_DELEGATION_REPORT_DIR", default))


def store_report(report: str, delegate_to: str) -> dict[str, str] | None:
    """Persist full delegation report and return a compact handle."""
    text = (report or "").strip()
    if not text:
        return None
    d = report_dir()
    d.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    rid = f"{delegate_to}-{ts}-{digest}"
    path = d / f"{rid}.txt"
    path.write_text(text)
    return {
        "id": rid,
        "path": str(path),
        "chars": str(len(text)),
        "sha16": digest,
    }


def extract_handle_id(text: str) -> str | None:
    """Extract report handle id from compact REPORT_HANDLE text."""
    m = _HANDLE_RE.search(text or "")
    return m.group(1) if m else None


def load_report(
    report_id: str,
    *,
    offset: int = 0,
    max_chars: int = 2400,
) -> dict[str, Any]:
    """Load a slice of a persisted delegation report."""
    rid = (report_id or "").strip()
    if not _REPORT_ID_RE.match(rid):
        return {"ok": False, "error": "invalid_report_id", "report_id": rid}

    safe_offset = max(0, int(offset))
    safe_max = max(64, min(int(max_chars), 12000))
    path = report_dir() / f"{rid}.txt"
    if not path.is_file():
        return {
            "ok": False,
            "error": "report_not_found",
            "report_id": rid,
            "path": str(path),
        }

    text = path.read_text()
    total_chars = len(text)
    if safe_offset >= total_chars:
        return {
            "ok": True,
            "report_id": rid,
            "path": str(path),
            "total_chars": total_chars,
            "offset": safe_offset,
            "next_offset": safe_offset,
            "truncated": False,
            "content": "",
        }

    end = min(total_chars, safe_offset + safe_max)
    chunk = text[safe_offset:end]
    return {
        "ok": True,
        "report_id": rid,
        "path": str(path),
        "total_chars": total_chars,
        "offset": safe_offset,
        "next_offset": end,
        "truncated": end < total_chars,
        "content": chunk,
    }
