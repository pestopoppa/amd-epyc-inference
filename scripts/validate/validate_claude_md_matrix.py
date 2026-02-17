#!/usr/bin/env python3
"""Validate CLAUDE.md governance matrix artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX_MD = ROOT / "docs/reference/agent-config/CLAUDE_MD_MATRIX.md"
MATRIX_JSON = ROOT / "docs/reference/agent-config/claude_md_matrix.json"

REQUIRED_GOVERNED = {
    "CLAUDE.md",
}


def main() -> int:
    if not MATRIX_MD.exists():
        print("missing", MATRIX_MD.relative_to(ROOT))
        return 1
    if not MATRIX_JSON.exists():
        print("missing", MATRIX_JSON.relative_to(ROOT))
        return 1

    text = MATRIX_MD.read_text(encoding="utf-8")
    for p in REQUIRED_GOVERNED:
        if p not in text:
            print("missing from md matrix:", p)
            return 1

    data = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    governed = {entry.get("path") for entry in data.get("governed", [])}
    missing = sorted(REQUIRED_GOVERNED - governed)
    if missing:
        print("missing from json matrix:", ", ".join(missing))
        return 1

    print("claude md matrix validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
