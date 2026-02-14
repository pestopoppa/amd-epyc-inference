#!/usr/bin/env python3
"""Simple checker for CLAUDE.md governance matrix presence and required entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[4]
    matrix = root / "docs/reference/agent-config/claude_md_matrix.json"
    if not matrix.exists():
        print("missing", matrix)
        return 1

    data = json.loads(matrix.read_text())
    governed = {item["path"] for item in data.get("governed", [])}
    required = {"CLAUDE.md", "kernel-dev/llama-cpp-dev/CLAUDE.md"}

    missing = sorted(required - governed)
    if missing:
        print("missing governed entries:", ", ".join(missing))
        return 1

    if args.check:
        print("claude-md matrix check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
