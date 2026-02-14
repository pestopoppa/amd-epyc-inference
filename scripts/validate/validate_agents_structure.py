#!/usr/bin/env python3
"""Validate required role schema in agents/*.md files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "agents"

EXCLUDED = {
    "README.md",
    "AGENT_INSTRUCTIONS.md",
    "research-writer-guide.md",
}
REQUIRED = [
    "## Mission",
    "## Use This Role When",
    "## Inputs Required",
    "## Outputs",
    "## Workflow",
    "## Guardrails",
]


def main() -> int:
    failures: list[str] = []
    for path in sorted(AGENTS.glob("*.md")):
        if path.name in EXCLUDED:
            continue
        text = path.read_text(encoding="utf-8")
        missing = [h for h in REQUIRED if h not in text]
        if missing:
            failures.append(f"{path.relative_to(ROOT)} missing: {', '.join(missing)}")

    if failures:
        print("agent structure validation failed")
        for item in failures:
            print("-", item)
        return 1

    print("agent structure validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
