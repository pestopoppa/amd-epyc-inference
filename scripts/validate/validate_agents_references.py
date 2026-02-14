#!/usr/bin/env python3
"""Validate local markdown references in agent governance files."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCAN_FILES = [
    ROOT / "agents" / "README.md",
    ROOT / "agents" / "AGENT_INSTRUCTIONS.md",
    ROOT / "handoffs" / "active" / "agent-files-refactor-complete.md",
    ROOT / "docs" / "guides" / "agent-workflows" / "INDEX.md",
    ROOT / "docs" / "reference" / "agent-config" / "CLAUDE_MD_MATRIX.md",
    ROOT / ".claude" / "commands" / "agent-files.md",
    ROOT / ".claude" / "commands" / "agent-governance.md",
]

CODE_REF = re.compile(r"`([^`]+\.md)`")
MD_LINK = re.compile(r"\[[^\]]+\]\(([^)]+\.md)\)")


def resolve(ref: str, src: Path) -> Path:
    if ref.startswith("http://") or ref.startswith("https://"):
        return Path("/dev/null")
    cleaned = ref.split(":", 1)[0]
    if "*" in cleaned or "<" in cleaned or ">" in cleaned:
        return Path("/dev/null")
    if cleaned.startswith("/"):
        return Path(cleaned)
    local_target = (src.parent / cleaned).resolve()
    if local_target.exists():
        return local_target
    return (ROOT / cleaned).resolve()


def main() -> int:
    missing: list[str] = []
    for path in SCAN_FILES:
        if not path.exists():
            missing.append(f"missing file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        refs = set(CODE_REF.findall(text)) | set(MD_LINK.findall(text))
        for ref in sorted(refs):
            if ref == "SKILL.md":
                continue
            target = resolve(ref, path)
            if str(target) == "/dev/null":
                continue
            if not target.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {ref}")

    if missing:
        print("agent reference validation failed")
        for item in missing:
            print("-", item)
        return 1

    print("agent reference validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
