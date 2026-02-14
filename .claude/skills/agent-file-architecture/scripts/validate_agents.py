#!/usr/bin/env python3
"""Run all agent-governance validators."""

from __future__ import annotations

import subprocess
import sys

CMDS = [
    ["python3", "scripts/validate/validate_agents_structure.py"],
    ["python3", "scripts/validate/validate_agents_references.py"],
]


def main() -> int:
    for cmd in CMDS:
        print("$", " ".join(cmd))
        res = subprocess.run(cmd)
        if res.returncode != 0:
            return res.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
