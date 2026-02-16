#!/usr/bin/env python3
"""Inventory numeric literals in runtime Python modules.

Produces a CSV-like report to help classify values as tunable vs invariant.
This script is intentionally non-blocking and used for planning/governance.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = [ROOT / "src", ROOT / "orchestration"]
EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tools/mined",
}


@dataclass
class Finding:
    path: str
    line: int
    literal: str
    classification_hint: str
    context: str


def _iter_py_files() -> Iterable[Path]:
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDE_PARTS for part in rel.parts):
                continue
            yield path


class LiteralVisitor(ast.NodeVisitor):
    def __init__(self, lines: list[str], path: str) -> None:
        self.lines = lines
        self.path = path
        self.findings: list[Finding] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        # Invariant-style constant assignment: UPPER_CASE = 123
        is_upper_target = any(
            isinstance(t, ast.Name) and t.id.isupper()
            for t in node.targets
        )
        if (
            isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
            and not isinstance(node.value.value, bool)
        ):
            val = node.value.value
            self.findings.append(
                Finding(
                    path=self.path,
                    line=node.lineno,
                    literal=str(val),
                    classification_hint="invariant_candidate" if is_upper_target else "review",
                    context=self.lines[node.lineno - 1].strip()[:180],
                )
            )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool):
            return
        if not isinstance(node.value, (int, float)):
            return
        # Skip common neutral literals to keep inventory readable.
        if node.value in {0, 1, 0.0, 1.0, -1}:
            return
        line = getattr(node, "lineno", 0)
        if line <= 0:
            return
        self.findings.append(
            Finding(
                path=self.path,
                line=line,
                literal=str(node.value),
                classification_hint="tunable_or_invariant",
                context=self.lines[line - 1].strip()[:180],
            )
        )


def main() -> int:
    rows: list[Finding] = []
    for path in _iter_py_files():
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
        except Exception:
            continue
        visitor = LiteralVisitor(text.splitlines(), str(path.relative_to(ROOT)))
        visitor.visit(tree)
        rows.extend(visitor.findings)

    print("path,line,literal,classification_hint,context")
    for row in sorted(rows, key=lambda r: (r.path, r.line)):
        context = row.context.replace('"', "'")
        print(f'{row.path},{row.line},"{row.literal}",{row.classification_hint},"{context}"')
    print(f"\n# total_findings={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
