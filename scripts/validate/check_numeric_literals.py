#!/usr/bin/env python3
"""Soft governance check for numeric literals in runtime Python code.

Modes:
- report: Non-blocking inventory summary (always exits 0)
- enforce-changed: Fails if changed files contain unclassified literals
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback path
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWLIST = ROOT / "scripts" / "validate" / "numeric_literal_allowlist.yaml"


@dataclass
class Finding:
    path: Path
    line: int
    literal: str
    context: str


def _load_allowlist(path: Path) -> dict:
    if not path.exists():
        return {"ignored_path_globs": [], "ignored_literals": [], "ignored_line_globs": []}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
        return {
            "ignored_path_globs": data.get("ignored_path_globs", []),
            "ignored_literals": data.get("ignored_literals", []),
            "ignored_line_globs": data.get("ignored_line_globs", []),
        }
    # JSON fallback if yaml dependency is unavailable.
    data = json.loads(text)
    return {
        "ignored_path_globs": data.get("ignored_path_globs", []),
        "ignored_literals": data.get("ignored_literals", []),
        "ignored_line_globs": data.get("ignored_line_globs", []),
    }


def _changed_files(diff_range: str) -> list[Path]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", diff_range],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    out = []
    for line in proc.stdout.splitlines():
        p = ROOT / line.strip()
        if (
            p.suffix == ".py"
            and p.exists()
            and (str(p.relative_to(ROOT)).startswith("src/") or str(p.relative_to(ROOT)).startswith("orchestration/"))
        ):
            out.append(p)
    return out


def _changed_lines(diff_range: str) -> dict[Path, set[int]]:
    proc = subprocess.run(
        ["git", "diff", "-U0", diff_range],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {}

    file_changes: dict[Path, set[int]] = {}
    current_file: Path | None = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in proc.stdout.splitlines():
        if line.startswith("+++ b/"):
            rel = line[len("+++ b/"):].strip()
            p = ROOT / rel
            rel_str = str(p.relative_to(ROOT)) if p.exists() else rel
            if p.suffix == ".py" and (rel_str.startswith("src/") or rel_str.startswith("orchestration/")):
                current_file = p
            else:
                current_file = None
            continue
        if current_file is None:
            continue
        m = hunk_re.match(line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2) or "1")
        if count == 0:
            continue
        bucket = file_changes.setdefault(current_file, set())
        for ln in range(start, start + count):
            bucket.add(ln)

    return file_changes


def _is_upper_constant_assignment(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign):
        return False
    for t in node.targets:
        if isinstance(t, ast.Name) and t.id.isupper():
            return True
    return False


def _iter_nodes_with_parent(tree: ast.AST) -> Iterable[tuple[ast.AST, ast.AST | None]]:
    stack: list[tuple[ast.AST, ast.AST | None]] = [(tree, None)]
    while stack:
        node, parent = stack.pop()
        yield node, parent
        for child in ast.iter_child_nodes(node):
            stack.append((child, node))


def _scan_file(path: Path, allowlist: dict, changed_line_filter: set[int] | None = None) -> list[Finding]:
    rel = path.relative_to(ROOT)
    rel_str = str(rel)
    for pat in allowlist["ignored_path_globs"]:
        if fnmatch.fnmatch(rel_str, pat):
            return []

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=rel_str)
    except SyntaxError:
        return []

    ignored_literals = {str(x) for x in allowlist["ignored_literals"]}
    findings: list[Finding] = []

    for node, parent in _iter_nodes_with_parent(tree):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bool):
            continue
        if not isinstance(node.value, (int, float)):
            continue

        lit = str(node.value)
        if lit in ignored_literals:
            continue

        # Upper-case module constants are acceptable invariant placement.
        if parent is not None and _is_upper_constant_assignment(parent):
            continue

        line = getattr(node, "lineno", 0)
        if line <= 0 or line > len(lines):
            continue
        if changed_line_filter is not None and line not in changed_line_filter:
            continue
        line_text = lines[line - 1].strip()

        # Skip common typing/dataclass/enum-style and schema bounds lines.
        safe_markers = ("Field(", "Enum", "ge=", "le=", "default=")
        if any(marker in line_text for marker in safe_markers):
            continue

        if any(fnmatch.fnmatch(line_text, pat) for pat in allowlist["ignored_line_globs"]):
            continue

        findings.append(Finding(path=rel, line=line, literal=lit, context=line_text[:180]))

    return findings


def _runtime_py_files() -> list[Path]:
    files = []
    for base in (ROOT / "src", ROOT / "orchestration"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(ROOT)
            if "tools/mined" in str(rel) or "__pycache__" in str(rel):
                continue
            files.append(path)
    return files


def _report(findings: list[Finding], max_findings: int) -> None:
    print(f"numeric-literal-findings: {len(findings)}")
    for f in findings[:max_findings]:
        print(f"- {f.path}:{f.line} literal={f.literal} :: {f.context}")
    if len(findings) > max_findings:
        print(f"... truncated {len(findings) - max_findings} additional findings")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("report", "enforce-changed"), default="report")
    parser.add_argument("--diff-range", default="origin/main...HEAD")
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    parser.add_argument("--max-findings", type=int, default=200)
    args = parser.parse_args()

    allowlist = _load_allowlist(Path(args.allowlist))

    if args.mode == "enforce-changed":
        files = _changed_files(args.diff_range)
        changed_map = _changed_lines(args.diff_range)
    else:
        files = _runtime_py_files()
        changed_map = {}

    findings: list[Finding] = []
    for path in files:
        changed_filter = changed_map.get(path) if args.mode == "enforce-changed" else None
        findings.extend(_scan_file(path, allowlist, changed_filter))

    _report(sorted(findings, key=lambda f: (str(f.path), f.line)), args.max_findings)

    if args.mode == "enforce-changed" and findings:
        print("ERROR: unclassified numeric literals found in changed files.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
