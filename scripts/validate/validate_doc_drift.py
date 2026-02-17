#!/usr/bin/env python3
"""Detect documentation drift between CLAUDE.md and source code.

Checks three high-value drift vectors:
  1. Port mappings: PORT_MAP in orchestrator_stack.py vs port table in CLAUDE.md
  2. File path references: relative markdown links in CLAUDE.md vs actual filesystem
  3. Makefile targets: `make <target>` references in CLAUDE.md vs .PHONY targets
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLAUDE_MD = ROOT / "CLAUDE.md"
ORCHESTRATOR = ROOT / "scripts" / "server" / "orchestrator_stack.py"
MAKEFILE = ROOT / "Makefile"


# ---------------------------------------------------------------------------
# 1. Port mapping drift
# ---------------------------------------------------------------------------

def extract_port_map_from_code(path: Path) -> dict[int, str]:
    """Parse PORT_MAP dict from orchestrator_stack.py.

    Returns {port_number: first_role_name_on_that_port}.
    """
    source = path.read_text(encoding="utf-8")
    # Find the PORT_MAP assignment via AST
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "PORT_MAP"
        ):
            port_to_role: dict[int, str] = {}
            if isinstance(node.value, ast.Dict):
                for key, val in zip(node.value.keys, node.value.values):
                    if isinstance(key, ast.Constant) and isinstance(val, ast.Constant):
                        role = str(key.value)
                        port = int(val.value)
                        if port not in port_to_role:
                            port_to_role[port] = role
            return port_to_role
    return {}


def extract_port_table_from_docs(text: str) -> dict[int, str]:
    """Extract port numbers from the Local Model Routing table in CLAUDE.md.

    Returns {port_number: task_type_label}.
    """
    ports: dict[int, str] = {}
    # Match table rows like: | Interactive chat | Qwen3-Coder-30B-A3B | 8080 | ...
    in_routing_table = False
    for line in text.splitlines():
        if "Local Model Routing" in line:
            in_routing_table = True
            continue
        if in_routing_table and line.startswith("|"):
            cells = [c.strip() for c in line.split("|")]
            # Skip separator rows and header
            if len(cells) < 5 or cells[1].startswith("-") or cells[1] == "Task Type":
                continue
            task_type = cells[1]
            port_cell = cells[3]
            # Port cell may contain "8086/8087" for multiple ports
            for m in re.finditer(r"\b(\d{4,5})\b", port_cell):
                port = int(m.group(1))
                if port not in ports:
                    ports[port] = task_type
        elif in_routing_table and not line.startswith("|") and line.strip() and not line.startswith(" "):
            # Exited the table
            break
    return ports


def check_port_drift() -> list[str]:
    """Compare ports in code vs docs, return list of drift messages."""
    errors: list[str] = []
    if not ORCHESTRATOR.exists():
        errors.append(f"port-drift: orchestrator_stack.py not found at {ORCHESTRATOR}")
        return errors

    code_ports = extract_port_map_from_code(ORCHESTRATOR)
    doc_text = CLAUDE_MD.read_text(encoding="utf-8")
    doc_ports = extract_port_table_from_docs(doc_text)

    if not code_ports:
        errors.append("port-drift: could not parse PORT_MAP from orchestrator_stack.py")
        return errors

    # Only check non-auxiliary ports documented in the routing table
    # (embedders, orchestrator, etc. are not in the routing table)
    code_model_ports = {
        p for p, role in code_ports.items()
        if not role.startswith("embedder") and role != "orchestrator" and role != "document_formalizer"
    }
    doc_port_set = set(doc_ports.keys())

    in_code_not_docs = code_model_ports - doc_port_set
    in_docs_not_code = doc_port_set - code_model_ports

    for port in sorted(in_code_not_docs):
        role = code_ports[port]
        errors.append(f"port-drift: port {port} ({role}) in PORT_MAP but missing from CLAUDE.md routing table")

    for port in sorted(in_docs_not_code):
        task = doc_ports[port]
        errors.append(f"port-drift: port {port} ({task}) in CLAUDE.md routing table but missing from PORT_MAP")

    return errors


# ---------------------------------------------------------------------------
# 2. File path references
# ---------------------------------------------------------------------------

def extract_markdown_links(text: str) -> list[tuple[str, str]]:
    """Extract (display_text, relative_path) from markdown links in CLAUDE.md.

    Only returns relative paths (not URLs, not anchors).
    """
    links: list[tuple[str, str]] = []
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", text):
        display, target = m.group(1), m.group(2)
        # Skip URLs and anchor-only links
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        # Strip any anchor fragment
        path_part = target.split("#")[0]
        if path_part:
            links.append((display, path_part))
    return links


def check_path_drift() -> list[str]:
    """Check that relative paths in CLAUDE.md actually exist."""
    errors: list[str] = []
    doc_text = CLAUDE_MD.read_text(encoding="utf-8")
    links = extract_markdown_links(doc_text)

    for display, rel_path in links:
        full = (ROOT / rel_path).resolve()
        if not full.exists():
            errors.append(f"path-drift: [{display}]({rel_path}) -> file not found")

    return errors


# ---------------------------------------------------------------------------
# 3. Makefile target drift
# ---------------------------------------------------------------------------

def extract_phony_targets(makefile: Path) -> set[str]:
    """Extract target names from .PHONY declarations in Makefile."""
    targets: set[str] = set()
    text = makefile.read_text(encoding="utf-8")
    for m in re.finditer(r"\.PHONY:\s*(.+)", text):
        targets.update(m.group(1).split())
    return targets


def extract_make_refs_from_docs(text: str) -> set[str]:
    """Extract `make <target>` references from CLAUDE.md."""
    refs: set[str] = set()
    for m in re.finditer(r"`make\s+([\w-]+)`", text):
        refs.add(m.group(1))
    return refs


def check_makefile_drift() -> list[str]:
    """Compare make targets referenced in docs vs actual Makefile .PHONY."""
    errors: list[str] = []
    if not MAKEFILE.exists():
        errors.append(f"makefile-drift: Makefile not found at {MAKEFILE}")
        return errors

    phony = extract_phony_targets(MAKEFILE)
    doc_text = CLAUDE_MD.read_text(encoding="utf-8")
    doc_refs = extract_make_refs_from_docs(doc_text)

    missing_targets = doc_refs - phony
    for target in sorted(missing_targets):
        errors.append(f"makefile-drift: `make {target}` referenced in CLAUDE.md but not a .PHONY target")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_checks() -> list[str]:
    """Run all drift checks, return combined error list."""
    errors: list[str] = []
    errors.extend(check_port_drift())
    errors.extend(check_path_drift())
    errors.extend(check_makefile_drift())
    return errors


def main() -> int:
    if not CLAUDE_MD.exists():
        print(f"doc-drift: CLAUDE.md not found at {CLAUDE_MD}")
        return 1

    errors = run_all_checks()

    if errors:
        print("doc drift validation FAILED")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("doc drift validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
