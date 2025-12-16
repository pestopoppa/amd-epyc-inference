#!/usr/bin/env python3
"""Validate TaskIR / ArchitectureIR JSON files against local JSON Schemas.

Usage:
  python orchestration/validate_ir.py task path/to/task_ir.json
  python orchestration/validate_ir.py arch path/to/architecture_ir.json
  echo '{"task_id": ...}' | python orchestration/validate_ir.py task -

Exit codes:
  0 = valid
  1 = usage error
  2 = invalid JSON or schema violation
  3 = missing file / schema
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Check for jsonschema dependency
try:
    from jsonschema import Draft202012Validator, ValidationError
except ImportError:
    print("ERROR: missing dependency 'jsonschema'")
    print("Install with: pip install jsonschema>=4.20")
    sys.exit(1)


# Resolve paths relative to this script
ROOT = Path(__file__).resolve().parent
SCHEMA_TASK = ROOT / "task_ir.schema.json"
SCHEMA_ARCH = ROOT / "architecture_ir.schema.json"


def load_json(path: Path | None, from_stdin: bool = False) -> dict[str, Any]:
    """Load JSON from file or stdin."""
    if from_stdin:
        content = sys.stdin.read()
    else:
        if path is None or not path.exists():
            raise FileNotFoundError(str(path))
        content = path.read_text(encoding="utf-8")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def format_error(err: ValidationError) -> str:
    """Format a validation error with JSON path."""
    # Build JSON pointer-style path
    path_parts = [str(p) for p in err.absolute_path]
    location = "$" + "".join(f".{p}" if isinstance(p, str) else f"[{p}]" for p in err.absolute_path)
    if not path_parts:
        location = "$ (root)"

    # Truncate long messages
    msg = err.message
    if len(msg) > 200:
        msg = msg[:200] + "..."

    return f"  {location}: {msg}"


def validate(instance: dict[str, Any], schema_path: Path, source_name: str) -> int:
    """Validate instance against schema. Returns exit code."""
    if not schema_path.exists():
        print(f"ERROR: schema not found: {schema_path}")
        return 3

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid schema JSON: {e}")
        return 3

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))

    if not errors:
        print(f"✓ {source_name} is valid")
        return 0

    print(f"✗ {source_name} has {len(errors)} validation error(s):")
    for err in errors[:20]:
        print(format_error(err))

    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more errors")

    return 2


def main(argv: list[str]) -> int:
    """Main entry point."""
    if len(argv) < 3:
        print("Usage: validate_ir.py (task|arch) <path.json | ->")
        print("  Use '-' to read from stdin")
        return 1

    kind = argv[1]
    if kind not in {"task", "arch"}:
        print(f"ERROR: unknown kind '{kind}', expected 'task' or 'arch'")
        return 1

    input_arg = argv[2]
    from_stdin = input_arg == "-"

    # Select schema
    schema_path = SCHEMA_TASK if kind == "task" else SCHEMA_ARCH
    source_name = "stdin" if from_stdin else input_arg

    # Load instance
    try:
        if from_stdin:
            instance = load_json(None, from_stdin=True)
        else:
            instance_path = Path(input_arg).expanduser().resolve()
            instance = load_json(instance_path)
    except FileNotFoundError as e:
        print(f"ERROR: file not found: {e}")
        return 3
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2

    return validate(instance, schema_path, source_name)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
