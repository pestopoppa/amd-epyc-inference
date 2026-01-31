"""TOON (Token-Oriented Object Notation) encoder for orchestrator performance.

TOON achieves 40-65% token reduction on structured data while maintaining
lossless round-trip conversion. This module provides TOON encoding for:
- Tool output formatting (file listings, grep results)
- Escalation context compression
- Stage 2 grep hits formatting

Reference: https://github.com/toon-format/spec
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid startup cost if TOON not used
_toon_format = None


def _get_toon():
    """Lazy-load toon_format module."""
    global _toon_format
    if _toon_format is None:
        try:
            import toon_format
            _toon_format = toon_format
        except ImportError:
            logger.warning("toon_format not installed, falling back to JSON")
            return None
    return _toon_format


def is_available() -> bool:
    """Check if TOON encoding is available.

    Returns:
        True if toon_format is installed.
    """
    return _get_toon() is not None


def encode(data: Any, fallback_to_json: bool = True) -> str:
    """Encode data to TOON format.

    Args:
        data: Python object to encode (dict, list, primitives).
        fallback_to_json: If True, return JSON when TOON unavailable.

    Returns:
        TOON-encoded string, or JSON if fallback enabled and TOON unavailable.

    Raises:
        ImportError: If TOON unavailable and fallback disabled.
    """
    toon = _get_toon()
    if toon is None:
        if fallback_to_json:
            return json.dumps(data, indent=2)
        raise ImportError("toon_format not installed")

    try:
        return toon.encode(data)
    except Exception as e:
        logger.warning(f"TOON encode failed: {e}, falling back to JSON")
        if fallback_to_json:
            return json.dumps(data, indent=2)
        raise


def decode(toon_str: str) -> Any:
    """Decode TOON string to Python object.

    Args:
        toon_str: TOON-encoded string.

    Returns:
        Decoded Python object.

    Raises:
        ImportError: If TOON not installed.
        ValueError: If decoding fails.
    """
    toon = _get_toon()
    if toon is None:
        raise ImportError("toon_format not installed")

    return toon.decode(toon_str)


def should_use_toon(data: Any, min_array_size: int = 3) -> bool:
    """Determine if TOON encoding is beneficial for the given data.

    TOON provides best savings for uniform arrays of objects.
    Returns False for small arrays or non-uniform structures.

    Args:
        data: Data to evaluate.
        min_array_size: Minimum array size to benefit from TOON.

    Returns:
        True if TOON encoding would provide significant savings.
    """
    if not is_available():
        return False

    # Check for uniform arrays (TOON sweet spot)
    if isinstance(data, dict):
        # Look for arrays in the dict
        for value in data.values():
            if isinstance(value, list) and len(value) >= min_array_size:
                if _is_uniform_object_array(value):
                    return True
    elif isinstance(data, list):
        if len(data) >= min_array_size and _is_uniform_object_array(data):
            return True

    return False


def _is_uniform_object_array(arr: list) -> bool:
    """Check if array contains uniform objects (same keys).

    Args:
        arr: List to check.

    Returns:
        True if all elements are dicts with identical keys.
    """
    if not arr:
        return False

    first = arr[0]
    if not isinstance(first, dict):
        return False

    keys = set(first.keys())
    return all(
        isinstance(item, dict) and set(item.keys()) == keys
        for item in arr[1:]
    )


def encode_list_dir(path: str, files: list[dict], total: int) -> str:
    """Encode directory listing with TOON optimization.

    Args:
        path: Directory path.
        files: List of file entries (name, type, size).
        total: Total number of entries.

    Returns:
        TOON or JSON encoded string based on efficiency.
    """
    result = {"path": path, "files": files, "total": total}

    if len(files) >= 3 and is_available():
        return encode(result)
    return json.dumps(result, indent=2)


def encode_grep_hits(grep_hits: list[dict]) -> str:
    """Encode grep hits as compact Markdown.

    NOTE: TOON encoding was tested but found to be LESS efficient than
    Markdown for grep hits due to pattern repetition on every row.
    Markdown's grouped structure (### Search: pattern) is more compact.

    Args:
        grep_hits: List of grep hit records from REPL.

    Returns:
        Markdown-formatted grep hits.
    """
    if not grep_hits:
        return ""

    # Markdown format is more compact than TOON for grep hits
    # because it groups hits by pattern instead of repeating pattern on each row
    parts = []
    for record in grep_hits:
        pattern = record.get("pattern", "")
        parts.append(f"### Search: `{pattern}`")
        for hit in record.get("hits", [])[:5]:
            line_num = hit.get("line_num", "?")
            match = hit.get("match", "")[:200]
            parts.append(f"Line {line_num}: {match}")
        parts.append("")
    return "\n".join(parts)


def encode_escalation_context(
    task_id: str,
    failure_count: int,
    error_category: str,
    gate_name: str | None,
    error_message: str | None,
    previous_attempts: list[dict] | None = None,
) -> str:
    """Encode escalation context with TOON optimization.

    Args:
        task_id: Task identifier.
        failure_count: Number of failures.
        error_category: Error category (FORMAT, EXECUTION, etc).
        gate_name: Optional gate that failed.
        error_message: Optional error details.
        previous_attempts: Optional list of previous attempt records.

    Returns:
        TOON-encoded escalation context.
    """
    ctx = {
        "task_id": task_id,
        "failure_count": failure_count,
        "error_category": error_category,
    }
    if gate_name:
        ctx["gate_name"] = gate_name
    if error_message:
        ctx["error_preview"] = error_message[:500]
    if previous_attempts:
        ctx["previous_attempts"] = previous_attempts

    if is_available() and previous_attempts and len(previous_attempts) >= 2:
        return encode(ctx)
    return json.dumps(ctx, indent=2)


def encode_procedures(procedures: list[dict]) -> str:
    """Encode procedure listing with TOON optimization.

    Args:
        procedures: List of procedure dicts from ProcedureRegistry.

    Returns:
        TOON or JSON encoded string based on list size.
    """
    if len(procedures) >= 3 and is_available():
        return encode({"procedures": procedures})
    return json.dumps(procedures, indent=2)


def encode_memory_results(results: list[dict]) -> str:
    """Encode memory recall results with TOON optimization.

    Args:
        results: List of memory result dicts from EpisodicStore.

    Returns:
        TOON or JSON encoded string based on list size.
    """
    if len(results) >= 3 and is_available():
        return encode({"results": results})
    return json.dumps({"results": results}, indent=2)
