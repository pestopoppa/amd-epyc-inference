"""Path validation for API routes.

Provides allowlist-based path validation to prevent path traversal attacks.
Mirrors the REPL environment's _validate_file_path() pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException

ALLOWED_PREFIXES = ["/mnt/raid0/llm/", "/mnt/raid0/llm/tmp/"]


def validate_api_path(raw: str) -> Path:
    """Resolve and validate a user-supplied file path against the allowlist.

    Uses os.path.realpath() to canonicalize the path (resolving symlinks
    and '..' sequences) before checking against ALLOWED_PREFIXES.

    Args:
        raw: User-supplied file path string.

    Returns:
        Resolved Path object.

    Raises:
        HTTPException: 403 if path is outside allowed prefixes.
    """
    resolved = Path(os.path.realpath(raw))
    if not any(str(resolved).startswith(p) for p in ALLOWED_PREFIXES):
        raise HTTPException(
            status_code=403,
            detail=f"Path not allowed: {resolved}",
        )
    return resolved
