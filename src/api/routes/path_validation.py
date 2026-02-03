"""Path validation for API routes.

Provides allowlist-based path validation to prevent path traversal attacks.
Mirrors the REPL environment's _validate_file_path() pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException


def _get_allowed_prefixes() -> list[str]:
    """Get allowed path prefixes from config or defaults."""
    import tempfile

    prefixes: list[str] = []
    try:
        from src.config import get_config

        config = get_config()
        llm_root = str(config.paths.llm_root)
        tmp_dir = str(config.paths.tmp_dir)
        # Ensure trailing slash for prefix matching
        prefixes = [
            llm_root if llm_root.endswith("/") else f"{llm_root}/",
            tmp_dir if tmp_dir.endswith("/") else f"{tmp_dir}/",
        ]
    except Exception:
        # Fallback to hardcoded defaults if config unavailable
        prefixes = ["/mnt/raid0/llm/", "/mnt/raid0/llm/tmp/"]

    # Always allow system temp directory (for CI and tests)
    sys_tmp = tempfile.gettempdir()
    sys_tmp_prefix = sys_tmp if sys_tmp.endswith("/") else f"{sys_tmp}/"
    if sys_tmp_prefix not in prefixes:
        prefixes.append(sys_tmp_prefix)

    return prefixes


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
    allowed_prefixes = _get_allowed_prefixes()
    if not any(str(resolved).startswith(p) for p in allowed_prefixes):
        raise HTTPException(
            status_code=403,
            detail=f"Path not allowed: {resolved}",
        )
    return resolved
