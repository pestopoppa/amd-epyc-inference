"""Tests for API route path validation."""

import pytest
from fastapi import HTTPException

from src.api.routes.path_validation import validate_api_path, _get_allowed_prefixes


def test_allowed_raid_path():
    """Paths under /mnt/raid0/llm/ are accepted."""
    # Use a path that resolves to itself (no symlinks needed)
    result = validate_api_path("/mnt/raid0/llm/claude/CLAUDE.md")
    assert str(result).startswith("/mnt/raid0/llm/")


def test_allowed_tmp_path(tmp_path):
    """Paths under /tmp/ are accepted."""
    f = tmp_path / "test.txt"
    f.write_text("hello")
    result = validate_api_path(str(f))
    assert result.exists()


def test_rejects_traversal():
    """Path traversal with ../../ is blocked after resolution."""
    with pytest.raises(HTTPException) as exc_info:
        validate_api_path("/mnt/raid0/llm/../../etc/passwd")
    assert exc_info.value.status_code == 403
    assert "not allowed" in exc_info.value.detail


def test_rejects_root_path():
    """Paths outside allowed prefixes are rejected."""
    with pytest.raises(HTTPException) as exc_info:
        validate_api_path("/etc/passwd")
    assert exc_info.value.status_code == 403


def test_rejects_home_path():
    """Paths under /home/ are rejected."""
    with pytest.raises(HTTPException) as exc_info:
        validate_api_path("/home/daniele/.bashrc")
    assert exc_info.value.status_code == 403


def test_allowed_prefixes_are_correct():
    """Verify the allowlist includes required prefixes."""
    import tempfile

    allowed = _get_allowed_prefixes()
    # Must include LLM root and tmp
    assert "/mnt/raid0/llm/" in allowed
    assert "/mnt/raid0/llm/tmp/" in allowed
    # Must include system temp for CI/tests
    sys_tmp = tempfile.gettempdir()
    sys_tmp_prefix = sys_tmp if sys_tmp.endswith("/") else f"{sys_tmp}/"
    assert sys_tmp_prefix in allowed
