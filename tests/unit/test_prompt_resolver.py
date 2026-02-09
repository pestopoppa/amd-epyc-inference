"""Tests for prompt hot-swap resolver."""

from __future__ import annotations

import pytest

from src.prompt_builders.resolver import (
    PROMPT_DIR,
    _get_variant,
    _safe_format,
    resolve_prompt,
)


# ── _safe_format ─────────────────────────────────────────────────────────


class TestSafeFormat:
    """Safe template interpolation."""

    def test_replaces_known_vars(self):
        result = _safe_format("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_missing_vars_stay_as_placeholders(self):
        result = _safe_format("Hello {name}, you are {role}!", {"name": "Alice"})
        assert result == "Hello Alice, you are {role}!"

    def test_empty_vars(self):
        result = _safe_format("Hello {name}!", {})
        assert result == "Hello {name}!"

    def test_no_placeholders(self):
        result = _safe_format("No placeholders here.", {"name": "Alice"})
        assert result == "No placeholders here."

    def test_malformed_template_returns_raw(self):
        result = _safe_format("Bad {template{nested}", {"template": "x"})
        assert "Bad" in result  # Should not crash

    def test_double_braces_preserved(self):
        """Double braces are literal braces in format strings."""
        result = _safe_format('{{"key": "{value}"}}', {"value": "42"})
        assert '"key"' in result
        assert "42" in result

    def test_empty_string(self):
        result = _safe_format("", {"key": "val"})
        assert result == ""


# ── _get_variant ─────────────────────────────────────────────────────────


class TestGetVariant:
    """Variant resolution from environment variables."""

    def test_no_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("PROMPT_VARIANT__test_prompt", raising=False)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        assert _get_variant("test_prompt") is None

    def test_per_prompt_env_var(self, monkeypatch):
        monkeypatch.setenv("PROMPT_VARIANT__architect_investigate", "v2")
        assert _get_variant("architect_investigate") == "v2"

    def test_global_env_var(self, monkeypatch):
        monkeypatch.delenv("PROMPT_VARIANT__my_prompt", raising=False)
        monkeypatch.setenv("PROMPT_VARIANT", "beta")
        assert _get_variant("my_prompt") == "beta"

    def test_per_prompt_overrides_global(self, monkeypatch):
        monkeypatch.setenv("PROMPT_VARIANT__my_prompt", "v3")
        monkeypatch.setenv("PROMPT_VARIANT", "beta")
        assert _get_variant("my_prompt") == "v3"


# ── resolve_prompt ───────────────────────────────────────────────────────


class TestResolvePrompt:
    """Core resolve_prompt() function."""

    def test_fallback_when_no_file(self, tmp_path, monkeypatch):
        """No file exists → uses fallback string."""
        monkeypatch.setattr(
            "src.prompt_builders.resolver.PROMPT_DIR", tmp_path / "nonexistent"
        )
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        result = resolve_prompt("missing", "fallback text")
        assert result == "fallback text"

    def test_file_overrides_fallback(self, tmp_path, monkeypatch):
        """File exists → reads from file, ignores fallback."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        (tmp_path / "my_prompt.md").write_text("From file!")
        result = resolve_prompt("my_prompt", "fallback")
        assert result == "From file!"

    def test_variant_file_overrides_default(self, tmp_path, monkeypatch):
        """Variant file exists → reads variant, ignores default file."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        (tmp_path / "my_prompt.md").write_text("Default file")
        (tmp_path / "my_prompt.v2.md").write_text("Variant v2")
        monkeypatch.setenv("PROMPT_VARIANT__my_prompt", "v2")
        result = resolve_prompt("my_prompt", "fallback")
        assert result == "Variant v2"

    def test_variant_missing_falls_to_default_file(self, tmp_path, monkeypatch):
        """Variant set but file missing → falls to default file."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        (tmp_path / "my_prompt.md").write_text("Default file")
        monkeypatch.setenv("PROMPT_VARIANT__my_prompt", "v99")
        result = resolve_prompt("my_prompt", "fallback")
        assert result == "Default file"

    def test_explicit_variant_param(self, tmp_path, monkeypatch):
        """Explicit variant= param overrides env var."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        (tmp_path / "my_prompt.beta.md").write_text("Beta version")
        monkeypatch.setenv("PROMPT_VARIANT__my_prompt", "v2")
        result = resolve_prompt("my_prompt", "fallback", variant="beta")
        assert result == "Beta version"

    def test_template_interpolation(self, tmp_path, monkeypatch):
        """Template vars get interpolated in file content."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        (tmp_path / "greet.md").write_text("Hello {user}, you are {role}!")
        result = resolve_prompt("greet", "fallback", user="Alice", role="admin")
        assert result == "Hello Alice, you are admin!"

    def test_template_interpolation_in_fallback(self, tmp_path, monkeypatch):
        """Template vars also work in fallback strings."""
        monkeypatch.setattr(
            "src.prompt_builders.resolver.PROMPT_DIR", tmp_path / "nonexistent"
        )
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        result = resolve_prompt(
            "missing", "Q: {question}\nA:", question="What is 2+2?"
        )
        assert result == "Q: What is 2+2?\nA:"

    def test_subdir_resolution(self, tmp_path, monkeypatch):
        """subdir= parameter resolves files in subdirectory."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        roles_dir = tmp_path / "roles"
        roles_dir.mkdir()
        (roles_dir / "frontdoor.md").write_text("I am frontdoor prompt")
        result = resolve_prompt("frontdoor", "fallback", subdir="roles")
        assert result == "I am frontdoor prompt"

    def test_subdir_fallback(self, tmp_path, monkeypatch):
        """subdir file missing → uses fallback."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        result = resolve_prompt("nonexistent_role", "fallback text", subdir="roles")
        assert result == "fallback text"

    def test_no_template_vars_no_interpolation(self, tmp_path, monkeypatch):
        """Without template vars, literal braces in file are preserved."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        (tmp_path / "raw.md").write_text('Reply JSON: {{"key": "value"}}')
        result = resolve_prompt("raw", "fallback")
        assert result == 'Reply JSON: {{"key": "value"}}'

    def test_prompt_dir_points_to_orchestration(self):
        """PROMPT_DIR should point to orchestration/prompts/."""
        assert PROMPT_DIR.name == "prompts"
        assert PROMPT_DIR.parent.name == "orchestration"


# ── Integration: actual prompt files ─────────────────────────────────────


class TestPromptFilesExist:
    """Verify the prompt files created in orchestration/prompts/ are loadable."""

    @pytest.mark.parametrize(
        "name",
        [
            "root_lm_system",
            "architect_investigate",
            "architect_synthesis",
            "review_verdict",
            "revision",
            "plan_review",
            "confidence_estimation",
            "task_decomposition",
            "formalizer",
        ],
    )
    def test_prompt_file_exists(self, name):
        path = PROMPT_DIR / f"{name}.md"
        assert path.exists(), f"Missing prompt file: {path}"
        content = path.read_text()
        assert len(content) > 10, f"Prompt file too short: {path}"

    @pytest.mark.parametrize(
        "role",
        [
            "frontdoor",
            "coder_primary",
            "coder_escalation",
            "architect_general",
            "architect_coding",
            "ingest_long_context",
            "worker_general",
            "worker_math",
            "worker_vision",
        ],
    )
    def test_role_prompt_file_exists(self, role):
        path = PROMPT_DIR / "roles" / f"{role}.md"
        assert path.exists(), f"Missing role prompt file: {path}"
        content = path.read_text()
        assert len(content) > 10, f"Role prompt file too short: {path}"
