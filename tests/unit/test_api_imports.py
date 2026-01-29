#!/usr/bin/env python3
"""Tests for API module imports and function signature compatibility.

These tests catch the class of bug where:
- chat.py imports a function from a deprecated wrapper that is missing new parameters
- Function signatures in wrappers diverge from the canonical source
- New parameters added to prompt_builders aren't forwarded by orchestrator.py

The build_root_lm_prompt import bug (commit 7b4b2da) caused ALL real_mode
requests to crash with TypeError, producing empty benchmark results.
"""

import inspect
from typing import get_type_hints

import pytest


class TestChatImportsResolve:
    """Verify all imports in chat.py resolve without errors."""

    def test_chat_module_imports(self):
        """Import chat module — catches any ImportError or circular import."""
        from src.api.routes import chat  # noqa: F401

    def test_prompt_builders_imports(self):
        """Import prompt_builders — catches missing functions."""
        from src.prompt_builders import (  # noqa: F401
            build_root_lm_prompt,
            build_stage2_review_prompt,
            build_long_context_exploration_prompt,
            build_routing_context,
            build_review_verdict_prompt,
            build_revision_prompt,
            extract_code_from_response,
            classify_error,
            auto_wrap_final,
            build_escalation_prompt,
        )

    def test_orchestrator_wrapper_imports(self):
        """Import orchestrator wrapper — catches stale re-exports."""
        from src.api.services.orchestrator import (  # noqa: F401
            build_root_lm_prompt,
            extract_code_from_response,
            auto_wrap_final,
            classify_error,
            build_escalation_prompt,
        )


class TestSignatureCompatibility:
    """Verify wrapper signatures match canonical source signatures."""

    def test_build_root_lm_prompt_signature_match(self):
        """Deprecated wrapper must accept all params the canonical function does.

        This is the exact bug that crashed all benchmarks: the wrapper in
        orchestrator.py was missing the `routing_context` parameter.
        """
        from src.prompt_builders import build_root_lm_prompt as canonical
        from src.api.services.orchestrator import build_root_lm_prompt as wrapper

        canonical_params = set(inspect.signature(canonical).parameters.keys())
        wrapper_params = set(inspect.signature(wrapper).parameters.keys())

        # Wrapper must accept AT LEAST all canonical params
        missing = canonical_params - wrapper_params
        assert not missing, (
            f"Deprecated wrapper is missing parameters: {missing}. "
            f"Canonical has: {sorted(canonical_params)}, "
            f"Wrapper has: {sorted(wrapper_params)}"
        )

    def test_build_root_lm_prompt_routing_context(self):
        """Specifically verify routing_context param exists everywhere."""
        from src.prompt_builders import build_root_lm_prompt as canonical
        from src.api.services.orchestrator import build_root_lm_prompt as wrapper

        canonical_sig = inspect.signature(canonical)
        wrapper_sig = inspect.signature(wrapper)

        assert "routing_context" in canonical_sig.parameters, \
            "routing_context missing from canonical build_root_lm_prompt"
        assert "routing_context" in wrapper_sig.parameters, \
            "routing_context missing from wrapper build_root_lm_prompt"

    def test_classify_error_signature_match(self):
        """classify_error wrapper matches canonical."""
        from src.prompt_builders import classify_error as canonical
        from src.api.services.orchestrator import classify_error as wrapper

        canonical_params = set(inspect.signature(canonical).parameters.keys())
        wrapper_params = set(inspect.signature(wrapper).parameters.keys())

        missing = canonical_params - wrapper_params
        assert not missing, f"classify_error wrapper missing params: {missing}"

    def test_build_escalation_prompt_signature_match(self):
        """build_escalation_prompt wrapper matches canonical."""
        from src.prompt_builders import build_escalation_prompt as canonical
        from src.api.services.orchestrator import build_escalation_prompt as wrapper

        canonical_params = set(inspect.signature(canonical).parameters.keys())
        wrapper_params = set(inspect.signature(wrapper).parameters.keys())

        missing = canonical_params - wrapper_params
        assert not missing, f"build_escalation_prompt wrapper missing params: {missing}"


class TestChatRouteImportSource:
    """Verify chat.py imports critical functions from the right module."""

    def test_build_root_lm_prompt_from_prompt_builders(self):
        """chat.py must import build_root_lm_prompt from prompt_builders, not the wrapper.

        The wrapper is deprecated and may lag behind on new parameters.
        """
        import ast
        from pathlib import Path

        chat_path = Path(__file__).parent.parent.parent / "src" / "api" / "routes" / "chat.py"
        tree = ast.parse(chat_path.read_text())

        # Find all ImportFrom nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "orchestrator" in node.module:
                    imported_names = [alias.name for alias in node.names]
                    assert "build_root_lm_prompt" not in imported_names, (
                        "chat.py imports build_root_lm_prompt from deprecated orchestrator wrapper. "
                        "Import from src.prompt_builders instead."
                    )


class TestCallableSmoke:
    """Verify key functions are callable with expected args."""

    def test_build_root_lm_prompt_callable(self):
        """build_root_lm_prompt works with all expected arguments."""
        from src.prompt_builders import build_root_lm_prompt

        result = build_root_lm_prompt(
            state="ready",
            original_prompt="test prompt",
            last_output="",
            last_error="",
            turn=0,
            routing_context="Role: frontdoor",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_code_from_response_callable(self):
        """extract_code_from_response works."""
        from src.prompt_builders import extract_code_from_response

        result = extract_code_from_response("```python\nprint('hello')\n```")
        assert isinstance(result, str)

    def test_auto_wrap_final_callable(self):
        """auto_wrap_final works."""
        from src.prompt_builders import auto_wrap_final

        result = auto_wrap_final("print('hello')")
        assert isinstance(result, str)
