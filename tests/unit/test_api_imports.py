#!/usr/bin/env python3
"""Tests for API module imports and function signature compatibility.

These tests catch the class of bug where:
- chat.py imports a function that doesn't exist in the canonical source
- New parameters added to prompt_builders break callers
- Decomposed chat modules have broken imports or circular dependencies

The build_root_lm_prompt import bug (commit 7b4b2da) caused ALL real_mode
requests to crash with TypeError, producing empty benchmark results.
The orchestrator.py facade that caused that bug was deleted in Phase 1.
"""

import inspect

import pytest


class TestChatImportsResolve:
    """Verify all imports in chat.py and decomposed modules resolve without errors."""

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

    def test_chat_utils_imports(self):
        """Import chat_utils — catches missing functions or constants."""
        from src.api.routes.chat_utils import (  # noqa: F401
            THREE_STAGE_CONFIG,
            TWO_STAGE_CONFIG,
            QWEN_STOP,
            LONG_CONTEXT_CONFIG,
            _estimate_tokens,
            _is_stub_final,
            _strip_tool_outputs,
            _resolve_answer,
            _truncate_looped_answer,
            _should_formalize,
            _formalize_output,
        )

    def test_chat_routing_imports(self):
        """Import chat_routing — catches missing functions."""
        from src.api.routes.chat_routing import (  # noqa: F401
            _should_use_direct_mode,
            _select_mode,
            _classify_and_route,
        )

    def test_chat_react_imports(self):
        """Import chat_react — catches missing functions."""
        from src.api.routes.chat_react import (  # noqa: F401
            _parse_react_args,
            _should_use_react_mode,
            _react_mode_answer,
        )

    def test_chat_delegation_imports(self):
        """Import chat_delegation — catches missing functions."""
        from src.api.routes.chat_delegation import (  # noqa: F401
            _parse_architect_decision,
            _architect_delegated_answer,
        )

    def test_chat_review_imports(self):
        """Import chat_review — catches missing functions."""
        from src.api.routes.chat_review import (  # noqa: F401
            _detect_output_quality_issue,
            _should_review,
            _architect_verdict,
            _fast_revise,
            _needs_plan_review,
            _architect_plan_review,
            _apply_plan_review,
            _store_plan_review_episode,
            _compute_plan_review_phase,
        )

    def test_chat_vision_imports(self):
        """Import chat_vision — catches missing functions."""
        from src.api.routes.chat_vision import (  # noqa: F401
            _is_ocr_heavy_prompt,
            _needs_structured_analysis,
            _handle_vision_request,
            _execute_vision_tool,
            _vision_react_mode_answer,
            _handle_multi_file_vision,
        )

    def test_chat_summarization_imports(self):
        """Import chat_summarization — catches missing functions."""
        from src.api.routes.chat_summarization import (  # noqa: F401
            _is_summarization_task,
            _should_use_two_stage,
            _run_two_stage_summarization,
        )


class TestSignatureCompatibility:
    """Verify key function signatures have expected parameters."""

    def test_build_root_lm_prompt_routing_context(self):
        """Verify routing_context param exists (the param whose absence crashed benchmarks)."""
        from src.prompt_builders import build_root_lm_prompt

        sig = inspect.signature(build_root_lm_prompt)
        assert "routing_context" in sig.parameters, \
            "routing_context missing from build_root_lm_prompt"


class TestChatRouteImportSource:
    """Verify chat.py imports critical functions from the right module."""

    def test_no_orchestrator_imports(self):
        """chat.py must not import from deleted orchestrator.py facade."""
        import ast
        from pathlib import Path

        chat_path = Path(__file__).parent.parent.parent / "src" / "api" / "routes" / "chat.py"
        tree = ast.parse(chat_path.read_text())

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "orchestrator" in node.module:
                    imported_names = [alias.name for alias in node.names]
                    assert False, (
                        f"chat.py still imports from deleted orchestrator facade: {imported_names}. "
                        f"Import from src.prompt_builders or decomposed modules instead."
                    )

    def test_orchestrator_facade_deleted(self):
        """Verify orchestrator.py facade no longer exists."""
        from pathlib import Path

        facade_path = Path(__file__).parent.parent.parent / "src" / "api" / "services" / "orchestrator.py"
        assert not facade_path.exists(), (
            "src/api/services/orchestrator.py still exists — it should have been deleted "
            "during Phase 1 decomposition."
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
