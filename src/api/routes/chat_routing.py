"""Intent classification and mode selection for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: direct mode detection, ReAct mode detection,
MemRL-informed mode selection, and proactive intent classification.
"""

from __future__ import annotations

from typing import Any

from src.features import features
from src.roles import Role


def _should_use_direct_mode(prompt: str, context: str = "") -> bool:
    """Decide if the prompt should bypass the REPL and get a direct LLM answer.

    The REPL wrapper forces the model to generate Python code and call FINAL(),
    which destroys quality on instruction-precision, formatting, and constraint-
    satisfaction tasks. When the task doesn't need tools (peek, grep, list_dir),
    direct mode produces much higher quality output.

    Bypass REPL when:
    - No context or short context (no files to explore)
    - Prompt doesn't reference file operations
    - Prompt doesn't ask for code execution

    Keep REPL when:
    - Large context (needs chunked exploration via peek/grep)
    - Prompt explicitly asks to read/write files
    - Prompt asks to execute or run code

    Args:
        prompt: The user's prompt.
        context: Optional context text.

    Returns:
        True if direct mode should be used.
    """
    prompt_lower = prompt.lower()

    # Keep REPL for large contexts (needs peek/grep/summarize_chunks)
    if context and len(context) > 20000:
        return False

    # Keep REPL when prompt explicitly needs file/tool operations
    repl_indicators = [
        "read the file", "list files", "list the files", "look at the file",
        "open the file", "read from", "write to", "save to",
        "execute", "run the", "run this",
        "search the codebase", "find in the", "grep for",
        "explore the", "scan the",
    ]
    if any(ind in prompt_lower for ind in repl_indicators):
        return False

    # Direct mode for everything else — reasoning, formatting, QA,
    # math proofs, instruction following, tool-call JSON generation, etc.
    return True


def _select_mode(
    prompt: str,
    context: str,
    state: "Any",
) -> str:
    """Select execution mode: direct, react, or repl.

    Uses MemRL route_with_mode() if available, falls back to heuristic chain:
    _should_use_direct_mode() → _should_use_react_mode() → repl.

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        state: Application state (may have hybrid_router).

    Returns:
        One of "direct", "react", or "repl".
    """
    from src.api.routes.chat_react import _should_use_react_mode

    # Try MemRL-based mode selection if available
    if hasattr(state, 'hybrid_router') and state.hybrid_router is not None:
        try:
            task_ir = {
                "task_type": "chat",
                "objective": prompt[:200],
                "priority": "interactive",
                "context_length": len(context) if context else 0,
            }
            _routing, _strategy, mode = state.hybrid_router.route_with_mode(task_ir)
            if mode in ("direct", "react", "repl"):
                return mode
        except Exception:
            pass  # Fall through to heuristic

    # Heuristic fallback: direct → react → repl
    if _should_use_direct_mode(prompt, context):
        if _should_use_react_mode(prompt, context):
            return "react"
        return "direct"
    return "repl"


def _classify_and_route(prompt: str, context: str = "", has_image: bool = False) -> tuple[str, str]:
    """Classify prompt intent and proactively route to the best specialist.

    Zero-latency keyword heuristic. Returns (role, strategy).
    Falls back to frontdoor if no strong signal.

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        has_image: Whether the request includes an image.

    Returns:
        Tuple of (role_name, routing_strategy).
    """
    # Vision: has image data — route to VL server (different model type)
    if has_image:
        return "worker_vision", "classified"

    # Specialist routing: when enabled, use keyword heuristics to route
    # code/architecture tasks to stronger specialists (32B, 235B, 480B).
    # Gated behind feature flag — only activate when Q-values demonstrate
    # clear benefit via comparative seeding (Phase 3).
    if features().specialist_routing:
        prompt_lower = prompt.lower()

        # Code generation / debugging → 32B coder (39 t/s with spec decode)
        code_keywords = [
            "implement", "write code", "function", "class ", "debug",
            "refactor", "fix the bug", "code review", "unit test",
            "algorithm", "data structure", "regex", "parse",
        ]
        if any(kw in prompt_lower for kw in code_keywords):
            return str(Role.CODER_PRIMARY), "classified"

        # Complex code requiring escalation → 32B coder escalation
        complex_code_keywords = [
            "concurrent", "lock-free", "distributed", "optimize performance",
            "memory leak", "race condition", "deadlock",
        ]
        if any(kw in prompt_lower for kw in complex_code_keywords):
            return str(Role.CODER_ESCALATION), "classified"

        # Architecture / system design → 235B architect (6.75 t/s)
        arch_keywords = [
            "architecture", "system design", "design pattern",
            "scalab", "microservice", "trade-off", "tradeoff",
            "invariant", "constraint", "cap theorem",
        ]
        if any(kw in prompt_lower for kw in arch_keywords):
            return str(Role.ARCHITECT_GENERAL), "classified"

    # Default: frontdoor (30B MoE) handles all text prompts
    return str(Role.FRONTDOOR), "rules"
