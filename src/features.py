#!/usr/bin/env python3
"""Feature flag system for optional orchestration modules.

This module defines all optional features that can be enabled/disabled
independently of core orchestration functionality. Each feature has:
- A clear description of what it does
- Dependencies (what other features/modules it requires)
- Environment variable to enable/disable

Usage:
    from src.features import Features, get_features

    # Get feature flags (reads from environment or config)
    features = get_features()

    # Check if a feature is enabled
    if features.memrl:
        from orchestration.repl_memory import TaskEmbedder
        # ... use MemRL components

    # Create features with explicit flags
    features = Features(memrl=False, tools=True)

Environment Variables:
    ORCHESTRATOR_MEMRL=1         Enable MemRL (learned routing, Q-scoring)
    ORCHESTRATOR_TOOLS=1         Enable tool registry (REPL tools)
    ORCHESTRATOR_SCRIPTS=1       Enable script registry (prepared scripts)
    ORCHESTRATOR_STREAMING=1     Enable SSE streaming endpoints
    ORCHESTRATOR_OPENAI_COMPAT=1 Enable OpenAI-compatible API
    ORCHESTRATOR_REPL=1          Enable REPL execution environment

Design Principles:
    1. Core orchestration works with ALL features disabled
    2. Features are opt-in by default in tests, opt-out in production
    3. Each feature can be toggled independently
    4. Dependencies are documented and checked at initialization

Adding New Features:
    1. Add field to Features dataclass with description
    2. Add environment variable check in get_features()
    3. Add dependency documentation if needed
    4. Guard feature code with if features.your_feature:
    5. Add tests for both enabled/disabled states
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Environment variable prefix for all feature flags
ENV_PREFIX = "ORCHESTRATOR_"


@dataclass
class Features:
    """Feature flags for optional orchestration modules.

    All features default to False for test isolation. Production code should
    use get_features() which reads from environment variables.

    Attributes:
        memrl: Memory-based Reinforcement Learning (Phase 4)
            - TaskEmbedder: 0.5B embedding model for task similarity
            - QScorer: Q-value scoring for escalation decisions
            - HybridRouter: Learned + rule-based routing
            - EpisodicStore: SQLite storage for task memories
            Dependencies: numpy, sqlite3, sentence-transformers (optional)

        tools: Tool Registry for REPL
            - TOOL() function in REPL environment
            - Role-based permission checking
            - Built-in tools (lint, test, search)
            Dependencies: None (tools are pure Python)

        scripts: Script Registry for prepared scripts
            - SCRIPT() function in REPL
            - Semantic search for script matching
            Dependencies: tools feature (scripts can invoke tools)

        streaming: SSE streaming for chat responses
            - /chat/stream endpoint
            - Server-sent events for incremental output
            Dependencies: None

        openai_compat: OpenAI-compatible API endpoints
            - /v1/chat/completions
            - /v1/models
            Dependencies: None

        repl: REPL execution environment
            - Sandboxed Python execution
            - Context-as-variable pattern
            - peek(), grep(), FINAL() built-ins
            Dependencies: None

        caching: Response caching with prefix routing
            - CachingBackend for LLM responses
            - Prefix-based routing to workers
            Dependencies: None

        restricted_python: Use RestrictedPython for REPL sandboxing
            - More battle-tested security model
            - compile_restricted for safer compilation
            - Built-in guards against attribute access exploits
            Dependencies: RestrictedPython>=7.0
    """

    # Phase 4: MemRL (Memory-based Reinforcement Learning)
    memrl: bool = False

    # Tool and Script Registries
    tools: bool = False
    scripts: bool = False

    # API Features
    streaming: bool = False
    openai_compat: bool = False

    # Core Features (usually enabled)
    repl: bool = True
    caching: bool = True

    # Security Features
    restricted_python: bool = False  # Use RestrictedPython for REPL (requires library)

    # Generation Monitoring (Phase 6)
    generation_monitor: bool = False  # Enable early failure detection

    # Debug/Development
    mock_mode: bool = True  # Default to mock mode for safety

    def validate(self) -> list[str]:
        """Validate feature dependencies.

        Returns:
            List of validation errors (empty if all valid).
        """
        errors = []

        # Scripts require tools
        if self.scripts and not self.tools:
            errors.append("scripts feature requires tools feature")

        # RestrictedPython requires the library
        if self.restricted_python:
            try:
                import RestrictedPython  # noqa: F401
            except ImportError:
                errors.append(
                    "restricted_python feature requires RestrictedPython library: "
                    "pip install RestrictedPython>=7.0"
                )

        return errors

    def summary(self) -> dict[str, bool]:
        """Get summary of all feature flags.

        Returns:
            Dictionary of feature name -> enabled status.
        """
        return {
            "memrl": self.memrl,
            "tools": self.tools,
            "scripts": self.scripts,
            "streaming": self.streaming,
            "openai_compat": self.openai_compat,
            "repl": self.repl,
            "caching": self.caching,
            "restricted_python": self.restricted_python,
            "generation_monitor": self.generation_monitor,
            "mock_mode": self.mock_mode,
        }

    def enabled_features(self) -> list[str]:
        """Get list of enabled feature names.

        Returns:
            List of enabled feature names.
        """
        return [name for name, enabled in self.summary().items() if enabled]


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from environment variable.

    Truthy values: 1, true, yes, on (case-insensitive)
    Falsy values: 0, false, no, off (case-insensitive)

    Args:
        name: Environment variable name (without prefix).
        default: Default value if not set.

    Returns:
        Boolean value.
    """
    key = f"{ENV_PREFIX}{name.upper()}"
    value = os.environ.get(key, "").lower()

    if not value:
        return default

    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False

    return default


def get_features(
    *,
    production: bool = False,
    override: dict[str, bool] | None = None,
) -> Features:
    """Get feature flags from environment variables.

    In production mode (production=True), most features default to enabled.
    In test mode (production=False), most features default to disabled.

    Args:
        production: If True, use production defaults (most features on).
        override: Explicit overrides for specific features.

    Returns:
        Features instance with flags set.

    Example:
        # Read from environment
        features = get_features()

        # Production defaults
        features = get_features(production=True)

        # Test with specific features
        features = get_features(override={"memrl": True, "tools": False})
    """
    # Base defaults depend on production vs test
    if production:
        defaults = {
            "memrl": True,
            "tools": True,
            "scripts": True,
            "streaming": True,
            "openai_compat": True,
            "repl": True,
            "caching": True,
            "restricted_python": True,  # Use safer sandbox in production
            "generation_monitor": True,  # Early failure detection in production
            "mock_mode": False,  # Real mode in production
        }
    else:
        defaults = {
            "memrl": False,
            "tools": False,
            "scripts": False,
            "streaming": False,
            "openai_compat": False,
            "repl": True,  # REPL is core functionality
            "caching": False,
            "restricted_python": False,  # Use custom sandbox in tests
            "generation_monitor": False,  # Disabled in tests by default
            "mock_mode": True,  # Mock mode in tests
        }

    # Read from environment (overrides defaults)
    flags = {
        "memrl": _env_bool("MEMRL", defaults["memrl"]),
        "tools": _env_bool("TOOLS", defaults["tools"]),
        "scripts": _env_bool("SCRIPTS", defaults["scripts"]),
        "streaming": _env_bool("STREAMING", defaults["streaming"]),
        "openai_compat": _env_bool("OPENAI_COMPAT", defaults["openai_compat"]),
        "repl": _env_bool("REPL", defaults["repl"]),
        "caching": _env_bool("CACHING", defaults["caching"]),
        "restricted_python": _env_bool("RESTRICTED_PYTHON", defaults["restricted_python"]),
        "generation_monitor": _env_bool("GENERATION_MONITOR", defaults["generation_monitor"]),
        "mock_mode": _env_bool("MOCK_MODE", defaults["mock_mode"]),
    }

    # Apply explicit overrides
    if override:
        flags.update(override)

    return Features(**flags)


# Singleton for global access (lazy-loaded)
_features: Features | None = None


def features() -> Features:
    """Get the global Features instance (lazy-loaded from environment).

    For most code, use this function:
        from src.features import features
        if features().memrl:
            ...

    Returns:
        Global Features instance.
    """
    global _features
    if _features is None:
        _features = get_features()
    return _features


def reset_features() -> None:
    """Reset the global Features instance (useful for tests).

    Call this to re-read feature flags from environment.
    """
    global _features
    _features = None


def set_features(new_features: Features) -> None:
    """Set the global Features instance (useful for tests).

    Args:
        new_features: Features instance to use globally.
    """
    global _features
    _features = new_features
