#!/usr/bin/env python3
"""Pytest configuration and fixtures for the orchestrator test suite.

Memory Safety:
    This file includes a memory guard that warns/exits if available RAM is below
    a safe threshold. This prevents crashes from running tests with pytest-xdist
    parallel workers that each load models.

    The 192-thread EPYC system with 1.13TB RAM can still crash if pytest spawns
    too many workers that each initialize the API (which loads the TaskEmbedder
    model). The lazy loading in src/api.py prevents this for mock mode tests,
    but this guard provides an additional safety net.

    The memory check is SKIPPED in CI environments (detected via CI or
    ORCHESTRATOR_MOCK_MODE env vars) since CI runners have limited RAM but
    only run mock-mode tests that don't load models.

Usage:
    pytest tests/              # Normal test run with memory check
    pytest tests/ -n 4         # Safe parallel execution (max 4 workers)
    pytest tests/ -n auto      # DANGEROUS on this machine - avoid

See also:
    - CLAUDE.md for memory constraints
    - research/ESCALATION_FLOW.md for memory pool architecture
"""

import os
import warnings
from unittest.mock import MagicMock

import pytest

from src.config import reset_config
from src.api.state import AppState


# Memory threshold in GB (fail if less available)
MEMORY_THRESHOLD_GB = 100

# Check if running in CI environment (GitHub Actions, etc.)
IS_CI = os.environ.get("CI") == "true" or os.environ.get("ORCHESTRATOR_MOCK_MODE") == "true"

# Maximum safe parallel workers for this machine
# -n 8 gives 4x speedup (67s → 17s), -n 4 gives 3x (67s → 23s)
MAX_SAFE_WORKERS = 8


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-server",
        action="store_true",
        default=False,
        help="Run tests that require a live llama-server instance",
    )
    parser.addoption(
        "--run-ocr-server",
        action="store_true",
        default=False,
        help="Run tests that require a live OCR server",
    )


def pytest_configure(config):
    """Check memory and register custom markers before running tests."""
    # Register custom markers
    config.addinivalue_line("markers", "heavy: marks tests that load models (may need more memory)")
    config.addinivalue_line("markers", "real_mode: marks tests that require real inference servers")
    config.addinivalue_line(
        "markers", "requires_server: marks tests that require a live llama-server"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")

    # Check available memory (skip in CI - mock mode tests don't load models)
    if IS_CI:
        pass  # Skip memory check in CI environments
    else:
        try:
            import psutil

            free_gb = psutil.virtual_memory().available / (1024**3)

            if free_gb < MEMORY_THRESHOLD_GB:
                # Hard fail if memory is critically low
                pytest.exit(
                    f"DANGER: Only {free_gb:.0f}GB free RAM. "
                    f"Need {MEMORY_THRESHOLD_GB}GB+ for safe testing.\n"
                    "This machine can crash if tests load too many models.\n"
                    "Free up memory or wait for other processes to complete.",
                    returncode=1,
                )
            elif free_gb < MEMORY_THRESHOLD_GB * 2:
                # Warn if memory is getting low
                warnings.warn(
                    f"Low memory: {free_gb:.0f}GB free. Tests may be slow. Consider freeing memory.",
                    UserWarning,
                )
        except ImportError:
            warnings.warn(
                "psutil not installed - cannot check memory. Install with: pip install psutil",
                UserWarning,
            )


@pytest.fixture(autouse=True)
def _reset_config_between_tests():
    """Ensure config cache is clean between tests.

    Prevents env var leaks from one test affecting another.
    Uses reset_config() which clears the lru_cache on get_config().
    """
    reset_config()
    yield
    reset_config()


def pytest_collection_modifyitems(config, items):
    """Add markers and warnings for parallel execution."""
    # Check if pytest-xdist is being used with too many workers
    try:
        num_workers = config.option.numprocesses
        if num_workers is not None:
            if num_workers == "auto":
                warnings.warn(
                    f"DANGER: pytest -n auto on 192-thread machine will spawn ~192 workers!\n"
                    f"Each worker may load models, causing memory exhaustion.\n"
                    f"Use: pytest -n {MAX_SAFE_WORKERS} instead.",
                    UserWarning,
                )
            elif isinstance(num_workers, int) and num_workers > MAX_SAFE_WORKERS:
                warnings.warn(
                    f"High parallelism: {num_workers} workers requested.\n"
                    f"Recommended max: {MAX_SAFE_WORKERS}. May cause memory issues.",
                    UserWarning,
                )
    except (AttributeError, TypeError):
        # pytest-xdist not installed or not using -n flag
        pass


@pytest.fixture
def mock_registry():
    """Shared registry mock fixture for chat pipeline tests."""
    registry = MagicMock()
    registry.routing_hints = {}
    return registry


@pytest.fixture
def mock_app_state(mock_registry):
    """Shared AppState mock fixture with common attributes used by route tests."""
    state = MagicMock(spec=AppState)
    state.progress_logger = MagicMock()
    state.hybrid_router = None
    state.tool_registry = MagicMock()
    state.script_registry = MagicMock()
    state.registry = mock_registry
    state.health_tracker = MagicMock()
    state.admission = MagicMock()
    state.increment_active = MagicMock()
    state.decrement_active = MagicMock()
    state.increment_request = MagicMock()
    return state


@pytest.fixture
def mock_llm_primitives():
    """Shared LLM primitives mock fixture with common telemetry fields."""
    primitives = MagicMock()
    primitives._backends = True
    primitives.total_tokens_generated = 100
    primitives.total_prompt_eval_ms = 50
    primitives.total_generation_ms = 200
    primitives._last_predicted_tps = 25.0
    primitives.total_http_overhead_ms = 10
    primitives.get_cache_stats.return_value = {"hits": 5, "misses": 2}
    primitives.llm_call.return_value = "Test response"
    return primitives
