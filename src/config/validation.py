"""Validation and registry-backed helper utilities for configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _env_optional_float(name: str, default: float | None) -> float | None:
    """Parse optional float from environment variable."""
    val = os.environ.get(name, "")
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    """Parse string from environment variable."""
    return os.environ.get(name, default)


# Registry timeout cache (avoids repeated YAML parsing)
_REGISTRY_TIMEOUTS_CACHE: dict[str, int | float] | None = None
_REGISTRY_RUNTIME_DEFAULTS_CACHE: dict[str, Any] | None = None
_LOADING_REGISTRY_TIMEOUTS = False
_LOADING_RUNTIME_DEFAULTS = False


def _load_registry_timeouts() -> dict[str, int | float]:
    """Load all timeouts from the registry (single source of truth).

    Returns a flat dict with keys like:
        "roles.architect_general", "server.request", "services.ocr_pdf"
    """
    global _REGISTRY_TIMEOUTS_CACHE
    global _LOADING_REGISTRY_TIMEOUTS
    if _REGISTRY_TIMEOUTS_CACHE is not None:
        return _REGISTRY_TIMEOUTS_CACHE
    if _LOADING_REGISTRY_TIMEOUTS:
        # Avoid recursive config->registry->config loops during bootstrap.
        return {}

    try:
        _LOADING_REGISTRY_TIMEOUTS = True
        runtime_defaults = _load_registry_runtime_defaults()
        raw_timeouts = runtime_defaults.get("timeouts", {}) if isinstance(runtime_defaults, dict) else {}

        # Flatten nested structure to "category.key" format
        flat: dict[str, int | float] = {"default": raw_timeouts.get("default", 600)}
        # All timeout categories defined in model_registry.yaml
        categories = [
            "roles", "server", "services", "pools", "benchmark",
            "repl", "tools", "external", "health", "scripts", "backends",
        ]
        for category in categories:
            cat_data = raw_timeouts.get(category, {})
            for key, value in cat_data.items():
                flat[f"{category}.{key}"] = value

        _REGISTRY_TIMEOUTS_CACHE = flat
        return flat
    except Exception as e:
        logger.debug("Registry timeouts unavailable, using hardcoded fallbacks: %s", e)
        _REGISTRY_TIMEOUTS_CACHE = {}
        return {}
    finally:
        _LOADING_REGISTRY_TIMEOUTS = False


def _registry_timeout(category: str, key: str, fallback: int | float) -> int | float:
    """Get timeout from registry, falling back to hardcoded default.

    Args:
        category: Timeout category (roles, server, services, pools, benchmark).
        key: Timeout key within category.
        fallback: Hardcoded fallback if registry unavailable.

    Returns:
        Timeout value from registry, or fallback.
    """
    timeouts = _load_registry_timeouts()
    full_key = f"{category}.{key}"
    if full_key in timeouts:
        return timeouts[full_key]
    # Try just the key for backward compat
    if key in timeouts:
        return timeouts[key]
    # Use hardcoded fallback (registry default only used for unknown keys without fallback)
    return fallback


def _load_registry_runtime_defaults() -> dict[str, Any]:
    """Load runtime_defaults block from registry (cached)."""
    global _REGISTRY_RUNTIME_DEFAULTS_CACHE
    global _LOADING_RUNTIME_DEFAULTS
    if _REGISTRY_RUNTIME_DEFAULTS_CACHE is not None:
        return _REGISTRY_RUNTIME_DEFAULTS_CACHE
    if _LOADING_RUNTIME_DEFAULTS:
        return {}
    try:
        _LOADING_RUNTIME_DEFAULTS = True
        registry_path = Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_REGISTRY_PATH",
                "orchestration/model_registry.yaml",
            )
        )
        if not registry_path.exists():
            _REGISTRY_RUNTIME_DEFAULTS_CACHE = {}
            return _REGISTRY_RUNTIME_DEFAULTS_CACHE

        import yaml

        with registry_path.open(encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        runtime_defaults = payload.get("runtime_defaults", {})
        _REGISTRY_RUNTIME_DEFAULTS_CACHE = (
            dict(runtime_defaults) if isinstance(runtime_defaults, dict) else {}
        )
        return _REGISTRY_RUNTIME_DEFAULTS_CACHE
    except Exception as e:
        logger.debug("Registry runtime defaults unavailable, using hardcoded fallbacks: %s", e)
        _REGISTRY_RUNTIME_DEFAULTS_CACHE = {}
        return {}
    finally:
        _LOADING_RUNTIME_DEFAULTS = False


def _registry_runtime_value(path: tuple[str, ...], fallback: Any) -> Any:
    """Get nested runtime_defaults value by path, with fallback."""
    cur: Any = _load_registry_runtime_defaults()
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return fallback
        cur = cur[key]
    return cur


def reset_validation_caches() -> None:
    """Reset internal helper caches used by config loading."""
    global _REGISTRY_TIMEOUTS_CACHE
    global _REGISTRY_RUNTIME_DEFAULTS_CACHE
    global _LOADING_REGISTRY_TIMEOUTS
    global _LOADING_RUNTIME_DEFAULTS
    _REGISTRY_TIMEOUTS_CACHE = None
    _REGISTRY_RUNTIME_DEFAULTS_CACHE = None
    _LOADING_REGISTRY_TIMEOUTS = False
    _LOADING_RUNTIME_DEFAULTS = False


def reset_runtime_defaults_cache() -> None:
    """Reset only runtime-default cache (preserves timeout cache behavior)."""
    global _REGISTRY_RUNTIME_DEFAULTS_CACHE
    _REGISTRY_RUNTIME_DEFAULTS_CACHE = None
