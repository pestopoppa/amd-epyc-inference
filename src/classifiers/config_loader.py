"""YAML configuration loader for classifiers."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_config_path() -> Path:
    """Get the path to classifier_config.yaml."""
    # Check environment variable first
    env_path = os.environ.get("ORCHESTRATOR_CLASSIFIER_CONFIG")
    if env_path:
        return Path(env_path)

    # Default path relative to project root
    project_root = os.environ.get(
        "ORCHESTRATOR_PATHS_PROJECT_ROOT", "/mnt/raid0/llm/claude"
    )
    return Path(project_root) / "orchestration" / "classifier_config.yaml"


@lru_cache(maxsize=1)
def get_classifier_config() -> dict[str, Any]:
    """Load and cache the classifier configuration.

    Returns:
        Parsed YAML configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    config_path = _get_config_path()

    if not config_path.exists():
        logger.warning(f"Classifier config not found at {config_path}, using defaults")
        return _get_default_config()

    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if not config:
            logger.warning("Empty classifier config, using defaults")
            return _get_default_config()

        return config
    except ImportError:
        logger.warning("PyYAML not available, using default config")
        return _get_default_config()
    except Exception as e:
        logger.warning(f"Failed to load classifier config: {e}, using defaults")
        return _get_default_config()


def reset_classifier_config() -> None:
    """Clear the cached config (useful for tests)."""
    get_classifier_config.cache_clear()


def _get_default_config() -> dict[str, Any]:
    """Return hardcoded default configuration as fallback.

    These match the original hardcoded values in the codebase.
    """
    return {
        "keyword_matchers": {
            "summarization": {
                "keywords": [
                    "summarize",
                    "summary",
                    "summarise",
                    "summarisation",
                    "executive summary",
                    "overview",
                    "key points",
                    "main ideas",
                    "tl;dr",
                    "tldr",
                    "synopsis",
                ],
                "case_sensitive": False,
            },
            "structured_analysis": {
                "keywords": [
                    "analyze",
                    "architecture",
                    "diagram",
                    "protocol",
                    "economic model",
                    "security audit",
                    "security analysis",
                    "whitepaper",
                    "smart contract",
                    "incentive",
                    "trust assumption",
                    "attack vector",
                    "forensic",
                    "entity extraction",
                    "business relationship",
                    "flow chart",
                    "flowchart",
                    "sequence diagram",
                    "system design",
                    "data flow",
                    "state machine",
                ],
                "case_sensitive": False,
            },
            "coding_task": {
                "keywords": [
                    "implement",
                    "code",
                    "function",
                    "class ",
                    "method",
                    "debug",
                    "refactor",
                    "bug",
                    "error",
                    "exception",
                    "compile",
                    "syntax",
                    "algorithm",
                    "data structure",
                    "api",
                    "endpoint",
                    "database",
                    "query",
                    "sql",
                    "test",
                    "unit test",
                    "integration",
                ],
                "case_sensitive": False,
            },
            "stub_final": {
                "patterns": [
                    "complete",
                    "see above",
                    "analysis complete",
                    "estimation complete",
                    "done",
                    "finished",
                    "see results above",
                    "see output above",
                    "see structured output above",
                    "see integrated results above",
                    "see the structured output above",
                ],
                "normalize": True,
            },
        },
        "routing_classifiers": {
            "direct_mode": {
                "context_threshold": 20000,
                "repl_indicators": [
                    "read the file",
                    "list files",
                    "list the files",
                    "look at the file",
                    "open the file",
                    "read from",
                    "write to",
                    "save to",
                    "execute",
                    "run the",
                    "run this",
                    "search the codebase",
                    "find in the",
                    "grep for",
                    "explore the",
                    "scan the",
                ],
                "use_memrl": False,
            },
            "specialist_routing": {
                "use_memrl": False,
                "categories": {
                    "coder_escalation": {
                        "keywords": [
                            "implement",
                            "write code",
                            "function",
                            "class ",
                            "debug",
                            "refactor",
                            "fix the bug",
                            "code review",
                            "unit test",
                            "algorithm",
                            "data structure",
                            "regex",
                            "parse",
                            "concurrent",
                            "lock-free",
                            "distributed",
                            "optimize performance",
                            "memory leak",
                            "race condition",
                            "deadlock",
                        ],
                    },
                    "architect_general": {
                        "keywords": [
                            "architecture",
                            "system design",
                            "design pattern",
                            "scalab",
                            "microservice",
                            "trade-off",
                            "tradeoff",
                            "invariant",
                            "constraint",
                            "cap theorem",
                        ],
                    },
                },
            },
        },
    }
