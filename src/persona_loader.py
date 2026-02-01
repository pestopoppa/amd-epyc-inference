"""Persona Registry: Load and resolve personas from YAML configuration.

Personas are orthogonal to roles. Role determines WHICH model runs
(server URL, tier). Persona determines HOW the model responds
(system prompt overlay, behavioral framing).

Usage:
    from src.persona_loader import get_persona_registry

    registry = get_persona_registry()
    cfg = registry.get("security_auditor")
    if cfg:
        print(cfg.system_prompt)

    # Heuristic matching against task text
    matches = registry.match("Review this code for SQL injection")
    # [("security_auditor", 1), ("code_reviewer", 1)]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent / "orchestration" / "persona_registry.yaml"


@dataclass(frozen=True)
class PersonaConfig:
    """Configuration for a single persona.

    Attributes:
        name: Persona identifier (e.g., "security_auditor").
        description: Human-readable purpose.
        system_prompt: Injected before the user prompt in llm_call().
        task_patterns: Substring patterns for heuristic matching.
        seed_q: Initial MemRL Q-value for persona seeds.
    """

    name: str
    description: str
    system_prompt: str
    task_patterns: list[str] = field(default_factory=list)
    seed_q: float = 0.85


class PersonaRegistry:
    """Loads and resolves personas from YAML registry.

    Thread-safe after initialization (read-only after __init__).
    """

    def __init__(self, yaml_path: Path | None = None) -> None:
        self._personas: dict[str, PersonaConfig] = {}
        path = yaml_path or DEFAULT_REGISTRY_PATH

        if not path.exists():
            logger.warning("Persona registry not found: %s", path)
            return

        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed; persona registry unavailable")
            return
        except Exception as e:
            logger.warning("Failed to load persona registry: %s", e)
            return

        if not data or "personas" not in data:
            logger.warning("No 'personas' key in %s", path)
            return

        for name, cfg in data["personas"].items():
            self._personas[name] = PersonaConfig(
                name=name,
                description=cfg.get("description", ""),
                system_prompt=cfg.get("system_prompt", ""),
                task_patterns=cfg.get("task_patterns", []),
                seed_q=cfg.get("seed_q", 0.85),
            )

        logger.info("Loaded %d personas from %s", len(self._personas), path)

    def get(self, name: str) -> PersonaConfig | None:
        """Get persona config by name.

        Args:
            name: Persona identifier.

        Returns:
            PersonaConfig if found, None otherwise.
        """
        return self._personas.get(name)

    def get_system_prompt(self, name: str) -> str | None:
        """Get system prompt for a persona.

        Args:
            name: Persona identifier.

        Returns:
            System prompt string if found, None otherwise.
        """
        cfg = self._personas.get(name)
        return cfg.system_prompt if cfg else None

    def match(self, task_text: str) -> list[tuple[str, int]]:
        """Find personas matching task text via substring patterns.

        Args:
            task_text: Task description to match against.

        Returns:
            List of (persona_name, match_count) tuples, sorted by
            match count descending. Only includes personas with at
            least one pattern match.
        """
        task_lower = task_text.lower()
        matches: list[tuple[str, int]] = []

        for name, cfg in self._personas.items():
            count = sum(1 for p in cfg.task_patterns if p.lower() in task_lower)
            if count > 0:
                matches.append((name, count))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def all_names(self) -> list[str]:
        """Get sorted list of all persona names.

        Returns:
            Sorted list of persona identifier strings.
        """
        return sorted(self._personas.keys())

    def __len__(self) -> int:
        return len(self._personas)

    def __contains__(self, name: str) -> bool:
        return name in self._personas


# ── Singleton ────────────────────────────────────────────────────────

_registry: PersonaRegistry | None = None


def get_persona_registry(yaml_path: Path | None = None) -> PersonaRegistry:
    """Get the global PersonaRegistry instance (lazy-loaded).

    Args:
        yaml_path: Override path to persona_registry.yaml.

    Returns:
        Global PersonaRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = PersonaRegistry(yaml_path)
    return _registry


def reset_persona_registry() -> None:
    """Reset the global PersonaRegistry (useful for tests)."""
    global _registry
    _registry = None
