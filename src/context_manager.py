#!/usr/bin/env python3
"""Context Manager for hierarchical local-agent orchestration.

This module manages shared context between execution steps, handling
artifacts, context limits, and summaries for multi-step workflows.

Usage:
    from src.context_manager import ContextManager

    ctx = ContextManager()
    ctx.set("analysis_result", "The code looks good")
    result = ctx.get("analysis_result")
    ctx.add_artifact("output.py", "/tmp/output.py")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ContextType(Enum):
    """Types of context entries."""

    TEXT = "text"  # Text output from a step
    ARTIFACT = "artifact"  # File artifact reference
    STRUCTURED = "structured"  # JSON/dict data
    SUMMARY = "summary"  # Summarized content


@dataclass
class ContextEntry:
    """A single entry in the context store."""

    key: str
    value: Any
    context_type: ContextType
    step_id: str | None = None  # Which step produced this
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self._serialize_value(),
            "context_type": self.context_type.value,
            "step_id": self.step_id,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "truncated": self.truncated,
            "metadata": self.metadata,
        }

    def _serialize_value(self) -> Any:
        """Serialize value for JSON output."""
        if self.context_type == ContextType.ARTIFACT:
            # For artifacts, just return the path
            return str(self.value) if isinstance(self.value, Path) else self.value
        elif self.context_type == ContextType.STRUCTURED:
            return self.value
        else:
            # For text, truncate if too long
            if isinstance(self.value, str) and len(self.value) > 1000:
                return self.value[:1000] + "... [truncated]"
            return self.value


@dataclass
class ContextConfig:
    """Configuration for the Context Manager."""

    max_entry_size: int = 10000  # Max chars per entry
    max_total_size: int = 100000  # Max total context size
    auto_summarize: bool = True  # Auto-summarize large entries
    summary_threshold: int = 5000  # Summarize entries larger than this
    artifact_base_path: Path = field(default_factory=lambda: _ctx_artifacts_path())


def _ctx_artifacts_path() -> Path:
    from src.config import get_config

    return get_config().paths.artifacts_dir


class ContextManagerError(Exception):
    """Error in context management."""

    pass


class ContextManager:
    """Manages shared context between execution steps.

    The ContextManager provides a key-value store for passing information
    between steps in a multi-step execution plan. It handles:
    - Text outputs from LLM inference
    - File artifacts (code, data files)
    - Structured data (JSON, dicts)
    - Context size limits and truncation
    """

    def __init__(self, config: ContextConfig | None = None):
        """Initialize the Context Manager.

        Args:
            config: Configuration options.
        """
        self.config = config or ContextConfig()
        self._entries: dict[str, ContextEntry] = {}
        self._order: list[str] = []  # Insertion order for iteration
        self._total_size: int = 0

    def set(
        self,
        key: str,
        value: Any,
        step_id: str | None = None,
        context_type: ContextType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextEntry:
        """Set a context value.

        Args:
            key: Unique key for this context entry.
            value: The value to store.
            step_id: ID of the step that produced this value.
            context_type: Type of context (auto-detected if None).
            metadata: Additional metadata for the entry.

        Returns:
            The created ContextEntry.

        Raises:
            ContextManagerError: If the value exceeds size limits.
        """
        # Auto-detect context type
        if context_type is None:
            context_type = self._detect_type(value)

        # Calculate size
        size_bytes = self._calculate_size(value)

        # Check size limits
        truncated = False
        if size_bytes > self.config.max_entry_size:
            if isinstance(value, str):
                value = value[: self.config.max_entry_size] + "\n... [truncated]"
                truncated = True
                size_bytes = len(value)
            elif self.config.auto_summarize:
                # For structured data, just note it's too large
                truncated = True

        # Create entry
        entry = ContextEntry(
            key=key,
            value=value,
            context_type=context_type,
            step_id=step_id,
            size_bytes=size_bytes,
            truncated=truncated,
            metadata=metadata or {},
        )

        # Update totals
        if key in self._entries:
            self._total_size -= self._entries[key].size_bytes
        else:
            self._order.append(key)

        self._entries[key] = entry
        self._total_size += size_bytes

        # Enforce total size limit by removing oldest entries
        self._enforce_total_limit()

        return entry

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value.

        Args:
            key: Key to look up.
            default: Default value if key not found.

        Returns:
            The stored value or default.
        """
        entry = self._entries.get(key)
        return entry.value if entry else default

    def get_entry(self, key: str) -> ContextEntry | None:
        """Get the full context entry.

        Args:
            key: Key to look up.

        Returns:
            The ContextEntry or None.
        """
        return self._entries.get(key)

    def has(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: Key to check.

        Returns:
            True if the key exists.
        """
        return key in self._entries

    def delete(self, key: str) -> bool:
        """Delete a context entry.

        Args:
            key: Key to delete.

        Returns:
            True if the key was deleted.
        """
        if key not in self._entries:
            return False

        entry = self._entries[key]
        self._total_size -= entry.size_bytes
        del self._entries[key]
        self._order.remove(key)
        return True

    def add_artifact(
        self,
        key: str,
        path: Path | str,
        step_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextEntry:
        """Add a file artifact to the context.

        Args:
            key: Key for the artifact.
            path: Path to the artifact file.
            step_id: ID of the step that produced this.
            metadata: Additional metadata.

        Returns:
            The created ContextEntry.
        """
        path = Path(path) if isinstance(path, str) else path

        return self.set(
            key=key,
            value=path,
            step_id=step_id,
            context_type=ContextType.ARTIFACT,
            metadata={"path": str(path), **(metadata or {})},
        )

    def get_for_step(self, step_id: str) -> dict[str, Any]:
        """Get all context entries produced by a specific step.

        Args:
            step_id: Step ID to filter by.

        Returns:
            Dictionary of key-value pairs from that step.
        """
        return {
            key: entry.value for key, entry in self._entries.items() if entry.step_id == step_id
        }

    def get_inputs(self, input_keys: list[str]) -> dict[str, Any]:
        """Get multiple inputs for a step.

        Args:
            input_keys: List of keys to retrieve.

        Returns:
            Dictionary of available inputs.
        """
        return {key: self.get(key) for key in input_keys if self.has(key)}

    def build_prompt_context(
        self,
        input_keys: list[str],
        max_chars: int = 4000,
    ) -> str:
        """Build a context string for inclusion in a prompt.

        Args:
            input_keys: Keys to include in the context.
            max_chars: Maximum characters in the output.

        Returns:
            Formatted context string.
        """
        parts = []
        total_chars = 0

        for key in input_keys:
            entry = self.get_entry(key)
            if not entry:
                continue

            # Format based on type
            if entry.context_type == ContextType.ARTIFACT:
                content = f"[File: {entry.value}]"
            elif entry.context_type == ContextType.STRUCTURED:
                content = json.dumps(entry.value, indent=2)
            else:
                content = str(entry.value)

            # Check size
            section = f"\n### {key}\n{content}"
            if total_chars + len(section) > max_chars:
                # Truncate this section
                remaining = max_chars - total_chars - len(f"\n### {key}\n")
                if remaining > 100:
                    content = content[:remaining] + "\n... [truncated]"
                    section = f"\n### {key}\n{content}"
                else:
                    section = f"\n### {key}\n[Content too large - {entry.size_bytes} bytes]"

            parts.append(section)
            total_chars += len(section)

            if total_chars >= max_chars:
                break

        return "".join(parts) if parts else ""

    def clear(self) -> None:
        """Clear all context entries."""
        self._entries.clear()
        self._order.clear()
        self._total_size = 0

    def keys(self) -> list[str]:
        """Get all context keys in insertion order.

        Returns:
            List of keys.
        """
        return list(self._order)

    def values(self) -> list[Any]:
        """Get all context values in insertion order.

        Returns:
            List of values.
        """
        return [self._entries[key].value for key in self._order]

    def items(self) -> list[tuple[str, Any]]:
        """Get all context items in insertion order.

        Returns:
            List of (key, value) tuples.
        """
        return [(key, self._entries[key].value) for key in self._order]

    def entries(self) -> list[ContextEntry]:
        """Get all context entries in insertion order.

        Returns:
            List of ContextEntry objects.
        """
        return [self._entries[key] for key in self._order]

    def size(self) -> int:
        """Get total context size in bytes.

        Returns:
            Total size of all entries.
        """
        return self._total_size

    def count(self) -> int:
        """Get number of context entries.

        Returns:
            Number of entries.
        """
        return len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        """Export context to dictionary.

        Returns:
            Dictionary representation of all context.
        """
        return {
            "entries": [entry.to_dict() for entry in self.entries()],
            "total_size": self._total_size,
            "count": len(self._entries),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Import context from dictionary.

        Args:
            data: Dictionary with context data.
        """
        self.clear()
        for entry_data in data.get("entries", []):
            self.set(
                key=entry_data["key"],
                value=entry_data["value"],
                step_id=entry_data.get("step_id"),
                context_type=ContextType(entry_data.get("context_type", "text")),
                metadata=entry_data.get("metadata", {}),
            )

    def _detect_type(self, value: Any) -> ContextType:
        """Auto-detect context type from value."""
        if isinstance(value, Path):
            return ContextType.ARTIFACT
        elif isinstance(value, (dict, list, tuple)):
            return ContextType.STRUCTURED
        else:
            return ContextType.TEXT

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of a value in bytes."""
        if isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, Path):
            return len(str(value))
        elif isinstance(value, (dict, list)):
            return len(json.dumps(value))
        else:
            return len(str(value))

    def _enforce_total_limit(self) -> None:
        """Remove oldest entries to stay within total size limit."""
        while self._total_size > self.config.max_total_size and self._order:
            oldest_key = self._order[0]
            self.delete(oldest_key)


def main() -> int:
    """CLI entry point for testing."""
    ctx = ContextManager()

    # Add some test entries
    ctx.set("step1_output", "This is the output from step 1", step_id="S1")
    ctx.set("analysis", {"score": 95, "issues": []}, step_id="S2")
    ctx.add_artifact("code", Path("/tmp/output.py"), step_id="S3")

    print("Context Manager Test")
    print("=" * 40)
    print(f"Entries: {ctx.count()}")
    print(f"Total size: {ctx.size()} bytes")
    print()

    for entry in ctx.entries():
        print(f"Key: {entry.key}")
        print(f"  Type: {entry.context_type.value}")
        print(f"  Step: {entry.step_id}")
        print(f"  Size: {entry.size_bytes} bytes")
        print()

    print("Prompt context:")
    print(ctx.build_prompt_context(["step1_output", "analysis"]))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
