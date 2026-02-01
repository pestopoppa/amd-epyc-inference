"""Structured logging support for the orchestrator API.

Provides a JSON log formatter and a helper to inject structured fields
into standard Python logging calls via the `extra` parameter.

Usage:
    import logging
    from src.api.structured_logging import configure_json_logging, task_extra

    # Enable JSON output (typically at startup)
    configure_json_logging()

    logger = logging.getLogger(__name__)
    logger.info("Request routed", extra=task_extra(
        task_id="chat-abc123",
        role="architect_general",
        latency_ms=42.5,
    ))
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


# Structured field names that are recognized by the JSON formatter
STRUCTURED_FIELDS = frozenset({
    "task_id", "role", "latency_ms", "stage", "strategy",
    "mode", "turn", "error_type", "tokens", "prompt_len",
})


def task_extra(
    task_id: str | None = None,
    role: str | None = None,
    latency_ms: float | None = None,
    stage: str | None = None,
    strategy: str | None = None,
    mode: str | None = None,
    turn: int | None = None,
    error_type: str | None = None,
    tokens: int | None = None,
    prompt_len: int | None = None,
) -> dict[str, Any]:
    """Build an `extra` dict for structured log fields.

    Only includes non-None values to keep log lines compact.
    """
    fields: dict[str, Any] = {}
    if task_id is not None:
        fields["task_id"] = task_id
    if role is not None:
        fields["role"] = role
    if latency_ms is not None:
        fields["latency_ms"] = round(latency_ms, 1)
    if stage is not None:
        fields["stage"] = stage
    if strategy is not None:
        fields["strategy"] = strategy
    if mode is not None:
        fields["mode"] = mode
    if turn is not None:
        fields["turn"] = turn
    if error_type is not None:
        fields["error_type"] = error_type
    if tokens is not None:
        fields["tokens"] = tokens
    if prompt_len is not None:
        fields["prompt_len"] = prompt_len
    return fields


class JSONFormatter(logging.Formatter):
    """JSON log formatter that includes structured fields from `extra`.

    Output format (one JSON object per line):
        {"ts": "...", "level": "INFO", "logger": "src.api...", "msg": "...", "task_id": "...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include recognized structured fields from extra
        for field in STRUCTURED_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                entry[field] = val
        return json.dumps(entry, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to use JSON output.

    Call once at application startup. Only affects the root logger handler;
    individual logger levels are preserved.
    """
    root = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
    root.setLevel(level)
