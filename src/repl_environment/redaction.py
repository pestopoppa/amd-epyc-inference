"""Credential redaction for tool and REPL output.

Scans text for known credential patterns (API keys, SSH keys, connection
strings, tokens) and replaces them with redaction markers. Applied as a
post-execution filter before output enters model context or session
transcripts.

Guarded by features().credential_redaction (default: True in production).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Minimum input length worth scanning (no credential fits in < 16 chars)
_MIN_SCAN_LENGTH = 16

# Maximum input length to scan (skip bulk data dumps)
_MAX_SCAN_LENGTH = 1_048_576  # 1 MB


@dataclass(frozen=True)
class RedactionResult:
    """Result of credential scanning."""

    text: str
    redacted_count: int
    categories: frozenset[str]  # e.g. {"aws_access_key", "ssh_private_key"}


# Each entry: (category_name, compiled_regex, replacement_string)
_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # === Cloud provider keys ===
    # AWS access key IDs (always start with AKIA)
    (
        "aws_access_key",
        re.compile(r"AKIA[0-9A-Z]{16}"),
        "[REDACTED:aws_access_key]",
    ),
    # Anthropic API keys
    (
        "anthropic_key",
        re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}"),
        "[REDACTED:anthropic_key]",
    ),
    # OpenAI API keys (sk- followed by 20+ alphanum, but NOT sk-ant-)
    (
        "openai_key",
        re.compile(r"sk-(?!ant-)[a-zA-Z0-9]{20,}"),
        "[REDACTED:openai_key]",
    ),
    # Google Cloud API keys
    (
        "gcloud_key",
        re.compile(r"AIza[0-9A-Za-z_-]{35}"),
        "[REDACTED:gcloud_key]",
    ),
    # === GitHub tokens ===
    (
        "github_pat",
        re.compile(r"ghp_[A-Za-z0-9]{36,}"),
        "[REDACTED:github_pat]",
    ),
    (
        "github_oauth",
        re.compile(r"gho_[A-Za-z0-9]{36,}"),
        "[REDACTED:github_oauth]",
    ),
    (
        "github_app",
        re.compile(r"(?:ghs|ghu)_[A-Za-z0-9]{36,}"),
        "[REDACTED:github_token]",
    ),
    (
        "github_fine_grained",
        re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
        "[REDACTED:github_fine_grained_pat]",
    ),
    # === SSH private keys (multiline) ===
    (
        "ssh_private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ED25519 )?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END (?:RSA |EC |DSA |OPENSSH |ED25519 )?PRIVATE KEY-----"
        ),
        "[REDACTED:ssh_private_key]",
    ),
    # === Bearer tokens ===
    (
        "bearer_token",
        re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
        "Bearer [REDACTED:token]",
    ),
    # === Database connection strings ===
    (
        "connection_string",
        re.compile(
            r"(?:postgres(?:ql)?|mysql|redis|mongodb(?:\+srv)?|amqp|amqps)"
            r":\/\/[^\s\"'<>]{10,}"
        ),
        "[REDACTED:connection_string]",
    ),
    # === Slack tokens ===
    (
        "slack_token",
        re.compile(r"xox[baprs]-[0-9a-zA-Z-]{10,}"),
        "[REDACTED:slack_token]",
    ),
    # === Stripe keys ===
    (
        "stripe_key",
        re.compile(r"(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{20,}"),
        "[REDACTED:stripe_key]",
    ),
    # === .env style KEY=secret on same line ===
    # IMPORTANT: This is the broadest/most generic pattern — must come LAST
    # so specific key patterns (OpenAI sk-, GCloud AIza, etc.) match first.
    (
        "env_secret",
        re.compile(
            r"(?:API_KEY|SECRET_KEY|ACCESS_TOKEN|AUTH_TOKEN|PASSWORD|PRIVATE_KEY|"
            r"DATABASE_URL|REDIS_URL|MONGO_URI|JWT_SECRET|ENCRYPTION_KEY|"
            r"CLIENT_SECRET|WEBHOOK_SECRET)"
            r"\s*=\s*[\"']?[^\s\"'\n]{8,}[\"']?",
            re.IGNORECASE,
        ),
        "[REDACTED:env_value]",
    ),
]


def redact_credentials(text: str) -> RedactionResult:
    """Scan text for credential patterns and replace matches.

    Args:
        text: Raw output text to scan.

    Returns:
        RedactionResult with redacted text, count, and categories.
    """
    if not isinstance(text, str):
        return RedactionResult(text=str(text), redacted_count=0, categories=frozenset())

    length = len(text)
    if length < _MIN_SCAN_LENGTH or length > _MAX_SCAN_LENGTH:
        return RedactionResult(text=text, redacted_count=0, categories=frozenset())

    count = 0
    categories: set[str] = set()
    result = text

    for name, pattern, replacement in _CREDENTIAL_PATTERNS:
        new_result, n = pattern.subn(replacement, result)
        if n > 0:
            result = new_result
            count += n
            categories.add(name)

    return RedactionResult(
        text=result,
        redacted_count=count,
        categories=frozenset(categories),
    )


def redact_if_enabled(text: str) -> str:
    """Convenience: redact only when feature flag is on, return text unchanged otherwise.

    Logs a warning when credentials are found.
    """
    try:
        from src.features import features as _get_features

        if not _get_features().credential_redaction:
            return text
    except Exception:
        # Feature system unavailable — redact defensively
        pass

    rr = redact_credentials(text)
    if rr.redacted_count > 0:
        logger.warning(
            "Redacted %d credential(s) from output: %s",
            rr.redacted_count,
            sorted(rr.categories),
        )
    return rr.text
