"""Portable resume tokens for crash recovery (Lobster pattern).

Encodes minimal graph state as a compact base64url token for
cross-session resume. Complements SQLiteStatePersistence.

Only active when features().resume_tokens is True.

Encoding: JSON → zlib compress → base64url. Target: <500 bytes.

Usage:
    from src.graph.resume_token import ResumeToken

    # Encode from TaskState
    token = ResumeToken.from_state(state, node_class="CoderNode")
    encoded = token.encode()  # compact base64url string

    # Decode back
    token = ResumeToken.decode(encoded)
    # Use token.node_class, token.current_role, etc. to reconstruct state
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import zlib
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ResumeToken:
    """Compact graph state for crash recovery.

    Contains only the fields needed to resume execution — no full
    prompt or context (those come from the session store).
    """

    task_id: str = ""
    node_class: str = ""  # e.g. "CoderNode"
    current_role: str = ""
    turns: int = 0
    escalation_count: int = 0
    consecutive_failures: int = 0
    role_history: list[str] = field(default_factory=list)
    last_error: str = ""  # truncated to 200 chars
    checksum: str = ""  # SHA-256[:8] integrity check

    @classmethod
    def from_state(
        cls,
        state: "TaskState",  # noqa: F821
        node_class: str,
    ) -> ResumeToken:
        """Create a ResumeToken from current TaskState.

        Args:
            state: Current graph state.
            node_class: Name of the current node class.

        Returns:
            ResumeToken ready for encoding.
        """
        token = cls(
            task_id=state.task_id,
            node_class=node_class,
            current_role=str(state.current_role),
            turns=state.turns,
            escalation_count=state.escalation_count,
            consecutive_failures=state.consecutive_failures,
            role_history=list(state.role_history),
            last_error=(state.last_error[:200] if state.last_error else ""),
        )
        # Compute checksum over content fields
        content = json.dumps(
            {k: v for k, v in asdict(token).items() if k != "checksum"},
            sort_keys=True,
        )
        token.checksum = hashlib.sha256(content.encode()).hexdigest()[:8]
        return token

    def encode(self) -> str:
        """Encode token to compact base64url string.

        Returns:
            Base64url-encoded string (typically <500 bytes).
        """
        payload = json.dumps(asdict(self), separators=(",", ":"))
        compressed = zlib.compress(payload.encode(), level=9)
        return base64.urlsafe_b64encode(compressed).decode()

    @classmethod
    def decode(cls, encoded: str) -> ResumeToken:
        """Decode a base64url token string.

        Args:
            encoded: Base64url-encoded token string.

        Returns:
            ResumeToken instance.

        Raises:
            ValueError: If token is invalid or checksum fails.
        """
        try:
            compressed = base64.urlsafe_b64decode(encoded)
            payload = zlib.decompress(compressed).decode()
            data = json.loads(payload)
        except Exception as e:
            raise ValueError(f"Invalid resume token: {e}") from e

        token = cls(**data)

        # Verify checksum
        content = json.dumps(
            {k: v for k, v in data.items() if k != "checksum"},
            sort_keys=True,
        )
        expected = hashlib.sha256(content.encode()).hexdigest()[:8]
        if token.checksum != expected:
            raise ValueError(
                f"Resume token checksum mismatch: {token.checksum} != {expected}"
            )

        return token
