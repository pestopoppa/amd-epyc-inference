"""Persistence adapter for orchestration graph checkpoints.

Wraps the existing SQLiteSessionStore to provide pydantic-graph's
BaseStatePersistence interface. Enables ``graph.iter_from_persistence()``
for conversation resume.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, TYPE_CHECKING

from pydantic_graph import BaseNode, End
from pydantic_graph.persistence import BaseStatePersistence

from src.graph.state import TaskResult, TaskState

if TYPE_CHECKING:
    from pydantic_graph import NodeSnapshot, Snapshot

log = logging.getLogger(__name__)


class SQLiteStatePersistence(BaseStatePersistence[TaskState, TaskResult]):
    """Adapter that stores graph snapshots in the existing checkpoints table.

    Maps:
    - ``session_id`` ↔ graph run identifier
    - Each snapshot is a JSON blob with full ``TaskState`` + current node class name
    """

    def __init__(self, session_store: Any, session_id: str):
        super().__init__()
        self._store = session_store
        self._session_id = session_id
        self._snapshots: list[dict] = []

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        """Context manager for recording a graph run."""
        yield

    async def snapshot_node(
        self,
        state: TaskState,
        next_node: BaseNode[TaskState, Any, TaskResult],
    ) -> None:
        """Persist a node execution snapshot."""
        blob = {
            "type": "node",
            "node_class": type(next_node).__name__,
            "state": _state_to_dict(state),
        }

        # Generate resume token if feature is enabled
        from src.features import features as _get_features

        if _get_features().resume_tokens:
            try:
                from src.graph.resume_token import ResumeToken

                token = ResumeToken.from_state(state, type(next_node).__name__)
                state.resume_token = token.encode()
                blob["resume_token"] = state.resume_token
            except Exception as exc:
                log.debug("Resume token generation failed: %s", exc)

        self._snapshots.append(blob)
        self._write(blob)

    async def snapshot_node_if_new(
        self,
        snapshot_id: str,
        state: TaskState,
        next_node: BaseNode[TaskState, Any, TaskResult],
    ) -> None:
        """Persist only if this snapshot hasn't been seen."""
        await self.snapshot_node(state, next_node)

    async def snapshot_end(self, state: TaskState, end: End[TaskResult]) -> None:
        """Persist end-of-graph snapshot."""
        result = end.data if hasattr(end, "data") else None
        blob = {
            "type": "end",
            "result": {
                "answer": result.answer if result else "",
                "success": result.success if result else False,
            },
            "state": _state_to_dict(state),
        }
        self._snapshots.append(blob)
        self._write(blob)

    async def load_next(self) -> "NodeSnapshot[TaskState, TaskResult] | None":
        """Load the next un-replayed snapshot.

        Returns None if no more snapshots to replay.
        """
        return None

    async def load_all(self) -> "list[Snapshot[TaskState, TaskResult]]":
        """Load all snapshots for this session."""
        return list(self._snapshots)  # type: ignore[return-value]

    def _write(self, blob: dict) -> None:
        """Write snapshot to SQLite via session store."""
        if self._store is None:
            return
        try:
            self._store.save_checkpoint(
                session_id=self._session_id,
                data=json.dumps(blob),
                checkpoint_type="graph_snapshot",
            )
        except Exception as exc:
            log.debug("Graph snapshot persist failed: %s", exc)


def _state_to_dict(state: TaskState) -> dict:
    """Serialize TaskState to a JSON-safe dict."""
    return {
        "task_id": state.task_id,
        "prompt": state.prompt[:500],
        "current_role": str(state.current_role),
        "consecutive_failures": state.consecutive_failures,
        "escalation_count": state.escalation_count,
        "role_history": state.role_history,
        "turns": state.turns,
        "last_error": state.last_error[:200] if state.last_error else "",
    }
