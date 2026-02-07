"""Orchestration graph construction and execution.

Provides the ``orchestration_graph`` singleton, ``run_task()`` for
one-shot execution, ``iter_task()`` for streaming/debugging, and
``generate_mermaid()`` for visual topology.
"""

from __future__ import annotations

from pydantic_graph import Graph

from src.graph.nodes import (
    ArchitectCodingNode,
    ArchitectNode,
    CoderEscalationNode,
    CoderNode,
    FrontdoorNode,
    IngestNode,
    WorkerNode,
    select_start_node,
)
from src.graph.state import (
    TaskDeps,
    TaskResult,
    TaskState,
)
from src.roles import Role

# ── Graph singleton ────────────────────────────────────────────────────

orchestration_graph: Graph[TaskState, TaskDeps, TaskResult] = Graph(
    nodes=[
        FrontdoorNode,
        WorkerNode,
        CoderNode,
        CoderEscalationNode,
        IngestNode,
        ArchitectNode,
        ArchitectCodingNode,
    ],
    name="orchestration",
)


# ── Execution helpers ──────────────────────────────────────────────────


async def run_task(
    state: TaskState,
    deps: TaskDeps,
    start_role: Role | str | None = None,
) -> TaskResult:
    """Run the full orchestration graph to completion.

    Args:
        state: Mutable task state.
        deps: Immutable infrastructure dependencies.
        start_role: Initial role (determines start node).
            Defaults to state.current_role.

    Returns:
        TaskResult with answer, success flag, and metadata.
    """
    role = start_role or state.current_role
    start_node = select_start_node(role)

    # Ensure role_history is initialized
    if not state.role_history:
        state.record_role(role)

    result = await orchestration_graph.run(
        start_node,
        state=state,
        deps=deps,
    )
    return result.output


async def iter_task(
    state: TaskState,
    deps: TaskDeps,
    start_role: Role | str | None = None,
):
    """Iterate through graph execution, yielding node snapshots.

    Useful for SSE streaming or debugging. Yields after each node
    completes so callers can emit events between turns.

    Args:
        state: Mutable task state.
        deps: Immutable infrastructure dependencies.
        start_role: Initial role (determines start node).

    Yields:
        Node instances at each step of execution.
    """
    role = start_role or state.current_role
    start_node = select_start_node(role)

    if not state.role_history:
        state.record_role(role)

    async with orchestration_graph.iter(start_node, state=state, deps=deps) as run:
        async for node in run:
            yield node


def generate_mermaid(direction: str = "LR") -> str:
    """Generate a Mermaid state diagram of the orchestration graph.

    Args:
        direction: Diagram direction ("LR", "TB", "RL", "BT").

    Returns:
        Mermaid code string.
    """
    return orchestration_graph.mermaid_code(direction=direction)
