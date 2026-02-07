"""Pydantic-graph based orchestration flow.

Public API:
    orchestration_graph  — the Graph singleton
    run_task()           — one-shot execution
    iter_task()          — streaming iteration
    generate_mermaid()   — visual topology
    select_start_node()  — role → node mapping

State types:
    TaskState, TaskDeps, TaskResult, GraphConfig

Node classes:
    FrontdoorNode, WorkerNode, CoderNode, CoderEscalationNode,
    IngestNode, ArchitectNode, ArchitectCodingNode
"""

from src.graph.graph import (
    generate_mermaid,
    iter_task,
    orchestration_graph,
    run_task,
)
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
    GraphConfig,
    TaskDeps,
    TaskResult,
    TaskState,
)

__all__ = [
    # Graph
    "orchestration_graph",
    "run_task",
    "iter_task",
    "generate_mermaid",
    "select_start_node",
    # State
    "TaskState",
    "TaskDeps",
    "TaskResult",
    "GraphConfig",
    # Nodes
    "FrontdoorNode",
    "WorkerNode",
    "CoderNode",
    "CoderEscalationNode",
    "IngestNode",
    "ArchitectNode",
    "ArchitectCodingNode",
]
