"""JSON Canvas import with diff detection and TOON encoding for LLM context.

Imports edited JSON Canvas files and detects changes vs a baseline:
- Node additions/deletions/modifications
- Edge changes
- Position-based priority inference (central = important)

TOON encoding provides 40-65% token reduction when passing canvas context to LLMs.
Files on disk remain in standard JSON for Obsidian compatibility.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of changes detected in a canvas."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    MOVED = "moved"


@dataclass
class NodeChange:
    """A change to a canvas node."""

    node_id: str
    change_type: ChangeType
    old_value: dict[str, Any] | None = None
    new_value: dict[str, Any] | None = None
    field_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class EdgeChange:
    """A change to a canvas edge."""

    edge_id: str
    change_type: ChangeType
    old_value: dict[str, Any] | None = None
    new_value: dict[str, Any] | None = None


@dataclass
class CanvasDiff:
    """Differences between two canvas states."""

    node_changes: list[NodeChange] = field(default_factory=list)
    edge_changes: list[EdgeChange] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Whether any changes were detected."""
        return bool(self.node_changes or self.edge_changes)

    @property
    def summary(self) -> str:
        """Human-readable summary of changes."""
        parts = []
        by_type = {}
        for nc in self.node_changes:
            by_type.setdefault(nc.change_type, []).append(nc.node_id)

        for ct, ids in by_type.items():
            parts.append(f"{ct.value}: {len(ids)} nodes")

        edge_by_type = {}
        for ec in self.edge_changes:
            edge_by_type.setdefault(ec.change_type, []).append(ec.edge_id)

        for ct, ids in edge_by_type.items():
            parts.append(f"{ct.value}: {len(ids)} edges")

        return "; ".join(parts) if parts else "No changes"


@dataclass
class CanvasConstraints:
    """Constraints inferred from canvas edits for use in planning.

    When a user edits a canvas, their changes encode priorities:
    - Central position = high importance
    - Larger nodes = more detail expected
    - Connected nodes = dependencies
    - Deleted nodes = not relevant
    - Added nodes = new requirements
    """

    priority_nodes: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    added_requirements: list[str] = field(default_factory=list)
    dependency_edges: list[tuple[str, str]] = field(default_factory=list)
    position_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "priority_nodes": self.priority_nodes,
            "removed_nodes": self.removed_nodes,
            "added_requirements": self.added_requirements,
            "dependency_edges": self.dependency_edges,
            "position_weights": self.position_weights,
        }


def parse_canvas(path: str | Path) -> dict[str, Any]:
    """Parse a JSON Canvas file.

    Args:
        path: Path to .canvas file

    Returns:
        Parsed canvas dict with 'nodes' and 'edges' keys

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Canvas file not found: {path}")

    content = path.read_text()
    canvas = json.loads(content)

    # Validate basic structure
    if "nodes" not in canvas:
        canvas["nodes"] = []
    if "edges" not in canvas:
        canvas["edges"] = []

    return canvas


def compute_canvas_diff(baseline: dict[str, Any], current: dict[str, Any]) -> CanvasDiff:
    """Compute differences between two canvas states.

    Args:
        baseline: Original canvas state
        current: Current (edited) canvas state

    Returns:
        CanvasDiff with all detected changes
    """
    diff = CanvasDiff()

    # Index nodes by ID
    base_nodes = {n["id"]: n for n in baseline.get("nodes", [])}
    curr_nodes = {n["id"]: n for n in current.get("nodes", [])}

    # Find node changes
    all_node_ids = set(base_nodes.keys()) | set(curr_nodes.keys())

    for node_id in all_node_ids:
        base_node = base_nodes.get(node_id)
        curr_node = curr_nodes.get(node_id)

        if base_node is None:
            # New node added
            diff.node_changes.append(
                NodeChange(
                    node_id=node_id,
                    change_type=ChangeType.ADDED,
                    new_value=curr_node,
                )
            )
        elif curr_node is None:
            # Node removed
            diff.node_changes.append(
                NodeChange(
                    node_id=node_id,
                    change_type=ChangeType.REMOVED,
                    old_value=base_node,
                )
            )
        else:
            # Check for modifications
            field_changes = {}
            position_changed = False

            for key in set(base_node.keys()) | set(curr_node.keys()):
                old_val = base_node.get(key)
                new_val = curr_node.get(key)
                if old_val != new_val:
                    field_changes[key] = (old_val, new_val)
                    if key in ("x", "y"):
                        position_changed = True

            if field_changes:
                # Determine if it's a move or content modification
                content_keys = {"text", "color", "label", "width", "height"}
                has_content_change = any(k in field_changes for k in content_keys)

                if has_content_change:
                    change_type = ChangeType.MODIFIED
                elif position_changed:
                    change_type = ChangeType.MOVED
                else:
                    change_type = ChangeType.MODIFIED

                diff.node_changes.append(
                    NodeChange(
                        node_id=node_id,
                        change_type=change_type,
                        old_value=base_node,
                        new_value=curr_node,
                        field_changes=field_changes,
                    )
                )

    # Index edges by ID
    base_edges = {e["id"]: e for e in baseline.get("edges", [])}
    curr_edges = {e["id"]: e for e in current.get("edges", [])}

    all_edge_ids = set(base_edges.keys()) | set(curr_edges.keys())

    for edge_id in all_edge_ids:
        base_edge = base_edges.get(edge_id)
        curr_edge = curr_edges.get(edge_id)

        if base_edge is None:
            diff.edge_changes.append(
                EdgeChange(
                    edge_id=edge_id,
                    change_type=ChangeType.ADDED,
                    new_value=curr_edge,
                )
            )
        elif curr_edge is None:
            diff.edge_changes.append(
                EdgeChange(
                    edge_id=edge_id,
                    change_type=ChangeType.REMOVED,
                    old_value=base_edge,
                )
            )
        elif base_edge != curr_edge:
            diff.edge_changes.append(
                EdgeChange(
                    edge_id=edge_id,
                    change_type=ChangeType.MODIFIED,
                    old_value=base_edge,
                    new_value=curr_edge,
                )
            )

    return diff


def extract_constraints(canvas: dict[str, Any], diff: CanvasDiff | None = None) -> CanvasConstraints:
    """Extract planning constraints from a canvas and optional diff.

    Args:
        canvas: Current canvas state
        diff: Optional diff from baseline (enables change-based constraints)

    Returns:
        CanvasConstraints for use in planning
    """
    constraints = CanvasConstraints()

    nodes = canvas.get("nodes", [])
    edges = canvas.get("edges", [])

    if not nodes:
        return constraints

    # Calculate canvas center
    xs = [n.get("x", 0) for n in nodes]
    ys = [n.get("y", 0) for n in nodes]
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)

    # Calculate position weights (closer to center = higher weight)
    max_dist = max(
        ((n.get("x", 0) - center_x) ** 2 + (n.get("y", 0) - center_y) ** 2) ** 0.5
        for n in nodes
    ) or 1

    for node in nodes:
        node_id = node.get("id", "")
        x = node.get("x", 0)
        y = node.get("y", 0)
        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

        # Normalize: 1.0 = at center, 0.0 = furthest
        weight = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
        constraints.position_weights[node_id] = round(weight, 3)

    # Top priority = high weight nodes
    sorted_nodes = sorted(
        constraints.position_weights.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    constraints.priority_nodes = [nid for nid, _ in sorted_nodes[:5]]

    # Extract dependencies from edges
    for edge in edges:
        from_id = edge.get("fromNode", "")
        to_id = edge.get("toNode", "")
        if from_id and to_id:
            constraints.dependency_edges.append((from_id, to_id))

    # Process diff if available
    if diff:
        for nc in diff.node_changes:
            if nc.change_type == ChangeType.REMOVED:
                constraints.removed_nodes.append(nc.node_id)
            elif nc.change_type == ChangeType.ADDED:
                # Extract text from new node as requirement
                text = nc.new_value.get("text", "") if nc.new_value else ""
                if text:
                    constraints.added_requirements.append(text[:200])

    return constraints


def load_canvas_for_llm(
    path: str | Path,
    baseline_path: str | Path | None = None,
    use_toon: bool = True,
) -> str:
    """Load a canvas and encode for LLM context.

    This is the primary entry point for getting canvas data into LLM context.
    Uses TOON encoding for 40-65% token reduction when available.

    Args:
        path: Path to current canvas file
        baseline_path: Optional path to baseline canvas (enables diff)
        use_toon: Whether to use TOON encoding (falls back to JSON if unavailable)

    Returns:
        Encoded string (TOON or JSON) suitable for LLM context
    """
    canvas = parse_canvas(path)

    diff = None
    if baseline_path:
        baseline = parse_canvas(baseline_path)
        diff = compute_canvas_diff(baseline, canvas)

    constraints = extract_constraints(canvas, diff)

    # Build context object
    context = {
        "canvas_summary": {
            "node_count": len(canvas.get("nodes", [])),
            "edge_count": len(canvas.get("edges", [])),
        },
        "constraints": constraints.to_dict(),
    }

    if diff and diff.has_changes:
        context["changes"] = diff.summary

    # Include high-priority nodes
    nodes_by_id = {n["id"]: n for n in canvas.get("nodes", [])}
    context["priority_nodes"] = []
    for node_id in constraints.priority_nodes[:5]:
        if node_id in nodes_by_id:
            node = nodes_by_id[node_id]
            context["priority_nodes"].append({
                "id": node_id,
                "text": node.get("text", "")[:300],
                "weight": constraints.position_weights.get(node_id, 0),
            })

    # Try TOON encoding
    if use_toon:
        try:
            from src.services.toon_encoder import encode, should_use_toon

            if should_use_toon(context):
                return encode(context)
        except ImportError:
            logger.debug("TOON not available, using JSON")

    # Fallback to compact JSON
    return json.dumps(context, separators=(",", ":"))


def apply_constraints_to_plan(
    plan: dict[str, Any],
    constraints: CanvasConstraints,
) -> dict[str, Any]:
    """Apply canvas-derived constraints to a task plan.

    Modifies plan priorities and filters based on canvas edits.

    Args:
        plan: Task plan dict (with 'steps' list)
        constraints: Constraints from canvas

    Returns:
        Modified plan with constraints applied
    """
    if "steps" not in plan:
        return plan

    # Remove steps for removed nodes
    if constraints.removed_nodes:
        plan["steps"] = [
            s for s in plan["steps"]
            if s.get("id") not in constraints.removed_nodes
        ]

    # Boost priority of high-weight steps
    for step in plan["steps"]:
        step_id = step.get("id", "")
        if step_id in constraints.position_weights:
            weight = constraints.position_weights[step_id]
            step["priority_boost"] = weight

    # Add new requirements as steps
    for i, req in enumerate(constraints.added_requirements):
        plan["steps"].append({
            "id": f"canvas_req_{i}",
            "description": req,
            "source": "canvas_edit",
            "priority_boost": 1.0,  # User-added = high priority
        })

    return plan


def validate_canvas_schema(canvas: dict[str, Any]) -> list[str]:
    """Validate a canvas against JSON Canvas schema.

    Args:
        canvas: Canvas dict to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not isinstance(canvas, dict):
        return ["Canvas must be a dict"]

    nodes = canvas.get("nodes", [])
    edges = canvas.get("edges", [])

    if not isinstance(nodes, list):
        errors.append("'nodes' must be a list")
    if not isinstance(edges, list):
        errors.append("'edges' must be a list")

    # Validate nodes
    node_ids = set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"Node {i}: must be a dict")
            continue

        if "id" not in node:
            errors.append(f"Node {i}: missing 'id'")
        else:
            if node["id"] in node_ids:
                errors.append(f"Node {i}: duplicate id '{node['id']}'")
            node_ids.add(node["id"])

        if "type" not in node:
            errors.append(f"Node {i}: missing 'type'")

        for required in ("x", "y", "width", "height"):
            if required not in node:
                errors.append(f"Node {i}: missing '{required}'")
            elif not isinstance(node[required], int | float):
                errors.append(f"Node {i}: '{required}' must be numeric")

    # Validate edges
    edge_ids = set()
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"Edge {i}: must be a dict")
            continue

        if "id" not in edge:
            errors.append(f"Edge {i}: missing 'id'")
        else:
            if edge["id"] in edge_ids:
                errors.append(f"Edge {i}: duplicate id '{edge['id']}'")
            edge_ids.add(edge["id"])

        for required in ("fromNode", "toNode"):
            if required not in edge:
                errors.append(f"Edge {i}: missing '{required}'")
            elif edge[required] not in node_ids:
                errors.append(f"Edge {i}: {required} '{edge[required]}' not found in nodes")

    return errors
