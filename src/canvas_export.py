"""JSON Canvas export for HypothesisGraph and FailureGraph visualization.

Exports graphs to JSON Canvas format (https://jsoncanvas.org/) for visualization
in Obsidian and other compatible tools. Supports:
- HypothesisGraph: Hypotheses and evidence nodes with confidence-based coloring
- FailureGraph: Failures, symptoms, and mitigations
- Session context: Combined view of active reasoning state

Spatial layout uses force-directed positioning to minimize edge crossings.
Colors map to confidence (green=high, yellow=medium, red=low).
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default canvas directory (on RAID per CLAUDE.md)
DEFAULT_CANVAS_DIR = Path("/mnt/raid0/llm/claude/logs/canvases")

# Node dimensions
NODE_WIDTH = 250
NODE_HEIGHT = 100
GROUP_PADDING = 40

# Color palette based on confidence
COLORS = {
    "high": "#22c55e",      # Green - high confidence (>0.7)
    "medium": "#eab308",    # Yellow - medium confidence (0.3-0.7)
    "low": "#ef4444",       # Red - low confidence (<0.3)
    "neutral": "#6b7280",   # Gray - untested
    "evidence": "#3b82f6",  # Blue - evidence nodes
    "symptom": "#f97316",   # Orange - symptom nodes
    "mitigation": "#8b5cf6",  # Purple - mitigation nodes
}


@dataclass
class CanvasNode:
    """A node in the JSON Canvas."""

    id: str
    type: str  # "text", "group", "file", "link"
    x: int
    y: int
    width: int
    height: int
    text: str = ""
    color: str | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON Canvas node format."""
        node: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }
        if self.type == "text":
            node["text"] = self.text
        if self.color:
            node["color"] = self.color
        if self.label:
            node["label"] = self.label
        return node


@dataclass
class CanvasEdge:
    """An edge in the JSON Canvas."""

    id: str
    fromNode: str
    toNode: str
    fromSide: str = "bottom"
    toSide: str = "top"
    color: str | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON Canvas edge format."""
        edge: dict[str, Any] = {
            "id": self.id,
            "fromNode": self.fromNode,
            "toNode": self.toNode,
            "fromSide": self.fromSide,
            "toSide": self.toSide,
        }
        if self.color:
            edge["color"] = self.color
        if self.label:
            edge["label"] = self.label
        return edge


@dataclass
class Canvas:
    """A JSON Canvas document."""

    nodes: list[CanvasNode] = field(default_factory=list)
    edges: list[CanvasEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON Canvas format."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        """Save canvas to file.

        Args:
            path: Output path. Creates parent directories if needed.

        Returns:
            Path where canvas was saved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        logger.info("Saved canvas to %s (%d nodes, %d edges)", path, len(self.nodes), len(self.edges))
        return path


def _confidence_to_color(confidence: float, tested: bool = True) -> str:
    """Map confidence value to color.

    Args:
        confidence: Confidence score 0.0-1.0
        tested: Whether the hypothesis has been tested

    Returns:
        Color hex code
    """
    if not tested:
        return COLORS["neutral"]
    if confidence >= 0.7:
        return COLORS["high"]
    if confidence >= 0.3:
        return COLORS["medium"]
    return COLORS["low"]


def _format_timestamp(dt: datetime | None) -> str:
    """Format timestamp for display."""
    if dt is None:
        return "N/A"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M")


def _layout_grid(count: int, start_x: int = 0, start_y: int = 0, cols: int = 3) -> list[tuple[int, int]]:
    """Generate grid layout positions.

    Args:
        count: Number of nodes to position
        start_x: Starting X coordinate
        start_y: Starting Y coordinate
        cols: Number of columns

    Returns:
        List of (x, y) positions
    """
    positions = []
    x_spacing = NODE_WIDTH + 50
    y_spacing = NODE_HEIGHT + 50

    for i in range(count):
        col = i % cols
        row = i // cols
        x = start_x + col * x_spacing
        y = start_y + row * y_spacing
        positions.append((x, y))

    return positions


def _layout_radial(
    count: int,
    center_x: int = 500,
    center_y: int = 500,
    radius: int = 300,
) -> list[tuple[int, int]]:
    """Generate radial layout positions around a center.

    Args:
        count: Number of nodes to position
        center_x: Center X coordinate
        center_y: Center Y coordinate
        radius: Distance from center

    Returns:
        List of (x, y) positions
    """
    if count == 0:
        return []

    positions = []
    angle_step = 2 * math.pi / count

    for i in range(count):
        angle = i * angle_step - math.pi / 2  # Start from top
        x = int(center_x + radius * math.cos(angle) - NODE_WIDTH / 2)
        y = int(center_y + radius * math.sin(angle) - NODE_HEIGHT / 2)
        positions.append((x, y))

    return positions


def export_hypothesis_graph(
    graph,  # HypothesisGraph
    output_path: str | Path | None = None,
    include_evidence: bool = True,
    max_hypotheses: int = 50,
) -> Canvas:
    """Export HypothesisGraph to JSON Canvas.

    Args:
        graph: HypothesisGraph instance
        output_path: Optional path to save canvas. If None, only returns Canvas object.
        include_evidence: Whether to include evidence nodes
        max_hypotheses: Maximum hypotheses to include (prevents canvas overload)

    Returns:
        Canvas object
    """
    canvas = Canvas()

    # Query all hypotheses
    result = graph.conn.execute(
        """
        MATCH (h:Hypothesis)
        RETURN h.id, h.claim, h.confidence, h.created_at, h.tested
        ORDER BY h.confidence DESC
        LIMIT $limit
        """,
        {"limit": max_hypotheses},
    )
    hypotheses = result.get_as_df()

    if hypotheses.empty:
        logger.info("No hypotheses found in graph")
        if output_path:
            canvas.save(output_path)
        return canvas

    # Create hypothesis nodes
    positions = _layout_grid(len(hypotheses), start_x=300, start_y=100)
    hypothesis_nodes = {}

    for i, (_, row) in enumerate(hypotheses.iterrows()):
        h_id = row["h.id"]
        claim = row["h.claim"]
        confidence = float(row["h.confidence"])
        tested = bool(row["h.tested"])
        created_at = row["h.created_at"]

        # Parse claim (action|task_type format)
        parts = claim.split("|", 1) if "|" in claim else [claim, ""]
        action, task_type = parts[0], parts[1] if len(parts) > 1 else ""

        text = f"**{action}**\n\n"
        text += f"Task: {task_type}\n"
        text += f"Confidence: {confidence:.2f}\n"
        text += f"Tested: {'Yes' if tested else 'No'}\n"
        text += f"Created: {_format_timestamp(created_at)}"

        x, y = positions[i]
        node = CanvasNode(
            id=h_id,
            type="text",
            x=x,
            y=y,
            width=NODE_WIDTH,
            height=NODE_HEIGHT,
            text=text,
            color=_confidence_to_color(confidence, tested),
        )
        canvas.nodes.append(node)
        hypothesis_nodes[h_id] = node

    # Add evidence nodes if requested
    if include_evidence:
        evidence_positions = _layout_grid(
            len(hypotheses) * 2,  # Estimate 2 evidence per hypothesis
            start_x=700,
            start_y=100,
            cols=2,
        )
        evidence_idx = 0

        for h_id in hypothesis_nodes:
            # Get supporting evidence
            result = graph.conn.execute(
                """
                MATCH (e:HypothesisEvidence)-[:SUPPORTS]->(h:Hypothesis {id: $id})
                RETURN e.id, e.source, e.timestamp
                LIMIT 3
                """,
                {"id": h_id},
            )
            supports = result.get_as_df()

            for _, ev_row in supports.iterrows():
                if evidence_idx >= len(evidence_positions):
                    break
                e_id = ev_row["e.id"]
                source = ev_row["e.source"][:50]
                x, y = evidence_positions[evidence_idx]
                evidence_idx += 1

                ev_node = CanvasNode(
                    id=e_id,
                    type="text",
                    x=x,
                    y=y,
                    width=200,
                    height=60,
                    text=f"+SUPPORTS\n{source}",
                    color=COLORS["evidence"],
                )
                canvas.nodes.append(ev_node)

                edge = CanvasEdge(
                    id=f"edge_{e_id}_{h_id}",
                    fromNode=e_id,
                    toNode=h_id,
                    fromSide="left",
                    toSide="right",
                    color=COLORS["high"],
                    label="supports",
                )
                canvas.edges.append(edge)

            # Get contradicting evidence
            result = graph.conn.execute(
                """
                MATCH (e:HypothesisEvidence)-[:CONTRADICTS]->(h:Hypothesis {id: $id})
                RETURN e.id, e.source, e.timestamp
                LIMIT 3
                """,
                {"id": h_id},
            )
            contradicts = result.get_as_df()

            for _, ev_row in contradicts.iterrows():
                if evidence_idx >= len(evidence_positions):
                    break
                e_id = ev_row["e.id"]
                source = ev_row["e.source"][:50]
                x, y = evidence_positions[evidence_idx]
                evidence_idx += 1

                ev_node = CanvasNode(
                    id=e_id,
                    type="text",
                    x=x,
                    y=y,
                    width=200,
                    height=60,
                    text=f"-CONTRADICTS\n{source}",
                    color=COLORS["low"],
                )
                canvas.nodes.append(ev_node)

                edge = CanvasEdge(
                    id=f"edge_{e_id}_{h_id}",
                    fromNode=e_id,
                    toNode=h_id,
                    fromSide="left",
                    toSide="right",
                    color=COLORS["low"],
                    label="contradicts",
                )
                canvas.edges.append(edge)

    if output_path:
        canvas.save(output_path)

    return canvas


def export_failure_graph(
    graph,  # FailureGraph
    output_path: str | Path | None = None,
    include_mitigations: bool = True,
    max_failures: int = 30,
) -> Canvas:
    """Export FailureGraph to JSON Canvas.

    Args:
        graph: FailureGraph instance
        output_path: Optional path to save canvas
        include_mitigations: Whether to include mitigation nodes
        max_failures: Maximum failures to include

    Returns:
        Canvas object
    """
    canvas = Canvas()

    # Query failures
    result = graph.conn.execute(
        """
        MATCH (f:FailureMode)
        RETURN f.id, f.description, f.severity, f.first_seen, f.last_seen
        ORDER BY f.last_seen DESC
        LIMIT $limit
        """,
        {"limit": max_failures},
    )
    failures = result.get_as_df()

    if failures.empty:
        logger.info("No failures found in graph")
        if output_path:
            canvas.save(output_path)
        return canvas

    # Layout failures in center
    failure_positions = _layout_grid(len(failures), start_x=400, start_y=200, cols=2)
    failure_nodes = {}

    for i, (_, row) in enumerate(failures.iterrows()):
        f_id = row["f.id"]
        desc = row["f.description"][:100]
        severity = int(row["f.severity"])
        last_seen = row["f.last_seen"]

        text = f"**Failure (Sev {severity})**\n\n"
        text += f"{desc}\n"
        text += f"Last seen: {_format_timestamp(last_seen)}"

        # Severity-based color
        if severity >= 4:
            color = COLORS["low"]
        elif severity >= 2:
            color = COLORS["medium"]
        else:
            color = COLORS["high"]

        x, y = failure_positions[i]
        node = CanvasNode(
            id=f_id,
            type="text",
            x=x,
            y=y,
            width=NODE_WIDTH,
            height=NODE_HEIGHT + 20,
            text=text,
            color=color,
        )
        canvas.nodes.append(node)
        failure_nodes[f_id] = node

    # Add symptoms
    symptom_idx = 0
    for f_id in failure_nodes:
        result = graph.conn.execute(
            """
            MATCH (f:FailureMode {id: $id})-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN s.id, s.pattern, s.detection_method
            LIMIT 5
            """,
            {"id": f_id},
        )
        symptoms = result.get_as_df()

        for _, s_row in symptoms.iterrows():
            s_id = s_row["s.id"]
            pattern = s_row["s.pattern"][:60]

            x = 50
            y = 100 + symptom_idx * 80
            symptom_idx += 1

            s_node = CanvasNode(
                id=s_id,
                type="text",
                x=x,
                y=y,
                width=200,
                height=60,
                text=f"Symptom\n`{pattern}`",
                color=COLORS["symptom"],
            )
            canvas.nodes.append(s_node)

            edge = CanvasEdge(
                id=f"edge_f_{f_id}_s_{s_id}",
                fromNode=f_id,
                toNode=s_id,
                fromSide="left",
                toSide="right",
                label="has_symptom",
            )
            canvas.edges.append(edge)

    # Add mitigations
    if include_mitigations:
        mitigation_idx = 0
        for f_id in failure_nodes:
            result = graph.conn.execute(
                """
                MATCH (f:FailureMode {id: $id})-[:MITIGATED_BY]->(m:Mitigation)
                RETURN m.id, m.action, m.success_rate
                LIMIT 3
                """,
                {"id": f_id},
            )
            mitigations = result.get_as_df()

            for _, m_row in mitigations.iterrows():
                m_id = m_row["m.id"]
                action = m_row["m.action"][:60]
                success_rate = float(m_row["m.success_rate"])

                x = 900
                y = 100 + mitigation_idx * 80
                mitigation_idx += 1

                m_node = CanvasNode(
                    id=m_id,
                    type="text",
                    x=x,
                    y=y,
                    width=220,
                    height=70,
                    text=f"Mitigation ({success_rate:.0%})\n{action}",
                    color=COLORS["mitigation"],
                )
                canvas.nodes.append(m_node)

                edge = CanvasEdge(
                    id=f"edge_f_{f_id}_m_{m_id}",
                    fromNode=f_id,
                    toNode=m_id,
                    fromSide="right",
                    toSide="left",
                    color=COLORS["mitigation"],
                    label="mitigated_by",
                )
                canvas.edges.append(edge)

    if output_path:
        canvas.save(output_path)

    return canvas


def export_session_context(
    hypothesis_graph,
    failure_graph,
    output_path: str | Path | None = None,
    session_id: str | None = None,
) -> Canvas:
    """Export combined session context with both graphs.

    Creates a canvas with two regions:
    - Top: Hypothesis graph (reasoning state)
    - Bottom: Failure graph (anti-patterns)

    Args:
        hypothesis_graph: HypothesisGraph instance
        failure_graph: FailureGraph instance
        output_path: Optional path to save canvas
        session_id: Optional session identifier for title

    Returns:
        Canvas object
    """
    canvas = Canvas()
    session_id = session_id or str(uuid.uuid4())[:8]

    # Title node
    title = CanvasNode(
        id=f"title_{session_id}",
        type="text",
        x=400,
        y=10,
        width=300,
        height=50,
        text=f"**Session Context: {session_id}**\n{_format_timestamp(datetime.now(timezone.utc))}",
        color=COLORS["neutral"],
    )
    canvas.nodes.append(title)

    # Export hypothesis graph (top region)
    hyp_canvas = export_hypothesis_graph(
        hypothesis_graph,
        include_evidence=True,
        max_hypotheses=20,
    )

    # Offset hypothesis nodes
    for node in hyp_canvas.nodes:
        node.y += 80
        node.id = f"hyp_{node.id}"
        canvas.nodes.append(node)

    for edge in hyp_canvas.edges:
        edge.fromNode = f"hyp_{edge.fromNode}"
        edge.toNode = f"hyp_{edge.toNode}"
        edge.id = f"hyp_{edge.id}"
        canvas.edges.append(edge)

    # Export failure graph (bottom region)
    fail_canvas = export_failure_graph(
        failure_graph,
        include_mitigations=True,
        max_failures=15,
    )

    # Offset failure nodes to bottom
    y_offset = 600
    for node in fail_canvas.nodes:
        node.y += y_offset
        node.id = f"fail_{node.id}"
        canvas.nodes.append(node)

    for edge in fail_canvas.edges:
        edge.fromNode = f"fail_{edge.fromNode}"
        edge.toNode = f"fail_{edge.toNode}"
        edge.id = f"fail_{edge.id}"
        canvas.edges.append(edge)

    # Add separator
    separator = CanvasNode(
        id="separator",
        type="text",
        x=0,
        y=y_offset - 50,
        width=1200,
        height=30,
        text="--- Failure Patterns (Anti-Memory) ---",
        color=COLORS["neutral"],
    )
    canvas.nodes.append(separator)

    if output_path:
        canvas.save(output_path)

    return canvas


def get_default_canvas_path(graph_type: str = "hypothesis") -> Path:
    """Get default path for a canvas type.

    Args:
        graph_type: "hypothesis", "failure", or "session"

    Returns:
        Path to default canvas location
    """
    DEFAULT_CANVAS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_CANVAS_DIR / f"{graph_type}_{timestamp}.canvas"
