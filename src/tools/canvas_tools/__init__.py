"""Canvas tools for MCP integration.

Provides MCP tools for exporting/importing JSON Canvas files:
- export_reasoning_canvas: Export hypothesis/failure graphs to canvas
- import_canvas_edits: Import edited canvas and extract constraints
- list_canvases: List available canvas files
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from src.tool_registry import ToolCategory, ToolRegistry

logger = logging.getLogger(__name__)

# Default canvas directory
CANVAS_DIR = Path("/mnt/raid0/llm/claude/logs/canvases")


def export_reasoning_canvas(
    graph_type: Literal["hypothesis", "failure", "session"] = "hypothesis",
    include_evidence: bool = True,
    output_path: str | None = None,
) -> str:
    """Export a reasoning graph to JSON Canvas format.

    Args:
        graph_type: Type of graph to export:
            - "hypothesis": HypothesisGraph with confidence scores
            - "failure": FailureGraph with symptoms/mitigations
            - "session": Combined session context
        include_evidence: Whether to include evidence/symptom nodes
        output_path: Optional custom output path. If None, uses default.

    Returns:
        Path to the created canvas file, or error message.
    """
    try:
        from src.canvas_export import (
            export_failure_graph,
            export_hypothesis_graph,
            export_session_context,
            get_default_canvas_path,
        )

        # Determine output path
        if output_path:
            path = Path(output_path)
        else:
            path = get_default_canvas_path(graph_type)

        if graph_type == "hypothesis":
            from orchestration.repl_memory.hypothesis_graph import HypothesisGraph

            graph = HypothesisGraph()
            canvas = export_hypothesis_graph(
                graph,
                output_path=path,
                include_evidence=include_evidence,
            )
            graph.close()

        elif graph_type == "failure":
            from orchestration.repl_memory.failure_graph import FailureGraph

            graph = FailureGraph()
            canvas = export_failure_graph(
                graph,
                output_path=path,
                include_mitigations=include_evidence,
            )
            graph.close()

        elif graph_type == "session":
            from orchestration.repl_memory.failure_graph import FailureGraph
            from orchestration.repl_memory.hypothesis_graph import HypothesisGraph

            hyp_graph = HypothesisGraph()
            fail_graph = FailureGraph()
            canvas = export_session_context(
                hyp_graph,
                fail_graph,
                output_path=path,
            )
            hyp_graph.close()
            fail_graph.close()

        else:
            return f"Unknown graph type: {graph_type}"

        return f"Canvas exported to: {path}\nNodes: {len(canvas.nodes)}, Edges: {len(canvas.edges)}"

    except ImportError as e:
        logger.warning("Missing dependency for canvas export: %s", e)
        return f"Error: Missing dependency - {e}"
    except Exception as e:
        logger.exception("Failed to export canvas: %s", e)
        return f"Error exporting canvas: {type(e).__name__}: {e}"


def import_canvas_edits(
    canvas_path: str,
    baseline_path: str | None = None,
) -> str:
    """Import an edited canvas and extract planning constraints.

    Args:
        canvas_path: Path to the edited canvas file
        baseline_path: Optional path to baseline canvas for diff

    Returns:
        JSON-encoded constraints extracted from the canvas.
    """
    import json

    try:
        from src.canvas_import import (
            compute_canvas_diff,
            extract_constraints,
            parse_canvas,
        )

        canvas = parse_canvas(canvas_path)

        diff = None
        if baseline_path:
            baseline = parse_canvas(baseline_path)
            diff = compute_canvas_diff(baseline, canvas)

        constraints = extract_constraints(canvas, diff)
        result = {
            "status": "success",
            "constraints": constraints.to_dict(),
        }

        if diff:
            result["changes_summary"] = diff.summary

        return json.dumps(result, indent=2)

    except FileNotFoundError as e:
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.exception("Failed to import canvas: %s", e)
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})


def list_canvases(directory: str | None = None) -> str:
    """List available canvas files.

    Args:
        directory: Directory to search. Defaults to logs/canvases.

    Returns:
        Formatted list of canvas files with metadata.
    """
    import json
    from datetime import datetime

    search_dir = Path(directory) if directory else CANVAS_DIR

    if not search_dir.exists():
        return json.dumps({"status": "success", "canvases": [], "message": "No canvases directory"})

    canvases = []
    for path in sorted(search_dir.glob("*.canvas"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            content = json.loads(path.read_text())
            node_count = len(content.get("nodes", []))
            edge_count = len(content.get("edges", []))
            mtime = datetime.fromtimestamp(path.stat().st_mtime)

            canvases.append({
                "path": str(path),
                "name": path.name,
                "nodes": node_count,
                "edges": edge_count,
                "modified": mtime.isoformat(),
            })
        except Exception as e:
            canvases.append({
                "path": str(path),
                "name": path.name,
                "error": str(e),
            })

    return json.dumps({"status": "success", "canvases": canvases}, indent=2)


def register_canvas_tools(registry: ToolRegistry) -> int:
    """Register canvas tools with the tool registry.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered.
    """
    registry.register_handler(
        name="export_reasoning_canvas",
        description="Export a reasoning graph (hypothesis/failure/session) to JSON Canvas format for visualization in Obsidian",
        category=ToolCategory.DATA,
        parameters={
            "graph_type": {
                "type": "string",
                "required": False,
                "description": "Type of graph: 'hypothesis', 'failure', or 'session'",
            },
            "include_evidence": {
                "type": "boolean",
                "required": False,
                "description": "Whether to include evidence/symptom nodes",
            },
            "output_path": {
                "type": "string",
                "required": False,
                "description": "Custom output path (optional)",
            },
        },
    )(export_reasoning_canvas)

    registry.register_handler(
        name="import_canvas_edits",
        description="Import an edited canvas file and extract planning constraints from user edits",
        category=ToolCategory.DATA,
        parameters={
            "canvas_path": {
                "type": "string",
                "required": True,
                "description": "Path to the edited canvas file",
            },
            "baseline_path": {
                "type": "string",
                "required": False,
                "description": "Path to baseline canvas for diff comparison",
            },
        },
    )(import_canvas_edits)

    registry.register_handler(
        name="list_canvases",
        description="List available canvas files in the canvases directory",
        category=ToolCategory.DATA,
        parameters={
            "directory": {
                "type": "string",
                "required": False,
                "description": "Directory to search (defaults to logs/canvases)",
            },
        },
    )(list_canvases)

    return 3
