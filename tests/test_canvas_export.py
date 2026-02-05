"""Tests for canvas_export module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.canvas_export import (
    Canvas,
    CanvasEdge,
    CanvasNode,
    COLORS,
    _confidence_to_color,
    _format_timestamp,
    _layout_grid,
    _layout_radial,
    export_failure_graph,
    export_hypothesis_graph,
    export_session_context,
    get_default_canvas_path,
)


class TestCanvasNode:
    """Tests for CanvasNode dataclass."""

    def test_text_node_to_dict(self):
        node = CanvasNode(
            id="node1",
            type="text",
            x=100,
            y=200,
            width=250,
            height=100,
            text="Test content",
            color="#22c55e",
        )
        result = node.to_dict()

        assert result["id"] == "node1"
        assert result["type"] == "text"
        assert result["x"] == 100
        assert result["y"] == 200
        assert result["width"] == 250
        assert result["height"] == 100
        assert result["text"] == "Test content"
        assert result["color"] == "#22c55e"

    def test_node_without_optional_fields(self):
        node = CanvasNode(
            id="node2",
            type="text",
            x=0,
            y=0,
            width=100,
            height=50,
        )
        result = node.to_dict()

        assert "color" not in result
        assert "label" not in result


class TestCanvasEdge:
    """Tests for CanvasEdge dataclass."""

    def test_edge_to_dict(self):
        edge = CanvasEdge(
            id="edge1",
            fromNode="node1",
            toNode="node2",
            fromSide="bottom",
            toSide="top",
            color="#3b82f6",
            label="supports",
        )
        result = edge.to_dict()

        assert result["id"] == "edge1"
        assert result["fromNode"] == "node1"
        assert result["toNode"] == "node2"
        assert result["fromSide"] == "bottom"
        assert result["toSide"] == "top"
        assert result["color"] == "#3b82f6"
        assert result["label"] == "supports"

    def test_edge_without_optional_fields(self):
        edge = CanvasEdge(
            id="edge2",
            fromNode="a",
            toNode="b",
        )
        result = edge.to_dict()

        assert "color" not in result
        assert "label" not in result


class TestCanvas:
    """Tests for Canvas dataclass."""

    def test_empty_canvas(self):
        canvas = Canvas()
        result = canvas.to_dict()

        assert result == {"nodes": [], "edges": []}

    def test_canvas_with_nodes_and_edges(self):
        canvas = Canvas(
            nodes=[
                CanvasNode(id="n1", type="text", x=0, y=0, width=100, height=50, text="Node 1"),
                CanvasNode(id="n2", type="text", x=200, y=0, width=100, height=50, text="Node 2"),
            ],
            edges=[
                CanvasEdge(id="e1", fromNode="n1", toNode="n2"),
            ],
        )
        result = canvas.to_dict()

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["nodes"][0]["id"] == "n1"
        assert result["edges"][0]["fromNode"] == "n1"

    def test_canvas_to_json(self):
        canvas = Canvas(
            nodes=[CanvasNode(id="n1", type="text", x=0, y=0, width=100, height=50)],
        )
        json_str = canvas.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "nodes" in parsed
        assert len(parsed["nodes"]) == 1

    def test_canvas_save(self, tmp_path):
        canvas = Canvas(
            nodes=[CanvasNode(id="n1", type="text", x=0, y=0, width=100, height=50, text="Test")],
        )
        output_path = tmp_path / "test.canvas"
        result_path = canvas.save(output_path)

        assert result_path == output_path
        assert output_path.exists()

        # Verify content
        content = json.loads(output_path.read_text())
        assert len(content["nodes"]) == 1

    def test_canvas_save_creates_directories(self, tmp_path):
        canvas = Canvas()
        nested_path = tmp_path / "deep" / "nested" / "test.canvas"
        canvas.save(nested_path)

        assert nested_path.exists()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_confidence_to_color_high(self):
        assert _confidence_to_color(0.9) == COLORS["high"]
        assert _confidence_to_color(0.7) == COLORS["high"]

    def test_confidence_to_color_medium(self):
        assert _confidence_to_color(0.5) == COLORS["medium"]
        assert _confidence_to_color(0.3) == COLORS["medium"]

    def test_confidence_to_color_low(self):
        assert _confidence_to_color(0.2) == COLORS["low"]
        assert _confidence_to_color(0.0) == COLORS["low"]

    def test_confidence_to_color_untested(self):
        assert _confidence_to_color(0.5, tested=False) == COLORS["neutral"]

    def test_format_timestamp(self):
        dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = _format_timestamp(dt)
        assert result == "2026-01-15 10:30"

    def test_format_timestamp_none(self):
        assert _format_timestamp(None) == "N/A"

    def test_layout_grid(self):
        positions = _layout_grid(6, start_x=0, start_y=0, cols=3)
        assert len(positions) == 6

        # First row
        assert positions[0][1] == positions[1][1] == positions[2][1]
        # Second row
        assert positions[3][1] == positions[4][1] == positions[5][1]
        # Different rows
        assert positions[0][1] != positions[3][1]

    def test_layout_radial(self):
        positions = _layout_radial(4, center_x=500, center_y=500, radius=200)
        assert len(positions) == 4

        # All positions should be different
        assert len(set(positions)) == 4

    def test_layout_radial_empty(self):
        positions = _layout_radial(0)
        assert positions == []


class TestExportHypothesisGraph:
    """Tests for export_hypothesis_graph function."""

    @pytest.fixture
    def mock_hypothesis_graph(self):
        """Create a mock HypothesisGraph."""
        graph = MagicMock()

        # Mock hypothesis query
        hyp_df = pd.DataFrame({
            "h.id": ["h1", "h2"],
            "h.claim": ["action1|task1", "action2|task2"],
            "h.confidence": [0.8, 0.3],
            "h.created_at": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "h.tested": [True, False],
        })

        # Mock evidence queries
        empty_df = pd.DataFrame(columns=["e.id", "e.source", "e.timestamp"])

        def mock_execute(query, params=None):
            result = MagicMock()
            if "Hypothesis" in query and "LIMIT" in query:
                result.get_as_df.return_value = hyp_df
            else:
                result.get_as_df.return_value = empty_df
            return result

        graph.conn.execute = mock_execute
        return graph

    def test_export_hypothesis_graph(self, mock_hypothesis_graph, tmp_path):
        output = tmp_path / "hypothesis.canvas"
        # Test without evidence to avoid mock complexity
        canvas = export_hypothesis_graph(mock_hypothesis_graph, output, include_evidence=False)

        assert len(canvas.nodes) == 2
        assert output.exists()

        # Check node colors
        node_colors = {n.id: n.color for n in canvas.nodes}
        assert node_colors["h1"] == COLORS["high"]  # 0.8 confidence, tested
        assert node_colors["h2"] == COLORS["neutral"]  # 0.3 confidence, untested

    def test_export_hypothesis_graph_empty(self, tmp_path):
        graph = MagicMock()
        empty_df = pd.DataFrame(columns=["h.id", "h.claim", "h.confidence", "h.created_at", "h.tested"])
        graph.conn.execute.return_value.get_as_df.return_value = empty_df

        canvas = export_hypothesis_graph(graph)
        assert len(canvas.nodes) == 0

    def test_export_hypothesis_graph_no_output_path(self, mock_hypothesis_graph):
        # Test without evidence to avoid mock complexity
        canvas = export_hypothesis_graph(mock_hypothesis_graph, output_path=None, include_evidence=False)
        assert isinstance(canvas, Canvas)
        assert len(canvas.nodes) == 2


class TestExportFailureGraph:
    """Tests for export_failure_graph function."""

    @pytest.fixture
    def mock_failure_graph(self):
        """Create a mock FailureGraph."""
        graph = MagicMock()

        # Mock failure query
        failure_df = pd.DataFrame({
            "f.id": ["f1"],
            "f.description": ["Test failure"],
            "f.severity": [3],
            "f.first_seen": [datetime.now(timezone.utc)],
            "f.last_seen": [datetime.now(timezone.utc)],
        })

        symptom_empty_df = pd.DataFrame(columns=["s.id", "s.pattern", "s.detection_method"])
        mitigation_empty_df = pd.DataFrame(columns=["m.id", "m.action", "m.success_rate"])

        def mock_execute(query, params=None):
            result = MagicMock()
            # Main failure query: MATCH (f:FailureMode) RETURN f.id... LIMIT
            if "RETURN f.id" in query:
                result.get_as_df.return_value = failure_df
            # Symptom query
            elif "HAS_SYMPTOM" in query:
                result.get_as_df.return_value = symptom_empty_df
            # Mitigation query
            elif "MITIGATED_BY" in query:
                result.get_as_df.return_value = mitigation_empty_df
            else:
                result.get_as_df.return_value = symptom_empty_df
            return result

        graph.conn.execute = mock_execute
        return graph

    def test_export_failure_graph(self, mock_failure_graph, tmp_path):
        output = tmp_path / "failure.canvas"
        # Test without mitigations to avoid mock complexity
        canvas = export_failure_graph(mock_failure_graph, output, include_mitigations=False)

        assert len(canvas.nodes) >= 1
        assert output.exists()

    def test_export_failure_graph_empty(self, tmp_path):
        graph = MagicMock()
        empty_df = pd.DataFrame(columns=["f.id", "f.description", "f.severity", "f.first_seen", "f.last_seen"])
        graph.conn.execute.return_value.get_as_df.return_value = empty_df

        canvas = export_failure_graph(graph)
        assert len(canvas.nodes) == 0


class TestExportSessionContext:
    """Tests for export_session_context function."""

    def test_export_session_context(self, tmp_path):
        # Create mock graphs
        hyp_graph = MagicMock()
        fail_graph = MagicMock()

        empty_df = pd.DataFrame()
        hyp_graph.conn.execute.return_value.get_as_df.return_value = empty_df
        fail_graph.conn.execute.return_value.get_as_df.return_value = empty_df

        output = tmp_path / "session.canvas"
        canvas = export_session_context(hyp_graph, fail_graph, output, session_id="test123")

        # Should have title and separator nodes at minimum
        assert len(canvas.nodes) >= 2
        assert output.exists()

        # Check for session ID in title
        title_nodes = [n for n in canvas.nodes if "test123" in n.text]
        assert len(title_nodes) == 1


class TestGetDefaultCanvasPath:
    """Tests for get_default_canvas_path function."""

    def test_get_default_path_hypothesis(self):
        path = get_default_canvas_path("hypothesis")
        assert "hypothesis_" in str(path)
        assert path.suffix == ".canvas"

    def test_get_default_path_failure(self):
        path = get_default_canvas_path("failure")
        assert "failure_" in str(path)

    def test_get_default_path_session(self):
        path = get_default_canvas_path("session")
        assert "session_" in str(path)
