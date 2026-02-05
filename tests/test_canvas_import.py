"""Tests for canvas_import module."""

from __future__ import annotations

import json

import pytest

from src.canvas_import import (
    CanvasConstraints,
    CanvasDiff,
    ChangeType,
    NodeChange,
    apply_constraints_to_plan,
    compute_canvas_diff,
    extract_constraints,
    load_canvas_for_llm,
    parse_canvas,
    validate_canvas_schema,
)


class TestParseCanvas:
    """Tests for parse_canvas function."""

    def test_parse_valid_canvas(self, tmp_path):
        canvas_data = {
            "nodes": [
                {"id": "n1", "type": "text", "x": 0, "y": 0, "width": 100, "height": 50}
            ],
            "edges": [],
        }
        canvas_file = tmp_path / "test.canvas"
        canvas_file.write_text(json.dumps(canvas_data))

        result = parse_canvas(canvas_file)

        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "n1"

    def test_parse_canvas_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_canvas("/nonexistent/path.canvas")

    def test_parse_canvas_adds_missing_keys(self, tmp_path):
        canvas_file = tmp_path / "test.canvas"
        canvas_file.write_text("{}")

        result = parse_canvas(canvas_file)

        assert "nodes" in result
        assert "edges" in result
        assert result["nodes"] == []
        assert result["edges"] == []


class TestComputeCanvasDiff:
    """Tests for compute_canvas_diff function."""

    def test_no_changes(self):
        canvas = {
            "nodes": [{"id": "n1", "type": "text", "x": 0, "y": 0}],
            "edges": [],
        }
        diff = compute_canvas_diff(canvas, canvas)

        assert not diff.has_changes
        assert diff.node_changes == []
        assert diff.edge_changes == []

    def test_node_added(self):
        baseline = {"nodes": [], "edges": []}
        current = {
            "nodes": [{"id": "n1", "type": "text", "x": 0, "y": 0}],
            "edges": [],
        }

        diff = compute_canvas_diff(baseline, current)

        assert diff.has_changes
        assert len(diff.node_changes) == 1
        assert diff.node_changes[0].change_type == ChangeType.ADDED
        assert diff.node_changes[0].node_id == "n1"

    def test_node_removed(self):
        baseline = {
            "nodes": [{"id": "n1", "type": "text", "x": 0, "y": 0}],
            "edges": [],
        }
        current = {"nodes": [], "edges": []}

        diff = compute_canvas_diff(baseline, current)

        assert diff.has_changes
        assert len(diff.node_changes) == 1
        assert diff.node_changes[0].change_type == ChangeType.REMOVED

    def test_node_modified(self):
        baseline = {
            "nodes": [{"id": "n1", "type": "text", "x": 0, "y": 0, "text": "old"}],
            "edges": [],
        }
        current = {
            "nodes": [{"id": "n1", "type": "text", "x": 0, "y": 0, "text": "new"}],
            "edges": [],
        }

        diff = compute_canvas_diff(baseline, current)

        assert diff.has_changes
        assert len(diff.node_changes) == 1
        assert diff.node_changes[0].change_type == ChangeType.MODIFIED
        assert "text" in diff.node_changes[0].field_changes

    def test_node_moved(self):
        baseline = {
            "nodes": [{"id": "n1", "type": "text", "x": 0, "y": 0}],
            "edges": [],
        }
        current = {
            "nodes": [{"id": "n1", "type": "text", "x": 100, "y": 200}],
            "edges": [],
        }

        diff = compute_canvas_diff(baseline, current)

        assert diff.has_changes
        assert len(diff.node_changes) == 1
        assert diff.node_changes[0].change_type == ChangeType.MOVED

    def test_edge_added(self):
        baseline = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [],
        }
        current = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"id": "e1", "fromNode": "n1", "toNode": "n2"}],
        }

        diff = compute_canvas_diff(baseline, current)

        assert diff.has_changes
        assert len(diff.edge_changes) == 1
        assert diff.edge_changes[0].change_type == ChangeType.ADDED

    def test_edge_removed(self):
        baseline = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"id": "e1", "fromNode": "n1", "toNode": "n2"}],
        }
        current = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [],
        }

        diff = compute_canvas_diff(baseline, current)

        assert diff.has_changes
        assert len(diff.edge_changes) == 1
        assert diff.edge_changes[0].change_type == ChangeType.REMOVED

    def test_diff_summary(self):
        baseline = {"nodes": [], "edges": []}
        current = {
            "nodes": [
                {"id": "n1", "type": "text", "x": 0, "y": 0},
                {"id": "n2", "type": "text", "x": 100, "y": 0},
            ],
            "edges": [{"id": "e1", "fromNode": "n1", "toNode": "n2"}],
        }

        diff = compute_canvas_diff(baseline, current)
        summary = diff.summary

        assert "added" in summary.lower()
        assert "2 nodes" in summary


class TestExtractConstraints:
    """Tests for extract_constraints function."""

    def test_extract_position_weights(self):
        # Three nodes: one at the computed center (333, 333), one close, one far
        canvas = {
            "nodes": [
                {"id": "center", "x": 333, "y": 333},  # Will be at center
                {"id": "close", "x": 400, "y": 400},   # Close to center
                {"id": "far", "x": 0, "y": 0},         # Far from center
            ],
            "edges": [],
        }

        constraints = extract_constraints(canvas)

        # Center node should have highest weight (closest to geometric center)
        assert constraints.position_weights["center"] > constraints.position_weights["far"]

    def test_extract_priority_nodes(self):
        canvas = {
            "nodes": [
                {"id": "n1", "x": 500, "y": 500},
                {"id": "n2", "x": 0, "y": 0},
                {"id": "n3", "x": 1000, "y": 1000},
            ],
            "edges": [],
        }

        constraints = extract_constraints(canvas)

        # n1 is closest to center
        assert "n1" in constraints.priority_nodes

    def test_extract_dependency_edges(self):
        canvas = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"id": "e1", "fromNode": "n1", "toNode": "n2"}],
        }

        constraints = extract_constraints(canvas)

        assert ("n1", "n2") in constraints.dependency_edges

    def test_extract_with_diff(self):
        canvas = {
            "nodes": [{"id": "n1", "x": 0, "y": 0, "text": "New requirement"}],
            "edges": [],
        }
        diff = CanvasDiff(
            node_changes=[
                NodeChange(
                    node_id="n1",
                    change_type=ChangeType.ADDED,
                    new_value={"id": "n1", "text": "New requirement"},
                ),
                NodeChange(
                    node_id="old",
                    change_type=ChangeType.REMOVED,
                    old_value={"id": "old"},
                ),
            ],
        )

        constraints = extract_constraints(canvas, diff)

        assert "old" in constraints.removed_nodes
        assert any("New requirement" in r for r in constraints.added_requirements)

    def test_empty_canvas(self):
        constraints = extract_constraints({"nodes": [], "edges": []})

        assert constraints.priority_nodes == []
        assert constraints.position_weights == {}


class TestCanvasConstraints:
    """Tests for CanvasConstraints dataclass."""

    def test_to_dict(self):
        constraints = CanvasConstraints(
            priority_nodes=["n1", "n2"],
            removed_nodes=["old"],
            added_requirements=["Do this"],
            dependency_edges=[("n1", "n2")],
            position_weights={"n1": 0.8},
        )
        result = constraints.to_dict()

        assert result["priority_nodes"] == ["n1", "n2"]
        assert result["removed_nodes"] == ["old"]
        assert ("n1", "n2") in result["dependency_edges"]


class TestLoadCanvasForLLM:
    """Tests for load_canvas_for_llm function."""

    def test_load_basic_canvas(self, tmp_path):
        canvas_data = {
            "nodes": [
                {"id": "n1", "type": "text", "x": 100, "y": 100, "width": 100, "height": 50, "text": "Test node"}
            ],
            "edges": [],
        }
        canvas_file = tmp_path / "test.canvas"
        canvas_file.write_text(json.dumps(canvas_data))

        result = load_canvas_for_llm(canvas_file, use_toon=False)

        # Should be valid JSON
        parsed = json.loads(result)
        assert "canvas_summary" in parsed
        assert parsed["canvas_summary"]["node_count"] == 1

    def test_load_with_diff(self, tmp_path):
        baseline_data = {"nodes": [], "edges": []}
        current_data = {
            "nodes": [{"id": "n1", "type": "text", "x": 0, "y": 0, "width": 100, "height": 50}],
            "edges": [],
        }

        baseline_file = tmp_path / "baseline.canvas"
        baseline_file.write_text(json.dumps(baseline_data))
        current_file = tmp_path / "current.canvas"
        current_file.write_text(json.dumps(current_data))

        result = load_canvas_for_llm(current_file, baseline_file, use_toon=False)
        parsed = json.loads(result)

        assert "changes" in parsed

    def test_load_without_toon(self, tmp_path):
        canvas_data = {"nodes": [], "edges": []}
        canvas_file = tmp_path / "test.canvas"
        canvas_file.write_text(json.dumps(canvas_data))

        result = load_canvas_for_llm(canvas_file, use_toon=False)

        # Should be compact JSON
        assert "{" in result
        assert "  " not in result  # No indentation


class TestApplyConstraintsToPlan:
    """Tests for apply_constraints_to_plan function."""

    def test_remove_steps_for_removed_nodes(self):
        plan = {
            "steps": [
                {"id": "keep", "description": "Keep this"},
                {"id": "remove", "description": "Remove this"},
            ],
        }
        constraints = CanvasConstraints(removed_nodes=["remove"])

        result = apply_constraints_to_plan(plan, constraints)

        assert len(result["steps"]) == 1
        assert result["steps"][0]["id"] == "keep"

    def test_add_priority_boost(self):
        plan = {
            "steps": [{"id": "n1", "description": "Step 1"}],
        }
        constraints = CanvasConstraints(position_weights={"n1": 0.9})

        result = apply_constraints_to_plan(plan, constraints)

        assert result["steps"][0]["priority_boost"] == 0.9

    def test_add_canvas_requirements(self):
        plan = {"steps": []}
        constraints = CanvasConstraints(added_requirements=["New feature"])

        result = apply_constraints_to_plan(plan, constraints)

        assert len(result["steps"]) == 1
        assert result["steps"][0]["source"] == "canvas_edit"
        assert "New feature" in result["steps"][0]["description"]


class TestValidateCanvasSchema:
    """Tests for validate_canvas_schema function."""

    def test_valid_canvas(self):
        canvas = {
            "nodes": [
                {"id": "n1", "type": "text", "x": 0, "y": 0, "width": 100, "height": 50}
            ],
            "edges": [
                {"id": "e1", "fromNode": "n1", "toNode": "n1"}
            ],
        }
        errors = validate_canvas_schema(canvas)
        assert errors == []

    def test_missing_node_id(self):
        canvas = {
            "nodes": [{"type": "text", "x": 0, "y": 0, "width": 100, "height": 50}],
            "edges": [],
        }
        errors = validate_canvas_schema(canvas)
        assert any("missing 'id'" in e for e in errors)

    def test_missing_node_type(self):
        canvas = {
            "nodes": [{"id": "n1", "x": 0, "y": 0, "width": 100, "height": 50}],
            "edges": [],
        }
        errors = validate_canvas_schema(canvas)
        assert any("missing 'type'" in e for e in errors)

    def test_missing_node_position(self):
        canvas = {
            "nodes": [{"id": "n1", "type": "text", "width": 100, "height": 50}],
            "edges": [],
        }
        errors = validate_canvas_schema(canvas)
        assert any("missing 'x'" in e for e in errors)
        assert any("missing 'y'" in e for e in errors)

    def test_duplicate_node_id(self):
        canvas = {
            "nodes": [
                {"id": "n1", "type": "text", "x": 0, "y": 0, "width": 100, "height": 50},
                {"id": "n1", "type": "text", "x": 100, "y": 0, "width": 100, "height": 50},
            ],
            "edges": [],
        }
        errors = validate_canvas_schema(canvas)
        assert any("duplicate id" in e for e in errors)

    def test_edge_references_nonexistent_node(self):
        canvas = {
            "nodes": [{"id": "n1", "type": "text", "x": 0, "y": 0, "width": 100, "height": 50}],
            "edges": [{"id": "e1", "fromNode": "n1", "toNode": "nonexistent"}],
        }
        errors = validate_canvas_schema(canvas)
        assert any("not found in nodes" in e for e in errors)

    def test_not_a_dict(self):
        errors = validate_canvas_schema("not a dict")
        assert errors == ["Canvas must be a dict"]
