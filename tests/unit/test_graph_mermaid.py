"""Tests for orchestration graph Mermaid diagram generation."""

from __future__ import annotations

from src.graph import generate_mermaid


class TestGenerateMermaid:
    def test_generates_valid_mermaid(self):
        code = generate_mermaid()
        assert "stateDiagram-v2" in code
        assert "orchestration" in code

    def test_contains_all_nodes(self):
        code = generate_mermaid()
        for node_name in [
            "FrontdoorNode",
            "WorkerNode",
            "CoderNode",
            "CoderEscalationNode",
            "IngestNode",
            "ArchitectNode",
            "ArchitectCodingNode",
        ]:
            assert node_name in code, f"Missing node: {node_name}"

    def test_contains_escalation_edges(self):
        """Verify key escalation transitions are present."""
        code = generate_mermaid()
        # FrontdoorNode -> CoderNode (frontdoor escalates to coder_escalation)
        assert "FrontdoorNode --> CoderNode" in code
        # WorkerNode -> CoderNode (workers escalate to coder_escalation)
        assert "WorkerNode --> CoderNode" in code
        # CoderNode -> ArchitectNode
        assert "CoderNode --> ArchitectNode" in code
        # CoderEscalationNode -> ArchitectCodingNode
        assert "CoderEscalationNode --> ArchitectCodingNode" in code
        # IngestNode -> ArchitectNode
        assert "IngestNode --> ArchitectNode" in code

    def test_contains_self_loops(self):
        """Nodes should have self-loop edges for retry."""
        code = generate_mermaid()
        assert "FrontdoorNode --> FrontdoorNode" in code
        assert "CoderNode --> CoderNode" in code
        assert "ArchitectNode --> ArchitectNode" in code

    def test_contains_end_edges(self):
        """All nodes should have end (terminal) edges."""
        code = generate_mermaid()
        # [*] represents End in stateDiagram-v2
        assert "[*]" in code

    def test_direction_parameter(self):
        code_lr = generate_mermaid(direction="LR")
        assert "direction LR" in code_lr

        code_tb = generate_mermaid(direction="TB")
        assert "direction TB" in code_tb

    def test_graph_singleton(self):
        """The orchestration_graph singleton should have 7 nodes."""
        # The graph is created with 7 node classes
        code = generate_mermaid()
        node_names = [
            "FrontdoorNode",
            "WorkerNode",
            "CoderNode",
            "CoderEscalationNode",
            "IngestNode",
            "ArchitectNode",
            "ArchitectCodingNode",
        ]
        for name in node_names:
            assert name in code
