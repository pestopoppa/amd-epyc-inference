"""Tests for ResearchContext tracker."""

import numpy as np
import pytest

from src.research_context import ResearchContext


class TestNodeCreation:
    """Test node creation and ID generation."""

    def test_node_creation_basic(self):
        """Test creating a basic node."""
        ctx = ResearchContext(use_semantic=False)
        node_id = ctx.add(tool="grep", query="error", content="Found 3 matches")

        assert node_id == "G1"
        assert node_id in ctx.nodes
        node = ctx.nodes[node_id]
        assert node.tool == "grep"
        assert node.query == "error"
        assert "Found 3 matches" in node.summary

    def test_id_generation_prefixes(self):
        """Test that different tools get different prefixes."""
        ctx = ResearchContext(use_semantic=False)

        grep_id = ctx.add(tool="grep", query="test", content="result")
        peek_id = ctx.add(tool="peek", query="n=500", content="content")
        llm_id = ctx.add(tool="llm_call", query="summarize", content="summary")

        assert grep_id.startswith("G")
        assert peek_id.startswith("P")
        assert llm_id.startswith("L")

    def test_id_incrementing(self):
        """Test that IDs increment within a prefix."""
        ctx = ResearchContext(use_semantic=False)

        id1 = ctx.add(tool="grep", query="a", content="1")
        id2 = ctx.add(tool="grep", query="b", content="2")
        id3 = ctx.add(tool="grep", query="c", content="3")

        assert id1 == "G1"
        assert id2 == "G2"
        assert id3 == "G3"

    def test_unknown_tool_uses_default_prefix(self):
        """Test that unknown tools use X prefix."""
        ctx = ResearchContext(use_semantic=False)
        node_id = ctx.add(tool="custom_tool", query="test", content="result")

        assert node_id.startswith("X")


class TestParentTracking:
    """Test parent-child relationship tracking."""

    def test_parent_id_stored(self):
        """Test that parent_id is stored correctly."""
        ctx = ResearchContext(use_semantic=False)

        parent_id = ctx.add(tool="grep", query="error", content="Found errors")
        child_id = ctx.add(tool="peek", query="line 42", content="Details", parent_id=parent_id)

        assert ctx.nodes[child_id].parent_id == parent_id
        assert ctx.nodes[parent_id].parent_id is None

    def test_lineage_tracking(self):
        """Test getting the full lineage of a node."""
        ctx = ResearchContext(use_semantic=False)

        root_id = ctx.add(tool="list_dir", query="/tmp", content="files")
        mid_id = ctx.add(tool="grep", query="pattern", content="match", parent_id=root_id)
        leaf_id = ctx.add(tool="peek", query="n=100", content="content", parent_id=mid_id)

        lineage = ctx.get_lineage(leaf_id)
        assert lineage == [root_id, mid_id, leaf_id]

    def test_lineage_orphan(self):
        """Test lineage with orphan node (parent not found)."""
        ctx = ResearchContext(use_semantic=False)
        node_id = ctx.add(tool="grep", query="test", content="result", parent_id="nonexistent")

        lineage = ctx.get_lineage(node_id)
        assert lineage == [node_id]


class TestStringCrossReferences:
    """Test string-based cross-reference detection."""

    def test_cross_reference_detection(self):
        """Test that explicit references like 'see G1' are detected."""
        ctx = ResearchContext(use_semantic=False)

        ctx.add(tool="grep", query="error", content="Found errors")  # G1
        ctx.add(tool="peek", query="file", content="See details")  # P1

        # This content references G1 and P1
        ref_id = ctx.add(
            tool="llm_call",
            query="analyze",
            content="Based on G1 and P1, the issue is...",
        )

        refs = ctx.nodes[ref_id].refs
        assert "G1" in refs
        assert "P1" in refs

    def test_no_self_reference(self):
        """Test that a node doesn't reference itself."""
        ctx = ResearchContext(use_semantic=False)

        # Content mentions "G1" but this is the first grep, so G1 is self
        node_id = ctx.add(tool="grep", query="test", content="Testing G1 reference")

        # G1 shouldn't be in refs since it refers to itself
        # Actually, when this node is created, G1 doesn't exist yet as a key
        # So no reference should be found
        assert len(ctx.nodes[node_id].refs) == 0


class TestSemanticCrossReferences:
    """Test semantic similarity-based cross-reference detection."""

    def test_semantic_disabled(self):
        """Test that semantic refs are not computed when disabled."""
        ctx = ResearchContext(use_semantic=False)

        ctx.add(tool="grep", query="database error", content="Connection failed")
        ctx.add(
            tool="llm_call",
            query="analyze",
            content="The database connection issue...",
        )

        # With semantic disabled, only string refs should be detected
        # No explicit reference to G1, so refs should be empty
        assert ctx.nodes["L1"].embedding is None

    def test_graceful_fallback_on_embedder_failure(self):
        """Test graceful fallback when embedder fails."""

        class FailingEmbedder:
            def embed_sync(self, text):
                raise RuntimeError("Embedding failed")

        ctx = ResearchContext(use_semantic=True, embedder=FailingEmbedder())

        # Should not raise, just fall back to string-only refs
        node_id = ctx.add(tool="grep", query="test", content="result")
        assert node_id == "G1"
        assert ctx.nodes[node_id].embedding is None


class MockEmbedder:
    """Mock embedder for testing semantic refs."""

    def __init__(self):
        self.call_count = 0

    def embed_sync(self, text: str) -> np.ndarray:
        self.call_count += 1
        # Return embeddings based on text content
        # Similar texts get similar embeddings
        if "database" in text.lower() or "connection" in text.lower():
            base = np.array([1.0, 0.0, 0.0, 0.0])
        elif "error" in text.lower() or "fail" in text.lower():
            base = np.array([0.9, 0.1, 0.0, 0.0])  # Similar to database
        else:
            base = np.array([0.0, 0.0, 1.0, 0.0])  # Different

        # Normalize
        return base / np.linalg.norm(base)


class TestSemanticRefsWithMock:
    """Test semantic references with a mock embedder."""

    def test_semantic_ref_detected(self):
        """Test that semantic refs are detected with high similarity."""
        ctx = ResearchContext(use_semantic=True, embedder=MockEmbedder())

        ctx.add(tool="grep", query="db", content="database connection failed")  # G1
        node_id = ctx.add(tool="llm_call", query="analyze", content="connection error")

        # L1 should reference G1 due to semantic similarity
        assert "G1" in ctx.nodes[node_id].refs

    def test_semantic_threshold(self):
        """Test that low similarity doesn't create refs."""
        ctx = ResearchContext(use_semantic=True, embedder=MockEmbedder(), semantic_threshold=0.99)

        ctx.add(tool="grep", query="db", content="database info")  # G1
        node_id = ctx.add(tool="peek", query="file", content="unrelated content")

        # With very high threshold, similar but not identical shouldn't match
        assert "G1" not in ctx.nodes[node_id].refs


class TestRenderTree:
    """Test tree rendering."""

    def test_render_empty(self):
        """Test rendering empty context."""
        ctx = ResearchContext(use_semantic=False)
        rendered = ctx.render()
        assert "No research nodes" in rendered

    def test_render_single_node(self):
        """Test rendering a single node."""
        ctx = ResearchContext(use_semantic=False)
        ctx.add(tool="grep", query="error", content="Found 3 errors in log")

        rendered = ctx.render()
        assert "G1" in rendered
        assert "grep" in rendered
        assert "error" in rendered

    def test_render_tree_structure(self):
        """Test that parent-child relationships show indentation."""
        ctx = ResearchContext(use_semantic=False)

        root = ctx.add(tool="list_dir", query="/tmp", content="files listed")
        ctx.add(tool="grep", query="pattern", content="match found", parent_id=root)

        rendered = ctx.render()
        lines = rendered.split("\n")

        # Find the lines with node IDs
        d1_line = next(line for line in lines if "D1" in line)
        g1_line = next(line for line in lines if "G1" in line)

        # G1 should be indented more than D1 (it's a child)
        d1_indent = len(d1_line) - len(d1_line.lstrip())
        g1_indent = len(g1_line) - len(g1_line.lstrip())
        assert g1_indent > d1_indent

    def test_render_progress_line(self):
        """Test that progress line shows analyzed/pending counts."""
        ctx = ResearchContext(use_semantic=False)

        ctx.add(tool="grep", query="a", content="1")
        ctx.add(tool="grep", query="b", content="2")
        ctx.mark_analyzed("G1")

        rendered = ctx.render()
        assert "1 analyzed" in rendered
        assert "1 pending" in rendered

    def test_render_refs(self):
        """Test that references are shown in render."""
        ctx = ResearchContext(use_semantic=False)

        ctx.add(tool="grep", query="error", content="errors found")  # G1
        ctx.add(tool="llm_call", query="analyze", content="Based on G1, we see...")  # L1

        rendered = ctx.render()
        assert "refs: G1" in rendered


class TestStatusManagement:
    """Test node status marking."""

    def test_mark_analyzed(self):
        """Test marking a node as analyzed."""
        ctx = ResearchContext(use_semantic=False)
        node_id = ctx.add(tool="grep", query="test", content="result")

        assert ctx.nodes[node_id].status == "unanalyzed"
        assert ctx.mark_analyzed(node_id)
        assert ctx.nodes[node_id].status == "analyzed"

    def test_mark_stale(self):
        """Test marking a node as stale."""
        ctx = ResearchContext(use_semantic=False)
        node_id = ctx.add(tool="grep", query="test", content="result")

        assert ctx.mark_stale(node_id)
        assert ctx.nodes[node_id].status == "stale"

    def test_mark_nonexistent_node(self):
        """Test marking a nonexistent node returns False."""
        ctx = ResearchContext(use_semantic=False)

        assert not ctx.mark_analyzed("nonexistent")
        assert not ctx.mark_stale("nonexistent")

    def test_get_unanalyzed(self):
        """Test getting list of unanalyzed nodes."""
        ctx = ResearchContext(use_semantic=False)

        ctx.add(tool="grep", query="a", content="1")  # G1
        ctx.add(tool="grep", query="b", content="2")  # G2
        ctx.add(tool="grep", query="c", content="3")  # G3
        ctx.mark_analyzed("G1")
        ctx.mark_stale("G2")

        unanalyzed = ctx.get_unanalyzed()
        assert unanalyzed == ["G3"]


class TestSerialization:
    """Test serialization and deserialization."""

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        ctx = ResearchContext(use_semantic=False)

        ctx.add(tool="grep", query="error", content="Found errors")
        ctx.add(tool="peek", query="n=100", content="Content preview", parent_id="G1")
        ctx.mark_analyzed("G1")

        # Serialize
        data = ctx.to_dict()

        # Deserialize
        restored = ResearchContext.from_dict(data, use_semantic=False)

        # Verify
        assert len(restored.nodes) == 2
        assert "G1" in restored.nodes
        assert "P1" in restored.nodes
        assert restored.nodes["G1"].status == "analyzed"
        assert restored.nodes["P1"].parent_id == "G1"

    def test_serialization_version_check(self):
        """Test that invalid version raises error."""
        with pytest.raises(ValueError, match="Unsupported version"):
            ResearchContext.from_dict({"version": 999})

    def test_counters_preserved(self):
        """Test that ID counters are preserved across roundtrip."""
        ctx = ResearchContext(use_semantic=False)

        ctx.add(tool="grep", query="a", content="1")  # G1
        ctx.add(tool="grep", query="b", content="2")  # G2

        data = ctx.to_dict()
        restored = ResearchContext.from_dict(data, use_semantic=False)

        # Add another grep - should be G3, not G1
        new_id = restored.add(tool="grep", query="c", content="3")
        assert new_id == "G3"


class TestGetNode:
    """Test node retrieval."""

    def test_get_existing_node(self):
        """Test getting an existing node."""
        ctx = ResearchContext(use_semantic=False)
        ctx.add(tool="grep", query="test", content="result")

        node = ctx.get_node("G1")
        assert node is not None
        assert node.tool == "grep"

    def test_get_nonexistent_node(self):
        """Test getting a nonexistent node returns None."""
        ctx = ResearchContext(use_semantic=False)
        assert ctx.get_node("nonexistent") is None


class TestClear:
    """Test clearing the context."""

    def test_clear(self):
        """Test clearing all nodes."""
        ctx = ResearchContext(use_semantic=False)

        ctx.add(tool="grep", query="a", content="1")
        ctx.add(tool="grep", query="b", content="2")

        count = ctx.clear()

        assert count == 2
        assert len(ctx.nodes) == 0
        assert len(ctx._counters) == 0


class TestREPLIntegration:
    """Test integration with REPLEnvironment."""

    def test_repl_has_research_context(self):
        """Test that REPLEnvironment initializes research context."""
        from src.repl_environment.environment import REPLEnvironment

        repl = REPLEnvironment(context="test content")
        assert hasattr(repl, "_research_context")
        assert hasattr(repl, "_last_research_node")
        assert repl._last_research_node is None

    def test_repl_get_state_with_nodes(self):
        """Test that get_state includes research context when >= 3 nodes."""
        from src.repl_environment.environment import REPLEnvironment

        repl = REPLEnvironment(context="test content with keyword pattern here")

        # Add some nodes by using REPL functions
        repl.execute("peek(100)")
        repl.execute("grep('pattern')")
        repl.execute("peek(50)")

        state = repl.get_state()
        # Should include research context since we have 3+ nodes
        assert "Research Context" in state
        assert "Progress:" in state

    def test_repl_get_state_without_enough_nodes(self):
        """Test that get_state doesn't include research context with < 3 nodes."""
        from src.repl_environment.environment import REPLEnvironment

        repl = REPLEnvironment(context="test content")

        # Only one node
        repl.execute("peek(100)")

        state = repl.get_state()
        # Should NOT include research context
        assert "Research Context" not in state

    def test_repl_reset_clears_research_context(self):
        """Test that reset() clears the research context."""
        from src.repl_environment.environment import REPLEnvironment

        repl = REPLEnvironment(context="test content")
        repl.execute("peek(100)")
        repl.execute("peek(50)")

        assert len(repl._research_context.nodes) == 2

        repl.reset()

        assert len(repl._research_context.nodes) == 0
        assert repl._last_research_node is None

    def test_repl_checkpoint_includes_research_context(self):
        """Test that checkpoint includes research context."""
        from src.repl_environment.environment import REPLEnvironment

        repl = REPLEnvironment(context="test content")
        repl.execute("peek(100)")

        checkpoint = repl.checkpoint()

        assert "research_context" in checkpoint
        assert checkpoint["research_context"]["version"] == 1
        assert "P1" in checkpoint["research_context"]["nodes"]


class TestSummaryExtraction:
    """Test content summary extraction."""

    def test_short_content_unchanged(self):
        """Test that short content is kept as-is."""
        ctx = ResearchContext(use_semantic=False)
        ctx.add(tool="grep", query="test", content="Short content")

        assert ctx.nodes["G1"].summary == "Short content"

    def test_long_content_truncated(self):
        """Test that long content is truncated."""
        ctx = ResearchContext(use_semantic=False)
        long_content = "A" * 500

        ctx.add(tool="grep", query="test", content=long_content)

        summary = ctx.nodes["G1"].summary
        assert len(summary) < 210  # ~200 + "..."
        assert summary.endswith("...")

    def test_empty_content(self):
        """Test handling of empty content."""
        ctx = ResearchContext(use_semantic=False)
        ctx.add(tool="grep", query="test", content="")

        assert ctx.nodes["G1"].summary == "[empty]"
