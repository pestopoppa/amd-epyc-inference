"""Research Context Tracker for REPL tool results.

Assigns IDs to tool invocations, tracks parent-child relationships,
detects cross-references (string + semantic), and renders structured
state for model context.

Example:
    ctx = ResearchContext()
    node_id = ctx.add(tool="grep", query="error", content="Found 3 matches...")
    child_id = ctx.add(tool="peek", query="line 42", content="...", parent_id=node_id)
    print(ctx.render())  # Tree view of research state
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Tool prefixes for ID generation
TOOL_PREFIXES = {
    "grep": "G",
    "peek": "P",
    "llm_call": "L",
    "llm_batch": "B",
    "TOOL": "T",
    "CALL": "C",
    "web_fetch": "W",
    "recall": "R",
    "list_dir": "D",
    "file_info": "F",
    "ocr_document": "O",
    "analyze_figure": "A",
    "run_shell": "S",
    "code_search": "CS",
    "doc_search": "DS",
}

DEFAULT_PREFIX = "X"


@dataclass
class ResearchNode:
    """A single research node representing one tool invocation."""

    id: str
    tool: str
    query: str  # Tool arguments summary
    summary: str  # Content summary (first ~200 chars)
    size: int  # Content size in chars
    parent_id: Optional[str] = None
    refs: list[str] = field(default_factory=list)  # Cross-references to other nodes
    status: str = "unanalyzed"  # unanalyzed, analyzed, stale
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None  # For semantic reference detection


@dataclass
class ResearchContext:
    """
    Lightweight tracker for REPL tool results.

    Features:
    - Assigns unique IDs to each tool invocation (G1, P2, L3, etc.)
    - Tracks parent-child relationships for research lineage
    - Detects cross-references via string matching (e.g., "see G1")
    - Optional semantic reference detection via embeddings
    - Renders tree view for model context injection

    Args:
        use_semantic: Whether to use semantic similarity for cross-refs.
        embedder: Optional ParallelEmbedderClient instance. If None and
            use_semantic=True, will attempt lazy initialization.
        semantic_threshold: Similarity threshold for semantic refs (0.0-1.0).
    """

    use_semantic: bool = True
    embedder: Any = None  # ParallelEmbedderClient or None
    semantic_threshold: float = 0.7

    # Internal state
    nodes: dict[str, ResearchNode] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)
    _embedder_initialized: bool = field(default=False, repr=False)
    _embedder_failed: bool = field(default=False, repr=False)

    def _get_next_id(self, tool: str) -> str:
        """Generate the next ID for a tool type."""
        prefix = TOOL_PREFIXES.get(tool, DEFAULT_PREFIX)
        count = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = count
        return f"{prefix}{count}"

    def _extract_summary(self, content: str, max_len: int = 200) -> str:
        """Extract a summary from content (first N chars, clean)."""
        if not content:
            return "[empty]"

        # Clean up whitespace
        clean = " ".join(content.split())

        if len(clean) <= max_len:
            return clean

        # Truncate at word boundary
        truncated = clean[:max_len]
        last_space = truncated.rfind(" ")
        if last_space > max_len // 2:
            truncated = truncated[:last_space]

        return truncated + "..."

    def _find_string_refs(self, content: str) -> list[str]:
        """Find explicit references to other nodes via string patterns.

        Matches patterns like: G1, P2, "see G1", "from L3", etc.
        """
        refs = []
        # Build pattern from known node IDs
        if not self.nodes:
            return refs

        node_ids = list(self.nodes.keys())
        # Pattern: word boundary + node ID
        for node_id in node_ids:
            pattern = rf"\b{re.escape(node_id)}\b"
            if re.search(pattern, content):
                refs.append(node_id)

        return refs

    def _get_embedder(self) -> Any:
        """Lazy-initialize the embedder if needed."""
        if self.embedder is not None:
            return self.embedder

        if self._embedder_failed or not self.use_semantic:
            return None

        if self._embedder_initialized:
            return self.embedder

        self._embedder_initialized = True

        try:
            from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

            self.embedder = ParallelEmbedderClient()
            return self.embedder
        except Exception as e:
            logger.debug("Failed to initialize embedder: %s", e)
            self._embedder_failed = True
            return None

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, returning None on failure."""
        embedder = self._get_embedder()
        if embedder is None:
            return None

        try:
            return embedder.embed_sync(text)
        except Exception as e:
            logger.debug("Embedding failed: %s", e)
            return None

    def _find_semantic_refs(
        self, embedding: Optional[np.ndarray], exclude_id: Optional[str] = None
    ) -> list[str]:
        """Find semantically similar nodes via embedding similarity."""
        if embedding is None or not self.nodes:
            return []

        refs = []
        for node_id, node in self.nodes.items():
            if node_id == exclude_id:
                continue
            if node.embedding is None:
                continue

            # Cosine similarity (embeddings are normalized)
            similarity = float(np.dot(embedding, node.embedding))
            if similarity >= self.semantic_threshold:
                refs.append(node_id)

        return refs

    def add(
        self,
        tool: str,
        query: str,
        content: str,
        parent_id: Optional[str] = None,
    ) -> str:
        """Add a new research node.

        Args:
            tool: Tool name (e.g., "grep", "peek", "TOOL")
            query: Tool arguments/query string
            content: Tool output content
            parent_id: Optional parent node ID for lineage tracking

        Returns:
            Generated node ID (e.g., "G1", "P2")
        """
        node_id = self._get_next_id(tool)

        # Generate embedding if semantic refs enabled
        embedding = None
        if self.use_semantic:
            # Combine query and content preview for embedding
            embed_text = f"{query} {content[:1000]}"
            embedding = self._get_embedding(embed_text)

        # Find cross-references
        refs = self._find_string_refs(content)

        # Find semantic refs (only if embedding succeeded)
        if embedding is not None:
            semantic_refs = self._find_semantic_refs(embedding, exclude_id=node_id)
            # Merge, avoiding duplicates
            for ref in semantic_refs:
                if ref not in refs:
                    refs.append(ref)

        node = ResearchNode(
            id=node_id,
            tool=tool,
            query=query[:100],  # Truncate long queries
            summary=self._extract_summary(content),
            size=len(content),
            parent_id=parent_id,
            refs=refs,
            embedding=embedding,
        )

        self.nodes[node_id] = node
        return node_id

    def mark_analyzed(self, node_id: str) -> bool:
        """Mark a node as analyzed.

        Args:
            node_id: Node ID to mark

        Returns:
            True if node exists and was marked, False otherwise
        """
        if node_id not in self.nodes:
            return False
        self.nodes[node_id].status = "analyzed"
        return True

    def mark_stale(self, node_id: str) -> bool:
        """Mark a node as stale (outdated).

        Args:
            node_id: Node ID to mark

        Returns:
            True if node exists and was marked, False otherwise
        """
        if node_id not in self.nodes:
            return False
        self.nodes[node_id].status = "stale"
        return True

    def get_unanalyzed(self) -> list[str]:
        """Get list of unanalyzed node IDs.

        Returns:
            List of node IDs with status='unanalyzed'
        """
        return [nid for nid, node in self.nodes.items() if node.status == "unanalyzed"]

    def get_node(self, node_id: str) -> Optional[ResearchNode]:
        """Get a node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            ResearchNode or None if not found
        """
        return self.nodes.get(node_id)

    def get_lineage(self, node_id: str) -> list[str]:
        """Get the lineage (ancestor chain) for a node.

        Args:
            node_id: Node ID to trace

        Returns:
            List of node IDs from root to the given node
        """
        lineage = []
        current_id = node_id
        seen = set()

        while current_id and current_id not in seen:
            seen.add(current_id)
            node = self.nodes.get(current_id)
            if node is None:
                break
            lineage.append(current_id)
            current_id = node.parent_id

        lineage.reverse()
        return lineage

    def render(self) -> str:
        """Render the research context as a tree structure.

        Returns:
            Formatted string showing nodes, relationships, and status
        """
        if not self.nodes:
            return "[No research nodes]"

        lines = ["## Research Context"]

        # Count by status
        unanalyzed = sum(1 for n in self.nodes.values() if n.status == "unanalyzed")
        analyzed = sum(1 for n in self.nodes.values() if n.status == "analyzed")
        stale = sum(1 for n in self.nodes.values() if n.status == "stale")

        lines.append(f"Progress: {analyzed} analyzed, {unanalyzed} pending, {stale} stale")
        lines.append("")

        # Build tree structure
        # Find root nodes (no parent)
        roots = [nid for nid, node in self.nodes.items() if node.parent_id is None]

        # Find children for each node
        children: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            if node.parent_id and node.parent_id in children:
                children[node.parent_id].append(nid)

        def render_node(node_id: str, indent: int = 0) -> list[str]:
            """Render a node and its children recursively."""
            node = self.nodes[node_id]
            prefix = "  " * indent

            # Status indicator
            status_mark = {"unanalyzed": "?", "analyzed": "+", "stale": "~"}.get(node.status, " ")

            # Main line
            line = f"{prefix}[{status_mark}] {node_id}: {node.tool}({node.query})"
            result = [line]

            # Summary (indented)
            result.append(f"{prefix}    -> {node.summary}")

            # References (if any)
            if node.refs:
                refs_str = ", ".join(node.refs)
                result.append(f"{prefix}    refs: {refs_str}")

            # Children
            for child_id in children.get(node_id, []):
                result.extend(render_node(child_id, indent + 1))

            return result

        # Render from roots
        for root_id in roots:
            lines.extend(render_node(root_id))

        # Render orphans (parent_id set but parent not found)
        orphans = [
            nid
            for nid, node in self.nodes.items()
            if node.parent_id and node.parent_id not in self.nodes and nid not in roots
        ]
        if orphans:
            lines.append("")
            lines.append("(Orphaned nodes)")
            for orphan_id in orphans:
                lines.extend(render_node(orphan_id))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the research context to a dictionary.

        Returns:
            JSON-serializable dict for persistence
        """
        nodes_data = {}
        for nid, node in self.nodes.items():
            nodes_data[nid] = {
                "id": node.id,
                "tool": node.tool,
                "query": node.query,
                "summary": node.summary,
                "size": node.size,
                "parent_id": node.parent_id,
                "refs": node.refs,
                "status": node.status,
                "timestamp": node.timestamp,
                # Note: embedding is not serialized (can be regenerated)
            }

        return {
            "version": 1,
            "nodes": nodes_data,
            "counters": self._counters,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        use_semantic: bool = True,
        embedder: Any = None,
    ) -> "ResearchContext":
        """Deserialize a research context from a dictionary.

        Args:
            data: Dict from to_dict()
            use_semantic: Whether to enable semantic refs
            embedder: Optional embedder instance

        Returns:
            ResearchContext instance
        """
        if data.get("version") != 1:
            raise ValueError(f"Unsupported version: {data.get('version')}")

        ctx = cls(use_semantic=use_semantic, embedder=embedder)
        ctx._counters = data.get("counters", {})

        for nid, node_data in data.get("nodes", {}).items():
            node = ResearchNode(
                id=node_data["id"],
                tool=node_data["tool"],
                query=node_data["query"],
                summary=node_data["summary"],
                size=node_data["size"],
                parent_id=node_data.get("parent_id"),
                refs=node_data.get("refs", []),
                status=node_data.get("status", "unanalyzed"),
                timestamp=node_data.get("timestamp", time.time()),
                embedding=None,  # Not serialized
            )
            ctx.nodes[nid] = node

        return ctx

    def clear(self) -> int:
        """Clear all nodes and reset counters.

        Returns:
            Number of nodes cleared
        """
        count = len(self.nodes)
        self.nodes.clear()
        self._counters.clear()
        return count
