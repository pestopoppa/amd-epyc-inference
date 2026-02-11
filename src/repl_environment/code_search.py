"""Code and document search via NextPLAID multi-vector retrieval.

Provides mixin with: code_search, doc_search.

These tools complement recall() (episodic memory) by searching actual source
code and documentation using token-level ColBERT matching — finding specific
function names, class definitions, and code patterns rather than past routing
decisions.

Requires: NextPLAID server on localhost:8088, pip install next-plaid-client.
Degrades gracefully when unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.repl_environment.types import wrap_tool_output

logger = logging.getLogger(__name__)

NEXTPLAID_URL = "http://localhost:8088"
VALID_INDICES = frozenset({"code", "docs"})


class _CodeSearchMixin:
    """Mixin providing multi-vector code/doc search tools.

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig
        artifacts: dict
        _exploration_calls: int
        _exploration_log: ExplorationLog
        _research_context: ResearchContext
        _last_research_node: str | None
    """

    _nextplaid_client: Any = None  # Lazy-loaded

    def _get_nextplaid_client(self) -> Any:
        """Lazy-load the NextPLAID client. Returns None if unavailable."""
        if self._nextplaid_client is None:
            try:
                from next_plaid_client import NextPlaidClient

                client = NextPlaidClient(NEXTPLAID_URL)
                # Lightweight health probe — fail fast if server is down
                health = client.health()
                if health.status != "healthy":
                    logger.warning("NextPLAID unhealthy: %s", health.status)
                    return None
                self._nextplaid_client = client
            except ImportError:
                logger.debug("next-plaid-client not installed")
                return None
            except Exception as e:
                logger.debug("NextPLAID unavailable: %s", e)
                return None
        return self._nextplaid_client

    def _code_search(self, query: str, limit: int = 5) -> str:
        """Search project source code for relevant passages.

        Uses multi-vector (ColBERT) retrieval with token-level matching.
        Finds specific function names, class definitions, and code patterns —
        not just semantic similarity.

        Unlike recall() which searches episodic memories (past routing decisions),
        code_search() finds actual source code in the project.

        Args:
            query: Natural language or code pattern to search for.
                   e.g., "escalation policy implementation",
                         "def embed_task_ir", "FAISS index configuration"
            limit: Maximum results to return (default 5).

        Returns:
            JSON with matching code passages, file paths, and line ranges.
        """
        return self._nextplaid_search(query, index="code", limit=limit)

    def _doc_search(self, query: str, limit: int = 5) -> str:
        """Search project documentation for relevant sections.

        Searches markdown docs, handoffs, model registry, and config files.
        For source code, use code_search() instead.

        Args:
            query: What to look for in documentation.
            limit: Maximum results (default 5).

        Returns:
            JSON with matching doc passages and metadata.
        """
        return self._nextplaid_search(query, index="docs", limit=limit)

    def _nextplaid_search(self, query: str, index: str, limit: int) -> str:
        """Internal: execute search against a NextPLAID index."""
        self._exploration_calls += 1

        if index not in VALID_INDICES:
            output = json.dumps(
                {"results": [], "error": f"Invalid index '{index}'. Valid: {sorted(VALID_INDICES)}"}
            )
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

        client = self._get_nextplaid_client()
        if client is None:
            output = json.dumps(
                {"results": [], "error": "NextPLAID not available"}
            )
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

        try:
            from next_plaid_client.models import SearchParams

            params = SearchParams(top_k=min(limit, 20))
            result = client.search_with_encoding(index, queries=[query], params=params)

            results = []
            if result.results:
                qr = result.results[0]  # Single query → first QueryResult
                for doc_id, score, meta in zip(qr.document_ids, qr.scores, qr.metadata):
                    if len(results) >= limit:
                        break
                    results.append({
                        "file": meta.get("file", "unknown"),
                        "lines": f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
                        "score": round(float(score), 3),
                    })

            response = {"results": results, "index": index, "query": query}

            self._exploration_log.add_event(
                "code_search" if index == "code" else "doc_search",
                {"query": query, "index": index},
                response,
            )

            # Track in research context
            tool_name = "code_search" if index == "code" else "doc_search"
            node_id = self._research_context.add(
                tool=tool_name,
                query=query[:100],
                content=json.dumps(results[:3]),
                parent_id=self._last_research_node,
            )
            self._last_research_node = node_id

            output = json.dumps(response, indent=2)
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

        except Exception as e:
            logger.warning("NextPLAID search failed: %s", e)
            output = json.dumps({"results": [], "error": str(e)})
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)
