"""Unit tests for NextPLAID code_search / doc_search REPL tools.

Phase 4: Tests cover dual-client routing (code→:8088, docs→:8089),
fallback behavior (docs container down → code container), and
both-down graceful degradation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal fakes for the NextPLAID client models
# ---------------------------------------------------------------------------

@dataclass
class FakeQueryResult:
    query_id: int
    document_ids: list[int]
    scores: list[float]
    metadata: list[dict[str, Any]]


@dataclass
class FakeSearchResult:
    results: list[FakeQueryResult]
    num_queries: int


@dataclass
class FakeHealthResponse:
    status: str = "healthy"
    version: str = "1.0.4"
    loaded_indices: int = 2
    index_dir: str = "/data/indices"
    memory_usage_bytes: int = 100_000_000
    indices: list[str] | None = None


class FakeNextPlaidClient:
    """Mock NextPLAID client returning deterministic results."""

    def __init__(self, url: str):
        self.url = url
        self._health_calls = 0
        self._search_calls: list[tuple[str, list[str]]] = []

    def health(self) -> FakeHealthResponse:
        self._health_calls += 1
        return FakeHealthResponse()

    def search_with_encoding(self, index, queries, params=None):
        self._search_calls.append((index, queries))
        return FakeSearchResult(
            results=[
                FakeQueryResult(
                    query_id=0,
                    document_ids=[42, 7],
                    scores=[8.5, 5.2],
                    metadata=[
                        {"file": "src/escalation.py", "start_line": "10", "end_line": "25"},
                        {"file": "tests/test_escalation.py", "start_line": "1", "end_line": "15"},
                    ],
                )
            ],
            num_queries=1,
        )


class FakeUnhealthyClient:
    """Mock NextPLAID client that reports unhealthy."""

    def __init__(self, url: str):
        self.url = url

    def health(self) -> FakeHealthResponse:
        return FakeHealthResponse(status="unhealthy")


class FakeConnectionErrorClient:
    """Mock NextPLAID client that raises on health check."""

    def __init__(self, url: str):
        self.url = url

    def health(self):
        raise ConnectionError(f"Cannot reach {self.url}")


# ---------------------------------------------------------------------------
# Fixture: build a minimal REPLEnvironment with code_search wired up
# ---------------------------------------------------------------------------

@pytest.fixture
def repl():
    """Create a REPLEnvironment with mocked NextPLAID client."""
    from src.repl_environment.environment import REPLEnvironment

    env = REPLEnvironment(context="test context", role="frontdoor")
    # Reset any cached clients from previous tests
    env._code_client = None
    env._docs_client = None
    return env


# ---------------------------------------------------------------------------
# Tests: Basic Functionality
# ---------------------------------------------------------------------------

class TestCodeSearchMixin:
    """Tests for _CodeSearchMixin integrated via REPLEnvironment."""

    def test_code_search_registered_in_globals(self, repl):
        """code_search and doc_search are available as sandbox globals."""
        assert "code_search" in repl._globals
        assert "doc_search" in repl._globals

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_code_search_happy_path(self, mock_get_client, repl):
        """code_search returns structured results from NextPLAID."""
        fake_client = FakeNextPlaidClient("http://localhost:8088")
        mock_get_client.return_value = fake_client

        raw = repl._code_search("escalation policy")
        # Strip tool output wrapper if present
        output = raw.replace("<<<TOOL_OUTPUT>>>", "").replace("<<<END_TOOL_OUTPUT>>>", "").strip()
        result = json.loads(output)

        assert result["index"] == "code"
        assert result["query"] == "escalation policy"
        assert len(result["results"]) == 2
        assert result["results"][0]["file"] == "src/escalation.py"
        assert result["results"][0]["lines"] == "10-25"
        assert result["results"][0]["score"] == 8.5

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_doc_search_uses_docs_index(self, mock_get_client, repl):
        """doc_search routes to the 'docs' index."""
        fake_client = FakeNextPlaidClient("http://localhost:8089")
        mock_get_client.return_value = fake_client

        repl._doc_search("benchmark results")

        assert len(fake_client._search_calls) == 1
        assert fake_client._search_calls[0][0] == "docs"

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_code_search_respects_limit(self, mock_get_client, repl):
        """Limit parameter caps number of results."""
        fake_client = FakeNextPlaidClient("http://localhost:8088")
        mock_get_client.return_value = fake_client

        raw = repl._code_search("test", limit=1)
        output = raw.replace("<<<TOOL_OUTPUT>>>", "").replace("<<<END_TOOL_OUTPUT>>>", "").strip()
        result = json.loads(output)

        assert len(result["results"]) == 1

    def test_code_search_unavailable_graceful(self, repl):
        """When NextPLAID is down, returns empty results without error."""
        repl._code_client = None
        repl._docs_client = None

        with patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client", return_value=None):
            raw = repl._code_search("anything")
            output = raw.replace("<<<TOOL_OUTPUT>>>", "").replace("<<<END_TOOL_OUTPUT>>>", "").strip()
            result = json.loads(output)

            assert result["results"] == []
            assert "error" in result

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_code_search_invalid_index(self, mock_get_client, repl):
        """Invalid index name returns error."""
        raw = repl._nextplaid_search("test", index="invalid", limit=5)
        output = raw.replace("<<<TOOL_OUTPUT>>>", "").replace("<<<END_TOOL_OUTPUT>>>", "").strip()
        result = json.loads(output)

        assert result["results"] == []
        assert "Invalid index" in result["error"]

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_code_search_increments_exploration_calls(self, mock_get_client, repl):
        """Each search increments the exploration counter."""
        fake_client = FakeNextPlaidClient("http://localhost:8088")
        mock_get_client.return_value = fake_client

        before = repl._exploration_calls
        repl._code_search("test query")
        assert repl._exploration_calls == before + 1

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_code_search_tracks_tool_output(self, mock_get_client, repl):
        """Results are appended to artifacts._tool_outputs."""
        fake_client = FakeNextPlaidClient("http://localhost:8088")
        mock_get_client.return_value = fake_client

        repl._code_search("test")
        outputs = repl.artifacts.get("_tool_outputs", [])
        assert len(outputs) >= 1

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_code_search_tracks_research_context(self, mock_get_client, repl):
        """Search results create research context nodes."""
        fake_client = FakeNextPlaidClient("http://localhost:8088")
        mock_get_client.return_value = fake_client

        repl._code_search("escalation logic")
        assert repl._last_research_node is not None
        assert repl._last_research_node.startswith("CS")

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_doc_search_tracks_research_context_ds_prefix(self, mock_get_client, repl):
        """doc_search nodes use DS prefix in research context."""
        fake_client = FakeNextPlaidClient("http://localhost:8089")
        mock_get_client.return_value = fake_client

        repl._doc_search("model quirks")
        assert repl._last_research_node is not None
        assert repl._last_research_node.startswith("DS")

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_search_exception_returns_error(self, mock_get_client, repl):
        """If search raises, error is captured gracefully."""
        mock_client = MagicMock()
        mock_client.search_with_encoding.side_effect = RuntimeError("connection reset")
        mock_get_client.return_value = mock_client

        raw = repl._code_search("crash test")
        output = raw.replace("<<<TOOL_OUTPUT>>>", "").replace("<<<END_TOOL_OUTPUT>>>", "").strip()
        result = json.loads(output)

        assert result["results"] == []
        assert "connection reset" in result["error"]


# ---------------------------------------------------------------------------
# Tests: Dual-Client Routing (Phase 4)
# ---------------------------------------------------------------------------

class TestDualClientRouting:
    """Tests for Phase 4 dual-container routing logic."""

    @patch("src.repl_environment.code_search._CodeSearchMixin._init_nextplaid_client")
    def test_code_search_uses_code_client(self, mock_init, repl):
        """code_search() routes to :8088 code client."""
        code_client = FakeNextPlaidClient("http://localhost:8088")
        mock_init.return_value = code_client

        repl._code_search("def retrieve")

        # _init_nextplaid_client called with code URL
        mock_init.assert_called_with("http://localhost:8088")
        assert len(code_client._search_calls) == 1
        assert code_client._search_calls[0][0] == "code"

    @patch("src.repl_environment.code_search._CodeSearchMixin._init_nextplaid_client")
    def test_doc_search_uses_docs_client(self, mock_init, repl):
        """doc_search() routes to :8089 docs client."""
        docs_client = FakeNextPlaidClient("http://localhost:8089")
        mock_init.return_value = docs_client

        repl._doc_search("escalation policy")

        # _init_nextplaid_client called with docs URL
        mock_init.assert_called_with("http://localhost:8089")
        assert len(docs_client._search_calls) == 1
        assert docs_client._search_calls[0][0] == "docs"

    @patch("src.repl_environment.code_search._CodeSearchMixin._init_nextplaid_client")
    def test_docs_fallback_to_code_client(self, mock_init, repl):
        """When docs container (:8089) is down, doc_search falls back to code container (:8088)."""
        code_client = FakeNextPlaidClient("http://localhost:8088")

        def init_side_effect(url):
            if url == "http://localhost:8089":
                return None  # Docs container down
            return code_client

        mock_init.side_effect = init_side_effect

        raw = repl._doc_search("escalation policy")
        output = raw.replace("<<<TOOL_OUTPUT>>>", "").replace("<<<END_TOOL_OUTPUT>>>", "").strip()
        result = json.loads(output)

        # Should still return results (from code client fallback)
        assert len(result["results"]) == 2
        assert result["index"] == "docs"
        # Code client was used as fallback
        assert len(code_client._search_calls) == 1

    @patch("src.repl_environment.code_search._CodeSearchMixin._init_nextplaid_client")
    def test_both_containers_down_graceful(self, mock_init, repl):
        """When both containers are down, returns graceful error."""
        mock_init.return_value = None  # Both unavailable

        raw = repl._doc_search("anything")
        output = raw.replace("<<<TOOL_OUTPUT>>>", "").replace("<<<END_TOOL_OUTPUT>>>", "").strip()
        result = json.loads(output)

        assert result["results"] == []
        assert "error" in result
        assert "not available" in result["error"]

    @patch("src.repl_environment.code_search._CodeSearchMixin._init_nextplaid_client")
    def test_both_containers_down_code_search(self, mock_init, repl):
        """code_search with code container down returns graceful error."""
        mock_init.return_value = None

        raw = repl._code_search("anything")
        output = raw.replace("<<<TOOL_OUTPUT>>>", "").replace("<<<END_TOOL_OUTPUT>>>", "").strip()
        result = json.loads(output)

        assert result["results"] == []
        assert "not available" in result["error"]

    @patch("src.repl_environment.code_search._CodeSearchMixin._init_nextplaid_client")
    def test_clients_are_cached(self, mock_init, repl):
        """Clients are lazy-loaded and cached — _init only called once per URL."""
        code_client = FakeNextPlaidClient("http://localhost:8088")
        docs_client = FakeNextPlaidClient("http://localhost:8089")

        def init_side_effect(url):
            if url == "http://localhost:8088":
                return code_client
            return docs_client

        mock_init.side_effect = init_side_effect

        # Two code searches → only one init call
        repl._code_search("query1")
        repl._code_search("query2")

        # Two doc searches → only one init call
        repl._doc_search("query3")
        repl._doc_search("query4")

        # _init called twice total: once for code URL, once for docs URL
        assert mock_init.call_count == 2


# ---------------------------------------------------------------------------
# Tests: REPL Sandbox Integration
# ---------------------------------------------------------------------------

class TestCodeSearchRepl:
    """Integration-style: execute code_search via REPL sandbox exec."""

    @patch("src.repl_environment.code_search._CodeSearchMixin._get_nextplaid_client")
    def test_exec_code_search(self, mock_get_client, repl):
        """code_search() is callable from within exec'd sandbox code."""
        fake_client = FakeNextPlaidClient("http://localhost:8088")
        mock_get_client.return_value = fake_client

        result = repl.execute('result = code_search("FAISS index")')
        assert result.error is None, f"REPL exec failed: {result.error}"
