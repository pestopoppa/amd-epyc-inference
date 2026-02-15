"""Corpus retrieval service for prompt-lookup acceleration.

Retrieves code snippets from a pre-built n-gram index and formats them
for injection into LLM prompts. When llama-server uses --lookup (prompt
n-gram matching), having relevant code snippets in the prompt provides
richer n-gram material for draft token proposals.

Usage:
    from src.services.corpus_retrieval import CorpusRetriever, CorpusConfig

    retriever = CorpusRetriever.get_instance()
    snippets = retriever.retrieve("def calculate_loss(predictions, targets)")
    context = retriever.format_for_prompt(snippets)

The service is a singleton — the index is loaded once and shared across
requests. Queries are sub-millisecond after the initial load.

Supports two index formats:
  - v1 (JSON): snippets.json + ngram_index.json (for small corpora)
  - v2 (SQLite): corpus.db (for 100GB+ corpora from The Stack)

Index built by: scripts/corpus/build_index.py (v1) or build_index_v2.py (v2)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class CorpusConfig:
    """Configuration for corpus retrieval."""

    enabled: bool = False
    index_path: str = "/mnt/raid0/llm/cache/corpus/mvp_index"
    max_snippets: int = 3
    max_chars: int = 3000  # ~750 tokens budget
    min_score: float = 0.5
    exact_only: bool = True  # GloVe irrelevant for code


@dataclass
class CodeSnippet:
    """A retrieved code snippet."""

    code: str
    file: str
    start_line: int
    score: float
    hash: str = ""


class CorpusRetriever:
    """Singleton retriever for corpus-augmented prompt stuffing.

    Loads the n-gram index once and provides fast lookup. Gracefully
    degrades if the index is missing or the service is not configured.
    """

    _instance: CorpusRetriever | None = None
    _lock = threading.Lock()

    def __init__(self, config: CorpusConfig | None = None):
        self.config = config or CorpusConfig()
        self._snippets: list[dict[str, Any]] = []
        self._ngram_index: dict[str, list[int]] = {}
        self._db: sqlite3.Connection | None = None
        self._format: str = ""  # "json" or "sqlite"
        self._loaded = False
        self._load_error: str | None = None
        self._ngram_size = 4

    @classmethod
    def get_instance(cls, config: CorpusConfig | None = None) -> CorpusRetriever:
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _ensure_loaded(self) -> bool:
        """Lazy-load the index on first query. Auto-detects v1 (JSON) or v2 (SQLite)."""
        if self._loaded:
            return True
        if self._load_error is not None:
            return False

        index_path = Path(self.config.index_path)
        db_file = index_path / "corpus.db"
        snippets_file = index_path / "snippets.json"
        ngram_file = index_path / "ngram_index.json"
        meta_file = index_path / "meta.json"

        # Prefer SQLite (v2) if available
        if db_file.exists():
            return self._load_sqlite(db_file, meta_file)
        if snippets_file.exists() and ngram_file.exists():
            return self._load_json(snippets_file, ngram_file, meta_file)

        self._load_error = f"Index not found at {index_path}"
        _log.info("Corpus index not found at %s — retrieval disabled", index_path)
        return False

    def _load_sqlite(self, db_file: Path, meta_file: Path) -> bool:
        """Load SQLite-backed v2 index."""
        try:
            t0 = time.perf_counter()
            conn = sqlite3.connect(
                str(db_file),
                check_same_thread=False,
            )
            conn.execute("PRAGMA mmap_size=1073741824")
            conn.execute("PRAGMA query_only=ON")

            # Read metadata
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                self._ngram_size = meta.get("ngram_size", 4)

            num_snippets = conn.execute(
                "SELECT COUNT(*) FROM snippets"
            ).fetchone()[0]

            self._db = conn
            self._format = "sqlite"
            self._loaded = True
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _log.info(
                "Corpus index loaded (SQLite): %d snippets in %.0fms",
                num_snippets, elapsed_ms,
            )
            return True

        except Exception as exc:
            self._load_error = str(exc)
            _log.warning("Failed to load SQLite corpus index: %s", exc)
            return False

    def _load_json(
        self, snippets_file: Path, ngram_file: Path, meta_file: Path,
    ) -> bool:
        """Load JSON-backed v1 index."""
        try:
            t0 = time.perf_counter()

            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                self._ngram_size = meta.get("ngram_size", 4)

            with open(snippets_file) as f:
                self._snippets = json.load(f)

            with open(ngram_file) as f:
                self._ngram_index = json.load(f)

            self._format = "json"
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._loaded = True
            _log.info(
                "Corpus index loaded (JSON): %d snippets, %d n-grams in %.0fms",
                len(self._snippets), len(self._ngram_index), elapsed_ms,
            )
            return True

        except Exception as exc:
            self._load_error = str(exc)
            _log.warning("Failed to load corpus index: %s", exc)
            return False

    def retrieve(self, query: str, max_results: int | None = None) -> list[CodeSnippet]:
        """Retrieve code snippets matching the query.

        Args:
            query: Code-related query text (task description, code identifiers).
            max_results: Override max_snippets from config.

        Returns:
            List of matching CodeSnippets, sorted by relevance score.
        """
        if not self.config.enabled:
            return []
        if not self._ensure_loaded():
            return []

        max_n = max_results or self.config.max_snippets
        query_grams = self._extract_ngrams(query.lower())

        if not query_grams:
            return []

        if self._format == "sqlite":
            return self._retrieve_sqlite(query_grams, max_n)
        return self._retrieve_json(query_grams, max_n)

    def _retrieve_sqlite(
        self, query_grams: list[str], max_n: int,
    ) -> list[CodeSnippet]:
        """Retrieve from SQLite-backed index."""
        assert self._db is not None
        n_query_grams = len(query_grams)

        # Build parameterized query for n-gram matching
        placeholders = ",".join("?" for _ in query_grams)
        rows = self._db.execute(
            f"""
            SELECT snippet_id, COUNT(*) as score
            FROM ngrams
            WHERE gram IN ({placeholders})
            GROUP BY snippet_id
            HAVING CAST(COUNT(*) AS REAL) / ? >= ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (*query_grams, n_query_grams, self.config.min_score, max_n * 2),
        ).fetchall()

        if not rows:
            return []

        # Fetch snippet details
        results: list[CodeSnippet] = []
        seen_hashes: set[str] = set()
        snippet_ids = [r[0] for r in rows]
        scores_map = {r[0]: r[1] / n_query_grams for r in rows}

        id_placeholders = ",".join("?" for _ in snippet_ids)
        snippets = self._db.execute(
            f"SELECT id, code, source, hash FROM snippets WHERE id IN ({id_placeholders})",
            snippet_ids,
        ).fetchall()

        # Index by id for ordered access
        snip_map = {s[0]: s for s in snippets}

        for sid, _ in rows:
            if len(results) >= max_n:
                break
            snip = snip_map.get(sid)
            if not snip:
                continue
            _, code, source, h = snip
            if h and h in seen_hashes:
                continue
            if h:
                seen_hashes.add(h)

            results.append(CodeSnippet(
                code=code,
                file=source or "",
                start_line=0,
                score=scores_map[sid],
                hash=h,
            ))

        return results

    def _retrieve_json(
        self, query_grams: list[str], max_n: int,
    ) -> list[CodeSnippet]:
        """Retrieve from JSON-backed v1 index."""
        # Score snippets by n-gram overlap
        scores: dict[int, float] = {}
        for gram in query_grams:
            snippet_ids = self._ngram_index.get(gram, [])
            for sid in snippet_ids:
                scores[sid] = scores.get(sid, 0) + 1.0

        if not scores:
            return []

        # Normalize scores by query n-gram count
        n_query_grams = len(query_grams)
        for sid in scores:
            scores[sid] /= n_query_grams

        # Filter by min_score and sort
        candidates = [
            (sid, score) for sid, score in scores.items()
            if score >= self.config.min_score
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Build CodeSnippet results, deduplicating near-identical
        results: list[CodeSnippet] = []
        seen_hashes: set[str] = set()

        for sid, score in candidates[:max_n * 2]:  # Over-fetch for dedup
            if len(results) >= max_n:
                break
            snip = self._snippets[sid]
            h = snip.get("hash", "")
            if h and h in seen_hashes:
                continue
            if h:
                seen_hashes.add(h)

            results.append(CodeSnippet(
                code=snip["code"],
                file=snip.get("file", ""),
                start_line=snip.get("start_line", 0),
                score=score,
                hash=h,
            ))

        return results

    def format_for_prompt(self, snippets: list[CodeSnippet]) -> str:
        """Format retrieved snippets for prompt injection.

        Uses <reference_code> delimiters and respects max_chars budget.

        Args:
            snippets: Retrieved code snippets.

        Returns:
            Formatted string for prompt injection, or "" if empty.
        """
        if not snippets:
            return ""

        parts: list[str] = []
        total_chars = 0

        for i, snip in enumerate(snippets):
            code = snip.code
            # Truncate individual snippet if needed
            remaining = self.config.max_chars - total_chars
            if remaining <= 100:
                break
            if len(code) > remaining:
                code = code[:remaining - 50] + "\n# ... truncated"

            source = Path(snip.file).name if snip.file else "unknown"
            part = (
                f"<reference_code source=\"{source}:{snip.start_line}\">\n"
                f"{code}\n"
                f"</reference_code>"
            )
            parts.append(part)
            total_chars += len(part)

        if not parts:
            return ""

        return "\n".join(parts)

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Strip non-alphanumeric chars (except underscore) from a token."""
        return re.sub(r"[^a-z0-9_]", "", token)

    def _extract_ngrams(self, text: str) -> list[str]:
        """Extract word-level n-grams from text with normalized tokens."""
        raw_words = text.split()
        words = [self._normalize_token(w) for w in raw_words]
        words = [w for w in words if w]  # drop empty after normalization
        n = self._ngram_size
        if len(words) < n:
            return []
        return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


# ── Query extraction helpers ──────────────────────────────────────────────

# Patterns for extracting code identifiers
_IDENT_RE = re.compile(
    r"[a-zA-Z_][a-zA-Z0-9_]*"  # Standard identifiers
    r"|[a-z]+(?=[A-Z])"         # camelCase split
)

_INSTRUCTION_PHRASES = frozenset({
    "write", "implement", "create", "build", "make", "add", "fix",
    "refactor", "optimize", "update", "modify", "change", "remove",
    "please", "can", "you", "the", "a", "an", "this", "that",
    "function", "method", "class", "module", "file", "code",
})


def extract_code_query(task_description: str) -> str:
    """Extract code-relevant query terms from a task description.

    Strips instruction phrases, keeps code identifiers (camelCase,
    snake_case, PascalCase) and language keywords.

    Args:
        task_description: The user's task description.

    Returns:
        Cleaned query string for corpus retrieval.
    """
    words = task_description.split()
    kept = []
    for w in words:
        lower = w.lower().strip(".,;:!?()[]{}\"'")
        if lower in _INSTRUCTION_PHRASES:
            continue
        # Keep identifiers and code-like tokens
        if "_" in w or any(c.isupper() for c in w[1:]) or lower.isidentifier():
            kept.append(lower)
    return " ".join(kept) if kept else task_description.lower()[:200]
