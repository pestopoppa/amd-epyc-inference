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

Supports three index formats:
  - v1 (JSON): snippets.json + ngram_index.json (for small corpora)
  - v2 (SQLite): corpus.db (for 100GB+ corpora from The Stack)
  - v3 (Sharded SQLite): snippets.db + shard_{00..15}.db (parallel build)

Index built by: scripts/corpus/build_index.py (v1), build_index_v2.py (v2),
or build_index_v3.py (v3 sharded)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections import defaultdict
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
    # Quality RAG settings (Phase 2B — active instruction to use snippets)
    rag_enabled: bool = False
    rag_max_snippets: int = 5
    rag_max_chars: int = 5000  # ~1250 tokens budget
    rag_min_score: float = 0.3
    rag_roles: list[str] | None = None  # Roles eligible for RAG injection


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
        self._word_index: dict[str, set[int]] | None = None  # keyword fallback
        self._db: sqlite3.Connection | None = None
        self._shard_conns: list[sqlite3.Connection] = []
        self._snippets_db: sqlite3.Connection | None = None
        self._num_shards: int = 0
        self._format: str = ""  # "json", "sqlite", or "sharded_sqlite"
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
        """Lazy-load the index on first query. Auto-detects v3 (sharded) → v2 (SQLite) → v1 (JSON)."""
        if self._loaded:
            return True
        if self._load_error is not None:
            return False

        index_path = Path(self.config.index_path)
        shard_file = index_path / "shard_00.db"
        db_file = index_path / "corpus.db"
        snippets_file = index_path / "snippets.json"
        ngram_file = index_path / "ngram_index.json"
        meta_file = index_path / "meta.json"

        # Prefer v3 sharded if shard_00.db exists
        if shard_file.exists():
            return self._load_sharded_sqlite(index_path, meta_file)
        # Fall back to v2 monolithic SQLite
        if db_file.exists():
            return self._load_sqlite(db_file, meta_file)
        # Fall back to v1 JSON
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

    def _load_sharded_sqlite(self, index_path: Path, meta_file: Path) -> bool:
        """Load v3 sharded SQLite index (snippets.db + shard_{00..N}.db).

        Tolerates in-progress builds: missing shards are stored as None
        in _shard_conns and skipped during queries.
        """
        try:
            t0 = time.perf_counter()

            # Read metadata to get shard count
            num_shards = 16  # default
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                num_shards = meta.get("num_shards", 16)
                self._ngram_size = meta.get("ngram_size", 4)

            # Open snippets DB
            snippets_db_path = index_path / "snippets.db"
            if not snippets_db_path.exists():
                self._load_error = f"snippets.db not found at {index_path}"
                _log.warning("Sharded index missing snippets.db at %s", index_path)
                return False

            self._snippets_db = sqlite3.connect(
                str(snippets_db_path), check_same_thread=False,
            )
            self._snippets_db.execute("PRAGMA mmap_size=1073741824")

            # Open shard connections — tolerate missing/in-progress shards
            self._shard_conns = []
            available_shards = 0
            for i in range(num_shards):
                shard_path = index_path / f"shard_{i:02d}.db"
                if not shard_path.exists():
                    self._shard_conns.append(None)
                    continue
                try:
                    conn = sqlite3.connect(str(shard_path), check_same_thread=False)
                    conn.execute("PRAGMA mmap_size=1073741824")
                    self._shard_conns.append(conn)
                    available_shards += 1
                except Exception:
                    self._shard_conns.append(None)

            if available_shards == 0:
                self._load_error = "No accessible shards"
                _log.warning("No shards accessible at %s", index_path)
                return False

            self._num_shards = num_shards

            try:
                num_snippets = self._snippets_db.execute(
                    "SELECT COUNT(*) FROM snippets"
                ).fetchone()[0]
            except Exception:
                num_snippets = 0  # Table may not exist yet during early build

            self._format = "sharded_sqlite"
            self._loaded = True
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _log.info(
                "Corpus index loaded (sharded SQLite): %d snippets, %d/%d shards in %.0fms",
                num_snippets, available_shards, num_shards, elapsed_ms,
            )
            return True

        except Exception as exc:
            self._load_error = str(exc)
            _log.warning("Failed to load sharded corpus index: %s", exc)
            return False

    @staticmethod
    def _gram_to_shard(gram: str, num_shards: int) -> int:
        """Deterministic shard routing via MD5 hash."""
        h = hashlib.md5(gram.encode()).digest()
        return int.from_bytes(h[:4], "little") % num_shards

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

    def retrieve(
        self,
        query: str,
        max_results: int | None = None,
        min_score_override: float | None = None,
    ) -> list[CodeSnippet]:
        """Retrieve code snippets matching the query.

        Uses n-gram matching first. Falls back to keyword overlap when
        n-gram matching returns nothing (common for natural language queries).

        Args:
            query: Code-related query text (task description, code identifiers).
            max_results: Override max_snippets from config.
            min_score_override: Override min_score from config (used by RAG).

        Returns:
            List of matching CodeSnippets, sorted by relevance score.
        """
        if not self.config.enabled:
            return []
        if not self._ensure_loaded():
            return []

        max_n = max_results or self.config.max_snippets
        min_score = min_score_override if min_score_override is not None else self.config.min_score
        query_lower = query.lower()
        query_grams = self._extract_ngrams(query_lower)

        results: list[CodeSnippet] = []
        if query_grams:
            if self._format == "sharded_sqlite":
                results = self._retrieve_sharded_sqlite(query_grams, max_n, min_score)
            elif self._format == "sqlite":
                results = self._retrieve_sqlite(query_grams, max_n, min_score)
            else:
                results = self._retrieve_json(query_grams, max_n, min_score)

        # Keyword fallback when n-gram matching returns nothing
        if not results and self._format == "json":
            keywords = self._extract_keywords(query_lower)
            if keywords:
                results = self._retrieve_keywords_json(keywords, max_n, min_score)

        return results

    def _retrieve_sqlite(
        self, query_grams: list[str], max_n: int, min_score: float | None = None,
    ) -> list[CodeSnippet]:
        """Retrieve from SQLite-backed index."""
        assert self._db is not None
        n_query_grams = len(query_grams)
        threshold = min_score if min_score is not None else self.config.min_score

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
            (*query_grams, n_query_grams, threshold, max_n * 2),
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

    def _retrieve_sharded_sqlite(
        self, query_grams: list[str], max_n: int, min_score: float | None = None,
    ) -> list[CodeSnippet]:
        """Retrieve from v3 sharded SQLite index.

        Tolerates in-progress builds: skips unavailable shards and handles
        query errors on shards that may be mid-write.
        """
        assert self._snippets_db is not None
        n_query_grams = len(query_grams)
        threshold = min_score if min_score is not None else self.config.min_score

        # Group grams by shard
        shard_groups: dict[int, list[str]] = defaultdict(list)
        for gram in query_grams:
            shard_id = self._gram_to_shard(gram, self._num_shards)
            shard_groups[shard_id].append(gram)

        # Query each relevant shard and aggregate scores
        scores: dict[int, int] = {}  # snippet_id → match count
        for shard_id, grams in shard_groups.items():
            conn = self._shard_conns[shard_id] if shard_id < len(self._shard_conns) else None
            if conn is None:
                continue  # Shard not available yet (in-progress build)
            try:
                placeholders = ",".join("?" for _ in grams)
                rows = conn.execute(
                    f"SELECT snippet_id, COUNT(*) FROM ngrams "
                    f"WHERE gram IN ({placeholders}) GROUP BY snippet_id",
                    grams,
                ).fetchall()
                for sid, cnt in rows:
                    scores[sid] = scores.get(sid, 0) + cnt
            except Exception:
                continue  # Shard may be mid-write or unindexed

        if not scores:
            return []

        # Normalize and filter
        candidates = [
            (sid, cnt / n_query_grams)
            for sid, cnt in scores.items()
            if cnt / n_query_grams >= threshold
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:max_n * 2]

        if not candidates:
            return []

        # Fetch snippet details from snippets.db
        snippet_ids = [c[0] for c in candidates]
        id_placeholders = ",".join("?" for _ in snippet_ids)
        snippets = self._snippets_db.execute(
            f"SELECT id, code, source, hash FROM snippets WHERE id IN ({id_placeholders})",
            snippet_ids,
        ).fetchall()

        snip_map = {s[0]: s for s in snippets}

        results: list[CodeSnippet] = []
        seen_hashes: set[str] = set()

        for sid, score in candidates:
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
                score=score,
                hash=h,
            ))

        return results

    def _retrieve_json(
        self, query_grams: list[str], max_n: int, min_score: float | None = None,
    ) -> list[CodeSnippet]:
        """Retrieve from JSON-backed v1 index."""
        threshold = min_score if min_score is not None else self.config.min_score

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
            if score >= threshold
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

    def _build_word_index(self) -> None:
        """Build word→snippet_ids reverse index for keyword fallback."""
        t0 = time.perf_counter()
        self._word_index = {}
        for gram, snippet_ids in self._ngram_index.items():
            for word in gram.split():
                if len(word) < 3:
                    continue
                if word not in self._word_index:
                    self._word_index[word] = set()
                self._word_index[word].update(snippet_ids)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _log.info(
            "Word index built: %d words in %.0fms",
            len(self._word_index), elapsed_ms,
        )

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from query text for fallback search."""
        words = text.split()
        keywords = set()
        for w in words:
            w = self._normalize_token(w)
            if len(w) >= 3 and w not in _KEYWORD_STOPWORDS:
                keywords.add(w)
        return keywords

    def _retrieve_keywords_json(
        self, keywords: set[str], max_n: int, min_score: float,
    ) -> list[CodeSnippet]:
        """Keyword-level fallback: score snippets by individual word overlap."""
        if self._word_index is None:
            self._build_word_index()
        assert self._word_index is not None

        scores: dict[int, float] = {}
        n_keywords = len(keywords)

        for word in keywords:
            sids = self._word_index.get(word, set())
            for sid in sids:
                scores[sid] = scores.get(sid, 0) + 1.0

        if not scores:
            return []

        # Normalize by keyword count
        for sid in scores:
            scores[sid] /= n_keywords

        # Filter and sort
        candidates = [
            (sid, s) for sid, s in scores.items() if s >= min_score
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        results: list[CodeSnippet] = []
        seen_hashes: set[str] = set()

        for sid, score in candidates[:max_n * 2]:
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

    def retrieve_for_rag(self, query: str) -> list[CodeSnippet]:
        """Retrieve snippets with RAG-tuned parameters (more results, lower threshold)."""
        if not self.config.rag_enabled:
            return []
        return self.retrieve(
            query,
            max_results=self.config.rag_max_snippets,
            min_score_override=self.config.rag_min_score,
        )

    def format_for_rag(self, snippets: list[CodeSnippet], task: str) -> str:
        """Format snippets with explicit RAG instruction for quality improvement.

        Unlike format_for_prompt() (speed-only, silent injection), this tells
        the model to study and adapt the retrieved patterns. Uses few-shot
        framing to show the model HOW to adapt reference code.
        """
        if not snippets:
            return task

        parts: list[str] = [
            "## Reference Code",
            "",
            "I found these relevant code examples. Use them to write better code:",
            "1. Reuse good patterns you see (error handling, type hints, docstrings)",
            "2. Fix any bugs or anti-patterns in the references",
            "3. Combine ideas from multiple references if helpful",
            "4. Write your own implementation — do not copy-paste",
            "",
        ]
        total_chars = 0
        budget = self.config.rag_max_chars

        for snip in snippets:
            code = snip.code
            remaining = budget - total_chars
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

        parts.extend(["", "## Task", task])
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

# Stopwords for keyword fallback (common words that match too broadly)
_KEYWORD_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "are", "was",
    "not", "but", "have", "has", "had", "will", "can", "should", "use",
    "using", "include", "also", "each", "both", "all", "any", "its",
    "write", "implement", "create", "build", "make", "add", "support",
    "return", "returns", "handle", "python", "def", "self", "none",
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
