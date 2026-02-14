"""
SkillBank: Structured knowledge abstraction layer for orchestration memory.

Sits above the EpisodicStore as a derived, compressed knowledge layer.
Raw trajectories remain in episodic.db for replay evaluation; skills are
a materialized view optimized for inference-time prompt injection.

Based on: SkillRL (Xia et al., arXiv:2602.08234, Feb 2026)
Enhanced with: provenance tracking, effectiveness scoring, lineage (parent_id),
               multi-teacher distillation, recursive evolution.

See also: ALMA (Xiong et al., 2026) — meta-learned memory designs
          (implemented in replay evaluation harness)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Storage paths (RAID array per CLAUDE.md)
DEFAULT_SKILLBANK_PATH = Path(
    "/mnt/raid0/llm/claude/orchestration/repl_memory/sessions"
)

SKILL_TYPES = ("general", "routing", "escalation", "failure_lesson")

# Limits
MAX_SKILLS = 500
WARN_SKILLS = 400


@dataclass
class Skill:
    """
    A structured knowledge abstraction distilled from raw trajectories.

    SkillRL uses 4 fields (id, title, principle, when_to_apply).
    We add provenance, lifecycle, and lineage for offline evaluation.

    Attributes:
        id: Unique identifier (e.g., "gen_001", "route_012", "fail_003")
        title: 3-7 word imperative title
        skill_type: One of "general", "routing", "escalation", "failure_lesson"
        principle: 1-3 sentence actionable strategy
        when_to_apply: Applicability conditions
        task_types: Applicable task types, or ["*"] for general
        source_trajectory_ids: Provenance — which raw trajectories produced this
        source_outcome: "success", "failure", or "mixed"
        confidence: 0.0-1.0, updated by recursive evolution
        retrieval_count: How often this skill has been retrieved
        effectiveness_score: Post-retrieval outcome correlation
        embedding_idx: Position in FAISS index (set on store)
        created_at: When the skill was first distilled
        updated_at: Last modification time
        revision: Incremented on evolution updates
        deprecated: Soft-delete for skills that degrade
        parent_id: Lineage — which skill this was evolved from
        teacher_model: Which teacher produced this
    """

    id: str
    title: str
    skill_type: str
    principle: str
    when_to_apply: str
    task_types: List[str] = field(default_factory=lambda: ["*"])
    source_trajectory_ids: List[str] = field(default_factory=list)
    source_outcome: str = "success"
    confidence: float = 0.5
    retrieval_count: int = 0
    effectiveness_score: float = 0.5
    embedding_idx: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    revision: int = 1
    deprecated: bool = False
    parent_id: Optional[str] = None
    teacher_model: str = "unknown"

    def __post_init__(self):
        if self.skill_type not in SKILL_TYPES:
            raise ValueError(
                f"Invalid skill_type '{self.skill_type}', must be one of {SKILL_TYPES}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (excludes embedding)."""
        return {
            "id": self.id,
            "title": self.title,
            "skill_type": self.skill_type,
            "principle": self.principle,
            "when_to_apply": self.when_to_apply,
            "task_types": self.task_types,
            "source_trajectory_ids": self.source_trajectory_ids,
            "source_outcome": self.source_outcome,
            "confidence": self.confidence,
            "retrieval_count": self.retrieval_count,
            "effectiveness_score": self.effectiveness_score,
            "embedding_idx": self.embedding_idx,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "revision": self.revision,
            "deprecated": self.deprecated,
            "parent_id": self.parent_id,
            "teacher_model": self.teacher_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Skill:
        """Deserialize from dict."""
        d = dict(data)
        for field_name in ("created_at", "updated_at"):
            if isinstance(d.get(field_name), str):
                d[field_name] = datetime.fromisoformat(d[field_name])
        if isinstance(d.get("task_types"), str):
            d["task_types"] = json.loads(d["task_types"])
        if isinstance(d.get("source_trajectory_ids"), str):
            d["source_trajectory_ids"] = json.loads(d["source_trajectory_ids"])
        if isinstance(d.get("deprecated"), int):
            d["deprecated"] = bool(d["deprecated"])
        return cls(**d)


class SkillBank:
    """
    SQLite + FAISS store for structured skills.

    Separate from episodic.db by design:
    - Different lifecycle (batch updates vs real-time writes)
    - Different FAISS index (skill embeddings vs task context embeddings)
    - Independent backup/restore (derived from episodic store)
    - Replay harness isolation (reads only episodic.db)

    Attributes:
        db_path: Path to skills.db SQLite file
        faiss_path: Directory for skill_embeddings.faiss + skill_id_map.npy
        embedding_dim: Dimensionality of skill embeddings (1024 for BGE-large)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        faiss_path: Optional[Path] = None,
        embedding_dim: int = 1024,
    ):
        if db_path is None:
            db_path = DEFAULT_SKILLBANK_PATH / "skills.db"
        if faiss_path is None:
            faiss_path = DEFAULT_SKILLBANK_PATH

        self.db_path = Path(db_path)
        self.faiss_path = Path(faiss_path)
        self.embedding_dim = embedding_dim

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

        # Initialize FAISS (lazy — only when first embedding is stored)
        self._embedding_store = None

    def _create_schema(self) -> None:
        """Create skills table if it doesn't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                skill_type TEXT NOT NULL CHECK(skill_type IN ('general', 'routing', 'escalation', 'failure_lesson')),
                principle TEXT NOT NULL,
                when_to_apply TEXT NOT NULL,
                task_types TEXT NOT NULL,
                source_trajectory_ids TEXT NOT NULL,
                source_outcome TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                retrieval_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.5,
                embedding_idx INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                revision INTEGER DEFAULT 1,
                deprecated INTEGER DEFAULT 0,
                parent_id TEXT,
                teacher_model TEXT NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES skills(id)
            );

            CREATE INDEX IF NOT EXISTS idx_skill_type ON skills(skill_type);
            CREATE INDEX IF NOT EXISTS idx_confidence_desc ON skills(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_deprecated ON skills(deprecated);
            CREATE INDEX IF NOT EXISTS idx_effectiveness ON skills(effectiveness_score DESC);
        """)
        self._conn.commit()

    def _get_embedding_store(self):
        """Lazy-init FAISS embedding store."""
        if self._embedding_store is None:
            from .faiss_store import FAISSEmbeddingStore

            self._embedding_store = FAISSEmbeddingStore(
                path=self.faiss_path,
                dim=self.embedding_dim,
                index_filename="skill_embeddings.faiss",
                id_map_filename="skill_id_map.npy",
            )
        return self._embedding_store

    # ── CRUD ──────────────────────────────────────────────────────────────

    def store(
        self,
        skill: Skill,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """
        Store a skill in the bank.

        Args:
            skill: Skill record to store
            embedding: Optional 1024-dim embedding for FAISS index

        Returns:
            Skill ID
        """
        # Warn on approaching limit
        current = self.count()
        if current >= MAX_SKILLS:
            logger.warning(
                "SkillBank at capacity (%d/%d). Deprecate low-confidence skills first.",
                current, MAX_SKILLS,
            )
        elif current >= WARN_SKILLS:
            logger.warning("SkillBank nearing capacity: %d/%d", current, MAX_SKILLS)

        # Store embedding if provided
        if embedding is not None:
            store = self._get_embedding_store()
            skill.embedding_idx = store.add(skill.id, embedding)
            store.save()

        # Upsert into SQLite
        self._conn.execute(
            """INSERT OR REPLACE INTO skills
               (id, title, skill_type, principle, when_to_apply, task_types,
                source_trajectory_ids, source_outcome, confidence, retrieval_count,
                effectiveness_score, embedding_idx, created_at, updated_at,
                revision, deprecated, parent_id, teacher_model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                skill.id,
                skill.title,
                skill.skill_type,
                skill.principle,
                skill.when_to_apply,
                json.dumps(skill.task_types),
                json.dumps(skill.source_trajectory_ids),
                skill.source_outcome,
                skill.confidence,
                skill.retrieval_count,
                skill.effectiveness_score,
                skill.embedding_idx,
                skill.created_at.isoformat(),
                skill.updated_at.isoformat(),
                skill.revision,
                int(skill.deprecated),
                skill.parent_id,
                skill.teacher_model,
            ),
        )
        self._conn.commit()

        logger.debug("Stored skill %s: %s", skill.id, skill.title)
        return skill.id

    def get_by_id(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        row = self._conn.execute(
            "SELECT * FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_skill(row)

    def get_skills(
        self,
        skill_type: Optional[str] = None,
        deprecated: Optional[bool] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        min_retrieval_count: Optional[int] = None,
        max_retrieval_count: Optional[int] = None,
        min_effectiveness: Optional[float] = None,
        max_effectiveness: Optional[float] = None,
        min_age_days: Optional[int] = None,
        task_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Skill]:
        """Flexible query with filtering."""
        clauses = []
        params: list[Any] = []

        if skill_type is not None:
            clauses.append("skill_type = ?")
            params.append(skill_type)
        if deprecated is not None:
            clauses.append("deprecated = ?")
            params.append(int(deprecated))
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        if max_confidence is not None:
            clauses.append("confidence <= ?")
            params.append(max_confidence)
        if min_retrieval_count is not None:
            clauses.append("retrieval_count >= ?")
            params.append(min_retrieval_count)
        if max_retrieval_count is not None:
            clauses.append("retrieval_count <= ?")
            params.append(max_retrieval_count)
        if min_effectiveness is not None:
            clauses.append("effectiveness_score >= ?")
            params.append(min_effectiveness)
        if max_effectiveness is not None:
            clauses.append("effectiveness_score <= ?")
            params.append(max_effectiveness)
        if min_age_days is not None:
            cutoff = datetime.now().isoformat()
            # SQLite date arithmetic: julianday diff
            clauses.append(
                "julianday(?) - julianday(created_at) >= ?"
            )
            params.extend([cutoff, min_age_days])
        if task_type is not None:
            # Match skills that apply to this task type or to all ("*")
            clauses.append(
                "(task_types LIKE ? OR task_types LIKE '%\"*\"%')"
            )
            params.append(f'%"{task_type}"%')

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM skills WHERE {where} ORDER BY confidence DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def update(self, skill_id: str, **fields) -> bool:
        """
        Update specific fields on a skill.

        Args:
            skill_id: Skill to update
            **fields: Fields to update (e.g., confidence=0.8, deprecated=True)

        Returns:
            True if updated, False if skill not found
        """
        allowed = {
            "title", "principle", "when_to_apply", "task_types",
            "source_trajectory_ids", "confidence", "retrieval_count",
            "effectiveness_score", "revision", "deprecated", "parent_id",
        }
        updates = {}
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(f"Cannot update field '{k}'")
            if k == "task_types" and isinstance(v, list):
                v = json.dumps(v)
            elif k == "source_trajectory_ids" and isinstance(v, list):
                v = json.dumps(v)
            elif k == "deprecated" and isinstance(v, bool):
                v = int(v)
            updates[k] = v

        if not updates:
            return False

        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        params = list(updates.values()) + [skill_id]

        cursor = self._conn.execute(
            f"UPDATE skills SET {set_clause} WHERE id = ?", params
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def deprecate(self, skill_id: str) -> bool:
        """Soft-delete a skill by marking it deprecated."""
        return self.update(skill_id, deprecated=True)

    def increment_retrieval(self, skill_ids: List[str]) -> None:
        """Bulk increment retrieval_count for given skills."""
        if not skill_ids:
            return
        self._conn.executemany(
            "UPDATE skills SET retrieval_count = retrieval_count + 1, updated_at = ? WHERE id = ?",
            [(datetime.now().isoformat(), sid) for sid in skill_ids],
        )
        self._conn.commit()

    # ── Search ────────────────────────────────────────────────────────────

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.3,
        exclude_deprecated: bool = True,
    ) -> List[tuple[Skill, float]]:
        """
        FAISS similarity search for skills.

        Args:
            query_embedding: 1024-dim query vector
            k: Max results
            min_similarity: Cosine floor
            exclude_deprecated: Skip deprecated skills

        Returns:
            List of (Skill, similarity_score) sorted by score descending
        """
        store = self._get_embedding_store()
        if store.count == 0:
            return []

        # Search for more than k to allow filtering
        raw_results = store.search(query_embedding, k=k * 2)

        results = []
        for skill_id, similarity in raw_results:
            if similarity < min_similarity:
                continue
            skill = self.get_by_id(skill_id)
            if skill is None:
                continue
            if exclude_deprecated and skill.deprecated:
                continue
            results.append((skill, similarity))
            if len(results) >= k:
                break

        return results

    def find_duplicates(
        self,
        principle_embedding: np.ndarray,
        threshold: float = 0.85,
    ) -> List[tuple[Skill, float]]:
        """
        Find skills with similar principles for deduplication.

        Args:
            principle_embedding: Embedding of the new skill's principle
            threshold: Cosine similarity threshold for "duplicate"

        Returns:
            List of (existing_skill, similarity) above threshold
        """
        return self.search_by_embedding(
            principle_embedding,
            k=5,
            min_similarity=threshold,
            exclude_deprecated=True,
        )

    # ── Statistics ────────────────────────────────────────────────────────

    def count(self, include_deprecated: bool = False) -> int:
        """Count skills."""
        if include_deprecated:
            row = self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM skills WHERE deprecated = 0"
            ).fetchone()
        return row[0]

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics for diagnostics/debugger."""
        rows = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN skill_type = 'general' THEN 1 ELSE 0 END) as general,
                SUM(CASE WHEN skill_type = 'routing' THEN 1 ELSE 0 END) as routing,
                SUM(CASE WHEN skill_type = 'escalation' THEN 1 ELSE 0 END) as escalation,
                SUM(CASE WHEN skill_type = 'failure_lesson' THEN 1 ELSE 0 END) as failure_lesson,
                SUM(CASE WHEN deprecated = 1 THEN 1 ELSE 0 END) as deprecated,
                AVG(confidence) as avg_confidence,
                AVG(effectiveness_score) as avg_effectiveness,
                SUM(retrieval_count) as total_retrievals,
                MAX(updated_at) as last_updated
            FROM skills
        """).fetchone()

        return {
            "total": rows["total"],
            "general": rows["general"] or 0,
            "routing": rows["routing"] or 0,
            "escalation": rows["escalation"] or 0,
            "failure_lesson": rows["failure_lesson"] or 0,
            "deprecated": rows["deprecated"] or 0,
            "avg_confidence": round(rows["avg_confidence"] or 0.0, 3),
            "avg_effectiveness": round(rows["avg_effectiveness"] or 0.0, 3),
            "total_retrievals": rows["total_retrievals"] or 0,
            "last_updated": rows["last_updated"],
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _row_to_skill(self, row: sqlite3.Row) -> Skill:
        """Convert a SQLite Row to a Skill dataclass."""
        return Skill(
            id=row["id"],
            title=row["title"],
            skill_type=row["skill_type"],
            principle=row["principle"],
            when_to_apply=row["when_to_apply"],
            task_types=json.loads(row["task_types"]),
            source_trajectory_ids=json.loads(row["source_trajectory_ids"]),
            source_outcome=row["source_outcome"],
            confidence=row["confidence"],
            retrieval_count=row["retrieval_count"],
            effectiveness_score=row["effectiveness_score"],
            embedding_idx=row["embedding_idx"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            revision=row["revision"],
            deprecated=bool(row["deprecated"]),
            parent_id=row["parent_id"],
            teacher_model=row["teacher_model"],
        )

    @staticmethod
    def generate_id(skill_type: str, seq: Optional[int] = None) -> str:
        """Generate a skill ID with type prefix."""
        prefix_map = {
            "general": "gen",
            "routing": "route",
            "escalation": "esc",
            "failure_lesson": "fail",
        }
        prefix = prefix_map.get(skill_type, "skill")
        if seq is not None:
            return f"{prefix}_{seq:03d}"
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def close(self) -> None:
        """Close database connections."""
        if self._embedding_store is not None:
            self._embedding_store.save()
        self._conn.close()
        logger.debug("SkillBank closed")
