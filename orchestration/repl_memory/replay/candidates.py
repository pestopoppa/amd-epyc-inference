"""Design candidates and archive for memory configuration evolution.

DesignCandidate wraps a (RetrievalConfig, ScoringConfig) pair with metadata
for lineage tracking. DesignArchive provides SQLite-backed storage for
candidate configs and their replay metrics.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..q_scorer import ScoringConfig
from ..retriever import RetrievalConfig
from ..staged_scorer import StagedConfig
from .metrics import ReplayMetrics
from .skill_replay import SkillBankConfig

logger = logging.getLogger(__name__)

# Default archive location on RAID
DEFAULT_ARCHIVE_PATH = Path(
    "/mnt/raid0/llm/claude/orchestration/repl_memory/meta_archive/archive.db"
)


@dataclass
class DesignCandidate:
    """A candidate memory configuration for replay evaluation."""

    candidate_id: str
    parent_id: Optional[str]  # Lineage — which candidate this was mutated from
    retrieval_config: RetrievalConfig
    scoring_config: ScoringConfig
    staged_config: Optional[StagedConfig] = None
    skill_config: Optional[SkillBankConfig] = None
    role_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def default(cls) -> DesignCandidate:
        """Return current production config as baseline candidate."""
        return cls(
            candidate_id=str(uuid.uuid4()),
            parent_id=None,
            retrieval_config=RetrievalConfig(),
            scoring_config=ScoringConfig(),
            staged_config=StagedConfig(),
            skill_config=SkillBankConfig(),
            role_overrides=None,
            notes="production baseline",
            created_at=datetime.utcnow(),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "retrieval_config": asdict(self.retrieval_config),
            "scoring_config": _scoring_config_to_dict(self.scoring_config),
            "staged_config": asdict(self.staged_config) if self.staged_config else None,
            "skill_config": asdict(self.skill_config) if self.skill_config else None,
            "role_overrides": self.role_overrides,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }
        return json.dumps(d, default=str)

    @classmethod
    def from_json(cls, s: str) -> DesignCandidate:
        """Deserialize from JSON string."""
        d = json.loads(s)
        ret_cfg = RetrievalConfig(**{
            k: v for k, v in d.get("retrieval_config", {}).items()
            if k in RetrievalConfig.__dataclass_fields__
        })
        scr_cfg = _scoring_config_from_dict(d.get("scoring_config", {}))
        staged = None
        if d.get("staged_config"):
            staged = StagedConfig(**{
                k: v for k, v in d["staged_config"].items()
                if k in StagedConfig.__dataclass_fields__
            })
        skill_cfg = None
        if d.get("skill_config"):
            skill_cfg = SkillBankConfig(**{
                k: v for k, v in d["skill_config"].items()
                if k in SkillBankConfig.__dataclass_fields__
            })
        return cls(
            candidate_id=d["candidate_id"],
            parent_id=d.get("parent_id"),
            retrieval_config=ret_cfg,
            scoring_config=scr_cfg,
            staged_config=staged,
            skill_config=skill_cfg,
            role_overrides=d.get("role_overrides"),
            notes=d.get("notes", ""),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.utcnow(),
        )


class DesignArchive:
    """SQLite-backed archive for design candidates and their metrics.

    Stores candidates with their evaluated replay metrics for historical
    comparison, lineage tracking, and meta-agent reflection prompts.
    """

    def __init__(self, db_path: Path = DEFAULT_ARCHIVE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the archive schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT,
                    created_at TEXT NOT NULL,
                    parent_id TEXT,
                    notes TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON candidates(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent_id ON candidates(parent_id)
            """)
            conn.commit()

    def store_result(
        self,
        candidate: DesignCandidate,
        metrics: Optional[ReplayMetrics] = None,
    ) -> None:
        """Store a candidate and optionally its metrics."""
        metrics_json = json.dumps(metrics.to_dict()) if metrics else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO candidates
                (id, config_json, metrics_json, created_at, parent_id, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
                    candidate.to_json(),
                    metrics_json,
                    candidate.created_at.isoformat(),
                    candidate.parent_id,
                    candidate.notes,
                ),
            )
            conn.commit()

    def get_top_candidates(
        self,
        metric: str = "cumulative_reward",
        limit: int = 10,
    ) -> List[Tuple[DesignCandidate, ReplayMetrics]]:
        """Get top candidates ranked by a metric.

        Args:
            metric: Metric name to rank by (must be a ReplayMetrics field).
            limit: Max results.

        Returns:
            List of (DesignCandidate, ReplayMetrics) sorted descending by metric.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT config_json, metrics_json FROM candidates WHERE metrics_json IS NOT NULL"
            ).fetchall()

        results = []
        for config_json, metrics_json in rows:
            try:
                candidate = DesignCandidate.from_json(config_json)
                metrics_obj = ReplayMetrics.from_dict(json.loads(metrics_json))
                results.append((candidate, metrics_obj))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Skipping malformed archive entry: %s", e)
                continue

        # Sort by the requested metric (descending)
        results.sort(
            key=lambda pair: getattr(pair[1], metric, 0.0),
            reverse=True,
        )
        return results[:limit]

    def get_lineage(self, candidate_id: str) -> List[DesignCandidate]:
        """Get ancestor chain for a candidate via parent_id links."""
        chain: List[DesignCandidate] = []
        current_id = candidate_id

        with sqlite3.connect(self.db_path) as conn:
            for _ in range(100):  # Safety limit
                row = conn.execute(
                    "SELECT config_json, parent_id FROM candidates WHERE id = ?",
                    (current_id,),
                ).fetchone()
                if not row:
                    break
                try:
                    candidate = DesignCandidate.from_json(row[0])
                    chain.append(candidate)
                except (json.JSONDecodeError, KeyError):
                    break
                if row[1] is None:
                    break
                current_id = row[1]

        return chain

    def sample_for_reflection(self, n: int = 5) -> List[Tuple[DesignCandidate, ReplayMetrics]]:
        """Sample diverse candidates for meta-agent reflection.

        Returns: top 2 + worst 1 + (n-3) random candidates.
        """
        all_candidates = self.get_top_candidates(metric="cumulative_reward", limit=1000)
        if len(all_candidates) <= n:
            return all_candidates

        import random
        result = []

        # Top 2
        result.extend(all_candidates[:2])

        # Worst 1
        result.append(all_candidates[-1])

        # Random from the middle
        middle = all_candidates[2:-1]
        if middle and n > 3:
            sampled = random.sample(middle, min(n - 3, len(middle)))
            result.extend(sampled)

        return result[:n]

    def get_baseline(self) -> Optional[Tuple[DesignCandidate, ReplayMetrics]]:
        """Get the production baseline candidate metrics (if evaluated)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT config_json, metrics_json FROM candidates "
                "WHERE notes = 'production baseline' AND metrics_json IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        if not row:
            return None

        try:
            candidate = DesignCandidate.from_json(row[0])
            metrics = ReplayMetrics.from_dict(json.loads(row[1]))
            return (candidate, metrics)
        except (json.JSONDecodeError, KeyError):
            return None

    def count(self) -> int:
        """Return total number of candidates in the archive."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()
            return row[0] if row else 0


# ---------------------------------------------------------------------------
# ScoringConfig serialization helpers
# ---------------------------------------------------------------------------

def _scoring_config_to_dict(cfg: ScoringConfig) -> Dict[str, Any]:
    """Serialize ScoringConfig to dict (handles dict fields with lambdas)."""
    return {
        "learning_rate": cfg.learning_rate,
        "success_reward": cfg.success_reward,
        "failure_reward": cfg.failure_reward,
        "partial_reward": cfg.partial_reward,
        "temporal_decay_rate": cfg.temporal_decay_rate,
        "use_claude_judge": cfg.use_claude_judge,
        "judge_model_path": str(cfg.judge_model_path) if cfg.judge_model_path else None,
        "judge_binary": str(cfg.judge_binary) if cfg.judge_binary else None,
        "cost_penalty_lambda": cfg.cost_penalty_lambda,
        "baseline_tps_by_role": cfg.baseline_tps_by_role,
        "baseline_quality_by_role": cfg.baseline_quality_by_role,
        "memory_cost_by_role": cfg.memory_cost_by_role,
        "cost_lambda_quality_gap": cfg.cost_lambda_quality_gap,
        "cost_lambda_memory": cfg.cost_lambda_memory,
        "delegation_misattribution_penalty": cfg.delegation_misattribution_penalty,
        "specialist_credit_bonus": cfg.specialist_credit_bonus,
        "teacher_regret_penalty": cfg.teacher_regret_penalty,
        "teacher_speedup_bonus": cfg.teacher_speedup_bonus,
        "min_score_interval_seconds": cfg.min_score_interval_seconds,
        "batch_size": cfg.batch_size,
    }


def _scoring_config_from_dict(d: Dict[str, Any]) -> ScoringConfig:
    """Deserialize ScoringConfig from dict."""
    kwargs: Dict[str, Any] = {}
    for key in ScoringConfig.__dataclass_fields__:
        if key in d and d[key] is not None:
            val = d[key]
            if key in ("judge_model_path", "judge_binary") and isinstance(val, str):
                val = Path(val)
            kwargs[key] = val
    return ScoringConfig(**kwargs)
