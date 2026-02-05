"""
HypothesisGraph: Kuzu-backed graph for tracking hypotheses and evidence.

Enables hypothetical reasoning by tracking:
- Hypothesis nodes: Claims with confidence scores
- Evidence nodes: Observations that support or contradict hypotheses
- Relationships: SUPPORTS, CONTRADICTS, GENERATED_FROM

Key insight: Actions succeed or fail in patterns. Tracking confidence in
action-task combinations allows warning when low-confidence actions are suggested.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default path (on RAID array per CLAUDE.md requirements)
# Note: Each graph needs its own Kuzu database directory to avoid collisions
DEFAULT_KUZU_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/kuzu_db/hypothesis_graph")


@dataclass
class Hypothesis:
    """A claim about action-task effectiveness with confidence."""

    id: str
    claim: str  # e.g., "speculative_decode + code_generation = effective"
    confidence: float  # 0.0-1.0
    created_at: datetime
    tested: bool
    evidence_count: int = 0


@dataclass
class Evidence:
    """An observation that supports or contradicts a hypothesis."""

    id: str
    evidence_type: str  # "supports" or "contradicts"
    source: str  # memory_id or other source
    timestamp: datetime


class HypothesisGraph:
    """
    Kuzu-backed graph for hypothesis tracking.

    Schema:
        Hypothesis(id, claim, confidence, created_at, tested)
        Evidence(id, type, source, timestamp)

        SUPPORTS: Evidence -> Hypothesis
        CONTRADICTS: Evidence -> Hypothesis
        GENERATED_FROM: Hypothesis -> MemoryEntry (links to episodic store)

    Confidence Update Formula:
        On success: confidence += 0.1 * (1 - confidence)  # asymptotic to 1.0
        On failure: confidence -= 0.1 * confidence        # asymptotic to 0.0
    """

    # Learning rate for confidence updates
    LEARNING_RATE = 0.1

    def __init__(self, path: Path = DEFAULT_KUZU_PATH):
        """
        Initialize the hypothesis graph.

        Args:
            path: Path for Kuzu database storage (will be created if doesn't exist)
        """
        try:
            import kuzu
        except ImportError as e:
            raise ImportError("kuzu not installed. Run: pip install kuzu") from e

        self._kuzu = kuzu
        self.path = Path(path)

        # Kuzu 0.11+ requires the path to not exist or be a valid Kuzu DB
        # Create parent directory but not the path itself
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db = kuzu.Database(str(self.path))
        self.conn = kuzu.Connection(self.db)

        # Create schema
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize graph schema if not exists."""
        node_schemas = [
            """CREATE NODE TABLE IF NOT EXISTS Hypothesis(
                id STRING,
                claim STRING,
                confidence DOUBLE,
                created_at TIMESTAMP,
                tested BOOLEAN,
                PRIMARY KEY(id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS HypothesisEvidence(
                id STRING,
                evidence_type STRING,
                source STRING,
                timestamp TIMESTAMP,
                PRIMARY KEY(id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS HypothesisMemoryLink(
                id STRING,
                memory_id STRING,
                PRIMARY KEY(id)
            )""",
        ]

        rel_schemas = [
            """CREATE REL TABLE IF NOT EXISTS SUPPORTS(
                FROM HypothesisEvidence TO Hypothesis
            )""",
            """CREATE REL TABLE IF NOT EXISTS CONTRADICTS(
                FROM HypothesisEvidence TO Hypothesis
            )""",
            """CREATE REL TABLE IF NOT EXISTS GENERATED_FROM(
                FROM Hypothesis TO HypothesisMemoryLink
            )""",
        ]

        for schema in node_schemas + rel_schemas:
            try:
                self.conn.execute(schema)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("Schema creation warning: %s", e)

    def _make_claim(self, action: str, task_type: str) -> str:
        """Create a standardized claim string."""
        return f"{action}|{task_type}"

    def create_hypothesis(
        self,
        claim: str,
        memory_id: str,
        initial_confidence: float = 0.5,
    ) -> str:
        """
        Create a new hypothesis.

        Args:
            claim: The hypothesis claim (e.g., "action X works for task Y")
            memory_id: ID of the memory that generated this hypothesis
            initial_confidence: Initial confidence (default 0.5 = neutral)

        Returns:
            Hypothesis ID
        """
        # Check if hypothesis with this claim exists
        result = self.conn.execute(
            """
            MATCH (h:Hypothesis {claim: $claim})
            RETURN h.id
            """,
            {"claim": claim},
        )
        rows = result.get_as_df()

        if len(rows) > 0:
            return rows.iloc[0]["h.id"]

        # Create new hypothesis
        hypothesis_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        self.conn.execute(
            """
            CREATE (h:Hypothesis {
                id: $id,
                claim: $claim,
                confidence: $confidence,
                created_at: $created_at,
                tested: false
            })
            """,
            {
                "id": hypothesis_id,
                "claim": claim,
                "confidence": initial_confidence,
                "created_at": now,
            },
        )

        # Link to memory
        link_id = str(uuid.uuid4())
        self.conn.execute(
            """
            CREATE (m:HypothesisMemoryLink {id: $lid, memory_id: $mid})
            """,
            {"lid": link_id, "mid": memory_id},
        )
        self.conn.execute(
            """
            MATCH (h:Hypothesis {id: $hid}), (m:HypothesisMemoryLink {id: $lid})
            CREATE (h)-[:GENERATED_FROM]->(m)
            """,
            {"hid": hypothesis_id, "lid": link_id},
        )

        return hypothesis_id

    def add_evidence(
        self,
        hypothesis_id: str,
        outcome: str,
        source: str,
    ) -> float:
        """
        Add evidence to a hypothesis and update confidence.

        Args:
            hypothesis_id: ID of the hypothesis
            outcome: "success" (supports) or "failure" (contradicts)
            source: Source of evidence (memory_id)

        Returns:
            Updated confidence score
        """
        # Determine evidence type
        evidence_type = "supports" if outcome == "success" else "contradicts"

        # Get current confidence
        result = self.conn.execute(
            """
            MATCH (h:Hypothesis {id: $id})
            RETURN h.confidence
            """,
            {"id": hypothesis_id},
        )
        rows = result.get_as_df()

        if len(rows) == 0:
            logger.warning("Hypothesis %s not found", hypothesis_id)
            return 0.5

        old_confidence = rows.iloc[0]["h.confidence"]

        # Calculate new confidence using asymptotic update
        if evidence_type == "supports":
            # Increases toward 1.0
            new_confidence = old_confidence + self.LEARNING_RATE * (1.0 - old_confidence)
        else:
            # Decreases toward 0.0
            new_confidence = old_confidence - self.LEARNING_RATE * old_confidence

        # Clamp to [0, 1]
        new_confidence = max(0.0, min(1.0, new_confidence))

        # Update hypothesis
        self.conn.execute(
            """
            MATCH (h:Hypothesis {id: $id})
            SET h.confidence = $confidence, h.tested = true
            """,
            {"id": hypothesis_id, "confidence": new_confidence},
        )

        # Create evidence node and link
        evidence_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        self.conn.execute(
            """
            CREATE (e:HypothesisEvidence {
                id: $id,
                evidence_type: $type,
                source: $source,
                timestamp: $timestamp
            })
            """,
            {"id": evidence_id, "type": evidence_type, "source": source, "timestamp": now},
        )

        # Link evidence to hypothesis
        rel_type = "SUPPORTS" if evidence_type == "supports" else "CONTRADICTS"
        self.conn.execute(
            f"""
            MATCH (e:HypothesisEvidence {{id: $eid}}), (h:Hypothesis {{id: $hid}})
            CREATE (e)-[:{rel_type}]->(h)
            """,
            {"eid": evidence_id, "hid": hypothesis_id},
        )

        return new_confidence

    def get_confidence(self, action: str, task_type: str) -> float:
        """
        Get confidence for an action-task combination.

        Args:
            action: The action string
            task_type: The task type

        Returns:
            Confidence score (0.0-1.0), or 0.5 if no hypothesis exists
        """
        claim = self._make_claim(action, task_type)

        result = self.conn.execute(
            """
            MATCH (h:Hypothesis {claim: $claim})
            RETURN h.confidence
            """,
            {"claim": claim},
        )
        rows = result.get_as_df()

        if len(rows) > 0:
            return rows.iloc[0]["h.confidence"]
        return 0.5  # Neutral confidence for unknown combinations

    def get_or_create_hypothesis(
        self,
        action: str,
        task_type: str,
        memory_id: str,
    ) -> str:
        """
        Get existing hypothesis or create new one for action-task combination.

        Args:
            action: The action string
            task_type: The task type
            memory_id: Memory ID for linking

        Returns:
            Hypothesis ID
        """
        claim = self._make_claim(action, task_type)
        return self.create_hypothesis(claim, memory_id)

    def get_untested_hypotheses(self, min_confidence: float = 0.7) -> List[Hypothesis]:
        """
        Get untested hypotheses with high confidence.

        These are hypotheses worth testing next.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of Hypothesis sorted by confidence descending
        """
        result = self.conn.execute(
            """
            MATCH (h:Hypothesis)
            WHERE h.tested = false AND h.confidence >= $min_confidence
            RETURN h.id, h.claim, h.confidence, h.created_at, h.tested
            ORDER BY h.confidence DESC
            """,
            {"min_confidence": min_confidence},
        )
        rows = result.get_as_df()

        return [
            Hypothesis(
                id=row["h.id"],
                claim=row["h.claim"],
                confidence=row["h.confidence"],
                created_at=row["h.created_at"],
                tested=row["h.tested"],
            )
            for _, row in rows.iterrows()
        ]

    def get_low_confidence_warnings(
        self,
        action: str,
        task_type: str,
        threshold: float = 0.2,
    ) -> List[str]:
        """
        Get warnings if confidence is low with cited evidence.

        Args:
            action: The action string
            task_type: The task type
            threshold: Confidence threshold below which to warn

        Returns:
            List of warning strings with evidence citations
        """
        claim = self._make_claim(action, task_type)

        result = self.conn.execute(
            """
            MATCH (h:Hypothesis {claim: $claim})
            WHERE h.confidence < $threshold
            OPTIONAL MATCH (e:HypothesisEvidence)-[:CONTRADICTS]->(h)
            RETURN h.confidence, COLLECT(e.source) as sources
            """,
            {"claim": claim, "threshold": threshold},
        )
        rows = result.get_as_df()

        warnings = []
        for _, row in rows.iterrows():
            confidence = row["h.confidence"]
            sources = row["sources"] or []
            source_str = ", ".join(sources[:3]) if sources else "no specific evidence"
            warnings.append(
                f"Low confidence ({confidence:.2f}) for '{action}' on '{task_type}'. "
                f"Evidence: {source_str}"
            )

        return warnings

    def get_contradicting_evidence(self, hypothesis_id: str) -> List[Evidence]:
        """Get all evidence that contradicts a hypothesis."""
        result = self.conn.execute(
            """
            MATCH (e:HypothesisEvidence)-[:CONTRADICTS]->(h:Hypothesis {id: $id})
            RETURN e.id, e.evidence_type, e.source, e.timestamp
            ORDER BY e.timestamp DESC
            """,
            {"id": hypothesis_id},
        )
        rows = result.get_as_df()

        return [
            Evidence(
                id=row["e.id"],
                evidence_type=row["e.evidence_type"],
                source=row["e.source"],
                timestamp=row["e.timestamp"],
            )
            for _, row in rows.iterrows()
        ]

    def get_supporting_evidence(self, hypothesis_id: str) -> List[Evidence]:
        """Get all evidence that supports a hypothesis."""
        result = self.conn.execute(
            """
            MATCH (e:HypothesisEvidence)-[:SUPPORTS]->(h:Hypothesis {id: $id})
            RETURN e.id, e.evidence_type, e.source, e.timestamp
            ORDER BY e.timestamp DESC
            """,
            {"id": hypothesis_id},
        )
        rows = result.get_as_df()

        return [
            Evidence(
                id=row["e.id"],
                evidence_type=row["e.evidence_type"],
                source=row["e.source"],
                timestamp=row["e.timestamp"],
            )
            for _, row in rows.iterrows()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {}

        # Count hypotheses
        result = self.conn.execute("MATCH (h:Hypothesis) RETURN COUNT(h) as count")
        rows = result.get_as_df()
        stats["hypothesis_count"] = rows.iloc[0]["count"] if len(rows) > 0 else 0

        # Count tested vs untested
        result = self.conn.execute(
            "MATCH (h:Hypothesis) WHERE h.tested = true RETURN COUNT(h) as count"
        )
        rows = result.get_as_df()
        stats["tested_count"] = rows.iloc[0]["count"] if len(rows) > 0 else 0

        # Count evidence
        result = self.conn.execute("MATCH (e:HypothesisEvidence) RETURN COUNT(e) as count")
        rows = result.get_as_df()
        stats["evidence_count"] = rows.iloc[0]["count"] if len(rows) > 0 else 0

        # Average confidence
        result = self.conn.execute("MATCH (h:Hypothesis) RETURN AVG(h.confidence) as avg")
        rows = result.get_as_df()
        stats["avg_confidence"] = rows.iloc[0]["avg"] if len(rows) > 0 and rows.iloc[0]["avg"] else 0.5

        return stats

    def close(self) -> None:
        """Close the database connection."""
        pass
