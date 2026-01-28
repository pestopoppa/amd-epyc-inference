"""
FailureGraph: Kuzu-backed graph for tracking failure patterns, symptoms, and mitigations.

Enables "failure anti-memory" - learning what NOT to do by tracking:
- FailureMode nodes: Specific failure patterns with severity
- Symptom nodes: Observable patterns that indicate failure
- Mitigation nodes: Actions that resolved failures
- Relationships: HAS_SYMPTOM, MITIGATED_BY, PRECEDED_BY, RECURRED_AFTER

Key insight: Current Q-learning optimizes for repeating success. But in debugging/optimization,
avoiding known failure modes is often more valuable.
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
DEFAULT_KUZU_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/kuzu_db/failure_graph")


@dataclass
class FailureMode:
    """A specific failure pattern with metadata."""

    id: str
    description: str
    severity: int  # 1-5 scale
    first_seen: datetime
    last_seen: datetime
    symptoms: List[str] = None  # Symptom IDs
    mitigations: List[str] = None  # Mitigation IDs

    def __post_init__(self):
        self.symptoms = self.symptoms or []
        self.mitigations = self.mitigations or []


@dataclass
class Symptom:
    """An observable pattern that indicates a failure."""

    id: str
    pattern: str  # Regex or keyword pattern
    detection_method: str  # "regex", "keyword", "llm"


@dataclass
class Mitigation:
    """An action that resolved a failure."""

    id: str
    action: str  # The action taken
    success_rate: float  # 0.0-1.0


class FailureGraph:
    """
    Kuzu-backed graph for failure pattern tracking.

    Schema:
        FailureMode(id, description, severity, first_seen, last_seen)
        Symptom(id, pattern, detection_method)
        Mitigation(id, action, success_rate)

        HAS_SYMPTOM: FailureMode -> Symptom
        MITIGATED_BY: FailureMode -> Mitigation
        PRECEDED_BY: FailureMode -> FailureMode (causal chain)
        RECURRED_AFTER: FailureMode -> Mitigation (mitigation didn't work)
        TRIGGERED: links to episodic memory IDs
    """

    def __init__(self, path: Path = DEFAULT_KUZU_PATH):
        """
        Initialize the failure graph.

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
        # Node tables
        node_schemas = [
            """CREATE NODE TABLE IF NOT EXISTS FailureMode(
                id STRING,
                description STRING,
                severity INT64,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                PRIMARY KEY(id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Symptom(
                id STRING,
                pattern STRING,
                detection_method STRING,
                PRIMARY KEY(id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Mitigation(
                id STRING,
                action STRING,
                success_rate DOUBLE,
                attempt_count INT64,
                success_count INT64,
                PRIMARY KEY(id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS MemoryLink(
                id STRING,
                memory_id STRING,
                PRIMARY KEY(id)
            )""",
        ]

        # Relationship tables
        rel_schemas = [
            """CREATE REL TABLE IF NOT EXISTS HAS_SYMPTOM(
                FROM FailureMode TO Symptom
            )""",
            """CREATE REL TABLE IF NOT EXISTS MITIGATED_BY(
                FROM FailureMode TO Mitigation
            )""",
            """CREATE REL TABLE IF NOT EXISTS PRECEDED_BY(
                FROM FailureMode TO FailureMode
            )""",
            """CREATE REL TABLE IF NOT EXISTS RECURRED_AFTER(
                FROM FailureMode TO Mitigation
            )""",
            """CREATE REL TABLE IF NOT EXISTS TRIGGERED_FROM(
                FROM MemoryLink TO FailureMode
            )""",
        ]

        for schema in node_schemas + rel_schemas:
            try:
                self.conn.execute(schema)
            except Exception as e:
                # Ignore "already exists" errors
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema creation warning: {e}")

    def record_failure(
        self,
        memory_id: str,
        symptoms: List[str],
        description: str = "",
        severity: int = 3,
        previous_failure_id: Optional[str] = None,
    ) -> str:
        """
        Record a failure with its symptoms.

        Args:
            memory_id: ID of the episodic memory entry that triggered this failure
            symptoms: List of symptom patterns detected
            description: Human-readable description of the failure
            severity: 1-5 severity scale (3 = medium)
            previous_failure_id: If this failure was preceded by another

        Returns:
            Failure ID
        """
        failure_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Check if similar failure already exists (by symptom overlap)
        existing = self.find_matching_failures(symptoms)
        if existing:
            # Update existing failure's last_seen
            best_match = existing[0]
            self.conn.execute(
                """
                MATCH (f:FailureMode {id: $id})
                SET f.last_seen = $last_seen
                """,
                {"id": best_match.id, "last_seen": now},
            )
            failure_id = best_match.id
        else:
            # Create new failure mode
            self.conn.execute(
                """
                CREATE (f:FailureMode {
                    id: $id,
                    description: $description,
                    severity: $severity,
                    first_seen: $first_seen,
                    last_seen: $last_seen
                })
                """,
                {
                    "id": failure_id,
                    "description": description or f"Failure with symptoms: {', '.join(symptoms)}",
                    "severity": severity,
                    "first_seen": now,
                    "last_seen": now,
                },
            )

        # Create/link symptoms
        for symptom_pattern in symptoms:
            symptom_id = self._get_or_create_symptom(symptom_pattern)
            # Link failure to symptom
            try:
                self.conn.execute(
                    """
                    MATCH (f:FailureMode {id: $fid}), (s:Symptom {id: $sid})
                    MERGE (f)-[:HAS_SYMPTOM]->(s)
                    """,
                    {"fid": failure_id, "sid": symptom_id},
                )
            except Exception as e:
                logger.debug(f"Symptom link exists or error: {e}")

        # Link to episodic memory
        link_id = str(uuid.uuid4())
        self.conn.execute(
            """
            CREATE (m:MemoryLink {id: $lid, memory_id: $mid})
            """,
            {"lid": link_id, "mid": memory_id},
        )
        self.conn.execute(
            """
            MATCH (m:MemoryLink {id: $lid}), (f:FailureMode {id: $fid})
            CREATE (m)-[:TRIGGERED_FROM]->(f)
            """,
            {"lid": link_id, "fid": failure_id},
        )

        # Link to previous failure if provided (causal chain)
        if previous_failure_id:
            try:
                self.conn.execute(
                    """
                    MATCH (f1:FailureMode {id: $f1}), (f2:FailureMode {id: $f2})
                    MERGE (f1)-[:PRECEDED_BY]->(f2)
                    """,
                    {"f1": failure_id, "f2": previous_failure_id},
                )
            except Exception as e:
                logger.debug(f"Failure chain link error: {e}")

        return failure_id

    def _get_or_create_symptom(self, pattern: str, detection_method: str = "regex") -> str:
        """Get existing symptom or create new one."""
        # Try to find existing
        result = self.conn.execute(
            """
            MATCH (s:Symptom {pattern: $pattern})
            RETURN s.id
            """,
            {"pattern": pattern},
        )
        rows = result.get_as_df()
        if len(rows) > 0:
            return rows.iloc[0]["s.id"]

        # Create new
        symptom_id = str(uuid.uuid4())
        self.conn.execute(
            """
            CREATE (s:Symptom {
                id: $id,
                pattern: $pattern,
                detection_method: $method
            })
            """,
            {"id": symptom_id, "pattern": pattern, "method": detection_method},
        )
        return symptom_id

    def find_matching_failures(self, symptoms: List[str]) -> List[FailureMode]:
        """
        Find failures that match the given symptoms.

        Args:
            symptoms: List of symptom patterns to match

        Returns:
            List of FailureMode sorted by match quality (most matching symptoms first)
        """
        if not symptoms:
            return []

        # Find failures that have any of these symptoms
        result = self.conn.execute(
            """
            MATCH (f:FailureMode)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE s.pattern IN $patterns
            RETURN f.id, f.description, f.severity, f.first_seen, f.last_seen,
                   COUNT(DISTINCT s) as match_count
            ORDER BY match_count DESC, f.last_seen DESC
            """,
            {"patterns": symptoms},
        )
        rows = result.get_as_df()

        failures = []
        for _, row in rows.iterrows():
            failures.append(
                FailureMode(
                    id=row["f.id"],
                    description=row["f.description"],
                    severity=row["f.severity"],
                    first_seen=row["f.first_seen"],
                    last_seen=row["f.last_seen"],
                )
            )
        return failures

    def record_mitigation(
        self,
        failure_id: str,
        action: str,
        worked: bool,
    ) -> str:
        """
        Record a mitigation attempt for a failure.

        Args:
            failure_id: ID of the failure being mitigated
            action: The action taken
            worked: Whether the mitigation resolved the failure

        Returns:
            Mitigation ID
        """
        # Check if mitigation already exists
        result = self.conn.execute(
            """
            MATCH (m:Mitigation {action: $action})
            RETURN m.id, m.attempt_count, m.success_count
            """,
            {"action": action},
        )
        rows = result.get_as_df()

        if len(rows) > 0:
            # Update existing mitigation
            row = rows.iloc[0]
            mitigation_id = row["m.id"]
            attempt_count = int(row["m.attempt_count"]) + 1
            success_count = int(row["m.success_count"]) + (1 if worked else 0)
            success_rate = float(success_count) / float(attempt_count)

            self.conn.execute(
                """
                MATCH (m:Mitigation {id: $id})
                SET m.attempt_count = $attempts,
                    m.success_count = $successes,
                    m.success_rate = $rate
                """,
                {
                    "id": mitigation_id,
                    "attempts": attempt_count,
                    "successes": success_count,
                    "rate": success_rate,
                },
            )
        else:
            # Create new mitigation
            mitigation_id = str(uuid.uuid4())
            self.conn.execute(
                """
                CREATE (m:Mitigation {
                    id: $id,
                    action: $action,
                    success_rate: $rate,
                    attempt_count: $attempts,
                    success_count: $successes
                })
                """,
                {
                    "id": mitigation_id,
                    "action": action,
                    "rate": 1.0 if worked else 0.0,
                    "attempts": 1,
                    "successes": 1 if worked else 0,
                },
            )

        # Link failure to mitigation
        try:
            self.conn.execute(
                """
                MATCH (f:FailureMode {id: $fid}), (m:Mitigation {id: $mid})
                MERGE (f)-[:MITIGATED_BY]->(m)
                """,
                {"fid": failure_id, "mid": mitigation_id},
            )
        except Exception as e:
            logger.debug(f"Mitigation link error: {e}")

        # If mitigation didn't work, record recurrence
        if not worked:
            try:
                self.conn.execute(
                    """
                    MATCH (f:FailureMode {id: $fid}), (m:Mitigation {id: $mid})
                    MERGE (f)-[:RECURRED_AFTER]->(m)
                    """,
                    {"fid": failure_id, "mid": mitigation_id},
                )
            except Exception as e:
                logger.debug(f"Recurrence link error: {e}")

        return mitigation_id

    def get_failure_chain(self, failure_id: str, depth: int = 5) -> List[FailureMode]:
        """
        Get the causal chain of failures via PRECEDED_BY edges.

        Args:
            failure_id: Starting failure ID
            depth: Maximum chain depth

        Returns:
            List of FailureMode in causal order (oldest first)
        """
        result = self.conn.execute(
            f"""
            MATCH path = (f1:FailureMode {{id: $id}})-[:PRECEDED_BY*1..{depth}]->(f2:FailureMode)
            RETURN f2.id, f2.description, f2.severity, f2.first_seen, f2.last_seen
            ORDER BY f2.first_seen ASC
            """,
            {"id": failure_id},
        )
        rows = result.get_as_df()

        chain = []
        for _, row in rows.iterrows():
            chain.append(
                FailureMode(
                    id=row["f2.id"],
                    description=row["f2.description"],
                    severity=row["f2.severity"],
                    first_seen=row["f2.first_seen"],
                    last_seen=row["f2.last_seen"],
                )
            )
        return chain

    def get_failure_risk(self, action: str) -> float:
        """
        Get failure risk score for an action (0.0-1.0 penalty).

        Based on:
        - Number of unmitigated failures associated with similar actions
        - Recurrence rate after mitigation attempts

        Args:
            action: The action string to assess

        Returns:
            Risk score 0.0 (safe) to 1.0 (high risk)
        """
        # Count failures triggered by memories with this action
        # This requires joining through MemoryLink
        result = self.conn.execute(
            """
            MATCH (m:MemoryLink)-[:TRIGGERED_FROM]->(f:FailureMode)
            WHERE m.memory_id CONTAINS $action_hint
            RETURN COUNT(DISTINCT f) as failure_count
            """,
            {"action_hint": action[:50]},  # Use action prefix as hint
        )
        rows = result.get_as_df()
        failure_count = rows.iloc[0]["failure_count"] if len(rows) > 0 else 0

        # Count mitigations that failed (recurred)
        result = self.conn.execute(
            """
            MATCH (f:FailureMode)-[:RECURRED_AFTER]->(m:Mitigation)
            WHERE m.action CONTAINS $action_hint
            RETURN COUNT(DISTINCT f) as recurrence_count
            """,
            {"action_hint": action[:50]},
        )
        rows = result.get_as_df()
        recurrence_count = rows.iloc[0]["recurrence_count"] if len(rows) > 0 else 0

        # Calculate risk score (sigmoid-like scaling)
        # More failures/recurrences = higher risk, asymptotic to 1.0
        raw_risk = failure_count + recurrence_count * 2  # Recurrences weighted higher
        risk = 1.0 - (1.0 / (1.0 + raw_risk * 0.1))  # Scale factor 0.1

        return min(1.0, max(0.0, risk))

    def get_effective_mitigations(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """
        Get mitigations that worked for failures with these symptoms.

        Args:
            symptoms: List of symptom patterns

        Returns:
            List of {action, success_rate} sorted by success rate
        """
        if not symptoms:
            return []

        result = self.conn.execute(
            """
            MATCH (f:FailureMode)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE s.pattern IN $patterns
            MATCH (f)-[:MITIGATED_BY]->(m:Mitigation)
            WHERE NOT EXISTS { MATCH (f)-[:RECURRED_AFTER]->(m) }
            RETURN DISTINCT m.action, m.success_rate
            ORDER BY m.success_rate DESC
            """,
            {"patterns": symptoms},
        )
        rows = result.get_as_df()

        return [
            {"action": row["m.action"], "success_rate": row["m.success_rate"]}
            for _, row in rows.iterrows()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {}

        for table in ["FailureMode", "Symptom", "Mitigation", "MemoryLink"]:
            result = self.conn.execute(f"MATCH (n:{table}) RETURN COUNT(n) as count")
            rows = result.get_as_df()
            stats[f"{table.lower()}_count"] = rows.iloc[0]["count"] if len(rows) > 0 else 0

        return stats

    def close(self) -> None:
        """Close the database connection."""
        # Kuzu handles cleanup automatically
        pass
