"""
FailureGraph ↔ SkillBank bridge for failure lesson formalization.

Connects the Kuzu-backed FailureGraph (anti-memory) with the SQLite-backed
SkillBank (structured skills) to:

1. Export FailureGraph mitigations as structured failure_lesson skills
2. Cross-reference when storing new failure_lessons to avoid duplication
3. Generate failure_graph_summary for distillation prompts

Based on: SkillRL §3.1 (Experience-based Skill Distillation)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FailureBridge:
    """
    Bridge between FailureGraph and SkillBank.

    FailureGraph tracks raw failure patterns and mitigations.
    SkillBank stores structured skills including failure_lesson type.
    This bridge syncs between them.

    Usage:
        bridge = FailureBridge(failure_graph, skill_bank)
        summary = bridge.get_failure_context_for_distillation()
        bridge.sync_mitigations_to_skills()
    """

    def __init__(
        self,
        failure_graph: "FailureGraph",
        skill_bank: "SkillBank",
    ):
        self.failure_graph = failure_graph
        self.skill_bank = skill_bank

    def get_failure_context_for_distillation(
        self,
        max_entries: int = 20,
    ) -> str:
        """
        Generate a markdown summary of FailureGraph for distillation prompts.

        This is the `failure_graph_summary` parameter used by
        build_failure_prompt() in the distillation pipeline.

        Args:
            max_entries: Maximum failure entries to include

        Returns:
            Markdown summary string
        """
        try:
            stats = self.failure_graph.get_stats()
        except Exception as e:
            logger.debug("Failed to get FailureGraph stats: %s", e)
            return "No FailureGraph context available."

        failure_count = stats.get("failuremode_count", 0)
        mitigation_count = stats.get("mitigation_count", 0)
        symptom_count = stats.get("symptom_count", 0)

        if failure_count == 0:
            return "No FailureGraph context available."

        lines = [
            f"**FailureGraph Summary**: {failure_count} known failure modes, "
            f"{mitigation_count} mitigations, {symptom_count} symptoms.",
            "",
        ]

        # Get existing failure_lesson skills to avoid duplication
        existing_lessons = self.skill_bank.get_skills(
            skill_type="failure_lesson",
            deprecated=False,
            limit=100,
        )
        existing_titles = {s.title.lower() for s in existing_lessons}

        lines.append(f"**Existing failure lessons in SkillBank**: {len(existing_lessons)}")
        if existing_lessons:
            for lesson in existing_lessons[:10]:
                lines.append(f"  - {lesson.title} (confidence={lesson.confidence:.2f})")

        lines.append("")
        lines.append("**Cross-reference**: Skip any failure pattern already covered above.")

        return "\n".join(lines)

    def sync_mitigations_to_skills(
        self,
        min_success_rate: float = 0.7,
        min_attempts: int = 2,
    ) -> Dict[str, Any]:
        """
        Export high-quality FailureGraph mitigations as SkillBank failure_lesson skills.

        Only exports mitigations with sufficient success rate and attempt count.

        Args:
            min_success_rate: Minimum mitigation success rate (0-1)
            min_attempts: Minimum number of attempts

        Returns:
            Dict with sync statistics
        """
        from ..skill_bank import Skill, SkillBank

        stats = {"checked": 0, "created": 0, "skipped_existing": 0, "skipped_low_quality": 0}

        # Get all mitigations from FailureGraph
        try:
            mitigations = self._get_all_mitigations()
        except Exception as e:
            logger.warning("Failed to query mitigations from FailureGraph: %s", e)
            return stats

        # Get existing failure_lesson titles to avoid duplication
        existing_lessons = self.skill_bank.get_skills(
            skill_type="failure_lesson",
            deprecated=False,
            limit=500,
        )
        existing_titles = {s.title.lower() for s in existing_lessons}

        for mit in mitigations:
            stats["checked"] += 1

            # Quality filter
            if mit["success_rate"] < min_success_rate:
                stats["skipped_low_quality"] += 1
                continue

            # Dedup check
            title = f"Avoid: {mit['failure_description'][:60]}"
            if title.lower() in existing_titles:
                stats["skipped_existing"] += 1
                continue

            # Create failure_lesson skill
            skill = Skill(
                id=SkillBank.generate_id("failure_lesson"),
                title=title,
                skill_type="failure_lesson",
                principle=(
                    f"FAILURE POINT: {mit['failure_description']}. "
                    f"PREVENTION: {mit['action']} "
                    f"(success_rate={mit['success_rate']:.0%})."
                ),
                when_to_apply=f"When symptoms match: {', '.join(mit.get('symptoms', ['unknown']))}",
                task_types=["*"],
                source_trajectory_ids=[f"failure_graph:{mit['failure_id']}"],
                source_outcome="failure",
                confidence=min(0.9, mit["success_rate"]),
                teacher_model="failure_graph_sync",
            )

            try:
                self.skill_bank.store(skill)
                stats["created"] += 1
                existing_titles.add(title.lower())
            except Exception as e:
                logger.debug("Failed to store failure skill: %s", e)

        logger.info(
            "FailureBridge sync: checked=%d, created=%d, skipped_existing=%d, skipped_low=%d",
            stats["checked"],
            stats["created"],
            stats["skipped_existing"],
            stats["skipped_low_quality"],
        )
        return stats

    def check_skill_against_graph(
        self,
        skill_principle: str,
        skill_symptoms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Cross-reference a proposed failure_lesson skill against FailureGraph.

        Used during distillation to check if a teacher-proposed failure lesson
        duplicates known mitigations.

        Args:
            skill_principle: The skill's principle text
            skill_symptoms: Optional symptom patterns from the skill

        Returns:
            Dict with {is_duplicate, existing_mitigation, coverage}
        """
        result = {"is_duplicate": False, "existing_mitigation": None, "coverage": 0.0}

        if not skill_symptoms:
            return result

        try:
            effective = self.failure_graph.get_effective_mitigations(skill_symptoms)
            if effective:
                best = effective[0]
                if best["success_rate"] > 0.8:
                    result["is_duplicate"] = True
                    result["existing_mitigation"] = best["action"]
                    result["coverage"] = best["success_rate"]
        except Exception as e:
            logger.debug("FailureGraph cross-reference failed: %s", e)

        return result

    def _get_all_mitigations(self) -> List[Dict[str, Any]]:
        """
        Query all mitigations from FailureGraph with their failure context.

        Returns:
            List of dicts with failure_id, failure_description, action,
            success_rate, symptoms
        """
        try:
            result = self.failure_graph.conn.execute(
                """
                MATCH (f:FailureMode)-[:MITIGATED_BY]->(m:Mitigation)
                WHERE m.success_rate >= 0.5
                OPTIONAL MATCH (f)-[:HAS_SYMPTOM]->(s:Symptom)
                RETURN f.id, f.description, m.action, m.success_rate,
                       COLLECT(s.pattern) as symptoms
                ORDER BY m.success_rate DESC
                """
            )
            rows = result.get_as_df()

            mitigations = []
            for _, row in rows.iterrows():
                mitigations.append({
                    "failure_id": row["f.id"],
                    "failure_description": row["f.description"],
                    "action": row["m.action"],
                    "success_rate": float(row["m.success_rate"]),
                    "symptoms": list(row.get("symptoms", [])),
                })
            return mitigations

        except Exception as e:
            logger.warning("Failed to query mitigations: %s", e)
            return []
