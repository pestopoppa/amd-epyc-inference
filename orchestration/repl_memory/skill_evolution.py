"""
Skill evolution monitor — recursive improvement of distilled skills.

Implements SkillRL §3.3 (Recursive Skill Evolution):
1. Track per-skill effectiveness via retrieval + outcome correlation
2. Promote high-performing skills (confidence → 1.0)
3. Deprecate low-performing skills (confidence < threshold → deprecated)
4. Trigger re-distillation when skill accuracy drops

Key insight from SkillRL: skills that reduce escalation rate or improve
success rate on their target task_types should be reinforced. Skills that
don't correlate with success should decay and eventually be replaced.

Based on: SkillRL (Xia et al., arXiv:2602.08234, Feb 2026)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .skill_bank import Skill, SkillBank

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for skill evolution monitoring.

    Attributes:
        promotion_threshold: Min effectiveness to promote (SkillRL: 0.8)
        deprecation_threshold: Max effectiveness before deprecation (SkillRL: 0.3)
        min_retrievals: Minimum retrievals before evaluating a skill
        decay_rate: Confidence decay per failed retrieval cycle
        promotion_boost: Confidence boost per successful retrieval cycle
        max_confidence: Ceiling for confidence scores
        stale_days: Days without retrieval before marking skill stale
    """

    promotion_threshold: float = 0.8
    deprecation_threshold: float = 0.3
    min_retrievals: int = 5
    decay_rate: float = 0.05
    promotion_boost: float = 0.03
    max_confidence: float = 0.95
    stale_days: int = 30


@dataclass
class EvolutionReport:
    """Summary of a single evolution cycle."""

    skills_evaluated: int = 0
    skills_promoted: int = 0
    skills_decayed: int = 0
    skills_deprecated: int = 0
    skills_stale: int = 0
    redistillation_candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skills_evaluated": self.skills_evaluated,
            "skills_promoted": self.skills_promoted,
            "skills_decayed": self.skills_decayed,
            "skills_deprecated": self.skills_deprecated,
            "skills_stale": self.skills_stale,
            "redistillation_candidates": self.redistillation_candidates,
        }


class EvolutionMonitor:
    """
    Monitors and evolves skill effectiveness over time.

    Runs periodically (e.g., after each distillation batch or daily)
    to update skill confidence scores based on observed outcomes.

    Workflow:
    1. For each active skill, compute effectiveness from outcome_tracker
    2. Promote skills above promotion_threshold (boost confidence)
    3. Decay skills below deprecation_threshold (reduce confidence)
    4. Deprecate skills with confidence < 0.1
    5. Flag stale skills (no retrievals in stale_days) for review
    6. Collect task_types needing re-distillation

    Usage:
        monitor = EvolutionMonitor(skill_bank)
        report = monitor.run_evolution_cycle(outcome_tracker)
    """

    def __init__(
        self,
        skill_bank: SkillBank,
        config: Optional[EvolutionConfig] = None,
    ):
        self.skill_bank = skill_bank
        self.config = config or EvolutionConfig()

    def run_evolution_cycle(
        self,
        outcome_tracker: Optional["OutcomeTracker"] = None,
    ) -> EvolutionReport:
        """
        Run one evolution cycle across all active skills.

        Args:
            outcome_tracker: Optional tracker providing per-skill effectiveness
                scores. If None, uses retrieval_count heuristics only.

        Returns:
            EvolutionReport with cycle statistics
        """
        report = EvolutionReport()

        skills = self.skill_bank.get_skills(deprecated=False, limit=500)
        now = datetime.now()

        for skill in skills:
            report.skills_evaluated += 1

            # Get effectiveness score
            if outcome_tracker:
                effectiveness = outcome_tracker.get_skill_effectiveness(skill.id)
            else:
                effectiveness = self._heuristic_effectiveness(skill)

            # Skip skills without enough data
            if skill.retrieval_count < self.config.min_retrievals:
                # Check staleness even for under-retrieved skills
                if self._is_stale(skill, now):
                    report.skills_stale += 1
                continue

            # Promote high-performing skills
            if effectiveness >= self.config.promotion_threshold:
                new_confidence = min(
                    self.config.max_confidence,
                    skill.confidence + self.config.promotion_boost,
                )
                if new_confidence != skill.confidence:
                    self.skill_bank.update(skill.id, confidence=new_confidence)
                    report.skills_promoted += 1

            # Decay underperforming skills
            elif effectiveness < self.config.deprecation_threshold:
                new_confidence = max(0.0, skill.confidence - self.config.decay_rate)
                self.skill_bank.update(skill.id, confidence=new_confidence)
                report.skills_decayed += 1

                # Deprecate skills with very low confidence
                if new_confidence < 0.1:
                    self.skill_bank.deprecate(skill.id)
                    report.skills_deprecated += 1
                    # Flag task_types for re-distillation
                    for tt in skill.task_types:
                        if tt not in report.redistillation_candidates:
                            report.redistillation_candidates.append(tt)

            # Check staleness
            if self._is_stale(skill, now):
                report.skills_stale += 1

        logger.info(
            "Evolution cycle: evaluated=%d, promoted=%d, decayed=%d, deprecated=%d, stale=%d",
            report.skills_evaluated,
            report.skills_promoted,
            report.skills_decayed,
            report.skills_deprecated,
            report.skills_stale,
        )
        return report

    def _heuristic_effectiveness(self, skill: Skill) -> float:
        """
        Estimate skill effectiveness from stored metadata when no outcome tracker.

        Uses a combination of:
        - effectiveness_score (if set by external scoring)
        - retrieval_count (popular skills are likely useful)
        - confidence (self-reinforcing — may need correction)

        Returns:
            Estimated effectiveness 0.0-1.0
        """
        if skill.effectiveness_score > 0:
            return skill.effectiveness_score

        # Heuristic: high retrieval + high existing confidence = likely effective
        retrieval_factor = min(1.0, skill.retrieval_count / 20.0)
        return 0.5 * skill.confidence + 0.5 * retrieval_factor

    def _is_stale(self, skill: Skill, now: datetime) -> bool:
        """Check if a skill hasn't been retrieved recently."""
        if skill.updated_at is None:
            return False
        try:
            if isinstance(skill.updated_at, str):
                updated = datetime.fromisoformat(skill.updated_at)
            else:
                updated = skill.updated_at
            delta = (now - updated).days
            return delta > self.config.stale_days
        except (ValueError, TypeError):
            return False

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get current skill population health metrics."""
        stats = self.skill_bank.get_stats()
        skills = self.skill_bank.get_skills(deprecated=False, limit=500)

        if not skills:
            return {
                "total_active": 0,
                "avg_confidence": 0.0,
                "avg_retrievals": 0.0,
                "by_type": {},
            }

        confidences = [s.confidence for s in skills]
        retrievals = [s.retrieval_count for s in skills]

        by_type: Dict[str, Dict[str, Any]] = {}
        for skill in skills:
            st = skill.skill_type
            if st not in by_type:
                by_type[st] = {"count": 0, "avg_confidence": 0.0, "total_retrievals": 0}
            by_type[st]["count"] += 1
            by_type[st]["avg_confidence"] += skill.confidence
            by_type[st]["total_retrievals"] += skill.retrieval_count

        for st in by_type:
            if by_type[st]["count"] > 0:
                by_type[st]["avg_confidence"] /= by_type[st]["count"]
                by_type[st]["avg_confidence"] = round(by_type[st]["avg_confidence"], 3)

        return {
            "total_active": len(skills),
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "avg_retrievals": round(sum(retrievals) / len(retrievals), 1),
            "by_type": by_type,
        }


class OutcomeTracker:
    """
    Tracks task outcomes correlated with skill retrievals.

    When skills are retrieved for a task and the task succeeds/fails,
    the outcome is recorded. Effectiveness is computed as the success
    rate of tasks where the skill was retrieved.

    Usage:
        tracker = OutcomeTracker()
        tracker.record_outcome(skill_id="sk_1", task_id="t1", success=True)
        effectiveness = tracker.get_skill_effectiveness("sk_1")
    """

    def __init__(self):
        self._outcomes: Dict[str, List[bool]] = {}

    def record_outcome(self, skill_id: str, task_id: str, success: bool) -> None:
        """Record a task outcome for a skill that was retrieved."""
        if skill_id not in self._outcomes:
            self._outcomes[skill_id] = []
        self._outcomes[skill_id].append(success)

    def get_skill_effectiveness(self, skill_id: str) -> float:
        """Get success rate for tasks where this skill was retrieved."""
        outcomes = self._outcomes.get(skill_id, [])
        if not outcomes:
            return 0.5  # Neutral when no data
        return sum(outcomes) / len(outcomes)

    def get_all_effectiveness(self) -> Dict[str, float]:
        """Get effectiveness scores for all tracked skills."""
        return {
            sid: self.get_skill_effectiveness(sid)
            for sid in self._outcomes
        }
