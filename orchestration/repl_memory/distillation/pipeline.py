"""
DistillationPipeline: batch distillation of trajectories into structured skills.

Workflow:
1. Extract trajectories via TrajectoryExtractor (reuses replay harness infra)
2. Group by outcome: success / failure / escalation
3. Batch and send to teacher model with appropriate prompt
4. Parse teacher response into Skill records
5. Deduplicate against existing SkillBank (cosine > 0.85 = merge)
6. Embed and store new skills

CLI: python3 -m orchestration.repl_memory.distillation.pipeline --days 25 --teacher mock --dry-run

Based on: SkillRL §3.1 (Experience-based Skill Distillation)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ..skill_bank import Skill, SkillBank

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 20


@dataclass
class DistillationBatch:
    """A batch of trajectories grouped for distillation."""

    batch_id: str
    skill_type: str  # "routing" | "failure_lesson" | "escalation"
    trajectories: List[Dict[str, Any]]
    task_type_filter: Optional[str] = None


@dataclass
class DistillationReport:
    """Summary of a distillation run."""

    total_trajectories: int = 0
    success_trajectories: int = 0
    failure_trajectories: int = 0
    escalation_trajectories: int = 0
    batches_processed: int = 0
    skills_proposed: int = 0
    skills_stored: int = 0
    skills_merged: int = 0
    skills_rejected: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    batch_latencies: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trajectories": self.total_trajectories,
            "success_trajectories": self.success_trajectories,
            "failure_trajectories": self.failure_trajectories,
            "escalation_trajectories": self.escalation_trajectories,
            "batches_processed": self.batches_processed,
            "skills_proposed": self.skills_proposed,
            "skills_stored": self.skills_stored,
            "skills_merged": self.skills_merged,
            "skills_rejected": self.skills_rejected,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
            "batch_latencies": self.batch_latencies,
        }


class DistillationPipeline:
    """
    Orchestrates the full distillation workflow.

    Attributes:
        teacher: Teacher model for skill extraction
        skill_bank: Target SkillBank for storing results
        embedder: Optional embedder for skill principle text
        batch_size: Trajectories per teacher call
    """

    def __init__(
        self,
        teacher,  # TeacherModel protocol
        skill_bank: SkillBank,
        embedder=None,  # Optional TaskEmbedder — None = skip FAISS indexing
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.teacher = teacher
        self.skill_bank = skill_bank
        self.embedder = embedder
        self.batch_size = batch_size

    async def run(
        self,
        trajectories: List[Dict[str, Any]],
        failure_graph_summary: str = "No FailureGraph context available.",
    ) -> DistillationReport:
        """
        Run full distillation on a set of trajectory dicts.

        Args:
            trajectories: List of trajectory dicts with at minimum:
                - task_id, task_type, objective, routing_decision, outcome
            failure_graph_summary: Markdown summary from FailureGraph for
                cross-referencing in failure lesson prompts

        Returns:
            DistillationReport with counts and errors
        """
        import time

        start = time.monotonic()
        report = DistillationReport(total_trajectories=len(trajectories))

        # Group by outcome
        successes = [t for t in trajectories if t.get("outcome") == "success"]
        failures = [t for t in trajectories if t.get("outcome") == "failure"]
        escalations = [t for t in trajectories if t.get("escalations")]

        report.success_trajectories = len(successes)
        report.failure_trajectories = len(failures)
        report.escalation_trajectories = len(escalations)

        # Distill each category
        all_skills: List[tuple[Skill, Optional[np.ndarray]]] = []

        if successes:
            skills = await self._distill_category(
                successes, "routing", failure_graph_summary, report
            )
            all_skills.extend(skills)

        if failures:
            skills = await self._distill_category(
                failures, "failure_lesson", failure_graph_summary, report
            )
            all_skills.extend(skills)

        if escalations:
            skills = await self._distill_category(
                escalations, "escalation", failure_graph_summary, report
            )
            all_skills.extend(skills)

        # Deduplicate and store
        for skill, embedding in all_skills:
            stored = self._deduplicate_and_store(skill, embedding, report)
            if stored:
                report.skills_stored += 1

        report.duration_seconds = time.monotonic() - start
        logger.info(
            "Distillation complete: %d proposed, %d stored, %d merged, %d rejected (%.1fs)",
            report.skills_proposed,
            report.skills_stored,
            report.skills_merged,
            report.skills_rejected,
            report.duration_seconds,
        )
        return report

    async def _distill_category(
        self,
        trajectories: List[Dict[str, Any]],
        skill_type: str,
        failure_graph_summary: str,
        report: DistillationReport,
    ) -> List[tuple[Skill, Optional[np.ndarray]]]:
        """Distill a category of trajectories into skills."""
        from .prompts import (
            build_success_prompt,
            build_failure_prompt,
            build_escalation_prompt,
        )
        from .teachers import parse_skills_from_response

        results: List[tuple[Skill, Optional[np.ndarray]]] = []

        # Batch trajectories
        for i in range(0, len(trajectories), self.batch_size):
            batch = trajectories[i : i + self.batch_size]
            report.batches_processed += 1

            # Build prompt
            if skill_type == "routing":
                prompt = build_success_prompt(batch)
            elif skill_type == "failure_lesson":
                prompt = build_failure_prompt(batch, failure_graph_summary)
            elif skill_type == "escalation":
                prompt = build_escalation_prompt(batch)
            else:
                continue

            # Call teacher (timed)
            import time as _time

            batch_start = _time.monotonic()
            try:
                response = await self.teacher.distill(prompt)
            except Exception as e:
                error_msg = f"Teacher error on {skill_type} batch {i}: {e}"
                logger.error(error_msg)
                report.errors.append(error_msg)
                continue
            finally:
                elapsed_ms = (_time.monotonic() - batch_start) * 1000
                batch_record = {
                    "skill_type": skill_type,
                    "batch_index": i,
                    "batch_size": len(batch),
                    "elapsed_ms": round(elapsed_ms, 1),
                    "teacher": getattr(self.teacher, "model_id", "unknown"),
                }
                report.batch_latencies.append(batch_record)
                logger.info(
                    "Batch %s/%d: %.1fms (%s)",
                    skill_type, i, elapsed_ms, batch_record["teacher"],
                )

            # Parse response
            raw_skills = parse_skills_from_response(response)
            if not raw_skills:
                logger.debug(
                    "No skills parsed from %s batch %d response", skill_type, i
                )
                continue

            # Convert to Skill objects
            for raw in raw_skills:
                try:
                    skill = self._raw_to_skill(raw, skill_type, batch)
                    embedding = self._embed_skill(skill)
                    results.append((skill, embedding))
                    report.skills_proposed += 1
                except Exception as e:
                    logger.debug("Failed to create skill from raw: %s", e)
                    report.errors.append(f"Skill creation error: {e}")

        return results

    def _raw_to_skill(
        self,
        raw: Dict[str, Any],
        default_skill_type: str,
        source_trajectories: List[Dict[str, Any]],
    ) -> Skill:
        """Convert a parsed JSON dict to a Skill object."""
        skill_type = raw.get("skill_type", default_skill_type)
        if skill_type not in ("general", "routing", "escalation", "failure_lesson"):
            skill_type = default_skill_type

        task_types = raw.get("task_types", ["*"])
        if isinstance(task_types, str):
            task_types = [task_types]

        source_ids = [
            t.get("task_id", "unknown") for t in source_trajectories[:10]
        ]

        return Skill(
            id=SkillBank.generate_id(skill_type),
            title=raw.get("title", "Untitled Skill"),
            skill_type=skill_type,
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            task_types=task_types,
            source_trajectory_ids=source_ids,
            source_outcome=raw.get("source_outcome", "success"),
            confidence=0.5,  # New skills start neutral
            teacher_model=self.teacher.model_id,
        )

    def _embed_skill(self, skill: Skill) -> Optional[np.ndarray]:
        """Embed a skill's principle text for FAISS indexing."""
        if self.embedder is None:
            return None
        try:
            text = f"{skill.title}: {skill.principle}"
            return self.embedder.embed(text)
        except Exception as e:
            logger.debug("Embedding failed for skill %s: %s", skill.id, e)
            return None

    def _deduplicate_and_store(
        self,
        skill: Skill,
        embedding: Optional[np.ndarray],
        report: DistillationReport,
    ) -> bool:
        """
        Check for duplicates and store or merge.

        Returns True if a new skill was stored.
        """
        if embedding is not None:
            duplicates = self.skill_bank.find_duplicates(embedding, threshold=0.85)
            if duplicates:
                existing, similarity = duplicates[0]
                if (
                    existing.skill_type == skill.skill_type
                    and _overlaps(existing.task_types, skill.task_types)
                ):
                    # Merge: update existing skill
                    merged_sources = list(
                        set(existing.source_trajectory_ids + skill.source_trajectory_ids)
                    )
                    self.skill_bank.update(
                        existing.id,
                        source_trajectory_ids=merged_sources,
                        revision=existing.revision + 1,
                    )
                    report.skills_merged += 1
                    logger.debug(
                        "Merged skill %s into %s (sim=%.3f)",
                        skill.id, existing.id, similarity,
                    )
                    return False

        # Validate minimum quality
        if not skill.principle or not skill.title:
            report.skills_rejected += 1
            return False

        self.skill_bank.store(skill, embedding=embedding)
        return True


def _overlaps(a: List[str], b: List[str]) -> bool:
    """Check if two task_type lists have any overlap (including wildcard)."""
    if "*" in a or "*" in b:
        return True
    return bool(set(a) & set(b))
