"""
Skill-aware replay: evaluate SkillBank skill retrieval during offline replay.

Extends the replay engine to assess whether SkillBank skills would have
improved routing decisions. Enables meta-learning over both memory config
AND skill config simultaneously.

Based on: SkillRL §3.3 (Recursive Skill Evolution) applied to offline replay.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..skill_bank import Skill, SkillBank
from ..skill_retriever import SkillRetriever, SkillRetrievalConfig
from .engine import ReplayEngine, ReplayStepResult
from .metrics import ReplayMetrics
from .trajectory import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class SkillBankConfig:
    """Configuration for SkillBank within a DesignCandidate.

    These parameters control how skills are retrieved and applied
    during replay evaluation. Tested alongside RetrievalConfig and
    ScoringConfig in the meta-learning loop.

    Attributes:
        enabled: Whether skill retrieval is active for this candidate
        general_skills_max: Max general skills (SkillRL default: 6)
        task_specific_k: Top-K task-specific skills
        min_similarity: Cosine floor for task-specific
        min_confidence: Skip skills below this confidence
        max_prompt_tokens: Token budget for skill prompt section
    """

    enabled: bool = True
    general_skills_max: int = 6
    task_specific_k: int = 6
    min_similarity: float = 0.4
    min_confidence: float = 0.3
    max_prompt_tokens: int = 1500

    def to_retrieval_config(self) -> SkillRetrievalConfig:
        """Convert to SkillRetrievalConfig."""
        return SkillRetrievalConfig(
            general_skills_max=self.general_skills_max,
            task_specific_k=self.task_specific_k,
            min_similarity=self.min_similarity,
            min_confidence=self.min_confidence,
            max_prompt_tokens=self.max_prompt_tokens,
        )


@dataclass
class SkillReplayStepResult:
    """Extended replay step result with skill retrieval data."""

    base_result: ReplayStepResult
    skills_retrieved: int  # Number of skills retrieved for this step
    skill_types_retrieved: List[str]  # Types of skills retrieved
    skill_context_tokens: int  # Estimated tokens in skill context


@dataclass
class SkillReplayMetrics:
    """Metrics specific to skill retrieval during replay."""

    base_metrics: ReplayMetrics
    avg_skills_per_step: float = 0.0
    total_skills_retrieved: int = 0
    skill_coverage: float = 0.0  # Fraction of steps with >=1 skill
    avg_skill_context_tokens: float = 0.0
    skills_by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = self.base_metrics.to_dict()
        d["skill_metrics"] = {
            "avg_skills_per_step": round(self.avg_skills_per_step, 2),
            "total_skills_retrieved": self.total_skills_retrieved,
            "skill_coverage": round(self.skill_coverage, 3),
            "avg_skill_context_tokens": round(self.avg_skill_context_tokens, 1),
            "skills_by_type": self.skills_by_type,
        }
        return d


class SkillAwareReplayEngine(ReplayEngine):
    """
    Replay engine that also evaluates skill retrieval effectiveness.

    Extends ReplayEngine to:
    1. Load SkillBank into the replay environment
    2. At each step, retrieve skills and measure coverage
    3. Track whether skill-augmented routing would have been different
    4. Produce SkillReplayMetrics alongside standard ReplayMetrics

    Usage:
        engine = SkillAwareReplayEngine(skill_db_path="/path/to/skills.db")
        metrics = engine.run_with_skill_metrics(
            retrieval_config, scoring_config, skill_config, trajectories
        )
    """

    def __init__(
        self,
        skill_db_path: Optional[Path] = None,
        skill_bank: Optional[SkillBank] = None,
        tmp_dir: Path = Path("/mnt/raid0/llm/tmp/replay"),
        embedding_dim: int = 1024,
    ):
        super().__init__(tmp_dir=tmp_dir, embedding_dim=embedding_dim)

        # Use provided SkillBank or load from path
        if skill_bank is not None:
            self._skill_bank = skill_bank
        elif skill_db_path is not None:
            self._skill_bank = SkillBank(
                db_path=skill_db_path,
                faiss_path=skill_db_path.parent,
                embedding_dim=embedding_dim,
            )
        else:
            self._skill_bank = None

    def run_with_skill_metrics(
        self,
        retrieval_config,
        scoring_config,
        skill_config: SkillBankConfig,
        trajectories: List[Trajectory],
        candidate_id: Optional[str] = None,
    ) -> SkillReplayMetrics:
        """
        Run replay with skill retrieval tracking.

        Args:
            retrieval_config: Standard retrieval parameters
            scoring_config: Standard scoring parameters
            skill_config: SkillBank-specific parameters
            trajectories: Pre-sorted trajectories
            candidate_id: Optional candidate identifier

        Returns:
            SkillReplayMetrics with both standard and skill-specific metrics
        """
        import time

        cid = candidate_id or str(uuid.uuid4())[:8]
        start_time = time.monotonic()

        # Run standard replay
        base_results = self.run(
            retrieval_config, scoring_config, trajectories, cid
        )
        elapsed = time.monotonic() - start_time

        # Compute base metrics
        base_metrics = self._compute_metrics(
            cid,
            trajectories,
            base_results,
            elapsed,
            retrieval_config,
            scoring_config,
        )

        # Evaluate skill retrieval for each step
        skill_results = self._evaluate_skills(
            skill_config, trajectories, base_results
        )

        # Compute skill metrics
        return self._compute_skill_metrics(base_metrics, skill_results)

    def _evaluate_skills(
        self,
        skill_config: SkillBankConfig,
        trajectories: List[Trajectory],
        base_results: List[ReplayStepResult],
    ) -> List[SkillReplayStepResult]:
        """Evaluate skill retrieval for each replay step."""
        results: List[SkillReplayStepResult] = []

        if not self._skill_bank or not skill_config.enabled:
            # No skills — return empty skill data
            for base in base_results:
                results.append(SkillReplayStepResult(
                    base_result=base,
                    skills_retrieved=0,
                    skill_types_retrieved=[],
                    skill_context_tokens=0,
                ))
            return results

        retriever = SkillRetriever(
            skill_bank=self._skill_bank,
            config=skill_config.to_retrieval_config(),
        )

        for trajectory, base in zip(trajectories, base_results):
            if trajectory.embedding is not None:
                try:
                    skill_results = retriever.retrieve_for_task(
                        trajectory.embedding,
                        task_type=trajectory.task_type,
                    )
                    prompt_text = retriever.format_for_prompt(skill_results)

                    results.append(SkillReplayStepResult(
                        base_result=base,
                        skills_retrieved=len(skill_results),
                        skill_types_retrieved=[
                            r.skill.skill_type for r in skill_results
                        ],
                        skill_context_tokens=len(prompt_text) // 4,
                    ))
                except Exception as e:
                    logger.debug("Skill retrieval failed for %s: %s", trajectory.task_id, e)
                    results.append(SkillReplayStepResult(
                        base_result=base,
                        skills_retrieved=0,
                        skill_types_retrieved=[],
                        skill_context_tokens=0,
                    ))
            else:
                results.append(SkillReplayStepResult(
                    base_result=base,
                    skills_retrieved=0,
                    skill_types_retrieved=[],
                    skill_context_tokens=0,
                ))

        return results

    def _compute_skill_metrics(
        self,
        base_metrics: ReplayMetrics,
        skill_results: List[SkillReplayStepResult],
    ) -> SkillReplayMetrics:
        """Compute skill-specific metrics from results."""
        if not skill_results:
            return SkillReplayMetrics(base_metrics=base_metrics)

        n = len(skill_results)
        total_skills = sum(r.skills_retrieved for r in skill_results)
        steps_with_skills = sum(1 for r in skill_results if r.skills_retrieved > 0)
        total_tokens = sum(r.skill_context_tokens for r in skill_results)

        by_type: Dict[str, int] = {}
        for r in skill_results:
            for st in r.skill_types_retrieved:
                by_type[st] = by_type.get(st, 0) + 1

        return SkillReplayMetrics(
            base_metrics=base_metrics,
            avg_skills_per_step=total_skills / n if n > 0 else 0.0,
            total_skills_retrieved=total_skills,
            skill_coverage=steps_with_skills / n if n > 0 else 0.0,
            avg_skill_context_tokens=total_tokens / n if n > 0 else 0.0,
            skills_by_type=by_type,
        )
