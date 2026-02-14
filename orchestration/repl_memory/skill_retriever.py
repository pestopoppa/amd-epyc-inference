"""
SkillRetriever: Runtime skill retrieval and prompt injection.

Implements SkillRL §3.2 (Adaptive Retrieval Strategy) with two-level retrieval:
1. General skills — always included, sorted by confidence
2. Task-specific skills — FAISS similarity search, filtered by confidence

Retrieval defaults from SkillRL: min_similarity=0.4, top-K=6.
Our additions: confidence filtering, token budget enforcement, retrieval tracking.

Based on: SkillRL (Xia et al., arXiv:2602.08234, Feb 2026)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .skill_bank import Skill, SkillBank

logger = logging.getLogger(__name__)


@dataclass
class SkillRetrievalConfig:
    """Configuration for skill retrieval.

    Defaults follow SkillRL §3.2 unless noted.

    Attributes:
        general_skills_max: Always-included general skills [SkillRL default: 6]
        task_specific_k: Top-K task-specific skills per query [SkillRL default: 6]
        min_similarity: Cosine floor for task-specific retrieval [SkillRL default: 0.4]
        min_confidence: Skip low-confidence skills [our addition]
        max_prompt_tokens: Token budget for injected skills [our addition]
    """

    general_skills_max: int = 6
    task_specific_k: int = 6
    min_similarity: float = 0.4
    min_confidence: float = 0.3
    max_prompt_tokens: int = 1500


@dataclass
class SkillRetrievalResult:
    """A single retrieved skill with metadata."""

    skill: Skill
    similarity: float  # 0.0 for general (not similarity-based)
    source: str  # "general" | "task_specific"


class SkillRetriever:
    """
    Retrieves and formats skills for prompt injection.

    Does NOT replace TwoPhaseRetriever. They serve different purposes:
    - TwoPhaseRetriever: raw memories + Q-values → HybridRouter routing decisions
    - SkillRetriever: compressed skill principles → prompt augmentation

    Usage:
        retriever = SkillRetriever(skill_bank)
        results = retriever.retrieve_for_task(embedding, "code_generation")
        prompt_section = retriever.format_for_prompt(results)
    """

    # Approximate tokens per character for budget enforcement
    _CHARS_PER_TOKEN = 4.0

    def __init__(
        self,
        skill_bank: SkillBank,
        config: Optional[SkillRetrievalConfig] = None,
    ):
        self.skill_bank = skill_bank
        self.config = config or SkillRetrievalConfig()

    def retrieve_for_task(
        self,
        task_embedding: np.ndarray,
        task_type: Optional[str] = None,
    ) -> List[SkillRetrievalResult]:
        """
        Two-level retrieval per SkillRL §3.2.

        Level 1: Load general skills (always included, sorted by confidence)
        Level 2: FAISS search for task-specific skills (filtered by similarity)

        Args:
            task_embedding: 1024-dim embedding of the task context
            task_type: Optional task type for filtering

        Returns:
            Combined list of SkillRetrievalResult (general first, then specific)
        """
        results: List[SkillRetrievalResult] = []

        # Level 1: General skills (always included)
        general_skills = self.skill_bank.get_skills(
            skill_type="general",
            deprecated=False,
            min_confidence=self.config.min_confidence,
            limit=self.config.general_skills_max,
        )
        for skill in general_skills:
            results.append(SkillRetrievalResult(
                skill=skill,
                similarity=1.0,  # General skills are always relevant
                source="general",
            ))

        # Level 2: Task-specific skills via FAISS
        if task_embedding is not None:
            specific_results = self.skill_bank.search_by_embedding(
                query_embedding=task_embedding,
                k=self.config.task_specific_k * 2,  # Over-fetch for filtering
                min_similarity=self.config.min_similarity,
                exclude_deprecated=True,
            )

            # Filter by confidence and task_type match
            seen_ids = {r.skill.id for r in results}
            specific_count = 0
            for skill, similarity in specific_results:
                if skill.id in seen_ids:
                    continue
                if skill.confidence < self.config.min_confidence:
                    continue
                if skill.skill_type == "general":
                    continue  # Already included above
                # Check task_type match
                if task_type and "*" not in skill.task_types and task_type not in skill.task_types:
                    continue

                results.append(SkillRetrievalResult(
                    skill=skill,
                    similarity=similarity,
                    source="task_specific",
                ))
                seen_ids.add(skill.id)
                specific_count += 1
                if specific_count >= self.config.task_specific_k:
                    break

        # Track retrievals
        skill_ids = [r.skill.id for r in results]
        if skill_ids:
            self.skill_bank.increment_retrieval(skill_ids)

        logger.debug(
            "Retrieved %d skills (%d general, %d specific) for task_type=%s",
            len(results),
            sum(1 for r in results if r.source == "general"),
            sum(1 for r in results if r.source == "task_specific"),
            task_type,
        )
        return results

    def format_for_prompt(self, results: List[SkillRetrievalResult]) -> str:
        """
        Format retrieved skills as a markdown prompt section.

        Respects max_prompt_tokens budget by truncating skills.

        Args:
            results: Retrieved skills from retrieve_for_task()

        Returns:
            Markdown string for prompt injection, or "" if no results
        """
        if not results:
            return ""

        max_chars = int(self.config.max_prompt_tokens * self._CHARS_PER_TOKEN)
        sections: dict[str, list[str]] = {
            "general": [],
            "routing": [],
            "escalation": [],
            "failure_lesson": [],
        }

        for r in results:
            entry = self._format_skill_entry(r)
            # Categorize by skill_type for prompt sections
            if r.source == "general":
                sections["general"].append(entry)
            elif r.skill.skill_type == "failure_lesson":
                sections["failure_lesson"].append(entry)
            elif r.skill.skill_type == "escalation":
                sections["escalation"].append(entry)
            else:
                sections["routing"].append(entry)

        # Build output with budget enforcement
        lines = ["## Learned Routing Skills\n"]
        total_chars = len(lines[0])

        section_headers = {
            "general": "### General Principles",
            "routing": "### Task-Specific Skills",
            "escalation": "### Escalation Avoidance",
            "failure_lesson": "### Failure Lessons",
        }

        for section_key, header in section_headers.items():
            entries = sections[section_key]
            if not entries:
                continue

            header_line = f"\n{header}\n"
            if total_chars + len(header_line) > max_chars:
                break

            lines.append(header_line)
            total_chars += len(header_line)

            for entry in entries:
                if total_chars + len(entry) > max_chars:
                    break
                lines.append(entry)
                total_chars += len(entry)

        output = "\n".join(lines).strip()
        return output if len(output) > len("## Learned Routing Skills") else ""

    def _format_skill_entry(self, result: SkillRetrievalResult) -> str:
        """Format a single skill as a markdown bullet."""
        s = result.skill
        return (
            f"- **{s.title}**: {s.principle}\n"
            f"  *Apply when*: {s.when_to_apply}\n"
        )
