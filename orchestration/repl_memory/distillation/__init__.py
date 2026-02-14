"""
Distillation pipeline: extract structured skills from raw trajectories.

Implements SkillRL §3.1 (Experience-based Skill Distillation) adapted for
our multi-teacher, offline-only workflow (no RL training loop).

Teachers analyze success/failure/escalation trajectories and produce
structured Skill records stored in the SkillBank.

Based on: SkillRL (Xia et al., arXiv:2602.08234, Feb 2026)
"""

from .teachers import TeacherModel, ClaudeTeacher, CodexTeacher, LocalLlamaTeacher
from .pipeline import DistillationPipeline, DistillationBatch, DistillationReport
from .failure_bridge import FailureBridge

__all__ = [
    "TeacherModel",
    "ClaudeTeacher",
    "CodexTeacher",
    "LocalLlamaTeacher",
    "DistillationPipeline",
    "DistillationBatch",
    "DistillationReport",
    "FailureBridge",
]
