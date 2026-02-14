#!/usr/bin/env python3
"""Bootstrap SkillBank from episodic memory or progress logs.

CLI script — NOT auto-run during seeding. User runs explicitly to populate
the SkillBank with distilled skills from historical trajectories.

Usage:
  python scripts/skillbank/seed_skills.py --teacher claude --max-trajectories 200
  python scripts/skillbank/seed_skills.py --teacher codex --from-progress-logs --days 14
  python scripts/skillbank/seed_skills.py --teacher mock --max-trajectories 5  # dry-run

Based on: SkillRL §3.1 (Experience-based Skill Distillation)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _mem_to_trajectory(mem) -> dict:
    """Convert a MemoryEntry to a trajectory dict for distillation."""
    ctx = mem.context if isinstance(mem.context, dict) else {}
    return {
        "task_id": mem.id,
        "task_type": ctx.get("task_type", "general"),
        "objective": ctx.get("task_description", mem.action),
        "routing_decision": mem.action,
        "outcome": mem.outcome or "unknown",
        "escalations": [],
        "cost_metrics": {},
    }


def _traj_to_dict(traj) -> dict:
    """Convert a Trajectory dataclass to a trajectory dict for distillation."""
    return {
        "task_id": traj.task_id,
        "task_type": traj.task_type,
        "objective": traj.objective,
        "routing_decision": traj.routing_decision,
        "outcome": traj.outcome,
        "escalations": traj.escalations,
        "cost_metrics": traj.cost_metrics,
    }


def _build_teacher(teacher_name: str):
    """Instantiate the requested teacher model."""
    from orchestration.repl_memory.distillation.teachers import (
        ClaudeTeacher,
        CodexTeacher,
        MockTeacher,
    )

    if teacher_name == "claude":
        return ClaudeTeacher()
    elif teacher_name == "codex":
        return CodexTeacher()
    elif teacher_name == "mock":
        return MockTeacher()
    else:
        raise ValueError(f"Unknown teacher: {teacher_name}. Use claude, codex, or mock.")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap SkillBank from historical trajectories",
    )
    parser.add_argument(
        "--teacher", required=True, choices=["claude", "codex", "mock"],
        help="Teacher model for distillation (claude=Opus 4.6, codex=gpt-5.3, mock=testing)",
    )
    parser.add_argument(
        "--max-trajectories", type=int, default=200,
        help="Maximum trajectories to distill (default: 200)",
    )
    parser.add_argument(
        "--from-progress-logs", action="store_true",
        help="Extract trajectories from progress logs (via TrajectoryExtractor)",
    )
    parser.add_argument(
        "--days", type=int, default=25,
        help="Days of progress logs to scan (default: 25, used with --from-progress-logs)",
    )
    parser.add_argument(
        "--with-failure-context", action="store_true",
        help="Enrich distillation with FailureGraph context",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="SkillBank database path (default: sessions/skills.db)",
    )
    args = parser.parse_args()

    # Initialize SkillBank
    from orchestration.repl_memory.skill_bank import SkillBank

    db_path = Path(args.db_path) if args.db_path else None
    sb = SkillBank(db_path=db_path)
    logger.info("SkillBank initialized: %d existing skills", sb.count())

    # Build teacher
    teacher = _build_teacher(args.teacher)
    logger.info("Teacher: %s", teacher.model_id)

    # Extract trajectories
    trajectories: list[dict] = []

    if args.from_progress_logs:
        from orchestration.repl_memory.replay.trajectory import TrajectoryExtractor
        from orchestration.repl_memory.progress_logger import ProgressReader

        reader = ProgressReader()
        extractor = TrajectoryExtractor(reader)
        raw = extractor.extract_complete(days=args.days, max_trajectories=args.max_trajectories)
        trajectories = [_traj_to_dict(t) for t in raw]
        logger.info("Extracted %d trajectories from progress logs (%d days)", len(trajectories), args.days)
    else:
        # Default: pull highest-Q memories from EpisodicStore
        import numpy as np
        from orchestration.repl_memory.episodic_store import EpisodicStore

        store = EpisodicStore()
        zero_vec = np.zeros(1024, dtype=np.float32)
        memories = store.retrieve_by_similarity(
            zero_vec, k=args.max_trajectories, min_q_value=0.7,
        )
        trajectories = [_mem_to_trajectory(m) for m in memories]
        logger.info("Retrieved %d high-Q memories from EpisodicStore", len(trajectories))

    if not trajectories:
        logger.warning("No trajectories found. Nothing to distill.")
        sb.close()
        return

    # Failure context enrichment
    failure_summary = "No FailureGraph context available."
    if args.with_failure_context:
        try:
            from orchestration.repl_memory.distillation.failure_bridge import FailureBridge
            from src.api.state import app_state

            if hasattr(app_state, "failure_graph") and app_state.failure_graph:
                bridge = FailureBridge(app_state.failure_graph, sb)
                failure_summary = bridge.get_failure_context_for_distillation()
                logger.info("FailureGraph context loaded")
        except Exception as e:
            logger.warning("FailureGraph context unavailable: %s", e)

    # Run distillation
    from orchestration.repl_memory.distillation.pipeline import DistillationPipeline

    pipeline = DistillationPipeline(teacher=teacher, skill_bank=sb)
    report = asyncio.run(pipeline.run(trajectories, failure_summary))

    # Print report
    print(f"\n{'='*60}")
    print("DISTILLATION REPORT")
    print(f"{'='*60}")
    print(f"  Teacher:             {teacher.model_id}")
    print(f"  Total trajectories:  {report.total_trajectories}")
    print(f"    Success:           {report.success_trajectories}")
    print(f"    Failure:           {report.failure_trajectories}")
    print(f"    Escalation:        {report.escalation_trajectories}")
    print(f"  Batches processed:   {report.batches_processed}")
    print(f"  Skills proposed:     {report.skills_proposed}")
    print(f"  Skills stored:       {report.skills_stored}")
    print(f"  Skills merged:       {report.skills_merged}")
    print(f"  Skills rejected:     {report.skills_rejected}")
    print(f"  Duration:            {report.duration_seconds:.1f}s")
    if report.errors:
        print(f"  Errors:              {len(report.errors)}")
        for err in report.errors[:5]:
            print(f"    - {err}")
    print(f"\n  SkillBank total:     {sb.count()} skills")
    print(f"{'='*60}")

    sb.close()


if __name__ == "__main__":
    main()
