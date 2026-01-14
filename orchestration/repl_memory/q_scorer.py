"""
QScorer: Async Q-value update agent for episodic memory.

Runs periodically (or on-demand) to:
1. Read progress logs for completed tasks
2. Compute rewards from outcomes
3. Update Q-values in the episodic store
4. Optionally run Claude-as-Judge for graded rewards

This implements the async scoring path from the MemRL architecture,
keeping Q-value computation off the critical inference path.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .embedder import TaskEmbedder
from .episodic_store import EpisodicStore
from .progress_logger import EventType, ProgressEntry, ProgressLogger, ProgressReader


@dataclass
class ScoringConfig:
    """Configuration for Q-scoring."""

    # Learning rate for Q-value updates
    learning_rate: float = 0.1

    # Reward values
    success_reward: float = 1.0
    failure_reward: float = -0.5
    partial_reward: float = 0.3

    # Claude-as-Judge settings (optional)
    use_claude_judge: bool = False
    judge_model_path: Optional[Path] = None
    judge_binary: Optional[Path] = None

    # Scoring frequency
    min_score_interval_seconds: int = 300  # 5 minutes

    # Batch size for processing
    batch_size: int = 50


class QScorer:
    """
    Async Q-value scoring agent.

    Workflow:
    1. Read progress logs for completed tasks
    2. For each task:
       a. Find associated memory entries
       b. Compute reward from outcome
       c. Update Q-values
    3. Log scoring events
    """

    def __init__(
        self,
        store: EpisodicStore,
        embedder: TaskEmbedder,
        logger: ProgressLogger,
        reader: ProgressReader,
        config: Optional[ScoringConfig] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.logger = logger
        self.reader = reader
        self.config = config or ScoringConfig()
        self._last_score_time: Optional[datetime] = None

    def score_pending_tasks(self) -> Dict[str, Any]:
        """
        Score all pending tasks from progress logs.

        Returns:
            Summary of scoring results
        """
        # Check minimum interval
        now = datetime.utcnow()
        if self._last_score_time:
            elapsed = (now - self._last_score_time).total_seconds()
            if elapsed < self.config.min_score_interval_seconds:
                return {
                    "skipped": True,
                    "reason": f"Too soon ({elapsed:.0f}s < {self.config.min_score_interval_seconds}s)",
                }

        # Find unscored tasks
        unscored_task_ids = self.reader.get_unscored_tasks()

        if not unscored_task_ids:
            return {"tasks_processed": 0, "message": "No pending tasks to score"}

        # Process in batches
        results = {
            "tasks_processed": 0,
            "memories_updated": 0,
            "memories_created": 0,
            "errors": [],
        }

        for task_id in unscored_task_ids[: self.config.batch_size]:
            try:
                task_result = self._score_task(task_id)
                results["tasks_processed"] += 1
                results["memories_updated"] += task_result.get("memories_updated", 0)
                results["memories_created"] += task_result.get("memories_created", 0)
            except Exception as e:
                results["errors"].append({"task_id": task_id, "error": str(e)})

        self._last_score_time = now
        self.logger.flush()

        return results

    def _score_task(self, task_id: str) -> Dict[str, Any]:
        """
        Score a single task.

        Args:
            task_id: Task ID to score

        Returns:
            Scoring results for this task
        """
        # Get task trajectory
        trajectory = self.reader.get_task_trajectory(task_id)

        if not trajectory:
            return {"error": "No trajectory found"}

        # Extract key events
        task_started = None
        routing_decision = None
        task_outcome = None
        gate_results = []
        escalations = []

        for entry in trajectory:
            if entry.event_type == EventType.TASK_STARTED:
                task_started = entry
            elif entry.event_type == EventType.ROUTING_DECISION:
                routing_decision = entry
            elif entry.event_type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
                task_outcome = entry
            elif entry.event_type in (EventType.GATE_PASSED, EventType.GATE_FAILED):
                gate_results.append(entry)
            elif entry.event_type == EventType.ESCALATION_TRIGGERED:
                escalations.append(entry)

        if not task_outcome:
            return {"error": "Task not completed yet"}

        # Compute reward
        reward = self._compute_reward(task_outcome, gate_results, escalations)

        result = {
            "memories_updated": 0,
            "memories_created": 0,
            "reward": reward,
        }

        # Update or create routing memory
        if routing_decision:
            memory_result = self._update_routing_memory(
                task_id,
                task_started,
                routing_decision,
                reward,
            )
            result.update(memory_result)

        # Update escalation memories
        for escalation in escalations:
            esc_result = self._update_escalation_memory(task_id, escalation, reward)
            result["memories_updated"] += esc_result.get("memories_updated", 0)
            result["memories_created"] += esc_result.get("memories_created", 0)

        return result

    def _compute_reward(
        self,
        task_outcome: ProgressEntry,
        gate_results: List[ProgressEntry],
        escalations: List[ProgressEntry],
    ) -> float:
        """
        Compute reward from task outcome.

        Reward formula:
        - Base: success=1.0, failure=-0.5
        - Penalty for gate failures: -0.1 per failure
        - Penalty for escalations: -0.15 per escalation
        """
        if task_outcome.outcome == "success":
            base_reward = self.config.success_reward
        elif task_outcome.outcome == "partial":
            base_reward = self.config.partial_reward
        else:
            base_reward = self.config.failure_reward

        # Gate failure penalties
        gate_failures = sum(1 for g in gate_results if g.event_type == EventType.GATE_FAILED)
        gate_penalty = gate_failures * 0.1

        # Escalation penalties (unnecessary escalations are wasteful)
        escalation_penalty = len(escalations) * 0.15

        # Final reward (clamped to [-1, 1])
        reward = base_reward - gate_penalty - escalation_penalty
        return max(-1.0, min(1.0, reward))

    def _update_routing_memory(
        self,
        task_id: str,
        task_started: Optional[ProgressEntry],
        routing_decision: ProgressEntry,
        reward: float,
    ) -> Dict[str, Any]:
        """Update or create routing memory."""
        result = {"memories_updated": 0, "memories_created": 0}

        # Check if memory already exists
        memory_id = routing_decision.memory_id

        if memory_id:
            # Update existing memory
            memory = self.store.get_by_id(memory_id)
            if memory:
                old_q = memory.q_value
                new_q = self.store.update_q_value(
                    memory_id, reward, self.config.learning_rate
                )
                self.logger.log_memory_update(memory_id, old_q, new_q, reward, task_id)
                result["memories_updated"] = 1
        else:
            # Create new memory from this routing decision
            if task_started and task_started.data:
                task_context = {
                    "task_type": task_started.data.get("task_type"),
                    "objective": task_started.data.get("objective"),
                    "priority": task_started.data.get("priority"),
                }

                # Generate embedding for task context
                embedding = self.embedder.embed_task_ir(task_context)

                # Store new memory
                routing = routing_decision.data.get("routing", [])
                action = ",".join(routing) if isinstance(routing, list) else str(routing)

                # Initial Q-value based on first observation
                initial_q = 0.5 + (reward * 0.5)  # Map reward to [0, 1]

                memory_id = self.store.store(
                    embedding=embedding,
                    action=action,
                    action_type="routing",
                    context=task_context,
                    outcome="success" if reward > 0 else "failure",
                    initial_q=initial_q,
                )

                self.logger.log(
                    ProgressEntry(
                        event_type=EventType.MEMORY_STORED,
                        task_id=task_id,
                        memory_id=memory_id,
                        data={"action_type": "routing", "initial_q": initial_q},
                    )
                )
                result["memories_created"] = 1

        return result

    def _update_escalation_memory(
        self,
        task_id: str,
        escalation: ProgressEntry,
        reward: float,
    ) -> Dict[str, Any]:
        """Update or create escalation memory."""
        result = {"memories_updated": 0, "memories_created": 0}

        memory_id = escalation.memory_id

        if memory_id:
            # Update existing memory
            memory = self.store.get_by_id(memory_id)
            if memory:
                old_q = memory.q_value
                new_q = self.store.update_q_value(
                    memory_id, reward, self.config.learning_rate
                )
                self.logger.log_memory_update(memory_id, old_q, new_q, reward, task_id)
                result["memories_updated"] = 1
        else:
            # Create new escalation memory
            failure_context = {
                "from_tier": escalation.data.get("from_tier"),
                "to_tier": escalation.data.get("to_tier"),
                "reason": escalation.data.get("reason"),
            }

            embedding = self.embedder.embed_failure_context(failure_context)
            action = f"escalate:{escalation.data.get('from_tier')}->{escalation.data.get('to_tier')}"

            initial_q = 0.5 + (reward * 0.5)

            memory_id = self.store.store(
                embedding=embedding,
                action=action,
                action_type="escalation",
                context=failure_context,
                outcome="success" if reward > 0 else "failure",
                initial_q=initial_q,
            )

            self.logger.log(
                ProgressEntry(
                    event_type=EventType.MEMORY_STORED,
                    task_id=task_id,
                    memory_id=memory_id,
                    data={"action_type": "escalation", "initial_q": initial_q},
                )
            )
            result["memories_created"] = 1

        return result


class ClaudeAsJudge:
    """
    Claude-as-Judge scoring for orchestrator quality.

    Provides graded rewards (0-3) instead of binary success/failure.
    Used optionally for richer Q-value updates.
    """

    def __init__(
        self,
        model_path: Path,
        binary_path: Path,
        threads: int = 8,
        timeout: int = 60,
    ):
        self.model_path = model_path
        self.binary_path = binary_path
        self.threads = threads
        self.timeout = timeout

    def score_routing(
        self,
        task_ir: Dict[str, Any],
        routing_decision: List[str],
        outcome: str,
    ) -> Tuple[int, str]:
        """
        Score a routing decision.

        Args:
            task_ir: Original TaskIR
            routing_decision: Routing decision made
            outcome: Task outcome

        Returns:
            (score, reason) tuple where score is 0-3
        """
        prompt = self._build_routing_prompt(task_ir, routing_decision, outcome)
        response = self._call_model(prompt)
        return self._parse_score(response)

    def score_plan(
        self,
        task_ir: Dict[str, Any],
        plan: Dict[str, Any],
        outcome: str,
    ) -> Tuple[int, str]:
        """
        Score a task plan.

        Args:
            task_ir: Original TaskIR
            plan: Plan generated
            outcome: Task outcome

        Returns:
            (score, reason) tuple where score is 0-3
        """
        prompt = self._build_plan_prompt(task_ir, plan, outcome)
        response = self._call_model(prompt)
        return self._parse_score(response)

    def _build_routing_prompt(
        self,
        task_ir: Dict[str, Any],
        routing_decision: List[str],
        outcome: str,
    ) -> str:
        """Build Claude-as-Judge prompt for routing evaluation."""
        return f"""You are evaluating the quality of a task routing decision.

TASK:
- Type: {task_ir.get('task_type')}
- Objective: {task_ir.get('objective', '')[:500]}
- Priority: {task_ir.get('priority')}

ROUTING DECISION: {', '.join(routing_decision)}

OUTCOME: {outcome}

Score the routing decision from 0-3:
3 = Perfect specialist selection for this task type
2 = Acceptable routing, could be optimized
1 = Suboptimal routing that likely hurt performance
0 = Completely wrong routing choice

Respond with exactly:
SCORE: <0-3>
REASON: <brief explanation>"""

    def _build_plan_prompt(
        self,
        task_ir: Dict[str, Any],
        plan: Dict[str, Any],
        outcome: str,
    ) -> str:
        """Build Claude-as-Judge prompt for plan evaluation."""
        steps = plan.get("steps", [])
        steps_str = "\n".join(
            f"  {s.get('id')}: {s.get('action')}" for s in steps[:10]
        )

        return f"""You are evaluating the quality of a task execution plan.

TASK:
- Type: {task_ir.get('task_type')}
- Objective: {task_ir.get('objective', '')[:500]}

PLAN STEPS:
{steps_str}

OUTCOME: {outcome}

Score the plan from 0-3:
3 = Complete, correctly ordered steps that address all requirements
2 = Mostly complete plan, missing 1-2 steps or minor ordering issues
1 = Major gaps in plan or incorrect dependencies
0 = Incoherent or completely wrong plan

Respond with exactly:
SCORE: <0-3>
REASON: <brief explanation>"""

    def _call_model(self, prompt: str) -> str:
        """Call the judge model."""
        try:
            result = subprocess.run(
                [
                    str(self.binary_path),
                    "-m", str(self.model_path),
                    "-p", prompt,
                    "-n", "100",
                    "--temp", "0",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return "SCORE: 1\nREASON: Judge model timed out"
        except Exception as e:
            return f"SCORE: 1\nREASON: Judge model error: {e}"

    def _parse_score(self, response: str) -> Tuple[int, str]:
        """Parse score from model response."""
        score = 1  # Default to middle-low
        reason = "Could not parse response"

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = int(line.split(":")[1].strip())
                    score = max(0, min(3, score))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip() if ":" in line else reason

        return (score, reason)
