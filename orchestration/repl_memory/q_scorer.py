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

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embedder import TaskEmbedder
from .episodic_store import EpisodicStore
from .progress_logger import EventType, ProgressEntry, ProgressLogger, ProgressReader
from .staged_scorer import StagedQScorer

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for Q-scoring."""

    # Learning rate for Q-value updates
    learning_rate: float = 0.1

    # Reward values
    success_reward: float = 1.0
    failure_reward: float = -0.5
    partial_reward: float = 0.3

    # Temporal decay: Q-values decay toward neutral (0.5) over time.
    # decay_rate ^ days_elapsed is applied before each TD update.
    temporal_decay_rate: float = 0.99

    # Claude-as-Judge settings (optional)
    use_claude_judge: bool = False
    judge_model_path: Optional[Path] = None
    judge_binary: Optional[Path] = None

    # Cost-aware reward (xRouter-style correctness-gated cost penalty).
    # reward_final = quality_reward - lambda * max(0, cost_ratio - 1.0)
    # where cost_ratio = actual_elapsed / expected_elapsed.
    # Only applied when answer is correct (incorrect = 0.0, no cost term).
    cost_penalty_lambda: float = 0.15

    # Per-role optimized tokens/second from production benchmarks.
    # Used to normalize cost: expected_elapsed = tokens_generated / baseline_tps.
    baseline_tps_by_role: Dict[str, float] = field(default_factory=lambda: {
        "frontdoor": 18.3,
        "coder_escalation": 39.44,
        "architect_general": 6.75,
        "architect_coding": 10.3,
        "ingest_long_context": 6.29,
        "worker_explore": 27.88,
        "worker_math": 48.5,
        "worker_vision": 15.28,
        "vision_escalation": 27.6,
    })

    # Per-role quality baselines (from RESULTS.md relative benchmark scores).
    # Used for quality-gap penalty: penalize using expensive model when cheap suffices.
    baseline_quality_by_role: Dict[str, float] = field(default_factory=lambda: {
        "frontdoor": 0.895,
        "coder_escalation": 0.915,
        "architect_general": 0.94,
        "architect_coding": 0.885,
        "worker_explore": 0.745,
        "worker_math": 0.85,
        "worker_vision": 0.81,
    })

    # Per-role memory tier cost (normalized: 1.0 = HOT baseline ~20GB).
    # WARM tier models incur mmap load penalty and higher memory pressure.
    memory_cost_by_role: Dict[str, float] = field(default_factory=lambda: {
        "frontdoor": 1.0,         # 19GB, HOT
        "coder_escalation": 1.05, # 20GB, HOT
        "worker_explore": 0.5,    # 4.4GB, HOT
        "worker_math": 0.5,       # 4.4GB, HOT
        "architect_general": 3.0,  # 133GB, WARM (mmap load penalty)
        "architect_coding": 5.0,   # 271GB, WARM (huge)
        "ingest_long_context": 1.5, # 46GB, WARM
    })

    # Multi-dimensional cost weights (tunable).
    # cost_lambda_latency is the existing cost_penalty_lambda.
    cost_lambda_quality_gap: float = 0.10  # Penalize using higher-quality model than needed
    cost_lambda_memory: float = 0.05       # Penalize WARM tier when HOT sufficient

    # Delegation/teacher attribution shaping.
    delegation_misattribution_penalty: float = 0.10
    specialist_credit_bonus: float = 0.05
    teacher_regret_penalty: float = 0.20
    teacher_speedup_bonus: float = 0.05

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
        staged_scorer: Optional[StagedQScorer] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.logger = logger
        self.reader = reader
        self.config = config or ScoringConfig()
        self.staged_scorer = staged_scorer
        self._last_score_time: Optional[datetime] = None

    def score_pending_tasks(self) -> Dict[str, Any]:
        """
        Score all pending tasks from progress logs.

        Returns:
            Summary of scoring results
        """
        # Check minimum interval
        now = datetime.now(timezone.utc)
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
        plan_reviews = []

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
            elif entry.event_type == EventType.PLAN_REVIEWED:
                plan_reviews.append(entry)

        if not task_outcome:
            return {"error": "Task not completed yet"}

        # Compute reward (pass completion data as optional cost/telemetry metrics)
        reward = self._compute_reward(
            task_outcome,
            gate_results,
            escalations,
            plan_reviews,
            cost_metrics=(task_outcome.data if task_outcome and task_outcome.data else None),
        )
        # Delegation credit assignment to avoid over-crediting envelope roles.
        reward = self._apply_delegation_credit(reward, routing_decision, task_outcome)

        # Apply staged reward shaping if enabled (explore early, exploit later)
        if self.staged_scorer is not None:
            task_type = ""
            if task_started and task_started.data:
                task_type = task_started.data.get("task_type", "")
            action_str = ""
            if routing_decision and routing_decision.data:
                routing = routing_decision.data.get("routing", [])
                action_str = ",".join(routing) if isinstance(routing, list) else str(routing)
            if action_str and task_type:
                reward = self.staged_scorer.compute_staged_reward(
                    reward, action_str, task_type, self.store,
                )

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

    def _apply_delegation_credit(
        self,
        reward: float,
        routing_decision: Optional[ProgressEntry],
        task_outcome: ProgressEntry,
    ) -> float:
        """Adjust reward for delegation lineage attribution.

        Penalize envelope over-credit (architect routed but specialist produced final answer).
        Slightly bonus direct specialist attribution when selected specialist finished task.
        """
        if routing_decision is None:
            return reward
        routing = routing_decision.data.get("routing", [])
        if isinstance(routing, str):
            routed_roles = [r.strip() for r in routing.split(",") if r.strip()]
        elif isinstance(routing, list):
            routed_roles = [str(r) for r in routing]
        else:
            routed_roles = []
        final_role = str(task_outcome.data.get("final_answer_role", "") or "")
        if not routed_roles or not final_role:
            return reward

        routed_architect = any(r.startswith("architect_") for r in routed_roles)
        final_is_architect = final_role.startswith("architect_")

        if routed_architect and not final_is_architect:
            reward -= self.config.delegation_misattribution_penalty
        elif final_role in routed_roles and not final_is_architect:
            reward += self.config.specialist_credit_bonus

        return max(-1.0, min(1.0, reward))

    def _compute_reward(
        self,
        task_outcome: ProgressEntry,
        gate_results: List[ProgressEntry],
        escalations: List[ProgressEntry],
        plan_reviews: List[ProgressEntry] | None = None,
        cost_metrics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute reward from task outcome with optional cost penalty.

        Quality reward formula:
        - Base: success=1.0, failure=-0.5
        - Penalty for gate failures: -0.1 per failure
        - Penalty for escalations: -0.15 per escalation
        - Plan review bonus: +0.1 if approved, -0.2 if corrected

        Cost penalty (xRouter-style, correctness-gated):
        - Only applied when quality reward > 0 (correct answers)
        - cost_ratio = actual_elapsed / expected_elapsed
        - penalty = lambda * max(0, cost_ratio - 1.0)
        - No penalty if running at or above expected speed

        Args:
            cost_metrics: Optional dict with keys:
                - tokens_generated (int): tokens produced
                - elapsed_seconds (float): wall-clock time
                - role (str): specialist role name for baseline lookup
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

        # Plan review adjustments (architect-in-the-loop)
        plan_review_adj = 0.0
        if plan_reviews:
            for pr in plan_reviews:
                decision = pr.data.get("decision", "")
                if decision == "ok":
                    plan_review_adj += 0.1  # Approved — routing was correct
                else:
                    plan_review_adj -= 0.2  # Corrected — routing needed fixing

        reward = base_reward - gate_penalty - escalation_penalty + plan_review_adj

        # Cost penalty: only penalize correct answers that were slower than expected.
        # Incorrect answers already receive low/zero reward — no cost signal needed.
        if cost_metrics and reward > 0:
            tokens_gen = cost_metrics.get("tokens_generated", 0)
            role = cost_metrics.get("role", "")
            baseline_tps = self.config.baseline_tps_by_role.get(role, 0)

            # Prefer generation_ms (clean generation time excluding prompt eval)
            # over elapsed_seconds (polluted by prompt processing time)
            gen_ms = cost_metrics.get("generation_ms", 0)
            if gen_ms > 0:
                elapsed = gen_ms / 1000.0
            else:
                elapsed = cost_metrics.get("elapsed_seconds", 0)

            # Dimension 1: Latency penalty (existing)
            if baseline_tps > 0 and tokens_gen > 0 and elapsed > 0:
                expected_elapsed = tokens_gen / baseline_tps
                cost_ratio = elapsed / expected_elapsed  # >1 = slower than expected
                cost_penalty = self.config.cost_penalty_lambda * max(0.0, cost_ratio - 1.0)
                reward -= cost_penalty

            # Dimension 2: Quality gap penalty — penalize using expensive model
            # when a cheaper one could suffice. If worker (0.745) got it right,
            # architect (0.94) was wasteful. Gap is relative to worker baseline.
            if role in self.config.baseline_quality_by_role:
                model_quality = self.config.baseline_quality_by_role[role]
                quality_gap = max(0.0, model_quality - 0.75)  # 0.75 ≈ worker baseline
                reward -= self.config.cost_lambda_quality_gap * quality_gap

            # Dimension 3: Memory tier penalty — discourage WARM models when HOT
            # tier can handle the task. Reduces memory pressure on the system.
            if role in self.config.memory_cost_by_role:
                mem_cost = self.config.memory_cost_by_role[role]
                if mem_cost > 1.0:
                    reward -= self.config.cost_lambda_memory * (mem_cost - 1.0)

            # Teacher telemetry shaping (optional, when available in completion data).
            # regret = int(pass_teacher)-int(pass_chosen) in {0,1}
            regret = float(cost_metrics.get("regret", 0.0) or 0.0)
            reward -= self.config.teacher_regret_penalty * max(0.0, regret)

            speedup = float(
                cost_metrics.get("speedup_vs_teacher", cost_metrics.get("speedup", 1.0)) or 1.0
            )
            if speedup > 1.0:
                # Cap bonus contribution to avoid runaway high-speedup artifacts.
                reward += self.config.teacher_speedup_bonus * min(speedup - 1.0, 1.0)

        # Final reward (clamped to [-1, 1])
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
                    memory_id, reward, self.config.learning_rate,
                    temporal_decay_rate=self.config.temporal_decay_rate,
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
                    memory_id, reward, self.config.learning_rate,
                    temporal_decay_rate=self.config.temporal_decay_rate,
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

    def score_external_result(
        self,
        task_description: str,
        action: str,
        reward: float,
        context: Dict[str, Any] | None = None,
        embedding: List[float] | None = None,
    ) -> Dict[str, Any]:
        """Score an externally-evaluated result.

        Accepts pre-computed rewards from external scoring (e.g., the MemRL
        learning loop or debug scorer). Bypasses progress log reader and
        directly creates/updates episodic memory.

        Args:
            task_description: Description of the task.
            action: The action taken (e.g., "frontdoor:direct").
            reward: Pre-computed reward (-1.0 to 1.0).
            context: Additional context to store with the memory.
            embedding: Precomputed embedding for task_description (avoids re-embedding).

        Returns:
            Dict with memories_created and memories_updated counts.
        """
        result = {"memories_updated": 0, "memories_created": 0}
        context = context or {}

        # Clamp reward to valid range
        reward = max(-1.0, min(1.0, reward))

        # Use precomputed embedding or compute it
        if embedding is not None:
            emb_array = np.array(embedding, dtype=np.float32)
        else:
            task_ir = {
                "task_type": context.get("task_type", "chat"),
                "objective": task_description[:200],
            }
            emb_array = self.embedder.embed_task_ir(task_ir)

        # Search for existing similar memory with same action
        # Note: retrieve_by_similarity returns memories sorted by similarity
        similar = self.store.retrieve_by_similarity(
            query_embedding=emb_array,
            k=5,
            action_type="routing",
        )
        # Filter to high-similarity matches (similarity_score >= 0.85)
        similar = [m for m in similar if m.similarity_score >= 0.85]

        # Update existing memory if action matches closely
        updated = False
        for mem in similar:
            if mem.action == action or (
                hasattr(mem, "action") and mem.action.startswith(action.split(":")[0])
            ):
                old_q = mem.q_value
                new_q = self.store.update_q_value(
                    mem.id, reward, self.config.learning_rate,
                    temporal_decay_rate=self.config.temporal_decay_rate,
                )
                self.logger.log_memory_update(
                    mem.id, old_q, new_q, reward, "external"
                )
                result["memories_updated"] += 1
                updated = True
                break

        # Create new memory if no similar one found
        if not updated:
            initial_q = 0.5 + (reward * 0.5)
            context["task_description"] = task_description
            context["source"] = "external"

            memory_id = self.store.store(
                embedding=emb_array,
                action=action,
                action_type="routing",
                context=context,
                outcome="success" if reward > 0 else "failure",
                initial_q=initial_q,
            )

            self.logger.log(
                ProgressEntry(
                    event_type=EventType.MEMORY_STORED,
                    task_id="external",
                    memory_id=memory_id,
                    data={
                        "action_type": "routing",
                        "initial_q": initial_q,
                        "source": "external_score",
                    },
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
