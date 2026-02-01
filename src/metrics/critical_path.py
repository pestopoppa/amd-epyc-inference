"""Critical path computation for TaskIR plan DAGs.

Given per-step timing data and dependency edges, computes the longest
dependency chain (critical path), per-step slack, and parallelism ratio.

Usage:
    from src.metrics.critical_path import compute_critical_path, StepTiming

    timings = [
        StepTiming("S1", 2.0),
        StepTiming("S2", 3.0, depends_on=("S1",)),
        StepTiming("S3", 1.0, depends_on=("S1",)),
        StepTiming("S4", 2.0, depends_on=("S2", "S3")),
    ]
    report = compute_critical_path(timings, wall_clock_seconds=7.5)
    # report.critical_path_steps == ["S1", "S2", "S4"]
    # report.critical_path_seconds == 7.0
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StepTiming:
    """Timing data for a single plan step.

    Attributes:
        step_id: Step identifier (e.g. "S1").
        elapsed_seconds: How long this step took to execute.
        depends_on: Step IDs that must complete before this step.
    """

    step_id: str
    elapsed_seconds: float
    depends_on: tuple[str, ...] = ()


@dataclass
class CriticalPathReport:
    """Result of critical path analysis.

    Attributes:
        critical_path_steps: Ordered step IDs on the critical path (root → leaf).
        critical_path_seconds: Sum of elapsed times on the critical path.
        total_work_seconds: Sum of ALL step elapsed times.
        wall_clock_seconds: Actual end-to-end elapsed time.
        parallelism_ratio: total_work / wall_clock (>1 means parallelism helped).
        step_slack: Per-step slack in seconds (0 for critical-path steps).
    """

    critical_path_steps: list[str] = field(default_factory=list)
    critical_path_seconds: float = 0.0
    total_work_seconds: float = 0.0
    wall_clock_seconds: float = 0.0
    parallelism_ratio: float = 1.0
    step_slack: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON logging."""
        return {
            "critical_path_steps": self.critical_path_steps,
            "critical_path_seconds": round(self.critical_path_seconds, 3),
            "total_work_seconds": round(self.total_work_seconds, 3),
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "parallelism_ratio": round(self.parallelism_ratio, 3),
            "step_slack": {k: round(v, 3) for k, v in self.step_slack.items()},
        }


def compute_critical_path(
    timings: list[StepTiming],
    wall_clock_seconds: float = 0.0,
) -> CriticalPathReport:
    """Compute the critical path through a step DAG.

    Uses dynamic programming on the topologically-sorted DAG to find the
    longest dependency chain (the critical path).

    Args:
        timings: Per-step timing data with dependency edges.
        wall_clock_seconds: Actual wall-clock time for the entire plan
            execution.  Used to compute parallelism_ratio.

    Returns:
        CriticalPathReport with critical path, slack, and ratios.
    """
    if not timings:
        return CriticalPathReport()

    # Build lookup maps
    timing_map: dict[str, StepTiming] = {t.step_id: t for t in timings}
    all_ids = set(timing_map)

    # Validate dependency references
    for t in timings:
        for dep in t.depends_on:
            if dep not in all_ids:
                raise ValueError(
                    f"Step {t.step_id} depends on unknown step {dep}"
                )

    # Topological sort via Kahn's algorithm
    in_degree: dict[str, int] = {sid: 0 for sid in all_ids}
    dependents: dict[str, list[str]] = defaultdict(list)

    for t in timings:
        in_degree[t.step_id] = len(t.depends_on)
        for dep in t.depends_on:
            dependents[dep].append(t.step_id)

    topo_order: list[str] = []
    queue = [sid for sid in all_ids if in_degree[sid] == 0]
    queue.sort()  # Deterministic ordering

    while queue:
        sid = queue.pop(0)
        topo_order.append(sid)
        for child in sorted(dependents[sid]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(topo_order) != len(all_ids):
        cycle_members = all_ids - set(topo_order)
        raise ValueError(
            f"Circular dependency detected among steps: "
            f"{', '.join(sorted(cycle_members))}"
        )

    # DP forward pass: dp[s] = earliest finish time for s
    # dp[s] = elapsed[s] + max(dp[dep] for dep in depends_on[s])
    dp: dict[str, float] = {}
    predecessor: dict[str, str | None] = {}

    for sid in topo_order:
        t = timing_map[sid]
        if not t.depends_on:
            dp[sid] = t.elapsed_seconds
            predecessor[sid] = None
        else:
            # Find the dependency with the latest finish time
            best_dep = max(t.depends_on, key=lambda d: dp[d])
            dp[sid] = t.elapsed_seconds + dp[best_dep]
            predecessor[sid] = best_dep

    # Critical path length = max finish time
    critical_path_seconds = max(dp.values())

    # Backtrack to recover the critical path (leaf → root)
    end_step = max(dp, key=lambda s: dp[s])
    path_reversed: list[str] = []
    current: str | None = end_step
    while current is not None:
        path_reversed.append(current)
        current = predecessor[current]

    critical_path_steps = list(reversed(path_reversed))

    # Total work
    total_work = sum(t.elapsed_seconds for t in timings)

    # Backward pass for proper slack computation (CPM)
    # EF[s] = dp[s] (from forward pass)
    # ES[s] = EF[s] - elapsed[s]
    # LF[s] = min(LS[child] for child in successors[s]), or CP_length if leaf
    # LS[s] = LF[s] - elapsed[s]
    # Slack = LS[s] - ES[s] = LF[s] - EF[s]
    lf: dict[str, float] = {}
    for sid in reversed(topo_order):
        t = timing_map[sid]
        successors = dependents[sid]
        if not successors:
            lf[sid] = critical_path_seconds
        else:
            # LF = min(LS[child]) = min(LF[child] - elapsed[child])
            lf[sid] = min(
                lf[child] - timing_map[child].elapsed_seconds
                for child in successors
            )

    step_slack = {
        sid: round(lf[sid] - dp[sid], 6)
        for sid in all_ids
    }

    # Parallelism ratio
    if wall_clock_seconds > 0:
        parallelism_ratio = total_work / wall_clock_seconds
    else:
        # Fallback: use critical path as wall clock estimate
        parallelism_ratio = (
            total_work / critical_path_seconds
            if critical_path_seconds > 0
            else 1.0
        )

    return CriticalPathReport(
        critical_path_steps=critical_path_steps,
        critical_path_seconds=critical_path_seconds,
        total_work_seconds=total_work,
        wall_clock_seconds=wall_clock_seconds,
        parallelism_ratio=parallelism_ratio,
        step_slack=step_slack,
    )
