# Chapter 25: Cost-Aware Reward Design for Model Routing

## Introduction

Prior to this work, the MemRL system used binary pass/fail rewards for routing decisions. When two specialists both answered correctly, they received identical rewards (+0.3), regardless of whether one took 6 seconds at 18 t/s or 294 seconds at 6.75 t/s. The learned policy could not distinguish efficiency.

This chapter documents the transition to a correctness-gated cost penalty, aligning with industry standard practice established by xRouter (Salesforce, 2025), RouteLLM (LMSYS, ICLR 2025), and related work.

## Problem Statement

The MemRL Q-value scoring system learns routing preferences through TD-style updates:

```
Q_new = Q_old + alpha * (reward - Q_old)
```

With flat rewards, the system converges to "any specialist that gets the answer right is equally good." In a multi-model stack where inference costs vary by 13.4x (frontdoor at 18.3 t/s vs architect_general at 6.75 t/s), this is a significant blind spot.

### Concrete Example

Consider a factual question correctly answered by both `frontdoor:direct` and `architect_general:direct`:

| Specialist | Correct | Elapsed | Tokens | t/s | Old Reward |
|---|---|---|---|---|---|
| frontdoor:direct | Yes | 10s | 183 | 18.3 | +0.3 |
| architect_general:direct | Yes | 100s | 675 | 6.75 | +0.3 |

Same reward despite 10x latency difference. The routing policy learns no preference. With cost-aware rewards:

| Specialist | Correct | cost_ratio | Penalty | New Reward |
|---|---|---|---|---|
| frontdoor:direct | Yes | 1.0 (at speed) | 0.0 | +0.50 |
| architect_general:direct | Yes | 1.0 (at speed) | 0.0 | +0.50 |

Both at expected speed — equal. But if the architect is overloaded:

| Specialist | Correct | cost_ratio | Penalty | New Reward |
|---|---|---|---|---|
| frontdoor:direct | Yes | 1.0 | 0.0 | +0.50 |
| architect_general:direct | Yes | 2.0 (2x slow) | 0.15 | +0.35 |

Now the router learns: for this task type, frontdoor is preferable when both can answer correctly.

## Literature Review

### xRouter (Salesforce AI Research, October 2025)

xRouter casts LLM routing as sequential decision-making and trains a 7B router end-to-end with reinforcement learning. The key contribution is an explicit cost-aware reward:

```
reward = quality - lambda * normalized_cost
```

**Correctness gate**: if the answer is wrong, reward = 0 regardless of cost. This prevents the optimizer from "gaming" rewards by answering quickly but incorrectly.

Three variants (xRouter-7B-1/2/3) are trained with different lambda values to explore the Pareto frontier. On Olympiad Bench, xRouter-7B-2 achieves near GPT-5 accuracy while using ~1/8 the evaluation cost.

**Reference**: Cheng Qian et al., "xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning," arXiv:2510.08439 (2025). https://arxiv.org/abs/2510.08439

### RouteLLM (LMSYS, ICLR 2025)

RouteLLM learns routers from preference data to select between a strong and weak model. A threshold alpha controls the cost-quality tradeoff: higher alpha favors cheaper models. Achieves 85%+ cost reduction on MT Bench while maintaining 95% of GPT-4 quality.

Four router architectures explored: similarity-weighted ranking, matrix factorization, BERT classifier, and causal LLM classifier. Matrix factorization models the scoring function as bilinear over model and query embeddings.

**Reference**: Isaac Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," ICLR 2025. https://openreview.net/forum?id=8sSqNntaMr

### Router-R1 (June 2025)

Combines routing with RL and multi-round aggregation. Reward function has three components: format reward, final outcome reward, and cost reward penalizing expensive model usage.

**Reference**: Wang et al., "Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning," arXiv:2506.09033 (2025). https://arxiv.org/abs/2506.09033

### FrugalGPT (Stanford, 2023)

Foundational cascade approach: query the cheapest model first, accept if confidence exceeds a learned threshold, otherwise escalate. Matches GPT-4 quality with up to 98% cost reduction.

**Reference**: Lingjiao Chen, Matei Zaharia, James Zou, "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance," arXiv:2305.05176 (2023). https://arxiv.org/abs/2305.05176

### Cost-Spectrum Contrastive Routing (August 2025)

Uses a novel contrastive objective (Cost-Spectrum InfoNCE) that selects correct positives within adaptive cost bands, temperature-scales each band, and down-weights negatives proportional to cost. Improves accuracy-cost tradeoff by up to 25%.

**Reference**: "Cost-Aware Contrastive Routing for LLMs," arXiv:2508.12491 (2025). https://arxiv.org/html/2508.12491

### LLMRank (October 2025)

Feature-driven routing that embeds model cost directly into training. Incorporates cost-aware routing by learning flexible deployment strategies that trade accuracy for efficiency.

**Reference**: "LLMRank: Understanding LLM Strengths for Model Routing," arXiv:2510.01234 (2025). https://arxiv.org/html/2510.01234

### Self-REF (October 2024)

Confidence-token routing: LoRA-finetuned 8B models learn to express uncertainty, routing only uncertain queries to 70B models. Achieves comparable system-level performance to non-finetuned 70B while dramatically reducing per-query cost.

**Reference**: "Learning to Route LLMs with Confidence Tokens," arXiv:2410.13284 (2024). https://arxiv.org/html/2410.13284v2

## Industry Consensus

All major routing systems converge on the same pattern:

1. **Correctness is a hard gate.** Wrong answers receive zero or negative reward, never adjusted by cost.
2. **Cost is a penalty term on correct answers.** The reward formula is `quality - lambda * normalized_cost`.
3. **Lambda is tunable.** Different operating points on the Pareto frontier serve different use cases (interactive vs batch).
4. **Normalization is essential.** Raw token counts or wall-clock times are meaningless across model sizes. Cost must be normalized relative to a baseline.

Our system maps directly to this pattern because it operates a heterogeneous model pool with known per-role speed baselines.

## Our Implementation

### Reward Formula

```
reward = quality_base - lambda * max(0, cost_ratio - 1.0)
```

Where:
- `quality_base` = existing quality reward (success=1.0, partial=0.3, minus gate/escalation penalties)
- `cost_ratio` = actual_elapsed / expected_elapsed
- `expected_elapsed` = tokens_generated / baseline_tps[role]
- `lambda` = 0.15 (configurable via `model_registry.yaml`)
- Result clamped to [-1.0, 1.0]

### Correctness Gate

Cost penalty is **only applied when reward > 0** (correct answers). Incorrect answers already receive low/zero reward — adding a cost penalty would be double-counting. This matches xRouter's design: "If the final answer is incorrect, the trajectory receives zero reward regardless of cost."

### Cost Normalization

We normalize by **expected elapsed time** rather than raw tokens or raw seconds:

```
expected_elapsed = tokens_generated / baseline_tps[role]
cost_ratio = actual_elapsed / expected_elapsed
```

This means:
- **cost_ratio = 1.0**: running at expected speed (no penalty)
- **cost_ratio < 1.0**: faster than expected (no penalty — `max(0, ...)` gates this)
- **cost_ratio > 1.0**: slower than expected (penalty proportional to slowdown)

Why this normalization:
- Raw elapsed is meaningless across roles (architect at 6.75 t/s vs frontdoor at 18.3 t/s)
- Normalizing by role baseline measures "did this specialist perform at its expected speed?" not "is this model inherently slow?"
- A slow model running at its expected speed is not penalized — the routing decision chose an appropriate model
- Penalty only fires when something is wrong (overloaded server, contention, thermal throttling)

### Baseline TPS Values

Per-role optimized speeds from production benchmarks (model_registry.yaml):

| Role | Model | Optimized t/s |
|---|---|---|
| frontdoor | Qwen3-Coder-30B-A3B Q4_K_M | 18.3 |
| coder_primary | Qwen3-Coder-30B-A3B Q4_K_M | 18.3 |
| coder_escalation | Qwen2.5-Coder-32B Q4_K_M + spec | 39.44 |
| architect_general | Qwen3-235B-A22B Q4_K_M | 6.75 |
| architect_coding | Qwen3-Coder-480B-A35B Q4_K_M | 10.3 |
| ingest_long_context | Qwen3-Next-80B-A3B Q4_K_M | 6.29 |
| worker_explore | Qwen2.5-7B-Instruct f16 | 27.88 |
| worker_math | Qwen2.5-7B-Instruct f16 + spec | 48.5 |
| worker_vision | Qwen2.5-VL-7B Q4_K_M | 15.28 |
| vision_escalation | Qwen3-VL-30B-A3B Q4_K_M | 27.6 |

### Configuration

In `model_registry.yaml`:

```yaml
repl_memory:
  scoring:
    cost_penalty_lambda: 0.15
    failure_reward: 0.0  # xRouter: incorrect = zero (was -0.5)
```

In `q_scorer.py`, `ScoringConfig` carries both `cost_penalty_lambda` and `baseline_tps_by_role` with defaults matching production values.

### Affected Code Paths

1. **Live Q-scoring** (`q_scorer.py::_compute_reward`): accepts optional `cost_metrics` dict with `tokens_generated`, `elapsed_seconds`, `role`.
2. **Seeding script** (`seed_specialist_routing.py::compute_comparative_rewards`): uses `RoleResult.tokens_generated` and `elapsed_seconds` already captured from API responses.
3. **Model registry** (`model_registry.yaml`): config source for lambda and failure_reward.

## Design Decisions

### Why lambda = 0.15

At lambda=0.15:
- Running at expected speed: no penalty
- 2x slower: penalty = 0.15 (reduces reward from 1.0 to 0.85)
- 5x slower: penalty = 0.60 (reduces reward from 1.0 to 0.40)
- 10x slower: penalty = 1.35 (clamped to -1.0 from quality base)

This creates meaningful differentiation without overwhelming quality signal. The routing system still strongly prefers correct answers over fast-but-wrong ones.

### Why failure_reward changed from -0.5 to 0.0

The xRouter pattern uses 0 for incorrect, not negative. This simplifies the reward space: positive = valuable, zero = not valuable, negative = harmful (escalation failures, gate violations). With -0.5 for failures, the Q-value update pushes memories well below neutral (0.5), making recovery slow even after a single bad result.

### Why `max(0, cost_ratio - 1.0)` not raw cost_ratio

Running faster than expected should not grant bonus reward — it's just normal operation. The penalty only activates for degraded performance. This prevents rewarding a specialist simply for having high baseline speed.

### Why floor at 0.1 in seeding

In the comparative seeding script, correct answers from any specialist get at least reward=0.1, even if extremely slow. This preserves the signal "this specialist can solve this task type" which is more valuable than "this specialist is fast." A slow-but-capable specialist is still useful for escalation.

## Expected Impact

With cost-aware rewards, MemRL will learn:

1. **For simple tasks**: prefer frontdoor (18.3 t/s, correct) over architect_general (6.75 t/s, also correct)
2. **For hard tasks**: prefer architect even though slower — only architect gets it right (+1.0 vs 0.0)
3. **Under contention**: penalize overloaded specialists (cost_ratio > 1.0), route to available ones
4. **Mode preferences**: ReAct with tools might be slower but more reliable — cost penalty balances this

## Mode-Advantage Task Enrichment (February 2026)

The cost-aware reward system is most effective when combined with tasks that produce strong comparative signal. The mode-advantage suite (90 tasks, see [Chapter 24](24-benchmark-suite-construction.md)) was specifically designed for this:

- **Computation-gated tasks** (15): Models hallucinate on modular arithmetic, but react mode with Python gets the correct answer. Cost penalty doesn't apply — react wins on correctness (+1.0).
- **Iterative-fix tasks** (15 + 30 SWE): REPL mode runs tests iteratively; direct mode patches blind. When both fail, cost penalty differentiates (-0.3 vs -0.3 - cost).
- **Multi-step tasks** (15): Chained calculations where cost penalty matters — delegation is slower but more reliable.
- **Escalation-gated tasks** (15): Specialist wins on correctness (+1.0); cost penalty is irrelevant because frontdoor fails entirely (0.0).

Three HuggingFace dataset adapters (GAIA, CRUXEval, BigCodeBench) provide additional volume where tool-use modes structurally outperform direct inference.

## Binary Rewards for Faithful Probability Estimation (February 2026)

### The Problem with Cost-Weighted Q-Values

The cost-aware reward system described above has a subtle issue: Q-values conflate two signals:
- **P(success|action)** - the probability that this action succeeds
- **Cost efficiency** - how expensive this action is

For Optuna threshold tuning, we need these signals separated. If Q-values incorporate cost during learning, we can't later re-tune cost-quality tradeoffs without retraining.

### Binary Reward Solution

The 3-way evaluation mode uses pure binary rewards for Q-value updates:

```python
def success_reward(passed: bool) -> float:
    """Binary reward for faithful probability estimation."""
    return 1.0 if passed else 0.0
```

**Why binary?**
- TD update: `Q_new = Q_old + α(reward - Q_old)`
- With binary rewards (1.0/0.0) and α=0.1, Q converges to empirical success rate
- Q-values become faithful P(success) estimates

### Cost Stored Separately

Cost metrics are stored in episodic memory context, not in rewards:

```python
metadata["cost_metrics"] = {
    "SELF:direct": {
        "elapsed_seconds": 2.3,
        "tokens_generated": 150,
        "predicted_tps": 18.3,
    },
    "SELF:repl": {
        "elapsed_seconds": 5.1,
        "tokens_generated": 280,
        "tools_used": 3,
    },
    "ARCHITECT": {
        "elapsed_seconds": 45.2,
        "tokens_generated": 1200,
        "role_history": ["architect_coding", "worker_explore"],
    },
}
```

### Cost Applied at Routing Time

At production routing time, cost is applied to Q-values:

```python
COST_TIER = {"SELF:direct": 2, "SELF:repl": 2, "ARCHITECT": 4, "WORKER": 1}

def route_with_cost(q_values: dict[str, float]) -> str:
    scores = {action: q / COST_TIER[action] for action, q in q_values.items()}
    return max(scores, key=scores.get)
```

### Optuna Threshold Optimization (Future)

With separated Q-values and cost metrics, Optuna can optimize:
- Cost tier weights per task type
- Confidence thresholds for each action
- Lambda values for cost-quality tradeoff

The stored cost metrics provide the ground truth for optimization without corrupting the Q-value estimates.

---

## Future Work

1. **Dynamic lambda by task priority**: interactive queries use higher lambda (latency-sensitive); batch uses lower lambda (quality-sensitive).

2. **Multi-objective Pareto frontier**: instead of scalarizing quality and cost, maintain a Pareto set and select based on current system state (load, queue depth).

3. **Claude-as-Judge integration**: the `ClaudeAsJudge` class already exists in q_scorer.py but is disabled. Graded quality scores (0-3) combined with cost penalty would provide much richer signal than binary pass/fail + cost.

4. **Token-level cost accounting**: distinguish prompt tokens (cheap) from completion tokens (expensive). Completion-heavy responses cost more compute per token.

5. **Cache-aware cost**: with RadixAttention, cache hits reduce real cost. A specialist that benefits from cached context should get lower effective cost.

## References

1. Cheng Qian et al., "xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning," arXiv:2510.08439 (2025). https://arxiv.org/abs/2510.08439
2. Isaac Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," ICLR 2025. https://openreview.net/forum?id=8sSqNntaMr
3. Wang et al., "Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning," arXiv:2506.09033 (2025). https://arxiv.org/abs/2506.09033
4. Lingjiao Chen, Matei Zaharia, James Zou, "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance," arXiv:2305.05176 (2023). https://arxiv.org/abs/2305.05176
5. "Cost-Aware Contrastive Routing for LLMs," arXiv:2508.12491 (2025). https://arxiv.org/html/2508.12491
6. "LLMRank: Understanding LLM Strengths for Model Routing," arXiv:2510.01234 (2025). https://arxiv.org/html/2510.01234
7. "Learning to Route LLMs with Confidence Tokens," arXiv:2410.13284 (2024). https://arxiv.org/html/2410.13284v2
