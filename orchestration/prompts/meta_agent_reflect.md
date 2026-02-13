# Memory Design Reflection

You are a meta-agent tasked with evolving the orchestration memory configuration.
Analyze the current config, archive history, and trajectory statistics below,
then propose 2-3 new candidate configurations.

## Current Production Config

{config}

## Archive History (Top + Worst + Random)

{archive}

## Recent Trajectory Statistics

{stats}

## Constraint Ranges

| Parameter | Min | Max | Notes |
|-----------|-----|-----|-------|
| semantic_k | 5 | 100 | Candidates retrieved per query |
| min_similarity | 0.1 | 0.9 | Cosine similarity floor |
| q_weight | 0.3 | 1.0 | Q-value vs similarity balance |
| confidence_threshold | 0.3 | 0.95 | Min score to trust learned routing |
| learning_rate | 0.01 | 0.5 | TD update step size |
| success_reward | 0.5 | 1.5 | Reward for successful task |
| failure_reward | -1.0 | 0.0 | Penalty for failed task |
| cost_penalty_lambda | 0.0 | 0.5 | Latency cost weight |

## Task

Propose 2-3 new candidate configs as JSON code blocks. Each must have:
- `retrieval_config`: dict with RetrievalConfig fields
- `scoring_config`: dict with ScoringConfig fields
- `notes`: string explaining the rationale

Focus on improving routing accuracy and cost efficiency. Consider:
- Adjusting q_weight if routing accuracy is low (more exploitation vs exploration)
- Tuning learning_rate if Q-values converge too slowly or oscillate
- Modifying cost_penalty_lambda if expensive models are overused
- Changing semantic_k if retrieval misses relevant memories

Example format:
```json
{
  "retrieval_config": {"semantic_k": 25, "q_weight": 0.8},
  "scoring_config": {"learning_rate": 0.15, "cost_penalty_lambda": 0.2},
  "notes": "Increase exploitation (q_weight 0.7→0.8) and penalize costly models more"
}
```
