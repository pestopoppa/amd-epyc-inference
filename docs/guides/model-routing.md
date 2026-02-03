# Model Routing Guide

How to choose the right model for your task.

## Quick Decision Tree

```
Is this code generation?
├── Yes → Qwen2.5-Coder-32B + spec K=24 + lookup (39 t/s, port 8081)
└── No
    Is this long context (>32K tokens)?
    ├── Yes → Qwen3-Next-80B + MoE4 (SSM, no speculation! port 8085)
    └── No
        Is this vision/image?
        ├── Yes → Qwen2.5-VL-7B (port 8086) or Qwen3-VL-30B (port 8087)
        └── No
            Is this reasoning/architecture?
            ├── Yes → Architect: Qwen3-235B (port 8083) or 480B (port 8084)
            └── No → Explore worker: Qwen2.5-7B + spec (44 t/s, port 8082)
```

## By Task Type

| Task | Model | Port | Acceleration | Speed |
|------|-------|------|--------------|-------|
| Interactive chat | Qwen3-Coder-30B-A3B | 8080 | MoE6 | 18 t/s |
| Code generation / escalation | Qwen2.5-Coder-32B | 8081 | Spec K=24 + lookup | 39 t/s |
| Explore / summarize | Qwen2.5-7B-Instruct-f16 | 8082 | Spec K=24 + lookup | 44 t/s |
| Long context | Qwen3-Next-80B-A3B | 8085 | MoE4 only | 6.3 t/s |
| Architecture (general) | Qwen3-235B-A22B | 8083 | MoE4 | 6.75 t/s |
| Architecture (coding) | Qwen3-Coder-480B-A35B | 8084 | MoE3 | 10.3 t/s |
| Vision (worker) | Qwen2.5-VL-7B + mmproj | 8086 | None | ~15 t/s |
| Vision (escalation) | Qwen3-VL-30B-A3B + mmproj | 8087 | MoE4 | ~10 t/s |
| Fast burst tasks | Qwen2.5-Coder-1.5B | 8102/8112 | None (WARM) | ~60 t/s |

## SSM Model Warning

**Qwen3-Next models** use SSM architecture. They **CANNOT** use:
- Speculative decoding
- Prompt lookup

Only use MoE expert reduction:
```bash
llama-cli -m Qwen3-Next-80B.gguf --override-kv qwen3next.expert_used_count=int:3
```

## Claude Code Tier Selection

When using Claude Code (Opus/Sonnet/Haiku):

| Tier | Use When |
|------|----------|
| **Opus** | Novel design, complex debugging, architecture decisions |
| **Sonnet** | Research, synthesis, routine code (default) |
| **Haiku** | Benchmark execution, log parsing, repetitive tasks |

## Escalation Rules

1. Start with appropriate tier for task
2. On first failure → retry with same model
3. On second failure → escalate one tier
4. On third failure → escalate to architect

## MemRL-Informed Routing

The orchestrator uses **learned routing** via MemRL (Memory-based Reinforcement Learning) when enabled:

1. **Episodic Memory**: Embeds incoming tasks and retrieves similar past experiences
2. **Q-Value Scoring**: Ranks candidate roles by expected success (reward history)
3. **Comparative Rewards**: When specialist outperforms frontdoor, +1.0 reward signal

### Routing Strategy Types

| Strategy | When Used | How It Works |
|----------|-----------|--------------|
| `learned` | MemRL enabled, sufficient history | Q-values from episodic memory retrieval |
| `rules` | MemRL disabled or cold start | Deterministic rules from escalation chains |
| `default` | Fallback | Route to frontdoor |

### Hard Benchmark Suites (2026-02-03)

Mode-advantage signal is strongest on these suites (30B frontdoor <50%, specialist >70%):

| Suite | Questions | Expected Mode-Advantage |
|-------|-----------|------------------------|
| `gpqa` | 448 | Graduate science → react/delegation |
| `simpleqa` | 4,326 | Factual lookup → search tools |
| `hotpotqa` | 7,405 | Multi-hop QA → search + reasoning |
| `livecodebench` | 2,360 | Competition code → REPL iteration |
| `debugbench` | 4,253 | Bug fixing → REPL + tests |
| `usaco` | 520 | Olympiad code → REPL + specialist |
| `mode_advantage_hard` | 60 | Hand-written hard tasks |

See [Chapter 15](../chapters/15-memrl-system.md) for MemRL architecture details.

---

*See [MODELS.md](../reference/models/MODELS.md) for detailed model specifications.*
