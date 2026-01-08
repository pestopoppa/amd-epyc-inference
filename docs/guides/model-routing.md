# Model Routing Guide

How to choose the right model for your task.

## Quick Decision Tree

```
Is this code generation?
├── Yes → Qwen2.5-Coder-32B + speculative (33 t/s)
└── No
    Is this long context (>32K tokens)?
    ├── Yes → Qwen3-Next-80B + MoE reduction (SSM, no speculation!)
    └── No
        Is this reasoning/thinking?
        ├── Yes → Thinking model or escalate to architect
        └── No → Worker model (Llama-3-8B, etc.)
```

## By Task Type

| Task | Model | Acceleration | Speed |
|------|-------|--------------|-------|
| Code generation | Qwen2.5-Coder-32B | Speculative K=24 | 33 t/s |
| Code editing | Qwen2.5-Coder-32B | Prompt lookup | 25.8 t/s |
| Summarization | Any + Prompt lookup | Prompt lookup | Up to 95 t/s |
| Long context | Qwen3-Next-80B | MoE only | 9 t/s |
| Math | Qwen2.5-Math-7B | Speculative K=8 | 28 t/s |
| General | Meta-Llama-3-8B | Speculative | 25 t/s |

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

---

*See [MODELS.md](../reference/models/MODELS.md) for detailed model specifications.*
