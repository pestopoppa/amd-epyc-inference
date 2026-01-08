# Model Reference

Comprehensive reference for all models used in the orchestration system.

## Production Models by Role

### Tier A - Front Door / Orchestrator

| Role | Model | Quantization | Speed | Acceleration |
|------|-------|--------------|-------|--------------|
| frontdoor | Qwen3-Coder-30B-A3B | Q4_K_M | 41.55 t/s | MoE 4 experts |

### Tier B - Specialists

| Role | Model | Quantization | Speed | Acceleration |
|------|-------|--------------|-------|--------------|
| coder_primary | Qwen2.5-Coder-32B | Q4_K_M | 33.0 t/s | Speculative K=24 |
| ingest_long_context | Qwen3-Next-80B-A3B | Q4_K_M | 9.1 t/s | MoE 3 experts |
| architect | Qwen3-235B-A22B | Q4_K_M | 6.75 t/s | MoE 4 experts |

### Tier C - Workers

| Role | Model | Quantization | Speed | Acceleration |
|------|-------|--------------|-------|--------------|
| worker_general | Meta-Llama-3-8B | Q4_K_M | ~25 t/s | Speculative |
| worker_math | Qwen2.5-Math-7B | Q4_K_M | ~28 t/s | Speculative K=8 |
| worker_vision | Qwen2.5-VL-7B | Q4_K_M | 57.1 t/s | Spec K=8, temp=0.7 |

### Tier D - Draft

| Role | Model | Quantization | Speed |
|------|-------|--------------|-------|
| draft | Qwen2.5-Coder-0.5B-Instruct | Q8_0 | 85 t/s |

## Model Compatibility Matrix

### Speculative Decoding Pairs

| Target Family | Compatible Drafts | Notes |
|---------------|-------------------|-------|
| Qwen2.5-* | Qwen2.5-0.5B, Qwen2.5-1.5B | Same tokenizer required |
| Qwen3-* | Qwen3-0.6B | Limited compatibility |
| Meta-Llama-3.* | PARD-Llama-3.2-1B | Verified working |
| DeepSeek-R1-Distill-* | **None found** | Vocab mismatches |

### Incompatible Pairs (Do Not Use)

| Target | Draft | Failure Mode |
|--------|-------|--------------|
| Qwen3-Coder-480B | Any | BOS token mismatch |
| Qwen3-Next-* | Any | SSM state corruption |
| DeepSeek-R1-Distill-* | Any | Vocab size mismatch |

## MoE Override Keys

| Model Family | Override Key | Example |
|--------------|--------------|---------|
| Qwen3 MoE | `qwen3moe.expert_used_count` | `int:4` |
| Qwen3-Next SSM | `qwen3next.expert_used_count` | `int:3` |
| GLM-4 | `glm4.expert_used_count` | `int:4` |

## Critical Constraints

### SSM Models (Qwen3-Next)

**NEVER use speculative decoding or prompt lookup.**

SSM architectures maintain recurrent state that cannot be rolled back. Use MoE expert reduction only.

```bash
# ⛔ WRONG - will corrupt model state
llama-speculative -m Qwen3-Next-80B.gguf -md draft.gguf

# ✅ CORRECT - expert reduction only
llama-cli -m Qwen3-Next-80B.gguf --override-kv qwen3next.expert_used_count=int:3
```

### Qwen3-Coder-480B

BOS token mismatch (`BOS=','`) breaks all speculation. Use expert reduction only.

## Model Locations

```
/mnt/raid0/llm/models/           # Primary GGUF storage
/mnt/raid0/llm/lmstudio/models/  # LM Studio format models
/mnt/raid0/llm/hf/               # HuggingFace format (raw)
```

## Quick Commands by Model Type

### Dense Model (with speculation)

```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  llama-speculative -m TARGET.gguf -md DRAFT.gguf \
  --draft-max K -t 96 -p "prompt"
```

### MoE Model (with expert reduction)

```bash
numactl --interleave=all \
  llama-cli -m MOE_MODEL.gguf \
  --override-kv ARCH.expert_used_count=int:N -t 96 -p "prompt"
```

### SSM Model (expert reduction ONLY)

```bash
numactl --interleave=all \
  llama-cli -m SSM_MODEL.gguf \
  --override-kv qwen3next.expert_used_count=int:3 -t 96 -p "prompt"
```

---

*See [QUIRKS.md](QUIRKS.md) for runtime issues and workarounds.*
*See [RESULTS.md](../benchmarks/RESULTS.md) for benchmark data.*
