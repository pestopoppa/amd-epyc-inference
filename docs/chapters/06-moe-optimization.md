# Chapter 06: MoE Optimization (Track 2)

## Introduction

Mixture-of-Experts (MoE) models use sparse activation - only a subset of "experts" (specialized sub-networks) are active for each token. Track 2 discovers that we can **force fewer experts** than the model was trained with, trading quality for speed.

This technique provides **21-52% speedup** on MoE models and is the **only safe optimization** for SSM-hybrid models like Qwen3-Next.

## How MoE Works

```
Standard Dense Model:
  Input → All parameters → Output

MoE Model (default 8 experts, top-2 routing):
  Input → Router → Select 2 of 8 experts → Output
  (Only 25% of expert parameters used per token)
```

**Expert Reduction Optimization**:
```
MoE Model (forced 4 experts total):
  Input → Router → Select 2 of 4 experts → Output
  (Same quality, faster inference)
```

## Best Results

| Model | Baseline | Top-4 Experts | Speedup | Quality |
|-------|----------|---------------|---------|---------|
| Qwen3-VL-30B-A3B | 24.8 t/s | 37.7 t/s (3 experts) | **+52%** | ✅ Good |
| Qwen3-Coder-480B-A35B | 2.5 t/s | 3.7 t/s | **+48%** | ✅ Good |
| GLM-4.6-355B-A32B | 2.2 t/s | 3.0 t/s | **+36%** | ✅ Good |
| Qwen3-Coder-30B-A3B | 26.6 t/s | 33.6 t/s | **+26%** | ✅ Good |
| Qwen3-Next-80B-A3B | 7.5 t/s | 9.1 t/s (3 experts) | **+21%** | ✅ Good |

## Expert Count Tuning

Testing different expert counts on Qwen3-VL-30B:

| Expert Count | Speed | Quality |
|--------------|-------|---------|
| 8 (default) | 24.8 t/s | Baseline |
| 6 experts | 28.4 t/s | ✅ Good |
| 4 experts | ~35 t/s | ✅ Good |
| 3 experts | **37.7 t/s** | ✅ Good |
| 2 experts | ~40 t/s | ⚠️ Degraded |

**Finding**: 3-4 experts is typically the sweet spot. Going below 3 often produces garbage output.

## Critical: SSM Models

**Qwen3-Next models (SSM hybrids)** cannot use speculative decoding or prompt lookup due to their recurrent architecture. Expert reduction is the **only safe optimization**:

```bash
# ⛔ DO NOT use --draft or --lookup with Qwen3-Next!
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Q4_K_M.gguf \
  --override-kv qwen3next.expert_used_count=int:3 \
  -t 96 -p "prompt"
```

**Why**: SSM architectures maintain recurrent state that cannot be rolled back. Draft token rejection corrupts this state irreversibly.

## Override Key Names

Different model families use different override keys:

| Model Family | Override Key |
|--------------|--------------|
| Qwen3 MoE | `qwen3moe.expert_used_count` |
| Qwen3-Next SSM | `qwen3next.expert_used_count` |
| GLM-4 | `glm4.expert_used_count` |

**Find the correct key**:
```bash
# List model metadata
llama-cli -m MODEL.gguf --verbose 2>&1 | grep expert
```

## Quick Start Command

```bash
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen3-235B-A22B-Q4_K_M.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  -t 96 -p "prompt"
```

## Combining with Other Techniques

Expert reduction is **orthogonal** to other optimizations:

| Combination | Compatible | Notes |
|-------------|------------|-------|
| MoE + Speculative Decoding | ⚠️ Partial | Only if base model supports spec decode |
| MoE + Prompt Lookup | ⚠️ Partial | Only if base model supports lookup |
| MoE + SSM | ✅ Yes | This IS the only option for SSM |

For models that support it, combining MoE reduction with prompt lookup achieves the best results:
- Qwen3-Coder-30B with 4 experts + prompt lookup: **47.5 t/s**

## Quality Monitoring

Always verify quality after applying expert reduction:

1. Run benchmark suite on new configuration
2. Compare Claude-as-Judge scores to baseline
3. If scores drop <10%, increase expert count

**Quality Degradation Signs**:
- Repetitive or looping output
- Incomplete sentences
- Factual errors on simple questions
- Instruction following failures

## References

### Foundational MoE Papers

1. Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. Journal of Machine Learning Research, 23(120), 1-39. https://arxiv.org/abs/2101.03961

2. Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., ... & Chen, Z. (2021). *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. ICLR 2021. https://arxiv.org/abs/2006.16668

3. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. https://arxiv.org/abs/1701.06538

### Expert Reduction and Efficiency

4. Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R. Y., Awan, A. A., ... & He, Y. (2022). *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale*. ICML 2022. https://arxiv.org/abs/2201.05596

5. Kim, Y., Awadalla, H. H., Muzio, A., Elbayad, M., & Esser, S. K. (2023). *Mixture of Experts with Capacity Factor Tuning*. arXiv preprint. https://arxiv.org/abs/2305.14705

### SSM and Hybrid Architectures

6. Gu, A., & Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. COLM 2024. https://arxiv.org/abs/2312.00752

7. Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., ... & Shoham, Y. (2024). *Jamba: A Hybrid Transformer-Mamba Language Model*. arXiv preprint. https://arxiv.org/abs/2403.19887

### Qwen Model Documentation

8. Qwen Team. (2024). *Qwen3 Technical Report*. Alibaba Group. https://arxiv.org/abs/2505.09388

9. Qwen Team. (2024). *Qwen2.5 Technical Report*. Alibaba Group. https://arxiv.org/abs/2412.15115

---

*Previous: [Chapter 05: Speculative Decoding](05-speculative-decoding.md)* | *Next: [Chapter 07: Prompt Lookup](07-prompt-lookup.md)*
