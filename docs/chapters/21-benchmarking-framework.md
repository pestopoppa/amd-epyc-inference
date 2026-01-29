# Chapter 21: Benchmarking Framework

## Introduction

We developed an 8-suite benchmarking framework to evaluate models for specific roles in our orchestration system. Unlike generic benchmarks (MMLU, etc.), our suites test task-specific capabilities: can a model follow precise formatting? Can it chain multi-step reasoning? Can it generate valid tool calls?

**Key Achievement**: 61 baseline models evaluated, with 381 total configurations (including MoE/speculative variants).

## The 8 Benchmark Suites

| Suite | Purpose | Key Test | Role Placement |
|-------|---------|----------|----------------|
| **Thinking** | Chain-of-thought reasoning | Multi-step logical deduction | oracle_reasoning, architect |
| **Coder** | Code generation & debugging | Working code with edge cases | coder_primary, coder_escalation |
| **Math** | Mathematical reasoning | Step-by-step proofs | Qwen2.5-Math for invariants |
| **General** | Instruction following | Summarization, reformatting | worker_general |
| **Agentic** | Tool calling | Valid JSON function calls | frontdoor, orchestrator |
| **VL** | Vision-language | OCR, image understanding | worker_vision |
| **Long Context** | Information retrieval | Needle-in-haystack (4K-50K tokens) | ingest_long_context |
| **Instruction Precision** | Format compliance | Exact output structure | **Critical for orchestration** |

## Claude-as-Judge Scoring

We use Claude as an independent judge rather than algorithmic rubrics. Early experiments showed algorithmic scoring severely underscored models (38% vs 89% for the same output) due to pattern matching failures.

### Scoring Rubric

| Score | Meaning |
|-------|---------|
| 3 | Correct answer with good reasoning |
| 2 | Partially correct or truncated |
| 1 | Wrong but reasonable attempt |
| 0 | Completely wrong, empty, or garbage |

### Why Claude-as-Judge?

- **Semantic understanding**: Recognizes correct answers in unexpected formats
- **Partial credit**: Awards 2 for "right approach, minor error"
- **Consistency**: Same model judges all, eliminating evaluator variance
- **Scalability**: Can score hundreds of responses efficiently

## Benchmark Hardening (December 2025)

Initial benchmarks had ceiling effects - top models scored 89-93%, preventing differentiation. We hardened all suites:

| Change | Before | After |
|--------|--------|-------|
| T1 questions | Easy | Medium (relabeled from T2) |
| T2 questions | Medium | Hard (relabeled from T3) |
| T3 questions | Hard | Post-doctoral level |

**New T3 Examples**:
- Thinking: Causal inference DAGs (collider bias)
- Math: Prove E[N] = e where S_n > 1 for uniform sum
- Coder: Lock-free stack ABA problem
- Agentic: Multi-agent coordination under time budget

### Expected Score Distribution (Post-Hardening)

| Model Class | Expected Score |
|-------------|----------------|
| 0.5B-1.5B draft models | 30-50% |
| 4B-8B general models | 50-70% |
| 8B+ specialized thinking | 60-80% |
| 14B+ large models | 70-85% |

Top models no longer hit 90%+ ceiling.

## Running Benchmarks

### Full Suite

```bash
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite all
```

### Specific Suite

```bash
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite thinking
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite coder
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite instruction_precision
```

### Results Location

```
benchmarks/
├── prompts/v1/          # Test cases by suite
├── results/
│   ├── runs/            # Raw benchmark outputs
│   ├── reviews/         # Claude-as-Judge scores
│   │   ├── {model}_baseline.csv
│   │   └── summary.csv
│   └── index.jsonl      # Benchmark index for comparison
```

## Instruction Precision Suite

**Critical for orchestration**: Models that fail instruction precision break TaskIR parsing.

| Test | What It Checks |
|------|----------------|
| Exact JSON structure | Can emit valid JSON with required fields |
| Format preservation | Respects specified output format |
| Constraint compliance | Follows "do not" instructions |
| Self-referential accuracy | Can accurately describe own output |

**Role Gate**: Models scoring <70% on instruction precision are not considered for orchestration roles (frontdoor, dispatcher).

## Quality vs Speed Trade-offs

Benchmarks capture both quality scores AND speed per question. This enables trade-off analysis:

| Configuration | Quality | Speed | Use Case |
|---------------|---------|-------|----------|
| Qwen2.5-Coder-32B baseline | 89% | 2.89 t/s | Quality critical |
| Qwen2.5-Coder-32B + spec decode | 89% | 28.79 t/s | **Best balance** |
| Qwen3-Coder-30B + MoE4 | 85% | 33.6 t/s | Speed critical |
| Qwen3-Coder-30B + MoE3 | 78% | 37.7 t/s | Speed extreme |

**Key Insight**: Speculative decoding preserves quality (same model). MoE reduction trades quality for speed.

## Orchestrator Benchmarks

The orchestrator benchmark pipeline compares orchestrated responses against direct large-model baselines, measuring quality retention and speed.

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/benchmark/run_orchestrator_benchmark.py` | Full 4-phase benchmark runner (smoke, compare, optimize, verify) |
| `scripts/benchmark/compare_orchestrator_direct.py` | Per-suite orchestrator vs baseline comparison |

### CLI Options

```bash
# Run Phase 2 (comparison) with API restart
./run_orchestrator_benchmark.py --phase 2 --restart-api

# Compare single suite
./compare_orchestrator_direct.py --suite thinking --use-baseline

# Create baseline from architect model
./compare_orchestrator_direct.py --create-baseline --suite all
```

The `--restart-api` flag restarts only the uvicorn API (port 8000), not the llama-server backends (8080-8090). Use after Python code changes.

### Output Format

Per-prompt line includes latency and tokens/sec:
```
  [thinking] t3_q1...   3042ms   16.3 t/s  speedup: 2.1x, quality: OK, turns: 1, routed: frontdoor
```

Per-suite mini-summary (in `run_orchestrator_benchmark.py`):
```
    thinking                10 prompts  ✓  92.0% quality   3042ms avg  16.3 t/s
```

Phase 2 aggregate:
```
  Phase 2 totals: 80 prompts across 8 suites in 342s
    Quality: ✓ 91.2% avg
    Speed:   19.4 t/s avg
    Latency: 4120ms avg
```

### Routing Telemetry

Each response includes `routed_to`, `role_history`, `routing_strategy`, and `tokens_generated` fields for debugging routing decisions.

### Results Location

```
benchmarks/results/orchestrator/
├── comparison_{suite}_{timestamp}.json  # Per-suite comparison
├── run_{timestamp}.json                 # Full run metadata
```

## Comparing Models

```bash
# List all benchmark runs
./scripts/benchmark/compare_results.sh --list-runs

# Compare two runs
./scripts/benchmark/compare_results.sh --baseline RUN_ID --current RUN_ID
```

## Permanent Results

Benchmark results persist in `benchmarks/results/`. Models can be deleted after benchmarking - results enable comparison with future models.

This is important because:
- Storage is limited even on our large system
- New models arrive frequently
- Historical comparison enables trend analysis
- Results include configs (MoE settings, K values) for reproduction

## References

### LLM Evaluation and Benchmarking

1. Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. https://arxiv.org/abs/2306.05685

2. Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). *Measuring Massive Multitask Language Understanding*. ICLR 2021. https://arxiv.org/abs/2009.03300

3. Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., ... & Zaremba, W. (2021). *Evaluating Large Language Models Trained on Code*. arXiv preprint. https://arxiv.org/abs/2107.03374

### Instruction Following and Format Compliance

4. Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., ... & Hui, K. (2023). *Instruction-Following Evaluation for Large Language Models*. arXiv preprint. https://arxiv.org/abs/2311.07911

5. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS 2022. https://arxiv.org/abs/2203.02155

### Long Context Evaluation

6. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). *Lost in the Middle: How Language Models Use Long Contexts*. TACL 2024. https://arxiv.org/abs/2307.03172

7. Kamradt, G. (2023). *Needle in a Haystack: Pressure Testing LLMs*. GitHub Repository. https://github.com/gkamradt/LLMTest_NeedleInAHaystack

### LLM-as-Judge Methodology

8. Chiang, W. L., & Zheng, L. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. https://chat.lmsys.org/

9. Dubois, Y., Li, X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., ... & Hashimoto, T. (2024). *AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback*. NeurIPS 2023. https://arxiv.org/abs/2305.14387

### Agentic and Tool Use Evaluation

10. Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). *Gorilla: Large Language Model Connected with Massive APIs*. arXiv preprint. https://arxiv.org/abs/2305.15334

11. Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., ... & Sun, M. (2024). *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*. ICLR 2024. https://arxiv.org/abs/2307.16789

---

**See Also:** [Master Benchmark Results](../reference/benchmarks/RESULTS.md) — Complete scores and speeds for all 61 models

---

*Previous: [Chapter 20: Session Persistence](20-session-persistence.md)* | *Next: [Chapter 22: Tool Registry & Agent Roles](22-tool-registry.md)*
