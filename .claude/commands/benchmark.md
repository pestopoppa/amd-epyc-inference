# Benchmarking & Eval Workflow

> If registering a new model, run `/new-model` first.

## New Model Testing Order

### Step 1: Establish Reliable Launch
1. Run: `llama-completion -m MODEL.gguf -p "Hello"`
2. Document launch quirks (flags, output format, piping issues)
3. Add quirks to `orchestration/model_registry.yaml` immediately

### Step 2: Run Quality Rubric (Captures Speed)
1. Run `run_thinking_rubric.sh` — captures quality AND speed per question
2. Apply optimizations: MoE `--override-kv ARCH.expert_used_count=int:4`, dense → spec decode
3. Assign role based on tier scores. Do NOT run separate speed benchmarks.

### Step 3: Full Benchmark Suites (Ask User First)

> **Note:** `run_overnight_benchmark_suite.sh` has been moved to `scripts/benchmark/deprecated/`.
> Use individual suite scripts or `run_thinking_rubric.sh` instead.

```bash
# Deprecated (still functional, moved to scripts/benchmark/deprecated/):
./scripts/benchmark/deprecated/run_overnight_benchmark_suite.sh --suite all
./scripts/benchmark/deprecated/run_overnight_benchmark_suite.sh --suite thinking
```

---

## Benchmarking Pitfalls

**ALWAYS use these flags** (prevents interactive mode hangs):
```bash
llama-cli -m MODEL.gguf -f prompt.txt -n 128 \
    --no-display-prompt --simple-io --no-warmup --temp 0
```

Never use `-i`/`--interactive` in scripts. If hung: `timeout 300 llama-cli ...` or `pkill -f llama-cli`.

### Document Quirks After Every Benchmark
Update `orchestration/model_registry.yaml`:
```yaml
performance:
  baseline_tps: <measured>
  optimized_tps: <measured>
  speedup: <calculated>
benchmark_date: YYYY-MM-DD
runtime_quirks:
  model_name:
    quirks:
      - issue: "description"
        workaround: "fix"
        discovered: YYYY-MM-DD
```
Required: spec decode acceptance rates, MoE override key names, BOS/EOS mismatches, SSM constraints.

---

## Claude-as-Judge Scoring

### Rubric

| Score | Meaning |
|-------|---------|
| 3 | Correct with good reasoning |
| 2 | Partially correct or truncated |
| 1 | Wrong but reasonable attempt |
| 0 | Wrong, empty, or no answer |

### Quick Heuristics

| Pattern | Score |
|---------|-------|
| Empty or `<think>` only | 0 |
| Valid tool call JSON | 3 |
| Valid JSON structure | 3 |
| Reformatting response | 2 |

### Review Workflow
1. Find results: `ls benchmarks/results/runs/*/`
2. Score each answer (0-3) with brief reason
3. Create CSV: `benchmarks/results/reviews/{model_name}_baseline.csv`
   ```csv
   suite,question_id,tokens_per_second,claude_score,score_reason
   ```
4. Update `benchmarks/results/reviews/summary.csv` (totals per suite, overall %, avg t/s)

Score inheritance: spec decode configs inherit quality scores from their baseline.
Reference: `benchmarks/results/reviews/BLIND_RESCORE_2026-01-16.md` (77 models)

### When to Score
- After new model completes benchmark suite
- When algorithmic scores seem low
- Before role assignment decisions

---

## Benchmark Hardening (2025-12-18)

Questions were hardened (removed trivial T1, added post-doctoral T3). Models benchmarked before this date need score conversion:
```
conversion_factor = new_score / old_score  (reference: DeepSeek-R1-8B 112/120 old → TBD new)
```

---

## Eval Log Analysis Protocol

1. Look up question data FIRST — no speculation: check `benchmarks/prompts/v1/` for question content and `benchmarks/results/runs/` for model answers
2. Show full output to user (question, expected answer, model answers, scores)
3. If scoring looks wrong, replay scorer against stored answers
4. Then analyze — only after raw data is presented
