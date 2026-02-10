# Benchmarking & Eval Workflow

> If registering a new model, run `/new-model` first.

---

## New Model Testing Workflow

**CRITICAL: When testing a NEW model, follow this order:**

### Step 1: Establish Reliable Launch (FIRST)
Before ANY testing:
1. Run a minimal test: `llama-completion -m MODEL.gguf -p "Hello"`
2. Identify and document launch quirks:
   - Does it need specific flags?
   - Does it auto-enable interactive/conversation mode?
   - Are there output format quirks (e.g., `<think>` tags)?
   - Does piping output cause errors?
3. Add quirks to `orchestration/model_registry.yaml` immediately

### Step 2: Run Quality Rubric (Captures Speed Automatically)
Once launch is reliable:
1. Run quality rubric script (e.g., `run_thinking_rubric.sh`)
2. Script captures BOTH quality scores AND speed per question
3. Apply known optimizations during testing:
   - MoE models: `--override-kv ARCH.expert_used_count=int:4`
   - Dense models: spec decode if compatible
4. Assign role based on tier scores

**DO NOT do separate speed benchmarks** - the rubric captures speed data.

### Step 3: Run Full Benchmark Suites (Optional — Ask User)
After registry entry and quirks are documented, **ask the user** if they want the full benchmark suite:

```bash
# All 8 suites (takes hours)
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite all

# Or specific suites based on model role:
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite thinking      # Reasoning models
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite coder         # Code models
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite instruction_precision  # Orchestration candidates
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite long_context  # Context window testing
```

**8 Benchmark Suites:**
1. **Thinking** - Chain-of-thought, multi-step reasoning
2. **Coder** - Code generation, debugging, refactoring
3. **VL** - Vision-language (OCR, image understanding)
4. **General** - Instruction following, summarization
5. **Agentic** - Tool calling, function extraction
6. **Math** - Mathematical reasoning, step verification
7. **Long Context** - Information retrieval across 4K-50K token contexts
8. **Instruction Precision** - Exact format compliance (critical for orchestration)

**Why this matters:**
- Results are stored permanently in `/mnt/raid0/llm/claude/benchmarks/results/`
- JSONL index enables faithful comparison with future models
- Models can be deleted after benchmarking - results persist for comparison
- Instruction Precision suite identifies models that will break orchestration parsing

**Why this order matters:**
- Debugging launch issues DURING quality tests wastes time
- Quality rubric captures speed - no separate benchmark needed
- Registry should always have working launch commands
- Full benchmark suite provides permanent record for future comparison

---

## Benchmarking Pitfalls

### Interactive Mode Hangs
**CRITICAL**: `llama-cli` can hang waiting for user input if not configured correctly.

**ALWAYS use these flags when benchmarking:**
```bash
llama-cli -m MODEL.gguf -f prompt.txt -n 128 \
    --no-display-prompt \
    --simple-io \
    --no-warmup \
    --temp 0
```

**Never use:**
- `-i` or `--interactive` in automated scripts
- Pipes without proper EOF handling

**If a benchmark hangs:**
1. Check for interactive mode prompts
2. Verify timeout is set: `timeout 300 llama-cli ...`
3. Kill stuck processes: `pkill -f llama-cli`

### MANDATORY: Document Model Quirks

**After every new model benchmark**, update `orchestration/model_registry.yaml`:

1. **Add performance data** under the appropriate role entry:
   ```yaml
   performance:
     baseline_tps: <measured>
     optimized_tps: <measured>
     speedup: <calculated>
   benchmark_date: YYYY-MM-DD
   ```

2. **Document any runtime quirks** in the `runtime_quirks` section:
   ```yaml
   runtime_quirks:
     model_name:
       description: "Model full name"
       quirks:
         - issue: "What breaks or behaves unexpectedly"
           workaround: "How to fix or avoid it"
           discovered: YYYY-MM-DD
   ```

3. **Required quirk documentation includes:**
   - Speculative decoding acceptance rates (if unusually low)
   - MoE override key names (`qwen3moe.*` vs `qwen3next.*` etc.)
   - BOS/EOS token mismatches that break draft compatibility
   - Timeout/wrapper issues specific to model or binary
   - Architecture-specific constraints (SSM incompatibility, etc.)

4. **Reference the model registry** before running benchmarks to avoid rediscovering known quirks.

---

## Claude-as-Judge Quality Review

### Overview

Claude-as-Judge is our framework for independent quality evaluation of model benchmark answers. The algorithmic rubric was found to severely underscore models (38% vs 89% for the same model) due to pattern matching failures.

**Use this framework to:**
- Score new model benchmark results
- Compare quality across models
- Identify models with empty output issues
- Make role assignment decisions

### Scoring Rubric

| Score | Meaning | Examples |
|-------|---------|----------|
| 3 | Correct answer with good reasoning | Complete solution, accurate math, valid logic |
| 2 | Partially correct or correct but truncated | Right approach but incomplete, minor errors |
| 1 | Wrong answer but reasonable attempt | Plausible but incorrect, misunderstood question |
| 0 | Completely wrong, empty, or no answer | Garbage output, empty response, unrelated text |

### File Locations

```
benchmarks/results/reviews/
├── {model_name}_baseline.csv      # Per-model review
├── {model_name}_{config}.csv      # Per-config review (if applicable)
├── summary.csv                    # Comparative summary
└── BLIND_RESCORE_2026-01-16.md    # Comprehensive blind rescore (77 models)
```

### Per-Model Review CSV Format

```csv
suite,question_id,tokens_per_second,claude_score,score_reason
thinking,t1_q1_logic,21.0,3,Correctly identified syllogism fallacy
thinking,t1_q2_sequence,20.8,3,Answer 42 is correct
general,t1_q1_reformat,18.5,2,Reformatted but truncated
agentic,t1_q1_single_tool,19.2,3,Tool call structure present
```

### Summary CSV Format

```csv
model,thinking,general,math,agentic,coder,instruction_precision,total,pct_str,avg_tps
thinking_deepseek_r1_distill_llama_8b,28/30,24/30,30/30,30/30,-,-,112/120,93%,7.2
```

### How to Review a New Model

1. **Locate benchmark results:**
   ```bash
   ls benchmarks/results/runs/*/  # Find the run directory
   # Look for {model_name}_baseline.json or similar
   ```

2. **Read the benchmark output:**
   - Each JSON file contains questions and model answers
   - Note the `tokens_per_second` from each answer

3. **Score each answer (0-3):**
   - Read the question and expected answer format
   - Evaluate the model's response
   - Assign score based on rubric above
   - Note brief reason

4. **Create review CSV:**
   ```bash
   # Create file at: benchmarks/results/reviews/{model_name}_baseline.csv
   ```

5. **Update summary.csv:**
   - Calculate totals per suite (e.g., "28/30")
   - Calculate overall percentage
   - Calculate average tokens/second
   - Add row to summary.csv (sorted by percentage descending)

### Batch Scoring Heuristics

For efficiency, use these heuristics for common patterns:

| Pattern | Score | Reason |
|---------|-------|--------|
| Empty or `<think>` only | 0 | Empty or minimal output |
| Tool call JSON present | 3 | Tool call structure present |
| JSON structure valid | 3 | JSON structure present |
| Reformatting response | 2 | Reformatting response |
| General text response | 2 | General response generated |

### Current Coverage (as of 2026-01-16)

- **77 baseline models blind-rescored** (comprehensive rescore of all benchmark results)
- **Blind Rescore Reference:** `benchmarks/results/reviews/BLIND_RESCORE_2026-01-16.md`
- **Top performers:** See RESULTS.md for current rankings
- **Score inheritance:** Speculative decoding configs inherit quality scores from their baseline (same model, different speed)
- **Note:** Blind rescore used stricter methodology - scores are 5-17% lower than summary.csv but relative rankings preserved

### When to Run Claude-as-Judge

- After any new model completes benchmark suite
- When algorithmic scores seem suspiciously low
- Before making role assignment decisions
- When comparing models for a specific role

---

## Benchmark Hardening (2025-12-18)

### Overview

Benchmark questions were hardened to address ceiling effects. Top models were scoring 89-93%, indicating questions were too easy for expert-level differentiation.

**Changes made:**
- Removed 3 trivial T1 questions from each of 8 suites
- Shifted T2 → T1, T3 → T2 (relabeling)
- Added 3 post-doctoral level T3 questions to each suite

### Reference Model for Score Conversion

Models benchmarked before 2025-12-18 were tested on easier questions. To compare old vs new scores:

| Reference Model | Old Score | New Score | Conversion Factor |
|-----------------|-----------|-----------|-------------------|
| DeepSeek-R1-Distill-Llama-8B | 112/120 (93%) | TBD | TBD |

**After testing reference model on new questions:**
```
conversion_factor = new_score / old_score
converted_score = old_claude_score × conversion_factor
```

### New T3 Question Difficulty

New T3 questions require expert-level reasoning:

| Suite | Example Question | Why It's Hard |
|-------|------------------|---------------|
| thinking | Causal inference DAG (collider bias) | Requires formal causal reasoning |
| thinking | Gödel/Penrose philosophy | Cross-domain philosophy of mind |
| math | f(x) = Σ(x^n/n!)sin(n) analysis | Closed-form via complex exponentials |
| math | E[N] where S_n > 1 (uniform sum) | Answer is e, requires two proof methods |
| coder | Lock-free stack ABA problem | Concurrent programming edge case |
| coder | Distributed consistency strategies | CAP theorem trade-offs |
| agentic | Multi-agent coordination | Time-budgeted agent orchestration |
| agentic | Adversarial input handling | Security-aware tool use |
| vl | Scientific figure analysis | Statistical critique of graphs |
| long_context | Multi-hop temporal reasoning | 4+ document chain reasoning |
| instruction_precision | Self-referential constraints | Meta-accurate self-description |

### Expected Score Distribution (Post-Hardening)

| Model Class | Expected Score |
|-------------|----------------|
| 0.5B-1.5B draft models | 30-50% |
| 4B-8B general models | 50-70% |
| 8B+ specialized thinking models | 60-80% |
| 14B+ large models | 70-85% |

Top models should no longer hit 90%+ ceiling.

---

## Eval Log Analysis Protocol (MANDATORY)

When asked to analyze eval log output (3-way eval, benchmark results, etc.):

1. **Run the deterministic lookup script FIRST** — no speculation before data:
   ```bash
   python3 scripts/benchmark/lookup_question.py <question_id>
   ```
2. **Show the full output to the user** — question text, expected answer, all model answers, scores
3. **If pass/fail looks wrong**, replay the scorer against stored answers:
   ```python
   python3 -c "
   import json, re
   # Extract full answer from JSONL, run multiple_choice / exact_match logic
   "
   ```
4. **Then analyze** — only after raw data is presented

**Never speculate about format issues, knowledge gaps, or scoring bugs without first running the scripts and showing actual model outputs.**
