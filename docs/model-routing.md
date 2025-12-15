# Model Routing Guide

This document consolidates the model routing strategy for the AMD EPYC 9655 inference project.

---

## Quick Reference


## TL;DR Decision Matrix

| Task Type | Model | Speed | Cost | When |
|-----------|-------|-------|------|------|
| **Novel design/debugging** | OPUS | Slow | High | Need deep analysis |
| **Web search/research** | SONNET | Fast | Medium | Finding info |
| **Repetitive benchmarking** | HAIKU | ⭐⭐⭐ Fast | Low | Known commands |
| **General questions** | SONNET | Fast | Medium | Default choice |
| **Output parsing/logs** | HAIKU | ⭐⭐⭐ Fast | Low | Routine work |

---

## Keyword-Based Routing

### → USE OPUS 4.5
**Keywords:** implement, write (code), design, architecture, debug, trace, root cause, novel, strategy, plan, proposal, complex, critical, security, high-stakes, refactor, integrate, develop (novel), investigate (deep)

**Examples:**
- ✓ "Implement Track 7 CAS-Spec with layer skipping"
- ✓ "Debug why EAGLE acceptance is 0%"
- ✓ "Design a new approach for SSM speculative decoding"
- ✓ "Trace through the token verification logic"
- ✗ "Run a benchmark" → Use HAIKU instead
- ✗ "What are the latest papers?" → Use SONNET instead

---

### → USE SONNET 4.5
**Keywords:** find, search, look for, fetch, compare, what, how, explain, describe, summarize (shallow), also, parallel, meanwhile, research (non-deep), analyze (non-deep), overview, list, check, compile, gather, synthesis, follow-up, clarify

**Examples:**
- ✓ "Find NeurIPS 2025 speculative decoding papers"
- ✓ "Search GitHub for llama.cpp implementations"
- ✓ "What are the current bottlenecks?"
- ✓ "Compare K=8 vs K=16 acceptance rates"
- ✓ "How does MoE soft masking work?"
- ✓ "Summarize the research findings (quick)"
- ✗ "Implement speculative decoding" → Use OPUS instead
- ✗ "Run the benchmark" → Use HAIKU instead

---

### → USE HAIKU 4.5
**Keywords:** run, execute, benchmark, measure, test, parse, extract, collect, read (logs), aggregate, tabulate, status, check (file), simple, known command

**Examples:**
- ✓ "Run bench_zen5.sh on Qwen2.5-Coder-32B"
- ✓ "Execute this command: [COMMAND]"
- ✓ "Parse research_report.md and extract speedups"
- ✓ "Collect metrics from the last 10 benchmarks"
- ✓ "Check if the model file exists"
- ✓ "Benchmark K=8,16,24 and create a table"
- ✗ "Why is acceptance so low?" → Use SONNET/OPUS instead
- ✗ "Design a new optimization" → Use OPUS instead

---

## Common Workflows

### 1. Research Phase
```
YOU: "Find the latest papers on [topic]"
→ SONNET: Quick web search, compile list

YOU: "Summarize what you found"
→ SONNET: Synthesis of results

YOU: "Now design an implementation based on these"
→ OPUS: Deep architectural thinking
```

### 2. Implementation Phase
```
YOU: "Implement [feature]"
→ OPUS: Write the code

YOU: "Run a benchmark to validate"
→ HAIKU: Execute the known command

YOU: "Analyze the results"
→ SONNET: Quick pattern analysis
(or OPUS if deep insights needed)
```

### 3. Optimization Phase
```
YOU: "Test K=8,16,24 and compare"
→ HAIKU: Run all 3 benchmarks

YOU: "Why is K=16 best?"
→ SONNET: Quick analysis of numbers
(or OPUS if architectural insight needed)

YOU: "Design a new K selection algorithm"
→ OPUS: Deep design work
```

### 4. Troubleshooting Phase
```
YOU: "The acceptance rate is 0%"
→ OPUS: Debug and trace the issue

YOU: "Did anything change in the logs?"
→ HAIKU: Quick log parsing

YOU: "Search for similar issues online"
→ SONNET: Web research
```

---

## Multi-Model Coordination

### Best Practice: Research → Design → Execute → Analyze

```
Step 1: SONNET gathers info
   └─ Fast, parallel research

Step 2: OPUS designs solution
   └─ Deep reasoning, architecture

Step 3: HAIKU executes implementation
   └─ Runs known commands, collects output

Step 4: SONNET summarizes results
   └─ Quick synthesis of findings

Step 5: (Optional) OPUS deep dives
   └─ Only if insights are unclear
```

### Cost Optimization

**High ROI:**
- Use SONNET for all research/synthesis
- Use HAIKU for all pre-validated commands
- Use OPUS only for truly novel/complex work

**Red Flags (avoid):**
- ❌ Using OPUS for simple status checks
- ❌ Using HAIKU for architecture design
- ❌ Using SONNET for complex debugging (needs OPUS)

---

## Real-World Examples

### Example 1: "Benchmark Track 1 vs Track 2"
```
Model: HAIKU
Why: Repetitive execution of known commands
Task: Run benchmarks, collect metrics, create table
```

### Example 2: "Find the latest speculative decoding papers"
```
Model: SONNET
Why: Web research + synthesis
Task: Search, compile list, summarize findings
```

### Example 3: "Why does SSM speculation fail on Qwen3-Next?"
```
Model: OPUS
Why: Complex debugging, root cause analysis
Task: Trace through architecture, identify incompatibilities
```

### Example 4: "Implement SuffixDecoding (Track 6) for CPU"
```
Model: OPUS
Why: Complex, novel implementation
Task: Design data structures, write integration code, handle edge cases
```

### Example 5: "Parse the benchmark results and extract speedup metrics"
```
Model: HAIKU
Why: Routine output collection
Task: Read file, extract numbers, format table
```

### Example 6: "Compare our current approach with CAS-Spec"
```
Model: SONNET
Why: Quick analysis, synthesis
Task: Review both approaches, list pros/cons, highlight differences
```

### Example 7: "Design a content-adaptive K controller"
```
Model: OPUS
Why: Novel architecture, complex reasoning
Task: Design algorithm, consider trade-offs, plan implementation
```

### Example 8: "Check if the model is loaded and ready"
```
Model: HAIKU
Why: Simple status check
Task: Run command, parse output, report status
```

---

## When in Doubt

**Prefer SONNET** for:
- Anything exploratory
- When you're unsure of the task type
- Fast feedback before committing to work
- Questions that might need revision

**Escalate to OPUS** if:
- SONNET's answer is incomplete
- You need deeper reasoning
- The task is novel/research-grade
- High-stakes decisions required

**Use HAIKU** only when:
- You know the command works
- Task is purely repetitive
- Output collection/parsing is needed
- Budget is a constraint

---

## Prompt Template for Explicit Routing

If you want to force a specific model, start your prompt with:

```
→ OPUS: [complex task requiring deep reasoning]
→ SONNET: [research/synthesis task]
→ HAIKU: [repetitive/benchmark task]
```

Example:
```
→ OPUS: Design a new speculative decoding approach for SSM models
```

---

## Performance Expectations

| Model | Response Time | Typical Wait | Parallelizable |
|-------|----------------|--------------|----------------|
| OPUS | ~30-60 seconds | Long | No (serial) |
| SONNET | ~5-15 seconds | Medium | Yes (parallel) |
| HAIKU | ~2-5 seconds | Short | Yes (parallel) |

**Implication:** 
- Use OPUS for blocking work (nothing else can proceed)
- Use SONNET while waiting for other tasks
- Use HAIKU for quick feedback loops

---

## Cost Estimates (Relative)

Assuming 1 HAIKU task = 1 unit cost:

| Model | Relative Cost | Example Task |
|-------|---------------|--------------|
| HAIKU | 1x | Run benchmark, parse logs |
| SONNET | 3-5x | Web search, synthesis |
| OPUS | 10-15x | Complex debugging, design |

**Strategy:** Maximize HAIKU usage, use SONNET as default, reserve OPUS for necessities.

---

## Summary Rules

1. **DEFAULT: SONNET** (safest, good balance)
2. **UPGRADE TO OPUS** if task is novel/complex/debugging
3. **DOWNGRADE TO HAIKU** if task is repetitive/known-working
4. **PARALLELIZE WITH SONNET** for information gathering
5. **USE HAIKU** to minimize costs on routine work


---

## Implementation Details


## How to Use This System

### For Your Workflow

When you have a task or question, **think through this 10-second check:**

1. Is this complex/novel/debugging? → **OPUS**
2. Is this research/synthesis/web search? → **SONNET**
3. Is this repetitive/known command? → **HAIKU**
4. If uncertain? → **SONNET** (default)

---

## Template Prompts for Each Tier

### OPUS 4.5 Template: Complex Implementation

**Use this for coding, debugging, architecture:**

```
You are an expert systems engineer and LLM researcher specializing in 
speculative decoding for CPU inference on AMD EPYC Zen 5.

Task: [DESCRIBE YOUR COMPLEX TASK]

Context:
- Target system: AMD EPYC 9655 (96 cores, 1.13TB RAM)
- Current approach: [CURRENT METHOD]
- Problem: [WHAT'S NOT WORKING]
- Constraints: [ANY LIMITATIONS]

Required outputs:
1. Root cause analysis
2. Step-by-step implementation plan
3. Potential pitfalls and mitigations
4. How to validate the solution

Begin your analysis:
```

**Examples:**
- Debugging EAGLE-1 acceptance rates
- Designing Track 7 CAS-Spec implementation
- Fixing Qwen3-Next SSM compatibility
- Integrating multiple optimization tracks

---

### SONNET 4.5 Template: Research & Synthesis

**Use this for finding information, comparison, synthesis:**

```
Research and compile information on: [TOPIC]

Find:
1. Latest papers/resources (focus on [YEAR/FIELD])
2. Key findings and metrics
3. Implementation examples from [REPOSITORIES/PROJECTS]
4. How this relates to [YOUR CONTEXT]

Format as:
- Brief summary (2-3 sentences)
- Bullet points of key findings
- Links to resources
- Relevance to AMD EPYC CPU inference

After gathering, be ready to answer follow-up questions.
```

**Examples:**
- "Find latest NeurIPS 2025 speculative decoding papers"
- "Search GitHub for llama.cpp speculative implementations"
- "Compare vLLM vs TensorRT-LLM spec decode approaches"
- "Research SuffixDecoding implementation options"

---

### HAIKU 4.5 Template: Execution & Collection

**Use this for running commands, parsing output, benchmarking:**

```
Execute benchmark: [TASK DESCRIPTION]

Commands to run:
1. [COMMAND 1]
2. [COMMAND 2]
...

Expected outputs:
- [METRIC 1]
- [METRIC 2]
- [METRIC 3]

Format results as: [TABLE/CSV/MARKDOWN]

Key metrics to extract: [SPECIFIC VALUES NEEDED]
```

**Examples:**
- "Benchmark K=8,16,24 on Qwen2.5-Coder-32B"
- "Parse research_report.md and extract speedup numbers"
- "Run health check and report system status"
- "Execute model compatibility matrix test"

---

## Workflow Walkthroughs

### Workflow 1: New Optimization Research

**Goal:** Understand and implement a new optimization technique

```
Step 1: SONNET - Gather Information
Prompt: "Find papers on [NEW TECHNIQUE]. What's the approach? 
         How does it differ from existing methods?"
Time: 30-60 seconds
Output: Summary of papers, key ideas, implementation notes

Step 2: OPUS - Design Implementation
Prompt: "Based on [SONNET'S FINDINGS], design how to implement 
         [NEW TECHNIQUE] for our CPU inference system.
         Consider: compatibility, memory, threading model."
Time: 2-5 minutes
Output: Architecture, implementation plan, validation strategy

Step 3: HAIKU - Execute
Prompt: "Implement the plan: [OPUS'S STEPS].
         Run benchmarks on [MODELS].
         Report metrics."
Time: Varies (5-60 minutes depending on complexity)
Output: Benchmark results, metrics table

Step 4: SONNET - Analyze & Summarize
Prompt: "Analyze these results: [HAIKU'S METRICS].
         How does it compare to our current approach?
         What's the speedup gain?"
Time: 30-60 seconds
Output: Quick analysis, comparison table, next steps

Step 5: (Optional) OPUS - Deep Analysis
Prompt: "Why did [TECHNIQUE] outperform on [MODEL] but 
         underperform on [OTHER MODEL]?
         What architectural properties explain this?"
Time: 2-3 minutes
Output: Detailed analysis, insights for future work
```

**Total cost:** ~1 OPUS + 2 SONNET + 1 HAIKU (efficient!)

---

### Workflow 2: Troubleshooting & Debugging

**Goal:** Fix a broken feature or optimize a bottleneck

```
Step 1: Describe Problem → Any model can help gather initial info
Prompt: "What's happening with [FAILING COMPONENT]? 
         Check logs: [LOG FILE]. What do you see?"
Model: HAIKU (fast check) OR SONNET (if analysis needed)
Time: 1-2 minutes
Output: Quick observation of problem

Step 2: OPUS - Deep Debugging
Prompt: "The issue is: [PROBLEM DESCRIPTION].
         Debug this: [CODE/LOGS].
         Trace through execution. Find root cause."
Time: 3-10 minutes
Output: Root cause, proposed fix, implementation steps

Step 3: HAIKU - Execute Fix
Prompt: "Implement this fix: [OPUS'S STEPS].
         Run validation tests: [TEST COMMANDS].
         Report metrics."
Time: 2-15 minutes
Output: Validation results, metrics

Step 4: SONNET - Verify & Document
Prompt: "The fix improved performance by [METRICS].
         How does it affect other areas?
         Is it stable across different models?"
Time: 1-2 minutes
Output: Risk assessment, recommendation
```

**Total cost:** 1 OPUS + 1 SONNET + 1 HAIKU (focused debugging!)

---

### Workflow 3: Routine Benchmarking

**Goal:** Measure performance of a known-working optimization

```
Step 1: HAIKU - Run Benchmark
Prompt: "Run benchmark on [MODEL] with [CONFIGURATION].
         Test K=8,16,24.
         Measure: acceptance rate, tokens/sec, speedup.
         Create comparison table."
Time: 5-30 minutes (depending on test duration)
Output: Benchmark results, metrics

Step 2: (Optional) SONNET - Quick Analysis
Prompt: "Analyze these benchmarks: [RESULTS TABLE].
         What patterns do you see?
         What's the optimal K value and why?"
Time: 30 seconds
Output: Quick insights, recommendation

Step 3: (Optional) OPUS - Deep Analysis
Prompt: "Why did K=16 outperform K=8 and K=24?
         What architectural properties explain this?
         How will this change with different model sizes?"
Time: 2-3 minutes
Output: Detailed insights for future optimization
```

**Total cost:** 1 HAIKU + (0-1 SONNET) + (0-1 OPUS) (varies by depth!)

---

### Workflow 4: Multi-Track Optimization

**Goal:** Implement and compare multiple optimization techniques together

```
Step 1: OPUS - Design Integration
Prompt: "Design how to combine Track 1 + Track 2 + Track 6.
         How do they interact?
         What's the implementation strategy?
         What are the potential conflicts?"
Time: 3-5 minutes
Output: Integration architecture, implementation plan

Step 2: HAIKU - Execute Implementation
Prompt: "Implement the integration plan: [OPUS'S STEPS].
         Run tests: [TEST SUITE].
         Benchmark: [MODELS].
         Report metrics and status."
Time: 30-120 minutes (depending on complexity)
Output: Implementation status, benchmark results

Step 3: SONNET - Analyze & Summarize
Prompt: "The results show: [METRICS].
         How much speedup from each component?
         What's the final combined speedup?"
Time: 1-2 minutes
Output: Contribution analysis, summary metrics

Step 4: (Optional) OPUS - Optimization Iteration
Prompt: "We achieved [SPEEDUP] with the combined approach.
         Where's the next bottleneck?
         What's the most impactful next optimization?
         Design it."
Time: 2-5 minutes
Output: Next optimization strategy
```

**Total cost:** 1-2 OPUS + 1 SONNET + 1 HAIKU (reasonable for complex work!)

---

## Cost-Conscious Strategy

### Budget Optimization (Minimize Expensive Calls)

**Rule 1: Use SONNET First**
- Ask SONNET exploratory questions
- Only escalate to OPUS if SONNET hits its limits
- Saves: 10-15x on some tasks

**Rule 2: Batch HAIKU Tasks**
- Run multiple benchmarks in sequence
- Parse multiple logs at once
- Costs: Nearly same as single task
- Saves: Repeated setup overhead

**Rule 3: Reuse OPUS Outputs**
- When OPUS designs something, HAIKU implements
- Don't re-ask OPUS same question
- When results change, use SONNET first to understand
- Saves: Duplicate analysis costs

**Rule 4: Cache Information**
- Document findings in research_report.md
- Reference in future OPUS/SONNET prompts
- Reduces context window usage
- Saves: 10-20% token budget

---

## Prompt Optimization Tips

### For OPUS (Deep Work):
- Provide full context upfront
- Be specific about constraints
- Ask for step-by-step breakdowns
- Request rollback/validation procedures
- Example format: "Given [CONTEXT], design [TASK] considering [CONSTRAINTS]. Output: [SPECIFIC STEPS]"

### For SONNET (Research/Synthesis):
- Ask multiple related questions together (parallelizable!)
- Request specific format (bullets, tables, etc.)
- Ask for comparisons or contrasts
- Example format: "Research [TOPIC]. Find: [3 specific things]. Format as: [TABLE/LIST]"

### For HAIKU (Execution):
- Be very specific about commands/steps
- Provide exact file paths
- Request specific output format
- Keep it focused (don't ask for analysis)
- Example format: "Run: [COMMAND]. Extract: [METRIC]. Report as: [TABLE]"

---

## Decision Tree (Visual)

```
                     ┌─ Your Prompt ──────┐
                     │                    │
        ┌────────────┴────────────┐      
        │                         │      
   [Complex?]                [Research/         
   (design,              Search/Parallel?]     
    debug,          (find, search,                
    implement,      parallel, web)                
    root cause)     │           │                
        │           │      ┌────┴──────────┐    
        │       YES │      │               │    
        │           ↓      │          [Routine?]
       YES       SONNET    │          (bench,   
        │                  │           test,    
        ↓                  │           parse,   
      OPUS                 │           collect) 
                           │          │   │     
                       YES │      YES │   NO    
                           ↓         ↓   │      
                        SONNET    HAIKU  │      
                                       DEFAULT  
                                       SONNET
```

---

## Quick Checklist Before Prompting

- [ ] Is this task well-defined or exploratory?
  - Well-defined → Right model for task type
  - Exploratory → SONNET (gather info first)
  
- [ ] Do I need parallel processing or sequential?
  - Parallel (multiple questions) → SONNET
  - Sequential (build on previous) → OPUS or HAIKU
  
- [ ] Is the command/approach known to work?
  - Yes → HAIKU
  - No → OPUS (design) then HAIKU (execute)
  
- [ ] What's my deadline?
  - Minutes → HAIKU or SONNET
  - Hours → OPUS is ok
  - Days → OPUS + full strategy
  
- [ ] What's my budget/cost constraint?
  - Tight → Maximize HAIKU, minimize OPUS
  - Normal → Mix as needed
  - Unlimited → Any approach ok

---

## Integration with Your System

### In CLAUDE.md:
The full routing strategy is documented in the "Model Routing Strategy" section

### Quick Reference:
Use `MODEL_ROUTING_QUICK_REFERENCE.md` for fast lookups

### In Practice:
- Start with the decision tree above
- Check Quick Reference for keywords
- Follow appropriate workflow template
- Monitor costs and adjust strategy as needed

---

## Example: Your Current Project State

**Current Status:**
- Track 1: 5.9x speedup ✓ (proven, use HAIKU for benchmarking)
- Track 2: 21-48% gain ✓ (proven, use HAIKU for testing)
- Track 6: Research phase → Use SONNET for papers
- Track 8: Implementation phase → Use OPUS to design, HAIKU to test

**Recommended Approach for Next Week:**
```
Day 1-2: SONNET research on Track 6 & 8
Day 2-3: OPUS design the implementations
Day 3-5: HAIKU benchmarking & validation
Day 5-6: SONNET analysis & comparison
```

**Estimated Cost:** ~3 OPUS + 4 SONNET + 10 HAIKU ≈ 60-80 HAIKU-equivalent units

---

## Final Rules of Thumb

1. **OPUS is expensive.** Use for blockers only.
2. **SONNET is your friend.** Use for exploratory work.
3. **HAIKU is cheap.** Use for all repetitive work.
4. **Default to SONNET.** When uncertain, pick SONNET.
5. **Batch operations.** Multiple items → same cost as one.
6. **Parallelize with SONNET.** Ask 3 questions at once if possible.
7. **Cache results.** Don't re-ask OPUS the same question.
8. **Validate with HAIKU.** Quick execution checks are cheap.


---

