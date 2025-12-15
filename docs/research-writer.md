# Research Writer Agent Guide

This document describes the research-writer agent's capabilities and workflows.

---

## Quick Reference


## Installation (One-Time)

```bash
# On Beelzebub
mkdir -p /mnt/raid0/llm/claude/agents
cp research-writer.md /mnt/raid0/llm/claude/agents/
cp report_update_workflow.sh /mnt/raid0/llm/claude/scripts/utils/
chmod +x /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh
```

---

## Common Workflows

### After Benchmarking

```bash
# 1. Collect benchmark data
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --benchmark /mnt/raid0/llm/LOGS/zen5_benchmark_20251215_143022.csv

# 2. Copy output, paste into Claude Code:
@research-writer update 'Benchmark Results' section with:
[paste workflow output]
```

### Track Completion

```bash
# 1. Mark track as complete
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --track "Track 1" "âœ… Production" "5.9x speedup validated"

# 2. Invoke agent
@research-writer update Track 1 section: [paste above]
```

### Full Report Refresh

```bash
# 1. Collect all data
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --summary

# 2. Request comprehensive update
@research-writer refresh entire research_report.md with latest data:
[paste summary]
```

### Validate Report

```bash
# Check for missing sections, stale timestamps, inconsistencies
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --validate

# Ask agent to fix issues
@research-writer fix the following report issues:
[paste validation output]
```

### View Current Report

```bash
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --show
```

---

## Agent Invocation Templates

### Template 1: Benchmark Update

```
@research-writer I completed a K-sweep benchmark. Please update research_report.md:

[paste workflow output from --benchmark]

Specifically:
1. Add new row to K-Value Optimization table
2. Analyze acceptance rate vs. speedup tradeoff
3. Update "Optimal K by Content Type" recommendation
4. Note source file: [CSV path]
```

### Template 2: Track Status

```
@research-writer Track [N] milestone reached: [description]

[paste workflow output from --track]

Update the Track [N] section with:
1. Status change to [new status]
2. Key metrics ([speedup], [acceptance rate])
3. Models tested
4. Command template for reproduction
5. Next milestone
```

### Template 3: Full Summary

```
@research-writer comprehensive report update needed.

[paste workflow output from --summary]

Please create/update research_report.md with:
1. Executive Summary (current status of all tracks)
2. System Configuration (hardware and software)
3. Tested Models (compatibility matrix)
4. Benchmark Results (all K-sweeps and results)
5. Key Findings (architecture insights, what works/doesn't)
6. Combined Optimization Stack (Tiers 1-4)
7. Future Work (implementation priorities)
8. Literature References (papers and resources)

Ensure:
- All speedups are from verified benchmarks
- Timestamps are current
- Cross-references are consistent
- Reproducibility information is complete
```

### Template 4: Literature Update

```
@research-writer we're implementing Track [N]. Please integrate academic context.

Paper: [Title/Link]
Key Findings: [Brief summary]

Update research_report.md with:
1. Add to Literature References section
2. Include in relevant Track description
3. Cite methodology parallels
4. Link to full paper
```

---

## Workflow Script Options

```bash
# Post-benchmark update
./report_update_workflow.sh --benchmark FILE.csv

# Track status change
./report_update_workflow.sh --track TRACK_NAME STATUS [details]

# Full data collection
./report_update_workflow.sh --summary

# Consistency check
./report_update_workflow.sh --validate

# View current report
./report_update_workflow.sh --show

# Help
./report_update_workflow.sh --help
```

---

## File Locations

| File | Purpose | Location |
|------|---------|----------|
| Agent Definition | Defines role & workflow | `/mnt/raid0/llm/claude/agents/research-writer.md` |
| Workflow Script | Automates data collection | `/mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh` |
| Research Report | Main output | `/mnt/raid0/llm/LOGS/research_report.md` |
| Benchmark Data | Input for updates | `/mnt/raid0/llm/LOGS/zen5_benchmark_*.csv` |
| Model Tests | Compatibility data | `/mnt/raid0/llm/LOGS/tested_models.json` |
| Agent Audit | Activity log | `/mnt/raid0/llm/LOGS/agent_audit.log` |

---

## Best Practices

✅ **DO:**
- Run `--validate` before major updates
- Cite source files in report updates
- Keep timestamps current
- Cross-reference sections consistently
- Use workflow script to collect data first

❌ **DON'T:**
- Update report manually without workflow data
- Paste unvalidated benchmark numbers
- Leave timestamps stale
- Create inconsistencies between sections
- Skip validation before finalizing

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| "Benchmark file not found" | Check path with `find /mnt/raid0/llm/LOGS -name "*.csv"` |
| "Validation FAILED" | Run `--validate` and ask @research-writer to fix issues |
| "Report section missing" | Manually add section or ask @research-writer to create it |
| "Stale timestamps" | Workflow script is responsible for this; @research-writer should update |

---

## Integration Ideas

### Daily Recap
```bash
# In a cron job or manual daily check:
bash report_update_workflow.sh --summary | \
  mail -s "Daily Research Recap" your_email@example.com
```

### Post-Benchmark Automation
```bash
# At end of bench_zen5.sh:
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --benchmark "$RESULTS_FILE"
```

### Git Hooks (if using version control)
```bash
# Pre-commit: validate report is consistent
bash report_update_workflow.sh --validate || exit 1
```

---

## First-Time Setup

1. **Copy files** → Install agent definition and workflow script
2. **Run validation** → `bash report_update_workflow.sh --validate`
3. **Generate summary** → `bash report_update_workflow.sh --summary`
4. **Invoke agent** → Copy summary, ask `@research-writer` to initialize report
5. **Verify output** → `bash report_update_workflow.sh --show`
6. **Bookmark this file** → Keep this quick reference handy

---

## Example Session

```
You: "Initialize research reporting for this project"

bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --summary

[copy output]

@research-writer create comprehensive research_report.md from:
[paste summary]

# Later...

You: "K-sweep completed. Update report with results."

bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --benchmark /mnt/raid0/llm/LOGS/zen5_benchmark_20251215_143022.csv

[copy output]

@research-writer add K-sweep results to Benchmark Results section:
[paste output]

# Later...

You: "Validate report consistency"

bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --validate

@research-writer fix any issues shown above

# Later...

You: "Show final report"

bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --show
```

---

**Ready to use! Copy these files to Beelzebub and start coordinating research reports with @research-writer.**

---

## Integration Guide


## Overview

This guide explains how to integrate the Research Writer agent into your AMD EPYC inference optimization project. The setup consists of:

1. **Agent Definition** (`research-writer.md`) — Defines the agent's role, responsibilities, and workflow
2. **Workflow Script** (`report_update_workflow.sh`) — Automates data collection and report update invocations
3. **Integration Points** — Where to invoke @research-writer in your Claude Code sessions

---

## Installation

### Step 1: Create Agent Directory

```bash
mkdir -p /mnt/raid0/llm/claude/agents
```

### Step 2: Copy Agent Definition

Copy `research-writer.md` to the agents directory:

```bash
cp research-writer.md /mnt/raid0/llm/claude/agents/
```

### Step 3: Copy Workflow Script

Copy `report_update_workflow.sh` to your scripts directory:

```bash
cp report_update_workflow.sh /mnt/raid0/llm/claude/scripts/utils/
chmod +x /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh
```

### Step 4: Verify Installation

```bash
ls -la /mnt/raid0/llm/claude/agents/research-writer.md
ls -la /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh
```

---

## Quick Start

### Scenario 1: Post-Benchmark Report Update

After running a K-sweep benchmark:

```bash
# Terminal 1: Run benchmark
bash /mnt/raid0/llm/claude/scripts/benchmark/bench_zen5.sh Qwen2.5-Coder-32B
# Produces: /mnt/raid0/llm/LOGS/zen5_benchmark_20251215_143022.csv

# Terminal 2: Trigger report update
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --benchmark /mnt/raid0/llm/LOGS/zen5_benchmark_20251215_143022.csv

# Output: Instructions and data summary for @research-writer
```

**Then in Claude Code:**

```
You: "Copy the output above and use it to update research_report.md"

[paste workflow script output]

@research-writer please update the 'Benchmark Results' section with 
these K-sweep findings. Include:
- New table entry with speedup and acceptance rates
- Analysis of the K=24 → 10.0x result
- Updated 'Optimal K by Content Type' recommendation
- Link to the CSV file for reproducibility
```

### Scenario 2: Track Milestone Completion

When Track 1 reaches production status:

```bash
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --track "Track 1" "âœ… Production" "5.9x speedup validated on Qwen2.5-Coder-32B"
```

**In Claude Code:**

```
@research-writer Track 1 (External Draft Model) is now production-ready.

Update research_report.md with:
1. Status changed to âœ… Production in the Track 1 section
2. Add working configuration: Qwen2.5-Coder-32B + Qwen2.5-0.5B-Instruct
3. Document 5.9x speedup with K=16 optimal
4. Add compatibility matrix showing what works/doesn't
5. Set next milestone: "Implement Track 6 + 8 for compound gains"
```

### Scenario 3: Generate Full Report Summary

When you want a comprehensive update across all data:

```bash
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --summary
```

**In Claude Code:**

```
[paste the full summary from the script]

@research-writer use this data snapshot to create a comprehensive 
research_report.md update. Specifically:

1. Update Executive Summary with current status of all tracks
2. Synthesize Benchmark Results across all tested models
3. Update Key Findings with the latest architectural insights
4. Document the combined optimization stack progress
5. Ensure all timestamps are current
```

### Scenario 4: Validate Report Consistency

Before finalizing a major update:

```bash
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --validate
```

This checks for:
- All required sections present
- Track references consistent
- Speedup values reasonable
- Last Updated timestamp fresh

---

## Integration with Existing Workflow

### Within a Single Claude Code Session

```
You: "Let's update the research report with today's findings.
     
     First, run validation"

bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --validate

You: "@research-writer validation output above shows report is missing
     'Future Work' section. Add it with:"
     
@research-writer: [adds section]

You: "Now collect today's benchmark data"

bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --benchmark /mnt/raid0/llm/LOGS/zen5_benchmark_20251215.csv

You: "@research-writer update Benchmark Results with output above"

@research-writer: [updates section]

You: "Track 2 (MoE soft mask) is now validated on 4 models with 21-48% gains"

bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --track "Track 2" "âœ… Production" "21-48% on MoE models"

You: "@research-writer update Track 2 status with output above"

@research-writer: [updates section]

You: "Show me the final report"

bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --show

You: "Validate it one more time"

bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh --validate
```

### Automation via Session Initialization

Add to your `session_init.sh`:

```bash
# At the end, check if report needs update
if [[ $(find /mnt/raid0/llm/LOGS -name "zen5_benchmark_*.csv" -newer /mnt/raid0/llm/LOGS/research_report.md 2>/dev/null) ]]; then
    echo ""
    echo "⚠️  New benchmark results available since last report update"
    echo "Run: bash report_update_workflow.sh --summary"
    echo ""
fi
```

---

## Best Practices

### 1. Always Validate Before Major Updates

```bash
bash report_update_workflow.sh --validate
# Fix any issues before running @research-writer updates
```

### 2. Use Descriptive Commit-Style Messages

When updating track status:

```bash
bash report_update_workflow.sh \
  --track "Track 6" "IN PROGRESS" \
  "Implementing SuffixDecoding. Read paper, created draft Python wrapper. Testing with Qwen2.5 next."
```

### 3. Source Control (if using git)

```bash
cd /mnt/raid0/llm
git diff LOGS/research_report.md
# Review changes before committing
git add LOGS/research_report.md
git commit -m "Update: Track 1 production (5.9x validated), start Track 6"
```

### 4. Link to Raw Data

When @research-writer updates a section, ensure they cite source files:

```
"Update 'K-Value Optimization' section with:
 - Table from /mnt/raid0/llm/LOGS/zen5_benchmark_20251215_143022.csv
 - Summary: K=24 achieves 10.0x speedup (83% acceptance)"
```

### 5. Cross-Reference Consistently

The report should be self-consistent. Use workflow validation to catch issues:

```
If you say "Track 1: 5.9x" in Executive Summary,
then "K=16 achieves 5.9x" in Benchmark Results,
and "Qwen2.5-0.5B draft" in Tested Models,
these must all align.
```

---

## Troubleshooting

### Issue: "Benchmark file not found"

```bash
# Check where benchmarks actually are
find /mnt/raid0/llm/LOGS -name "*.csv" -type f

# Run with correct path
bash report_update_workflow.sh --benchmark /path/to/actual/file.csv
```

### Issue: "Report validation FAILED (X issues found)"

The script will tell you what's missing. Common fixes:

```bash
# Missing section? Manually add it or ask @research-writer to add it
# "Missing: Future Work section"

@research-writer add a "Future Work" section to research_report.md with:
- Short-term priorities
- Medium-term goals
- Publication targets
```

### Issue: "Audit log shows stale entries"

The report may have old timestamps. Update them:

```
@research-writer update all "Last Updated" timestamps in research_report.md
to today's date (2025-12-15)
```

---

## Advanced: Custom Workflow Triggers

Create scripts that automatically invoke @research-writer:

### Option A: Daily Recap Script

```bash
#!/bin/bash
# daily_research_recap.sh

REPORT_SCRIPT="/mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh"

echo "=== Daily Research Recap ==="
bash $REPORT_SCRIPT --summary > /tmp/recap_$$.txt

# Print for copy-paste into Claude
echo "Paste this into Claude Code with @research-writer:"
echo ""
cat /tmp/recap_$$.txt
```

### Option B: Post-Benchmark Hook

In `bench_zen5.sh`, add at the end:

```bash
# After benchmark completes
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --benchmark "$RESULTS_FILE"
```

---

## Integration Checklist

- [ ] Created `/mnt/raid0/llm/claude/agents/` directory
- [ ] Copied `research-writer.md` to agents directory
- [ ] Copied `report_update_workflow.sh` to scripts/utils
- [ ] Made script executable: `chmod +x report_update_workflow.sh`
- [ ] Tested workflow: `bash report_update_workflow.sh --show`
- [ ] Read `research-writer.md` to understand invocation patterns
- [ ] Added note to `.bashrc` or session init about new workflow
- [ ] Identified first use case (e.g., post-benchmark update)
- [ ] Ran `--validate` on existing research_report.md to identify gaps
- [ ] Ready to invoke `@research-writer` in Claude Code

---

## Next Steps

1. **First Invocation**: Copy the agent definition and run a summary
2. **Initial Report**: Ask @research-writer to create comprehensive research_report.md
3. **Establish Rhythm**: Use workflow script after each major milestone
4. **Automate**: Add hooks to benchmarking scripts
5. **Publish**: Once report is comprehensive, consider blog post or technical publication

---

**Example First Use:**

```bash
# On Beelzebub, after copying files:
cd /mnt/raid0/llm
bash ./scripts/utils/report_update_workflow.sh --summary

# Copy output, paste into Claude Code:
claude

# In Claude Code:
@research-writer initialize the research report for our EPYC inference 
optimization project using this data snapshot:

[paste output from workflow script]

Create a comprehensive research_report.md with sections for:
- Executive Summary (Track 1+2 production, next: Track 6+8)
- System Configuration (EPYC 9655, 1.13TB DDR5, llama.cpp)
- Tested Models (Qwen2.5 family working, SSM incompatible)
- Benchmark Results (K-sweep data)
- Key Findings (tokenizer compatibility critical)
- Combined Optimization Stack
- Future Work (Tier 1-4 tracks)
- Literature References

Save to /mnt/raid0/llm/LOGS/research_report.md
```

Done! You now have a dedicated research agent coordinating report generation.

---

## Summary


## What Was Created

You now have a **dedicated research-writer agent** for your AMD EPYC inference optimization project. This framework consists of three components:

### 1. **research-writer.md** (Agent Definition)
- Defines the agent's expertise, responsibilities, and workflow
- Specifies interaction patterns with other agents
- Documents quality standards and best practices
- Includes invocation templates and resource references

### 2. **report_update_workflow.sh** (Automation Script)
- Collects data from benchmarks, logs, and test results
- Validates data consistency before report updates
- Generates structured summaries for the agent to work from
- Provides validation and verification capabilities

### 3. **Integration Documentation** (Setup & Usage)
- Installation instructions
- Quick reference card for common workflows
- Troubleshooting guide
- Example invocation patterns

---

## Why This Matters

### Before (Current State)
- Research reports are written **ad hoc** without dedicated coordination
- Benchmark data, logs, and findings exist in separate files
- No systematic process for keeping reports updated
- Risk of inconsistencies between sections (e.g., Track 1 status differs between sections)
- Manual data collection each time @research-writer is invoked

### After (With This Framework)
```
You run benchmark → Script collects data → Invokes @research-writer 
→ Report updated automatically with source citations → Validation 
ensures consistency
```

---

## Key Design Decisions

### 1. **Explicit Invocation** (Not Auto-Magic)
The agent must be explicitly invoked with `@research-writer`. This keeps you in control and prevents accidental report modifications.

### 2. **Data-Driven Updates**
The workflow script always collects source data first. This ensures:
- Benchmarks are cited with file paths
- Results are verified before inclusion
- Timestamps are accurate
- Reproducibility is documented

### 3. **Composable Workflows**
Four main invocation patterns handle different scenarios:
- **Post-benchmark**: Update specific results section
- **Track milestone**: Update track status and metrics
- **Full refresh**: Comprehensive report update
- **Validation**: Consistency checking

### 4. **Integration with Logging**
Reports integrate with your existing `agent_audit.log` system. The workflow script can see what each agent has done.

---

## How to Use (30-Second Overview)

### Setup (One-Time)
```bash
# Copy files to project
mkdir -p /mnt/raid0/llm/claude/agents
cp research-writer.md /mnt/raid0/llm/claude/agents/
cp report_update_workflow.sh /mnt/raid0/llm/claude/scripts/utils/
chmod +x /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh
```

### After Benchmarking
```bash
# Collect data
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --benchmark /mnt/raid0/llm/LOGS/zen5_benchmark_*.csv

# In Claude Code, paste output:
@research-writer update Benchmark Results section with [pasted data]
```

### After Major Milestone
```bash
# Mark completion
bash /mnt/raid0/llm/claude/scripts/utils/report_update_workflow.sh \
  --track "Track 1" "âœ… Production" "5.9x speedup validated"

# In Claude Code:
@research-writer update Track 1 status with [pasted data]
```

---

## Agent Invocation Examples

### Example 1: K-Sweep Results

```bash
# Terminal
bash report_update_workflow.sh --benchmark \
  /mnt/raid0/llm/LOGS/zen5_benchmark_20251215_143022.csv

# Output shows:
# - K values tested: 4, 8, 12, 16, 20, 24
# - Best result: K=24 → 28.79 t/s (10.0x speedup)
# - Acceptance rates: 80-96% range
```

**Claude Code:**

```
@research-writer update 'K-Value Optimization' table with these results:

K=4: 11.73 t/s, 96.77% acceptance
K=8: 17.32 t/s, 86.84% acceptance
...
K=24: 28.79 t/s, 83.33% acceptance

Key insight: Aggressive K (20-24) optimal for code generation.
Update "Optimal K by Content Type" to recommend K=24 for structured,
K=8-12 for general, K=4-8 for prose.

Source: /mnt/raid0/llm/LOGS/zen5_benchmark_20251215_143022.csv
```

### Example 2: Track Completion

```bash
# Terminal
bash report_update_workflow.sh --track "Track 1" "âœ… Production" \
  "Qwen2.5-Coder-32B + Qwen2.5-0.5B-Instruct tested with K=16 for 5.9x speedup"
```

**Claude Code:**

```
@research-writer Track 1 is now production-ready. Update the report:

Status: âœ… Production (was ðŸ"„ In Progress)
Primary Configuration:
- Target: Qwen2.5-Coder-32B-Q4_K_M (18.5GB)
- Draft: Qwen2.5-0.5B-Instruct-Q8_0 (507MB)
- Optimal K: 16
- Speedup: 5.9x
- Acceptance Rate: 66%

Add compatibility matrix showing:
✅ Works: Qwen2.5 family → Qwen2.5-0.5B
❌ Fails: DeepSeek-R1 family (vocab mismatch)
❌ Fails: Qwen3 (BOS token mismatch)

Next milestone: Implement Track 6 (SuffixDecoding) + Track 8 (Prompt Lookup)
```

### Example 3: Full Report Refresh

```bash
# Terminal
bash report_update_workflow.sh --summary

# Output includes:
# - 47 new benchmark results since last report
# - 5 tracks tested
# - 2 new architectural insights (SSM incompatibility)
# - 3 new papers added to literature
```

**Claude Code:**

```
@research-writer comprehensive report update.

[pasted summary from script]

Create/update research_report.md with:

1. **Executive Summary**
   - Track 1: âœ… Production (5.9x)
   - Track 2: âœ… Production (21-48%)
   - Track 6: ðŸ†• In Progress (Suffix trees)
   - Track 8: ðŸ†• In Progress (Prompt lookup)

2. **Benchmark Results** 
   - Include all K-sweeps from summary
   - Show context-dependent variation (code vs prose)

3. **Key Findings**
   - Tokenizer compatibility is critical (same family only)
   - NUMA interleaving essential for bandwidth saturation
   - SSM architectures fundamentally incompatible

4. **Combined Optimization Stack**
   - Tier 1: Prompt Lookup + SuffixDecoding (stacking!)
   - Tier 2: MoE Soft Mask (orthogonal)
   - Expected: 8-15x on grounded/agentic tasks

5. **Future Work**
   - Track 6/8 implementation (1-2 days)
   - Track 7 (CAS-Spec) if needed for DeepSeek-R1
   - Publication: "Maximizing LLM Inference on AMD EPYC Turin"

Ensure all timestamps are current and cross-references consistent.
```

### Example 4: Validation & Fixes

```bash
# Terminal
bash report_update_workflow.sh --validate

# Output shows:
# ✅ Executive Summary
# ✅ System Configuration
# ❌ Tested Models (MISSING)
# ✅ Benchmark Results
# ❌ Future Work (MISSING)
# ✅ Literature References
# Errors: 2 issues found
```

**Claude Code:**

```
@research-writer report validation found 2 missing sections.
Please add:

1. **Tested Models** section with:
   - Table: Model | Format | Size | Spec Decode Status
   - Working: Qwen2.5-Coder-32B, Meta-Llama-70B
   - Blocked: Qwen3-Next (SSM), DeepSeek-R1 (no draft)

2. **Future Work** section with:
   - Tier 1: Track 6/8 implementation (1-2 days, 8-15x expected)
   - Tier 2: Track 7 (CAS-Spec) for non-MoE models
   - Tier 3: Publish findings (blog + paper)

Also update "Last Updated" timestamp to today.
```

---

## Integration with Your Existing Workflow

### Your Current Setup
```
Session 1: Run benchmark (bash)
Session 2: Ask Claude agents questions (claude code)
Session 3: Manual report updates (claude code)
```

### With This Framework
```
Session 1: Run benchmark (bash)
Session 2: Run workflow script (bash) → Collect data
Session 3: Invoke @research-writer (claude code) → Auto-update report
Session 4: Validate with workflow script (bash) → Verify consistency
```

**Key benefit:** The workflow script becomes the "source of truth" for what data is available. No more guessing at report structure.

---

## Files Provided

| File | Purpose | Where to Copy |
|------|---------|---------------|
| `research-writer.md` | Agent definition | `/mnt/raid0/llm/claude/agents/` |
| `report_update_workflow.sh` | Automation script | `/mnt/raid0/llm/claude/scripts/utils/` |
| `RESEARCH_WRITER_INTEGRATION.md` | Detailed setup guide | For reference (or `/mnt/raid0/llm/claude/`) |
| `RESEARCH_WRITER_QUICK_REF.md` | Quick commands | Print or bookmark |
| This file | Overview | For reference |

---

## What's Next?

### Immediate (Today)
1. Copy the three files to your Beelzebub project
2. Run `report_update_workflow.sh --validate` on existing report
3. Ask `@research-writer` to initialize comprehensive report if needed

### Short-term (This Week)
1. After each benchmark, run `--benchmark` workflow
2. Invoke `@research-writer` to update specific sections
3. Run `--validate` before finalizing major changes

### Medium-term (Next Month)
1. Integrate workflow into `session_init.sh` to show stale data warnings
2. Create git commits with `report_update_workflow.sh --summary` in messages
3. Consider blog post: "Maximizing LLM Inference on AMD EPYC Turin"

---

## FAQ

### Q: Do I have to use the workflow script?
**A:** No, but it's highly recommended. It ensures data consistency and prevents manual errors.

### Q: Can I invoke @research-writer without the script?
**A:** Yes, but you'll need to manually collect benchmark data, verify paths, and ensure timestamps are correct. The script automates this.

### Q: Does this replace my existing report?
**A:** No. It enhances your process by adding systematic data collection and agent coordination.

### Q: How often should I update the report?
**A:** After each major milestone:
- After benchmarking a new model or K configuration
- When a track reaches a new status (âœ… Production, ðŸ†• In Progress, etc.)
- When a major architectural insight emerges
- Weekly as a best practice

### Q: Can I validate reports without Claude Code?
**A:** Yes, `bash report_update_workflow.sh --validate` runs independently.

### Q: What if the script doesn't find my benchmark file?
**A:** Check the exact path: `find /mnt/raid0/llm/LOGS -name "*.csv"` and use the full path.

---

## Example Workflow (Full Day)

```
08:00 - Start benchmark
       bash /mnt/raid0/llm/claude/scripts/benchmark/bench_zen5.sh Qwen2.5-Coder-32B
       (Runs in background for 1-2 hours)

09:00 - Meanwhile, in Claude Code
       @research-engineer review Track 6 (SuffixDecoding) implementation strategy
       @sysadmin monitor system during benchmark

11:00 - Benchmark completes
       Collect results:
       bash report_update_workflow.sh --benchmark \
         /mnt/raid0/llm/LOGS/zen5_benchmark_20251215_143022.csv

11:15 - Update report
       @research-writer update benchmark results section with [pasted data]
       @benchmark-analyst analyze acceptance rate curve

12:00 - Mark track progress
       bash report_update_workflow.sh --track "Track 1" "âœ… Production" "5.9x validated"
       @research-writer update Track 1 status with [pasted data]

13:00 - Comprehensive validation
       bash report_update_workflow.sh --validate
       @research-writer fix any issues found above

14:00 - Planning next work
       bash report_update_workflow.sh --summary
       @research-engineer based on current status, recommend Track 6/8 priority order
```

---

## Support

If you have questions after reading this:
1. Check `RESEARCH_WRITER_QUICK_REF.md` for common commands
2. Read `research-writer.md` for detailed agent role documentation
3. Review `RESEARCH_WRITER_INTEGRATION.md` for setup troubleshooting
4. Ask `@research-writer` directly in Claude Code for guidance

---

**You're now ready to manage research reports systematically with a dedicated agent. Welcome to organized inference research! 🚀**
