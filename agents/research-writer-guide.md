# Research Writer Quickstart

This guide replaces the older long-form handbook with a practical workflow for keeping research docs current.
Canonical workflow location: `docs/guides/agent-workflows/research-writer.md`.

## Primary Script

Use `scripts/utils/report_update_workflow.sh` to gather validated inputs before writing.

## Core Commands

```bash
# benchmark-driven update
bash scripts/utils/report_update_workflow.sh --benchmark logs/<file>.csv

# track status update
bash scripts/utils/report_update_workflow.sh --track "Track N" "<status>" "<details>"

# full refresh context
bash scripts/utils/report_update_workflow.sh --summary

# consistency check
bash scripts/utils/report_update_workflow.sh --validate

# print current report
bash scripts/utils/report_update_workflow.sh --show
```

## Prompt Templates

### Benchmark Update

```text
@research-writer update research_report.md using:
[paste workflow output]

Required updates:
1. Add benchmark table rows
2. Interpret acceptance-vs-speed tradeoff
3. Update recommendation section
4. Cite source files and timestamps
```

### Track Milestone Update

```text
@research-writer update Track N status in research_report.md:
[paste workflow output]

Required updates:
1. Status label and rationale
2. Verified metrics and tested models
3. Reproduction command template
4. Next milestone
```

### Full Report Refresh

```text
@research-writer refresh research_report.md from:
[paste workflow output]

Required updates:
1. Executive summary
2. Benchmark sections
3. Key findings
4. Priority next steps
```

## Definition of Done

- Every quantitative claim maps to a source artifact.
- Report status labels match latest validated outcomes.
- Reproduction information is present for major results.
- Validation checks pass or exceptions are explicitly documented.
