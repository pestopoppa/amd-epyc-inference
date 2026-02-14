# Research Writer Workflow

## Primary Script

Use `scripts/utils/report_update_workflow.sh` to gather validated source data before writing updates.

## Common Commands

```bash
bash scripts/utils/report_update_workflow.sh --benchmark logs/<file>.csv
bash scripts/utils/report_update_workflow.sh --track "Track N" "<status>" "<details>"
bash scripts/utils/report_update_workflow.sh --summary
bash scripts/utils/report_update_workflow.sh --validate
bash scripts/utils/report_update_workflow.sh --show
```

## Required Output Quality

1. Every quantitative claim maps to a source artifact.
2. Status labels match latest validated outcomes.
3. Reproduction details are present for key results.
4. Validation failures are either fixed or explicitly documented.

## Prompt Template

```text
@research-writer update research_report.md using this workflow output:
[paste output]

Required updates:
1. New benchmark rows
2. Interpretation of acceptance-speed tradeoff
3. Updated recommendation section
4. Source paths and timestamps
```
