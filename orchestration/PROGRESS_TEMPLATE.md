# Orchestration Progress Report Template

Use this template for weekly/milestone progress updates on the Hierarchical Local-Agent Orchestration system.

---

## Report Header

```markdown
# Orchestration Progress Report

**Date:** YYYY-MM-DD
**Sprint/Milestone:** [Name or Number]
**Status:** [On Track / At Risk / Blocked]
```

---

## Sections

### 1. Executive Summary
Brief 2-3 sentence overview of progress and key decisions.

### 2. Implementation Status

| Component | Status | Progress | Notes |
|-----------|--------|----------|-------|
| TaskIR Schema | [Done/In Progress/Planned] | [0-100%] | |
| ArchitectureIR Schema | [Done/In Progress/Planned] | [0-100%] | |
| Model Registry | [Done/In Progress/Planned] | [0-100%] | |
| Registry Loader | [Done/In Progress/Planned] | [0-100%] | |
| Dispatcher | [Done/In Progress/Planned] | [0-100%] | |
| Front Door Prompt | [Done/In Progress/Planned] | [0-100%] | |
| Model Server | [Done/In Progress/Planned] | [0-100%] | |
| Executor | [Done/In Progress/Planned] | [0-100%] | |
| Context Manager | [Done/In Progress/Planned] | [0-100%] | |

### 3. Model Status

| Role | Model | Status | Performance |
|------|-------|--------|-------------|
| frontdoor | [Model Name] | [Active/Testing/Planned] | [X t/s] |
| coder_escalation | [Model Name] | [Active/Testing/Planned] | [X t/s] |
| ingest_long_context | [Model Name] | [Active/Testing/Planned] | [X t/s] |
| worker_general | [Model Name] | [Active/Testing/Planned] | [X t/s] |

### 4. Benchmark Updates
New benchmark results since last report.

| Model | Quant | Baseline | Optimized | Method |
|-------|-------|----------|-----------|--------|
| | | | | |

### 5. Blockers & Risks

| Issue | Impact | Mitigation | Owner |
|-------|--------|------------|-------|
| | [High/Medium/Low] | | |

### 6. Completed This Period
- [ ] Item 1
- [ ] Item 2

### 7. Planned Next Period
- [ ] Item 1
- [ ] Item 2

### 8. Decisions Made
Document key architectural or implementation decisions.

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| | | |

### 9. Test Status

| Test Suite | Pass | Fail | Skip | Coverage |
|------------|------|------|------|----------|
| Unit Tests | | | | |
| Integration | | | | |
| E2E | | | | |

---

## File Naming Convention

`PROGRESS_YYYY-MM-DD.md` or `PROGRESS_Sprint-N.md`

Store in: `orchestration/progress/`
