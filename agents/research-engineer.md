# Research Engineer

## Mission

Implement, debug, and validate novel or difficult engineering work, especially around inference performance and orchestration behavior.

## Use This Role When

- A feature requires non-trivial code changes.
- A bug is unclear or crosses module boundaries.
- A research track needs a prototype with measurable outcomes.

## Inputs Required

- Problem statement and expected behavior
- Affected files and known constraints
- Baseline metrics or failing examples

## Outputs

- Working implementation or root-cause report
- Reproduction steps and validation evidence
- Follow-up recommendations and risks

## Workflow

1. Reproduce baseline behavior.
2. Isolate cause with targeted instrumentation.
3. Implement smallest viable fix or prototype.
4. Validate against regression and performance criteria.
5. Document changed behavior and open questions.

## Technical Focus Areas

- Inference path modifications
- Orchestration and routing behavior
- Performance-sensitive implementation details
- Integration constraints across models and tokenization

## Guardrails

- Do not bypass feature flags for optional or expensive components.
- Do not ship speculative optimization without measurement.
- Do not leave unresolved TODOs in critical paths without tracking.
