# Routing Intelligence: Semantic Classifiers + Factual-Risk Routing

**Created**: 2026-02-18 (consolidated from `classifier-refactoring.md` + `delegation-escalation-factual-risk-routing-track.md`)
**Status**: PHASE 1 COMPLETE — Phases 2-6 deferred as separate future work
**Priority**: HIGH
**Blocked by**: Nothing

---

## Problem

Two interrelated quality gaps in routing/escalation decisions:

1. **9 brittle heuristics** scattered across route modules (`_is_summarization_task`, `_should_use_direct_mode`, `_detect_output_quality_issue`, etc.) — keyword-based, fragile, caused regressions (e.g., date injection: 70% → 45.5%).
2. **No factual-risk signal** in routing — cheap-first accepts/rejects without considering hallucination risk, review triggers use `answer[:100]` as proxy, escalation is retry-count-based only.

These are the same problem: routing decisions lack semantic understanding of the input and output.

---

## Solution: Unified `src/classifiers/` Module

One classifier module serves both needs. Three classifier categories:

| Category | Purpose | Backing | Replaces |
|----------|---------|---------|----------|
| **A: Input** | Classify prompt intent, detect summarization/vision/direct-mode/factual-risk | MemRL embeddings + keyword fallback | `_is_summarization_task`, `_needs_structured_analysis`, `_should_use_direct_mode`, `_is_ocr_heavy_prompt` |
| **B: Output** | Parse verdicts, detect stubs, strip tool noise | Config-driven regex from YAML | `_is_stub_final`, `_strip_tool_outputs`, verdict parsing |
| **C: Quality** | Detect repetition, garble, quality issues + factual risk scoring | Configurable thresholds + risk features | `_detect_output_quality_issue` + new factual-risk scorer |

The factual-risk scorer (from LaCy research) becomes a Category C classifier producing `risk_score`, `risk_band`, and `risk_features` — used by cheap-first, review trigger, and escalation policy.

**Key principle**: New categories = YAML edits, not code changes. Everything in `orchestration/classifier_config.yaml`.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/classifiers/__init__.py` | Public API + lazy singletons |
| `src/classifiers/types.py` | Dataclasses (ClassificationResult, VerdictResult, RiskResult) |
| `src/classifiers/input_classifier.py` | MemRL-backed prompt classification |
| `src/classifiers/output_parser.py` | Structured output parsing (stub, verdict, tool stripping) |
| `src/classifiers/quality_detector.py` | Quality detection + factual-risk scoring |
| `orchestration/classifier_config.yaml` | All exemplars, patterns, thresholds, risk features |
| `tests/unit/test_classifiers.py` | Unit tests (40+ tests) |

## Files to Modify

| File | Changes |
|------|---------|
| `src/api/routes/chat_summarization.py` | Replace `_is_summarization_task()` with classifier call |
| `src/api/routes/chat_vision.py` | Replace `_needs_structured_analysis()`, `_is_ocr_heavy_prompt()` |
| `src/api/routes/chat_routing.py` | Replace `_should_use_direct_mode()` |
| `src/api/routes/chat_utils.py` | Replace `_is_stub_final()`, `_strip_tool_outputs()` |
| `src/api/routes/chat_review.py` | Replace `_detect_output_quality_issue()`, add risk-aware review trigger |
| `src/api/routes/chat.py` | Add risk-aware cheap-first condition in `_try_cheap_first` |
| `src/escalation.py` | Add risk-aware escalation branch (keep retry-count as fallback) |
| `src/features.py` | Add `semantic_classifiers: bool` flag |
| `orchestration/repl_memory/retriever.py` | Add `retrieve_for_classification()` method |
| `src/api/__init__.py` | Seed classification exemplars on API init |

---

## Implementation Phases

### Phase 0: Telemetry Integrity (prerequisite)

Fix existing broken logging before adding new telemetry:
- `orchestration/repl_memory/progress_logger.py`: implement missing `log_delegation()` method
- `src/api/routes/chat_pipeline/stream_adapter.py`: fix `log_exploration()` calls missing `tokens_spent`
- Add `delegation_policy_version` to routing logs

### Phase 1: Types + Config + Output Parsers (no MemRL, no risk) — ✅ COMPLETE (2026-02-19)

All 9 heuristics now delegate to `src/classifiers/`:
- `src/classifiers/types.py` — ClassificationResult, RoutingDecision, MatcherConfig
- `src/classifiers/keyword_matcher.py` — is_summarization_task, is_coding_task, is_stub_final, needs_structured_analysis, should_use_direct_mode, classify_and_route
- `src/classifiers/output_parser.py` — strip_tool_outputs, truncate_looped_answer
- `src/classifiers/quality_detector.py` — detect_output_quality_issue
- `src/classifiers/config_loader.py` — YAML config loading
- `src/classifiers/classification_retriever.py` — MemRL-backed classification
- `orchestration/classifier_config.yaml` — all keywords, patterns, thresholds, output_parsing
- Original functions in chat_utils.py and chat_review.py are thin delegating wrappers (zero import breakage)
- 61 unit tests in test_classifiers.py, all passing
- **Exit criteria met**: all heuristics delegate to classifiers, existing behavior preserved

### Phase 2: Input Classifier + MemRL Integration

- `src/classifiers/input_classifier.py` — embedding similarity + keyword fallback
- `orchestration/repl_memory/retriever.py` — add `retrieve_for_classification()`
- Seed exemplars from YAML into EpisodicStore on API init
- **Exit**: Input classification uses embeddings when available, falls back to keywords

### Phase 3: Factual-Risk Scorer (shadow mode)

Add risk features to quality_detector.py:
- Query intent flags (asks-for-facts/dates/names/citations)
- Claim density estimate
- Uncertainty lexical markers
- Optional spaCy mode behind config flag

Deploy as `shadow` mode — compute and log risk score/band/features, no routing changes.

Config additions:
- `factual_risk_mode: off|shadow|enforce`
- `factual_risk_threshold_low/high`
- `factual_risk_extractor: regex|spacy`

**Exit**: Risk features logged on every request, p95 overhead < 5ms (regex mode)

### Phase 4: Risk-Aware Routing (enforce mode)

Wire risk outputs into routing decisions:
- Cheap-first: `risk >= high` → bypass or strict pass criteria
- Review: `risk_band=high` → force review even if generic heuristics pass
- Escalation: high risk + uncertainty → earlier escalation
- Replace review objective proxy (`answer[:100]`) with structured objective

**Exit**: High-risk prompts show higher review/escalation rates without broad inflation

### Phase 5: Seeding/Eval/Debugger Integration

- Persist risk fields in `RoleResult` and diagnostics
- Add threshold sweep support in seeding harness
- Emit Pareto reports: factuality vs cost vs latency
- ClaudeDebugger can recommend threshold updates

### Phase 6: Controlled Rollout

`off` → `shadow` (7 days) → `enforce` on selected roles → global enable

---

## Factual-Risk Scorer Design

```python
class FactualRiskResult:
    risk_score: float       # [0, 1]
    risk_band: str          # "low" | "medium" | "high"
    risk_features: dict     # for telemetry/debugging

class QualityDetector:
    def assess_risk(self, prompt: str, context: str = "") -> FactualRiskResult:
        features = self._extract_features(prompt)
        score = self._compute_score(features)
        band = self._band(score)
        return FactualRiskResult(score, band, features)
```

Features (fast, deterministic, regex mode):
- `has_date_question`, `has_entity_question`, `has_citation_request`
- `claim_density` (assertion-like sentence ratio)
- `uncertainty_markers` (hedging language count)
- `factual_keyword_ratio`

---

## Config Shape (`orchestration/classifier_config.yaml`)

```yaml
input_classifiers:
  summarization:
    keywords: ["summarize", "summary", "tldr", "key points"]
    exemplars:
      - prompt: "Summarize this document"
        label: "summarization"
  direct_mode:
    keywords: ["what is", "who is", "when did"]
    negative_keywords: ["implement", "write code"]

output_parsers:
  verdict:
    patterns: { ok: "^OK$", wrong: "^WRONG:\\s*(.+)$" }
  stub:
    patterns: ["I don't have", "I cannot", "As an AI"]

quality_thresholds:
  repetition_ratio: 0.3
  min_answer_length: 10

factual_risk:
  mode: shadow  # off|shadow|enforce
  extractor: regex
  threshold_low: 0.3
  threshold_high: 0.7
  force_review_high: true
  early_escalation_high: false
```

---

## Research References

- LaCy: `arXiv:2602.12005` (factual-risk-aware call placement)
- FrugalGPT: `arXiv:2305.05176` (LLM cascades/routing)
- Language Model Cascades: `arXiv:2207.10342`
- Chain-of-Verification: `arXiv:2309.11495` (factuality checks)
- Learning to Route LLMs: `arXiv:2501.12388`

---

## KPIs

- **Primary**: Factuality@Cost (weighted factual score / token+latency budget)
- **Secondary**: escalation rate by risk band, cheap-first pass rate by risk band, p50/p95 latency deltas
- **Safety gates**: no >10% p95 latency regression, no >5% cost regression at equal factuality

---

## Verification

```bash
python3 -m pytest tests/unit/test_classifiers.py -v
python3 -m pytest tests/unit/ -x -q
python scripts/benchmark/compare_orchestrator_direct.py --debug --suite all
make gates
```
