# SimpleQA 0% Fix + Debugger Agency Improvement

**Created**: 2026-02-17
**Status**: COMPLETE
**Priority**: HIGH

## Root Causes (4 layers)

1. **web_search()** DDG HTML scraper failed silently → models couldn't look up facts
2. **Scorer** used `exact_match` → missed answers buried in prose (6 confirmed near-misses)
3. **Architect prompt** forbade factual delegation → 37 near-empty hallucinated answers
4. **Debugger** too passive → diagnosed 25+ simpleqa batches as "model knowledge gap"

## Changes Made

| File | Change |
|------|--------|
| `orchestration/tools/web.py` | Retry (2 attempts, 1s delay), snippet extraction, Wikipedia opensearch fallback, typed errors |
| `scripts/benchmark/dataset_adapters.py` | SimpleQA scoring: `exact_match` → `f1` with threshold=0.8 |
| `orchestration/prompts/architect_investigate.md` | Allow factual delegation to worker_explore when uncertain |
| `orchestration/prompts/debugger_system.md` | Tool/Scorer investigation section, suite-level analysis, action bias, reload guidance |
| `src/pipeline_monitor/claude_debugger.py` | Suite-level failure alerts in prompt builder, action-biased tail prompt |

## Verification

- `fetch_wikipedia()` uses `exsentences` param for accurate truncation
- `_fallback_wikipedia_search()` returns structured results matching DDG format
- F1 scorer already existed (`_score_f1`), just needed wiring
- Prompt resolver confirms both prompts load with new content
- 1708 tests pass, 3 pre-existing failures unrelated

## Resume Commands

```bash
# Verify web search
python3 -c "from orchestration.tools.web import web_search; import json; print(json.dumps(web_search('test query'), indent=2))"

# Verify F1 scoring
python3 -c "from scripts.benchmark.debug_scorer import _score_f1; print(_score_f1('The prize was awarded to Ye Tian', 'Ye Tian'))"

# Re-run seeding to validate
python3 scripts/benchmark/seed_specialist_routing.py --suite simpleqa --batch-size 10 --debug-replay
```
