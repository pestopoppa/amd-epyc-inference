#!/usr/bin/env python3
"""Corpus-Augmented Prompt Stuffing — Claude-as-Judge Quality Gate.

Runs the same code generation prompts with and without corpus injection,
then uses Claude to judge whether corpus injection degrades output quality.

Usage:
    python scripts/benchmark/corpus_quality_gate.py --models 7b 32b
    python scripts/benchmark/corpus_quality_gate.py --models 32b --dry-run
    python scripts/benchmark/corpus_quality_gate.py --models 7b 32b --results-only

Quality gate: PASS if average quality delta >= -0.5 (on 1-10 scale).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Model configs
MODELS = {
    "7b": {"port": 8082, "name": "Qwen2.5-7B", "role": "worker"},
    "32b": {"port": 8081, "name": "Qwen2.5-Coder-32B", "role": "coder_escalation"},
}

# Code generation prompts — novel tasks where corpus could help or hurt
PROMPTS = [
    {
        "id": "async_retry",
        "prompt": "Write a Python async HTTP client with retry logic, exponential backoff, and circuit breaker pattern. Include type hints and a usage example.",
        "language": "python",
    },
    {
        "id": "bst_iterator",
        "prompt": "Implement a binary search tree in Python with an in-order iterator that uses O(h) memory where h is the height of the tree. Include insert, search, delete, and the iterator protocol (__iter__, __next__).",
        "language": "python",
    },
    {
        "id": "lru_cache",
        "prompt": "Write a thread-safe LRU cache in Python using a doubly-linked list and a dictionary. Support get, put, and resize operations. Include proper locking and a decorator version.",
        "language": "python",
    },
    {
        "id": "json_parser",
        "prompt": "Write a recursive descent JSON parser in Python from scratch (no json module). Handle strings (with escapes), numbers (int and float), booleans, null, arrays, and objects. Return native Python types.",
        "language": "python",
    },
    {
        "id": "rate_limiter",
        "prompt": "Implement a token bucket rate limiter in Python that supports per-key limits, burst capacity, and automatic refill. Make it work both synchronously and with asyncio.",
        "language": "python",
    },
    {
        "id": "graph_shortest",
        "prompt": "Write Dijkstra's algorithm and A* search in Python. Support weighted directed graphs with an adjacency list representation. Include a priority queue implementation and path reconstruction.",
        "language": "python",
    },
]

# Claude-as-Judge prompt template
JUDGE_PROMPT = """You are evaluating code generation quality. You will see two code outputs for the same prompt — Output A and Output B. One was generated with corpus-augmented prompt stuffing (injected reference code snippets), the other without.

You do NOT know which is which. Judge each output independently on these criteria:

1. **Correctness** (1-10): Does the code work? Are there bugs?
2. **Completeness** (1-10): Does it address all requirements in the prompt?
3. **Code quality** (1-10): Clean style, good naming, proper error handling, type hints?
4. **Originality** (1-10): Does it feel like a thoughtful solution vs. copied boilerplate?

Return your scores in this exact JSON format (no other text):
{{"a_correctness": N, "a_completeness": N, "a_quality": N, "a_originality": N, "b_correctness": N, "b_completeness": N, "b_quality": N, "b_originality": N, "notes": "brief comparison"}}

## Task Prompt
{task_prompt}

## Output A
```python
{output_a}
```

## Output B
```python
{output_b}
```

Return ONLY the JSON object, no other text."""


@dataclass
class GenerationResult:
    model: str
    prompt_id: str
    corpus_enabled: bool
    output: str
    speed_tps: float
    tokens_generated: int
    draft_n: int = 0
    draft_accepted: int = 0
    wall_time: float = 0.0


@dataclass
class JudgeResult:
    prompt_id: str
    model: str
    baseline_score: float  # avg of 4 criteria
    corpus_score: float
    delta: float
    raw_scores: dict = field(default_factory=dict)


def generate(port: int, prompt: str, max_tokens: int = 1024) -> dict:
    """Send generation request to llama-server."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # Deterministic for fair comparison
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=300)
    wall = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    timings = data.get("timings", {})

    return {
        "output": content,
        "tokens": usage.get("completion_tokens", len(content.split())),
        "speed": timings.get("predicted_per_second", 0),
        "draft_n": timings.get("draft_n", 0),
        "draft_accepted": timings.get("draft_n_accepted", 0),
        "wall_time": wall,
    }


def build_corpus_prompt(prompt: str, corpus_config: dict) -> str:
    """Build prompt with corpus context injected."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.services.corpus_retrieval import CorpusConfig, CorpusRetriever, extract_code_query

    config = CorpusConfig(
        enabled=True,
        index_path=corpus_config.get("index_path", "/mnt/raid0/llm/cache/corpus/mvp_index"),
        max_snippets=corpus_config.get("max_snippets", 3),
        max_chars=corpus_config.get("max_chars", 3000),
    )
    retriever = CorpusRetriever.get_instance(config)
    query = extract_code_query(prompt)
    snippets = retriever.retrieve(query)
    corpus_ctx = retriever.format_for_prompt(snippets)

    if corpus_ctx:
        return f"{corpus_ctx}\n\n{prompt}"
    return prompt


def run_generation_pairs(
    model_key: str,
    corpus_config: dict,
    dry_run: bool = False,
) -> list[tuple[GenerationResult, GenerationResult]]:
    """Run all prompts with and without corpus for a model."""
    cfg = MODELS[model_key]
    port = cfg["port"]
    pairs = []

    for p in PROMPTS:
        log.info("  [%s] %s — baseline...", model_key, p["id"])

        if dry_run:
            baseline = GenerationResult(model_key, p["id"], False, "# dry run", 0, 0)
            corpus = GenerationResult(model_key, p["id"], True, "# dry run", 0, 0)
            pairs.append((baseline, corpus))
            continue

        # Baseline (no corpus)
        result_b = generate(port, p["prompt"])
        baseline = GenerationResult(
            model=model_key,
            prompt_id=p["id"],
            corpus_enabled=False,
            output=result_b["output"],
            speed_tps=result_b["speed"],
            tokens_generated=result_b["tokens"],
            draft_n=result_b["draft_n"],
            draft_accepted=result_b["draft_accepted"],
            wall_time=result_b["wall_time"],
        )

        log.info("  [%s] %s — with corpus...", model_key, p["id"])

        # With corpus
        corpus_prompt = build_corpus_prompt(p["prompt"], corpus_config)
        result_c = generate(port, corpus_prompt)
        corpus = GenerationResult(
            model=model_key,
            prompt_id=p["id"],
            corpus_enabled=True,
            output=result_c["output"],
            speed_tps=result_c["speed"],
            tokens_generated=result_c["tokens"],
            draft_n=result_c["draft_n"],
            draft_accepted=result_c["draft_accepted"],
            wall_time=result_c["wall_time"],
        )

        pairs.append((baseline, corpus))
        log.info(
            "    baseline: %.1f t/s, corpus: %.1f t/s, accept: %d/%d → %d/%d",
            baseline.speed_tps, corpus.speed_tps,
            baseline.draft_accepted, baseline.draft_n,
            corpus.draft_accepted, corpus.draft_n,
        )

    return pairs


def judge_pair(
    prompt_id: str,
    task_prompt: str,
    baseline_output: str,
    corpus_output: str,
) -> JudgeResult | None:
    """Use Claude to judge output quality. Randomizes A/B assignment."""
    import random

    # Randomize which is A vs B to avoid position bias
    corpus_is_a = random.random() < 0.5

    if corpus_is_a:
        output_a, output_b = corpus_output, baseline_output
    else:
        output_a, output_b = baseline_output, corpus_output

    judge_input = JUDGE_PROMPT.format(
        task_prompt=task_prompt,
        output_a=output_a,
        output_b=output_b,
    )

    try:
        result = subprocess.run(
            ["claude", "-p", judge_input, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if result.returncode != 0:
            log.warning("Claude judge failed for %s: %s", prompt_id, result.stderr[:200])
            return None

        # Parse JSON from response
        response = result.stdout.strip()
        # Try to extract JSON if wrapped in markdown
        if "```" in response:
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)

        scores = json.loads(response)

        # Map back: which was baseline vs corpus?
        if corpus_is_a:
            corpus_avg = (scores["a_correctness"] + scores["a_completeness"] + scores["a_quality"] + scores["a_originality"]) / 4
            baseline_avg = (scores["b_correctness"] + scores["b_completeness"] + scores["b_quality"] + scores["b_originality"]) / 4
        else:
            baseline_avg = (scores["a_correctness"] + scores["a_completeness"] + scores["a_quality"] + scores["a_originality"]) / 4
            corpus_avg = (scores["b_correctness"] + scores["b_completeness"] + scores["b_quality"] + scores["b_originality"]) / 4

        return JudgeResult(
            prompt_id=prompt_id,
            model="",
            baseline_score=baseline_avg,
            corpus_score=corpus_avg,
            delta=corpus_avg - baseline_avg,
            raw_scores=scores,
        )
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("Failed to parse judge response for %s: %s", prompt_id, e)
        return None
    except subprocess.TimeoutExpired:
        log.warning("Claude judge timed out for %s", prompt_id)
        return None


def main():
    parser = argparse.ArgumentParser(description="Corpus quality gate")
    parser.add_argument("--models", nargs="+", default=["7b", "32b"], choices=list(MODELS.keys()))
    parser.add_argument("--index-path", default="/mnt/raid0/llm/cache/corpus/mvp_index")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-only", help="Path to existing results JSON to re-judge")
    parser.add_argument("--output", default="/mnt/raid0/llm/tmp/corpus_quality_gate.json")
    args = parser.parse_args()

    corpus_config = {
        "index_path": args.index_path,
        "max_snippets": 3,
        "max_chars": 3000,
    }

    all_results = {}

    if args.results_only:
        with open(args.results_only) as f:
            all_results = json.load(f)
    else:
        for model_key in args.models:
            log.info("=== Generating for %s (%s) ===", model_key, MODELS[model_key]["name"])
            pairs = run_generation_pairs(model_key, corpus_config, args.dry_run)

            model_results = []
            for baseline, corpus in pairs:
                model_results.append({
                    "prompt_id": baseline.prompt_id,
                    "baseline": {
                        "output": baseline.output,
                        "speed": baseline.speed_tps,
                        "tokens": baseline.tokens_generated,
                        "draft_n": baseline.draft_n,
                        "draft_accepted": baseline.draft_accepted,
                        "wall_time": baseline.wall_time,
                    },
                    "corpus": {
                        "output": corpus.output,
                        "speed": corpus.speed_tps,
                        "tokens": corpus.tokens_generated,
                        "draft_n": corpus.draft_n,
                        "draft_accepted": corpus.draft_accepted,
                        "wall_time": corpus.wall_time,
                    },
                })
            all_results[model_key] = model_results

        # Save generation results
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info("Generation results saved to %s", args.output)

    if args.dry_run:
        log.info("[DRY RUN] Would judge %d pairs per model", len(PROMPTS))
        return

    # Judge phase
    log.info("\n=== Claude-as-Judge Quality Scoring ===")
    all_judge_results = {}
    gate_pass = True

    for model_key in args.models:
        results = all_results.get(model_key, [])
        judge_results = []

        for r in results:
            prompt_text = next((p["prompt"] for p in PROMPTS if p["id"] == r["prompt_id"]), "")
            log.info("  Judging %s / %s...", model_key, r["prompt_id"])

            jr = judge_pair(
                r["prompt_id"],
                prompt_text,
                r["baseline"]["output"],
                r["corpus"]["output"],
            )
            if jr:
                jr.model = model_key
                judge_results.append(jr)
                log.info(
                    "    baseline=%.1f  corpus=%.1f  delta=%+.1f  %s",
                    jr.baseline_score, jr.corpus_score, jr.delta,
                    "PASS" if jr.delta >= -0.5 else "FAIL",
                )

        if judge_results:
            avg_delta = sum(j.delta for j in judge_results) / len(judge_results)
            avg_baseline = sum(j.baseline_score for j in judge_results) / len(judge_results)
            avg_corpus = sum(j.corpus_score for j in judge_results) / len(judge_results)

            model_pass = avg_delta >= -0.5
            if not model_pass:
                gate_pass = False

            log.info(
                "\n  %s SUMMARY: baseline=%.2f  corpus=%.2f  delta=%+.2f  %s",
                model_key.upper(),
                avg_baseline,
                avg_corpus,
                avg_delta,
                "GATE PASS" if model_pass else "GATE FAIL",
            )

            all_judge_results[model_key] = {
                "avg_baseline": avg_baseline,
                "avg_corpus": avg_corpus,
                "avg_delta": avg_delta,
                "gate_pass": model_pass,
                "per_prompt": [
                    {
                        "prompt_id": j.prompt_id,
                        "baseline": j.baseline_score,
                        "corpus": j.corpus_score,
                        "delta": j.delta,
                        "raw": j.raw_scores,
                    }
                    for j in judge_results
                ],
            }

    # Save judge results
    judge_output = args.output.replace(".json", "_judge.json")
    with open(judge_output, "w") as f:
        json.dump(all_judge_results, f, indent=2)
    log.info("\nJudge results saved to %s", judge_output)

    # Final verdict
    log.info("\n" + "=" * 60)
    if gate_pass:
        log.info("QUALITY GATE: PASS — corpus injection does not degrade quality")
    else:
        log.info("QUALITY GATE: FAIL — corpus injection degrades quality beyond -0.5 threshold")
    log.info("=" * 60)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
