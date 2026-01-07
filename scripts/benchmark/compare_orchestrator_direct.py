#!/usr/bin/env python3
"""
Orchestrator vs Direct Model Comparison

Compares orchestrator responses against a pre-computed baseline from
direct large model responses. Measures quality retention and speedup.

Usage:
    python scripts/benchmark/compare_orchestrator_direct.py --suite thinking --use-baseline
    python scripts/benchmark/compare_orchestrator_direct.py --create-baseline --suite all
    python scripts/benchmark/compare_orchestrator_direct.py --config-from checkpoint.yaml
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import httpx
    import yaml
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install httpx pyyaml")
    sys.exit(1)


# Constants
BASELINE_PATH = PROJECT_ROOT / "orchestration" / "orchestrator_baseline.json"
PROMPT_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "v1"
DEFAULT_ORCHESTRATOR_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120


@dataclass
class ComparisonResult:
    """Result of comparing orchestrator vs direct model."""
    prompt_id: str
    suite: str
    # Direct model
    direct_answer: str
    direct_latency_ms: float
    direct_score: Optional[float]  # Pre-computed Claude-as-judge score
    # Orchestrator
    orchestrator_answer: str
    orchestrator_latency_ms: float
    orchestrator_turns: int
    # Comparison
    quality_match: bool  # Manual assessment placeholder
    speedup: float


def load_baseline() -> dict:
    """Load pre-computed baseline results."""
    if not BASELINE_PATH.exists():
        return {}
    with open(BASELINE_PATH) as f:
        return json.load(f)


def save_baseline(baseline: dict):
    """Save baseline results."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_comparison_prompts(suite: str) -> list[dict]:
    """Get prompts for comparison from existing benchmark suites."""

    prompts = []

    # Use a subset of existing benchmarks that have Claude-as-judge scores
    suite_dirs = {
        "thinking": PROMPT_DIR / "thinking",
        "coder": PROMPT_DIR / "coder",
        "general": PROMPT_DIR / "general",
        "math": PROMPT_DIR / "math",
    }

    if suite == "all":
        dirs_to_check = suite_dirs.values()
    elif suite in suite_dirs:
        dirs_to_check = [suite_dirs[suite]]
    else:
        print(f"Unknown suite: {suite}")
        return []

    for suite_dir in dirs_to_check:
        if not suite_dir.exists():
            continue

        # Get first 5 prompts from each tier
        for tier_file in sorted(suite_dir.glob("t*_*.txt"))[:5]:
            try:
                content = tier_file.read_text()
                parts = content.split("---")
                prompt_text = parts[0].strip() if parts else content.strip()

                prompts.append({
                    "id": tier_file.stem,
                    "suite": suite_dir.name,
                    "prompt": prompt_text,
                    "file": str(tier_file)
                })
            except Exception as e:
                print(f"Error loading {tier_file}: {e}")

    return prompts


def call_orchestrator(
    prompt: str,
    api_url: str,
    timeout: int,
    config: Optional[dict] = None
) -> dict:
    """Call orchestrator API."""

    payload = {
        "prompt": prompt,
        "real_mode": True,
    }
    if config:
        payload.update(config)

    try:
        with httpx.Client(timeout=timeout) as client:
            start = time.perf_counter()
            response = client.post(
                f"{api_url}/chat",
                json=payload
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                result = response.json()
                result["latency_ms"] = latency_ms
                return result
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "latency_ms": latency_ms
                }
    except Exception as e:
        return {"error": str(e), "latency_ms": 0}


def compare_prompt(
    prompt_data: dict,
    baseline: dict,
    api_url: str,
    timeout: int,
    config: Optional[dict] = None
) -> ComparisonResult:
    """Compare orchestrator response to baseline for a single prompt."""

    prompt_id = prompt_data["id"]
    suite = prompt_data["suite"]

    # Get baseline data
    baseline_entry = baseline.get("prompts", {}).get(prompt_id, {})
    direct_answer = baseline_entry.get("answer", "")
    direct_latency = baseline_entry.get("latency_ms", 0)
    direct_score = baseline_entry.get("claude_score")

    # Call orchestrator
    result = call_orchestrator(prompt_data["prompt"], api_url, timeout, config)

    orchestrator_answer = result.get("answer", "")
    orchestrator_latency = result.get("latency_ms", 0)
    orchestrator_turns = result.get("turns", 0)

    # Calculate speedup
    if orchestrator_latency > 0 and direct_latency > 0:
        speedup = direct_latency / orchestrator_latency
    else:
        speedup = 0.0

    return ComparisonResult(
        prompt_id=prompt_id,
        suite=suite,
        direct_answer=direct_answer[:500],  # Truncate for storage
        direct_latency_ms=direct_latency,
        direct_score=direct_score,
        orchestrator_answer=orchestrator_answer[:500],
        orchestrator_latency_ms=orchestrator_latency,
        orchestrator_turns=orchestrator_turns,
        quality_match=True,  # Placeholder - requires manual review
        speedup=speedup
    )


def create_baseline_entry(prompt_data: dict, direct_url: str, timeout: int) -> dict:
    """Create baseline entry by calling direct model."""

    # This would call the direct large model
    # For now, create a placeholder structure
    return {
        "prompt_id": prompt_data["id"],
        "suite": prompt_data["suite"],
        "prompt": prompt_data["prompt"][:500],
        "answer": "",  # To be filled by running direct model
        "latency_ms": 0,
        "claude_score": None,  # To be filled by manual review
        "created": datetime.now().isoformat()
    }


def run_comparison(
    suite: str,
    api_url: str,
    timeout: int,
    use_baseline: bool,
    config: Optional[dict] = None
) -> dict:
    """Run comparison between orchestrator and baseline."""

    prompts = get_comparison_prompts(suite)
    baseline = load_baseline() if use_baseline else {}

    if not prompts:
        print(f"No prompts found for suite: {suite}")
        return {}

    results = []
    print(f"\nComparing {len(prompts)} prompts from {suite}...")
    print("-" * 60)

    for prompt_data in prompts:
        print(f"  {prompt_data['id']}...", end=" ", flush=True)

        result = compare_prompt(prompt_data, baseline, api_url, timeout, config)
        results.append(result)

        speedup_str = f"{result.speedup:.1f}x" if result.speedup > 0 else "N/A"
        print(f"speedup: {speedup_str}, turns: {result.orchestrator_turns}")

    # Compute summary
    valid_speedups = [r.speedup for r in results if r.speedup > 0]
    avg_speedup = sum(valid_speedups) / len(valid_speedups) if valid_speedups else 0

    avg_latency = sum(r.orchestrator_latency_ms for r in results) / len(results) if results else 0
    avg_turns = sum(r.orchestrator_turns for r in results) / len(results) if results else 0

    summary = {
        "suite": suite,
        "prompts_compared": len(results),
        "avg_speedup": avg_speedup,
        "avg_orchestrator_latency_ms": avg_latency,
        "avg_turns": avg_turns,
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat(),
        "note": "Quality assessment requires manual review"
    }

    return summary


def print_summary(summary: dict):
    """Print comparison summary."""

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Suite: {summary.get('suite', 'all')}")
    print(f"Prompts compared: {summary.get('prompts_compared', 0)}")
    print(f"Average speedup: {summary.get('avg_speedup', 0):.2f}x")
    print(f"Average orchestrator latency: {summary.get('avg_orchestrator_latency_ms', 0):.0f}ms")
    print(f"Average turns: {summary.get('avg_turns', 0):.1f}")
    print(f"\nNote: {summary.get('note', '')}")

    # Targets
    speedup = summary.get('avg_speedup', 0)
    speedup_status = "PASS" if speedup >= 3.0 else "FAIL"
    print(f"\nTarget: >3x speedup: {speedup_status} ({speedup:.2f}x)")
    print("Target: >90% quality retention: REQUIRES MANUAL REVIEW")


def main():
    parser = argparse.ArgumentParser(description="Orchestrator vs Direct Comparison")
    parser.add_argument(
        "--suite",
        choices=["all", "thinking", "coder", "general", "math"],
        default="all",
        help="Benchmark suite to compare"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_ORCHESTRATOR_URL,
        help="Orchestrator API URL"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout per request"
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="Use pre-computed baseline"
    )
    parser.add_argument(
        "--create-baseline",
        action="store_true",
        help="Create baseline entries (requires direct model)"
    )
    parser.add_argument(
        "--config-from",
        help="Load optimized config from checkpoint"
    )
    parser.add_argument(
        "--output",
        help="Output file for results"
    )

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config_from:
        checkpoint = load_config(args.config_from)
        # Extract optimal params from all completed layers
        config = {}
        for layer_data in checkpoint.get("layers", {}).values():
            if layer_data.get("optimal_params"):
                config.update(layer_data["optimal_params"])
        print(f"Loaded config: {config}")

    if args.create_baseline:
        print("Creating baseline entries...")
        prompts = get_comparison_prompts(args.suite)
        baseline = load_baseline()

        if "prompts" not in baseline:
            baseline["prompts"] = {}
        if "meta" not in baseline:
            baseline["meta"] = {"created": datetime.now().isoformat()}

        for prompt_data in prompts:
            if prompt_data["id"] not in baseline["prompts"]:
                entry = create_baseline_entry(prompt_data, "", args.timeout)
                baseline["prompts"][prompt_data["id"]] = entry
                print(f"  Added: {prompt_data['id']}")

        save_baseline(baseline)
        print(f"\nBaseline saved to: {BASELINE_PATH}")
        print("NOTE: Run direct model and update answers/scores manually")
        return

    # Run comparison
    summary = run_comparison(
        args.suite,
        args.api_url,
        args.timeout,
        args.use_baseline,
        config
    )

    print_summary(summary)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "benchmarks" / "results" / "orchestrator" / f"comparison_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
