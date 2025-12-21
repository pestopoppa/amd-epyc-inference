#!/usr/bin/env python3
"""
Unified Benchmark Runner

Single entry point for all benchmarks with clean nested loops:

    for role in registry.roles:
        for config in get_configs(role.architecture):
            for suite in get_suites(role):
                for question in suite.questions:
                    run_and_save(role, config, suite, question)

Usage:
    ./run_benchmark.py                    # Run all (skips existing by default)
    ./run_benchmark.py --force            # Force re-run (don't skip)
    ./run_benchmark.py --model coder_primary  # Run specific model
    ./run_benchmark.py --suite thinking   # Run specific suite
    ./run_benchmark.py --dry-run          # Show what would run
    ./run_benchmark.py --process-queue    # Process queued models
"""

import argparse
import fcntl
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

# Add parent directory for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.registry import ModelRegistry, load_registry
from lib.executor import Executor, Config, ServerManager
from lib.output_parser import parse_output
from lib.scorer import score_response

from suites import load_suite, get_suites_for_role, get_inference_params, get_all_suite_names
from results import ResultsManager, QuestionResult, result_exists


# Lock file for single instance
LOCK_FILE = "/mnt/raid0/llm/tmp/benchmark.lock"
QUEUE_FILE = "/mnt/raid0/llm/tmp/benchmark_queue.txt"

# Speed test prompt for configs that only need speed measurement (quality inherited from baseline)
SPEED_TEST_PROMPT = """Write a Python function that calculates the Fibonacci sequence up to n terms. Include a docstring explaining the function and type hints for all parameters and return value. Then write a brief example showing how to use the function."""

# Reference TPS for timeout multiplier calculation (20 t/s = 1.0x multiplier)
REFERENCE_TPS = 20.0
# Minimum timeout multiplier (even fast models don't get shorter timeouts)
MIN_TIMEOUT_MULTIPLIER = 1.0
# Default multiplier when speed test fails
DEFAULT_TIMEOUT_MULTIPLIER = 2.0


def acquire_lock() -> Optional[int]:
    """Acquire exclusive lock for single-instance execution.

    Returns:
        File descriptor if lock acquired, None if another instance is running.
    """
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except (OSError, BlockingIOError):
        return None


def release_lock(fd: int) -> None:
    """Release the lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        os.unlink(LOCK_FILE)
    except OSError:
        pass


def print_progress(
    role: str,
    config: str,
    suite: str,
    question: str,
    status: str,
    tokens_per_second: Optional[float] = None,
    score: Optional[int] = None,
) -> None:
    """Print progress line."""
    tps_str = f"{tokens_per_second:.1f} t/s" if tokens_per_second else "---"
    score_str = f"{score}/3" if score is not None else "---"
    print(f"[{status:8}] {role:25} {config:15} {suite:20} {question:15} {tps_str:>10} {score_str:>5}")


def build_work_items(
    registry: ModelRegistry,
    executor: Executor,
    model_filter: Optional[str] = None,
    suite_filter: Optional[str] = None,
) -> list[dict]:
    """Build flat list of work items from nested loops.

    Returns list of dicts with: role, model_path, architecture, config, suite_name, suite, question, params
    """
    work_items = []

    roles = registry.get_all_roles(include_deprecated=False)
    if model_filter:
        roles = [r for r in roles if r == model_filter]

    for role in roles:
        model_path = registry.get_model_path(role)
        if not model_path or not os.path.exists(model_path):
            continue

        architecture = registry.get_architecture(role)
        configs = executor.get_configs_for_architecture(architecture, role, registry)

        for config in configs:
            suite_names = get_suites_for_role(role, registry)
            if suite_filter:
                suite_names = [s for s in suite_names if s == suite_filter]

            for suite_name in suite_names:
                suite = load_suite(suite_name)
                if not suite:
                    continue

                params = get_inference_params(suite)

                for question in suite.questions:
                    work_items.append({
                        "role": role,
                        "model_path": model_path,
                        "architecture": architecture,
                        "config": config,
                        "suite_name": suite_name,
                        "suite": suite,
                        "question": question,
                        "params": params,
                    })

    return work_items


def run_benchmark(
    registry: ModelRegistry,
    executor: Executor,
    results_manager: ResultsManager,
    run_id: str,
    model_filter: Optional[str] = None,
    suite_filter: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    server_mode: bool = False,
    no_mmap: bool = False,
) -> dict:
    """Run the benchmark with nested progress bars.

    Outer bar: models
    Inner bar: configs × suites × questions for current model
    """
    stats = {"total": 0, "skipped": 0, "passed": 0, "failed": 0, "errors": 0}

    # Get models to test
    roles = registry.get_all_roles(include_deprecated=False)
    if model_filter:
        roles = [r for r in roles if r == model_filter]

    # Group roles by model path to avoid duplicate display
    model_to_roles: dict[str, list[str]] = {}
    role_sizes: dict[str, float] = {}

    for role in roles:
        model_path = registry.get_model_path(role)
        if model_path and os.path.exists(model_path):
            if model_path not in model_to_roles:
                model_to_roles[model_path] = []
            model_to_roles[model_path].append(role)

            # Get model size from registry or file
            config = registry.get_role_config(role)
            size_gb = config.get("model", {}).get("size_gb", 0) if config else 0
            if size_gb == 0:
                size_gb = os.path.getsize(model_path) / (1024**3)
            role_sizes[role] = size_gb

    # Build list of (model_path, size, roles) sorted by size (largest first)
    models_sorted = []
    for model_path, role_list in model_to_roles.items():
        size_gb = role_sizes[role_list[0]]  # All roles for same model have same size
        models_sorted.append((model_path, size_gb, role_list))
    models_sorted.sort(key=lambda x: x[1], reverse=True)

    # Flatten back to role list but preserve model grouping order
    valid_roles = []
    for model_path, size_gb, role_list in models_sorted:
        for role in role_list:
            valid_roles.append((role, size_gb))

    print(f"\nBenchmark: {run_id} | {len(models_sorted)} models, {len(valid_roles)} roles (largest first)")

    # Track which models we've already printed and speed-tested
    printed_models: set[str] = set()
    model_tps: dict[str, float] = {}  # model_path -> measured TPS
    active_server: Optional[ServerManager] = None  # Server for current model
    current_server_model: Optional[str] = None  # Model path loaded in server
    current_server_experts: Optional[int] = None  # Expert count loaded in server (None = default)

    # Outer progress bar: roles
    role_iter = tqdm(valid_roles, desc="Roles") if TQDM_AVAILABLE else valid_roles

    try:
      for role, size_gb in role_iter:
        model_path = registry.get_model_path(role)
        arch = registry.get_architecture(role)
        configs = executor.get_configs_for_architecture(arch, role, registry)

        suite_names = get_suites_for_role(role, registry)
        if suite_filter:
            suite_names = [s for s in suite_names if s == suite_filter]

        # Run baseline speed test for each model (once per model, not per role)
        if model_path not in model_tps:
            if dry_run:
                # In dry run, use registry baseline_tps or default
                reg_tps = registry.get_baseline_tps(role)
                model_tps[model_path] = reg_tps if reg_tps else REFERENCE_TPS
            else:
                # Find the baseline config to measure actual speed
                baseline_config = next((c for c in configs if c.name == "baseline"), None)
                if baseline_config:
                    try:
                        speed_result = executor.run_inference(
                            model_path=model_path,
                            config=baseline_config,
                            prompt=SPEED_TEST_PROMPT,
                            max_tokens=256,
                            temperature=0.6,
                            timeout=180,  # Conservative timeout for speed test
                        )
                        if speed_result.success and not speed_result.timed_out:
                            parsed = parse_output(speed_result.raw_output)
                            if parsed.tokens_per_second and parsed.tokens_per_second > 0:
                                model_tps[model_path] = parsed.tokens_per_second
                            else:
                                model_tps[model_path] = REFERENCE_TPS
                        else:
                            model_tps[model_path] = REFERENCE_TPS
                    except Exception:
                        model_tps[model_path] = REFERENCE_TPS
                else:
                    # No baseline config, use registry or default
                    reg_tps = registry.get_baseline_tps(role)
                    model_tps[model_path] = reg_tps if reg_tps else REFERENCE_TPS

        # Calculate timeout multiplier based on measured speed
        measured_tps = model_tps.get(model_path, REFERENCE_TPS)
        timeout_multiplier = max(MIN_TIMEOUT_MULTIPLIER, REFERENCE_TPS / measured_tps)

        # Preload suites and count questions (with timeout multiplier applied)
        suites_data = {}
        for sname in suite_names:
            suite = load_suite(sname)
            if suite:
                suites_data[sname] = {"suite": suite, "params": get_inference_params(suite, timeout_multiplier)}

        total_questions = sum(len(suites_data[s]["suite"].questions) for s in suites_data)
        inner_total = len(configs) * total_questions

        # Print model header only once, then show roles underneath
        model_name = Path(model_path).stem if model_path else role
        if model_path not in printed_models:
            printed_models.add(model_path)
            roles_for_model = model_to_roles[model_path]
            tps_str = f"{measured_tps:.1f} t/s"
            mult_str = f"{timeout_multiplier:.1f}x" if timeout_multiplier > 1.0 else "1x"
            print(f"\n  {model_name} ({size_gb:.1f}GB) @ {tps_str} → timeout {mult_str}", flush=True)
            print(f"    roles: {', '.join(roles_for_model)}", flush=True)

            # SERVER MODE: Start server for this model (keeps it in RAM)
            if server_mode and not dry_run:
                # Stop previous server if different model
                if active_server and current_server_model != model_path:
                    print(f"    [SERVER] Stopping server for previous model", flush=True)
                    active_server.stop()
                    active_server = None

                if active_server is None:
                    # Start with no MoE override (use model defaults for baseline)
                    # Server will be restarted with specific expert counts when needed
                    print(f"    [SERVER] Starting llama-server (model will stay in RAM)...", flush=True)
                    active_server = ServerManager(port=8080)
                    active_server.start(model_path, moe_override=None, registry=registry, no_mmap=no_mmap)

                    # Dynamic timeout: 3s per GB + 2 min buffer, minimum 600s
                    server_timeout = max(600, int(size_gb * 3) + 120)
                    if not active_server.wait_ready(timeout=server_timeout):
                        print(f"    [SERVER] Failed to start, falling back to subprocess mode", flush=True)
                        active_server = None
                    else:
                        current_server_model = model_path
                        current_server_experts = None  # Default expert count
                        print(f"    [SERVER] Ready, model loaded in RAM (default experts)", flush=True)

        print(f"    [{role}] {inner_total} tests ({len(configs)} configs × {len(suite_names)} suites)", flush=True)

        for config in configs:
            # SERVER RESTART for different MoE expert counts
            # Each MoE config needs the server restarted with its specific expert count
            if server_mode and active_server is not None and active_server.is_running():
                # Determine required expert count for this config
                if config.config_type == "moe":
                    required_experts = config.moe_experts
                elif config.config_type == "baseline":
                    required_experts = None  # Use model default
                else:
                    required_experts = current_server_experts  # Keep current (for non-server configs)

                # Restart if expert count differs
                if required_experts != current_server_experts:
                    if required_experts is None:
                        print(f"      [SERVER] Restarting for baseline (default experts)...", flush=True)
                        moe_override = None
                    else:
                        moe_key = registry.get_moe_override_key(role) or "qwen3moe.expert_used_count"
                        moe_override = f"{moe_key}=int:{required_experts}"
                        print(f"      [SERVER] Restarting for {config.name} ({required_experts} experts)...", flush=True)

                    active_server.stop()
                    active_server = ServerManager(port=8080)
                    active_server.start(model_path, moe_override=moe_override, registry=registry, no_mmap=no_mmap)

                    server_timeout = max(600, int(size_gb * 3) + 120)
                    if not active_server.wait_ready(timeout=server_timeout):
                        print(f"      [SERVER] Failed to restart, falling back to subprocess", flush=True)
                        active_server = None
                        current_server_experts = None
                    else:
                        current_server_experts = required_experts
                        print(f"      [SERVER] Ready", flush=True)

            # SPEED-TEST OPTIMIZATION: For configs that only need speed measurement
            # (e.g., spec decode variants), run a single speed test instead of all questions.
            # Quality is inherited from baseline since target model is the same.
            if config.speed_test_only:
                stats["total"] += 1

                if dry_run:
                    print(f"      [SPEED] {config.name} (inherits quality from {config.inherits_quality_from})", flush=True)
                    continue

                # Skip if already done
                if not force and result_exists(run_id, role, config.name):
                    stats["skipped"] += 1
                    continue

                try:
                    # Speed tests use subprocess (spec decode configs need external draft)
                    result = executor.run_inference(
                        model_path=model_path,
                        config=config,
                        prompt=SPEED_TEST_PROMPT,
                        max_tokens=256,
                        temperature=0.6,
                        timeout=120,
                    )

                    if result.timed_out:
                        stats["errors"] += 1
                        print(f"    [TIMEOUT] {role}/{config.name} (speed test)")
                        continue

                    if not result.success:
                        stats["errors"] += 1
                        print(f"    [ERROR] {role}/{config.name} (speed test)")
                        continue

                    parsed = parse_output(result.raw_output)

                    # Store speed-only result
                    results_manager.add_speed_result(
                        run_id=run_id,
                        model_role=role,
                        config_name=config.name,
                        model_path=model_path,
                        tokens_per_second=parsed.tokens_per_second or 0,
                        inherits_quality_from=config.inherits_quality_from or "baseline",
                        acceptance_rate=parsed.acceptance_rate,
                    )

                    tps = parsed.tokens_per_second
                    tps_str = f"{tps:.1f}t/s" if tps else "---"
                    acc_str = f"acc={parsed.acceptance_rate:.1%}" if parsed.acceptance_rate else ""
                    print(f"      ⚡ {config.name}: {tps_str} {acc_str} (speed only, quality from {config.inherits_quality_from})", flush=True)
                    stats["passed"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    print(f"    [ERROR] {role}/{config.name}: {e}")

                continue  # Skip to next config

            # FULL QUALITY BENCHMARK for baseline and non-spec configs
            for suite_name, sdata in suites_data.items():
                # Skip lookup configs for short-prompt suites
                # Only long_context has prompts long enough for lookup to be effective
                if config.config_type in ("lookup", "moe_lookup") and suite_name != "long_context":
                    continue

                suite = sdata["suite"]
                params = sdata["params"]

                for question in suite.questions:
                    stats["total"] += 1

                    # Dry run: just count, don't execute
                    if dry_run:
                        continue

                    # Skip check
                    if not force and result_exists(run_id, role, config.name, suite_name, question.id):
                        stats["skipped"] += 1
                        continue

                    try:
                        # Use server for baseline/MoE configs (model stays in RAM)
                        # Use subprocess for spec decode (needs external draft model)
                        use_server = (
                            active_server is not None
                            and active_server.is_running()
                            and config.config_type in ("baseline", "moe")
                        )

                        if use_server:
                            result = active_server.run_inference(
                                prompt=question.prompt,
                                max_tokens=params["max_tokens"],
                                temperature=params["temperature"],
                                timeout=params["timeout"],
                            )
                        else:
                            result = executor.run_inference(
                                model_path=model_path,
                                config=config,
                                prompt=question.prompt,
                                max_tokens=params["max_tokens"],
                                temperature=params["temperature"],
                                timeout=params["timeout"],
                            )

                        if result.timed_out:
                            stats["errors"] += 1
                            print(f"    [TIMEOUT] {role}/{config.name}/{question.id}")
                            continue

                        if not result.success:
                            stats["errors"] += 1
                            print(f"    [ERROR] {role}/{config.name}/{question.id}")
                            continue

                        parsed = parse_output(result.raw_output)
                        score_result = score_response(suite_name, question.id, parsed.response)

                        qresult = QuestionResult(
                            question_id=question.id,
                            prompt=question.prompt,
                            response=parsed.response,
                            tokens_per_second=parsed.tokens_per_second,
                            prompt_tokens=parsed.prompt_tokens,
                            completion_tokens=parsed.completion_tokens,
                            total_time_ms=parsed.total_time_ms,
                            algorithmic_score=score_result.score,
                            score_reason=score_result.reason,
                            acceptance_rate=parsed.acceptance_rate,
                        )

                        results_manager.add_question_result(
                            run_id=run_id,
                            model_role=role,
                            config_name=config.name,
                            model_path=model_path,
                            suite=suite_name,
                            question_result=qresult,
                        )

                        # Show progress for every test
                        tps = parsed.tokens_per_second
                        tps_str = f"{tps:.1f}t/s" if tps else "---"
                        score_icon = "✓" if score_result.score >= 2 else "✗"
                        print(f"      {score_icon} {config.name}/{suite_name}/{question.id}: {tps_str} score={score_result.score}/3", flush=True)

                        if score_result.score >= 2:
                            stats["passed"] += 1
                        else:
                            stats["failed"] += 1

                    except Exception as e:
                        stats["errors"] += 1
                        print(f"    [ERROR] {role}/{config.name}/{question.id}: {e}")

    finally:
        # Clean up server if running
        if active_server is not None:
            print(f"\n  [SERVER] Stopping server...", flush=True)
            active_server.stop()

    print(f"\nDone: {stats['passed']} passed, {stats['failed']} failed, {stats['skipped']} skipped, {stats['errors']} errors")
    return stats


def process_queue(
    registry: ModelRegistry,
    executor: Executor,
    results_manager: ResultsManager,
) -> None:
    """Process queued models from the queue file."""
    if not os.path.exists(QUEUE_FILE):
        print("No benchmark queue found.")
        return

    with open(QUEUE_FILE) as f:
        queued_models = [line.strip() for line in f if line.strip()]

    if not queued_models:
        print("Benchmark queue is empty.")
        return

    print(f"Processing {len(queued_models)} queued models...")

    for model in queued_models:
        run_id = results_manager.generate_run_id()
        run_benchmark(
            registry=registry,
            executor=executor,
            results_manager=results_manager,
            run_id=run_id,
            model_filter=model,
        )

    # Clear queue
    os.unlink(QUEUE_FILE)
    print("Queue processed and cleared.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./run_benchmark.py                     # Run all benchmarks
  ./run_benchmark.py --force             # Force re-run
  ./run_benchmark.py --model coder_primary --suite thinking
  ./run_benchmark.py --dry-run           # Preview what would run
  ./run_benchmark.py --process-queue     # Run queued models
        """,
    )
    parser.add_argument("--model", "-m", help="Only run this model role")
    parser.add_argument("--suite", "-s", help="Only run this suite")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-run (don't skip existing)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would run without executing")
    parser.add_argument("--process-queue", action="store_true", help="Process queued models")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume the latest run (skip completed, retry errors)")
    parser.add_argument("--server-mode", action="store_true", help="Keep model in RAM via llama-server (faster for large models)")
    parser.add_argument("--no-mmap", action="store_true", help="Use bulk read instead of mmap (may be faster for cold loads)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-suites", action="store_true", help="List available suites")

    args = parser.parse_args()

    # Initialize components
    registry = load_registry()
    executor = Executor(registry)
    results_manager = ResultsManager()

    # List modes
    if args.list_models:
        print("Available models:")
        for role in registry.get_all_roles():
            tier = registry.get_tier(role)
            arch = registry.get_architecture(role)
            path = registry.get_model_path(role)
            exists = "✓" if path and os.path.exists(path) else "✗"
            print(f"  [{tier}] {role:30} {arch:15} {exists}")
        return

    if args.list_suites:
        print("Available suites:")
        for name in get_all_suite_names():
            suite = load_suite(name)
            if suite:
                print(f"  {name:25} ({len(suite.questions)} questions) - {suite.description[:50]}...")
        return

    # Process queue mode
    if args.process_queue:
        lock_fd = acquire_lock()
        if lock_fd is None:
            print("ERROR: Another benchmark is already running.")
            sys.exit(1)
        try:
            process_queue(registry, executor, results_manager)
        finally:
            release_lock(lock_fd)
        return

    # Acquire lock for benchmark run
    if not args.dry_run:
        lock_fd = acquire_lock()
        if lock_fd is None:
            print("ERROR: Another benchmark is already running.")
            print("       Kill the existing process or wait for it to complete.")
            sys.exit(1)
    else:
        lock_fd = None

    try:
        if args.resume:
            run_id = results_manager.get_latest_run()
            if not run_id:
                print("ERROR: No previous run to resume. Start a new run without --resume.")
                sys.exit(1)
            print(f"Resuming run: {run_id}")
        else:
            run_id = results_manager.generate_run_id()

        run_benchmark(
            registry=registry,
            executor=executor,
            results_manager=results_manager,
            run_id=run_id,
            model_filter=args.model,
            suite_filter=args.suite,
            force=args.force,
            dry_run=args.dry_run,
            server_mode=args.server_mode,
            no_mmap=args.no_mmap,
        )
    finally:
        if lock_fd is not None:
            release_lock(lock_fd)


if __name__ == "__main__":
    main()
