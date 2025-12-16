#!/bin/bash
# =============================================================================
# OVERNIGHT BENCHMARK SUITE - Complete Model Quality & Speed Testing
# =============================================================================
# Runs ALL benchmark suites with ALL optimization configurations
#
# Suites:
#   1. Thinking (reasoning, CoT)
#   2. Coder (code generation, debugging)
#   3. VL (vision-language)
#   4. General (instruction following)
#   5. Agentic (tool calling, orchestration)
#   6. Math (mathematical reasoning)
#   7. Long Context (information retrieval across large contexts)
#   8. Instruction Precision (exact format compliance)
#
# Optimization Modes:
#   - baseline: Standard inference
#   - moe4: MoE expert reduction (4 experts)
#   - spec_decode: Speculative decoding (external draft)
#   - lookup: Prompt lookup decoding
#
# Usage:
#   ./run_overnight_benchmark_suite.sh [--suite SUITE] [--skip-slow]
#
# Options:
#   --suite SUITE   Run only specified suite (thinking|coder|vl|general|agentic|math|long_context|instruction_precision|all)
#   --skip-slow     Skip very large models (>100B params)
#   --dry-run       Show what would run without executing
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
BASE_OUTPUT_DIR="/tmp/claude/overnight_benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$BASE_OUTPUT_DIR/$TIMESTAMP"
LOG_FILE="$RUN_DIR/benchmark.log"

# Parse arguments
SUITE="all"
SKIP_SLOW=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --suite)
            SUITE="$2"
            shift 2
            ;;
        --skip-slow)
            SKIP_SLOW=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup
mkdir -p "$RUN_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

run_suite() {
    local suite_name="$1"
    local script="$2"
    local suite_dir="$RUN_DIR/$suite_name"

    mkdir -p "$suite_dir"

    log "=========================================="
    log "STARTING SUITE: $suite_name"
    log "=========================================="

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would execute: $script"
        return 0
    fi

    local start_time=$(date +%s)

    if "$script" >> "$suite_dir/output.log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "COMPLETED: $suite_name (${duration}s)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "FAILED: $suite_name (${duration}s)"
        return 1
    fi
}

# =============================================================================
# DYNAMIC MODEL QUEUE PROCESSING
# =============================================================================
QUEUE_FILE="/tmp/claude/benchmark_queue.txt"
QUEUE_LOCK="/tmp/claude/benchmark_queue.lock"
QUEUE_DONE="/tmp/claude/benchmark_queue_done.txt"
QUEUED_MODELS_PROCESSED=0

# Check if there are models in the queue
has_queued_models() {
    [[ -f "$QUEUE_FILE" ]] && [[ -s "$QUEUE_FILE" ]]
}

# Process all models currently in the queue
process_queue() {
    if ! has_queued_models; then
        return 0
    fi

    log "=========================================="
    log "PROCESSING DYNAMICALLY QUEUED MODELS"
    log "=========================================="

    # Read queue atomically
    local queue_snapshot=$(mktemp)
    (
        flock -x 200
        if [[ -f "$QUEUE_FILE" ]]; then
            cat "$QUEUE_FILE" > "$queue_snapshot"
            # Clear the queue (entries will be moved to done file after processing)
            > "$QUEUE_FILE"
        fi
    ) 200>"$QUEUE_LOCK"

    if [[ ! -s "$queue_snapshot" ]]; then
        rm -f "$queue_snapshot"
        return 0
    fi

    local count=$(wc -l < "$queue_snapshot")
    log "Found $count model(s) in queue"

    while IFS='|' read -r model_path model_name arch moe_key timestamp; do
        [[ -z "$model_path" ]] && continue

        log ""
        log "Processing queued model: $model_name"
        log "  Path: $model_path"
        log "  Arch: $arch"
        log "  MoE Key: ${moe_key:-none}"
        log "  Queued at: $timestamp"

        if [[ ! -f "$model_path" ]]; then
            log "  ERROR: Model file not found, skipping"
            continue
        fi

        if [[ "$DRY_RUN" == "true" ]]; then
            log "  [DRY RUN] Would run benchmarks for $model_name"
            continue
        fi

        # Run all applicable suites for this model
        run_suites_for_queued_model "$model_path" "$model_name" "$arch" "$moe_key"

        # Mark as processed
        echo "${model_path}|${model_name}|${arch}|${moe_key}|${timestamp}|processed_$(date +%Y%m%d_%H%M%S)" >> "$QUEUE_DONE"
        ((QUEUED_MODELS_PROCESSED++)) || true

    done < "$queue_snapshot"

    rm -f "$queue_snapshot"
    log "Queue processing complete"
}

# Run all benchmark suites for a single queued model
run_suites_for_queued_model() {
    local model_path="$1"
    local model_name="$2"
    local arch="$3"
    local moe_key="$4"

    local model_dir="$RUN_DIR/queued_models/$model_name"
    mkdir -p "$model_dir"

    # Build MoE override if needed
    local moe_override=""
    if [[ -n "$moe_key" ]]; then
        moe_override="--override-kv ${moe_key}=int:4"
    fi

    log "  Running benchmark suites for $model_name..."

    # Thinking rubric (if not VL-only model)
    if [[ "$SUITE" == "all" || "$SUITE" == "thinking" ]]; then
        log "    Running thinking rubric..."
        "$SCRIPT_DIR/run_thinking_rubric.sh" "$model_path" "$model_name" "$arch" \
            >> "$model_dir/thinking.log" 2>&1 || log "    Thinking rubric failed"
    fi

    # Coder rubric
    if [[ "$SUITE" == "all" || "$SUITE" == "coder" ]]; then
        log "    Running coder rubric..."
        "$SCRIPT_DIR/run_coder_rubric.sh" "$model_path" "$model_name" "$arch" \
            >> "$model_dir/coder.log" 2>&1 || log "    Coder rubric failed"
    fi

    # General rubric
    if [[ "$SUITE" == "all" || "$SUITE" == "general" ]]; then
        log "    Running general rubric..."
        "$SCRIPT_DIR/run_general_rubric.sh" "$model_path" "$model_name" "$arch" \
            >> "$model_dir/general.log" 2>&1 || log "    General rubric failed"
    fi

    # Math rubric
    if [[ "$SUITE" == "all" || "$SUITE" == "math" ]]; then
        log "    Running math rubric..."
        "$SCRIPT_DIR/run_math_rubric.sh" "$model_path" "$model_name" "$arch" \
            >> "$model_dir/math.log" 2>&1 || log "    Math rubric failed"
    fi

    # Long context rubric
    if [[ "$SUITE" == "all" || "$SUITE" == "long_context" ]]; then
        log "    Running long context rubric..."
        "$SCRIPT_DIR/run_long_context_rubric.sh" "$model_path" "$model_name" "$arch" \
            >> "$model_dir/long_context.log" 2>&1 || log "    Long context rubric failed"
    fi

    # Instruction precision rubric
    if [[ "$SUITE" == "all" || "$SUITE" == "instruction_precision" ]]; then
        log "    Running instruction precision rubric..."
        "$SCRIPT_DIR/run_instruction_precision_rubric.sh" "$model_path" "$model_name" "$arch" \
            >> "$model_dir/instruction_precision.log" 2>&1 || log "    Instruction precision rubric failed"
    fi

    # Agentic rubric
    if [[ "$SUITE" == "all" || "$SUITE" == "agentic" ]]; then
        log "    Running agentic rubric..."
        "$SCRIPT_DIR/run_agentic_rubric.sh" "$model_path" "$model_name" "$arch" \
            >> "$model_dir/agentic.log" 2>&1 || log "    Agentic rubric failed"
    fi

    # Note: VL rubric requires mmproj file, skip for non-VL models
    # VL models should be added manually to the VL benchmark script

    log "  Completed suites for $model_name"
    log "  Logs: $model_dir/"
}

# =============================================================================
# BANNER
# =============================================================================
cat << 'EOF'

 ██████╗ ██╗   ██╗███████╗██████╗ ███╗   ██╗██╗ ██████╗ ██╗  ██╗████████╗
██╔═══██╗██║   ██║██╔════╝██╔══██╗████╗  ██║██║██╔════╝ ██║  ██║╚══██╔══╝
██║   ██║██║   ██║█████╗  ██████╔╝██╔██╗ ██║██║██║  ███╗███████║   ██║
██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██║╚██╗██║██║██║   ██║██╔══██║   ██║
╚██████╔╝ ╚████╔╝ ███████╗██║  ██║██║ ╚████║██║╚██████╔╝██║  ██║   ██║
 ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝

    ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗
    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝
    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║██╔████╔██║███████║██████╔╝█████╔╝
    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗
    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗
    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝

EOF

log "=============================================="
log "OVERNIGHT BENCHMARK SUITE"
log "Started: $(date)"
log "Run ID: $TIMESTAMP"
log "Output: $RUN_DIR"
log "Suite: $SUITE"
log "Skip slow: $SKIP_SLOW"
log "Dry run: $DRY_RUN"
log "=============================================="

# Record system info
{
    echo "System Information"
    echo "=================="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
    echo "Cores: $(nproc)"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
    echo ""
    echo "llama.cpp version:"
    /mnt/raid0/llm/llama.cpp/build/bin/llama-cli --version 2>&1 | head -5 || echo "N/A"
} > "$RUN_DIR/system_info.txt"

# =============================================================================
# RUN SUITES
# =============================================================================
TOTAL_START=$(date +%s)
SUITES_RUN=0
SUITES_PASSED=0
SUITES_FAILED=0

# Check queue at start (models queued before run began)
process_queue

# Thinking benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "thinking" ]]; then
    ((SUITES_RUN++)) || true
    if run_suite "thinking" "$SCRIPT_DIR/run_all_thinking_benchmarks.sh"; then
        ((SUITES_PASSED++)) || true
    else
        ((SUITES_FAILED++)) || true
    fi
fi
process_queue  # Check for newly queued models

# Coder benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "coder" ]]; then
    ((SUITES_RUN++)) || true
    if run_suite "coder" "$SCRIPT_DIR/run_all_coder_benchmarks.sh"; then
        ((SUITES_PASSED++)) || true
    else
        ((SUITES_FAILED++)) || true
    fi
fi

# VL benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "vl" ]]; then
    ((SUITES_RUN++)) || true
    if run_suite "vl" "$SCRIPT_DIR/run_all_vl_benchmarks.sh"; then
        ((SUITES_PASSED++)) || true
    else
        ((SUITES_FAILED++)) || true
    fi
fi
process_queue  # Check for newly queued models

# General benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "general" ]]; then
    ((SUITES_RUN++)) || true
    if run_suite "general" "$SCRIPT_DIR/run_all_general_benchmarks.sh"; then
        ((SUITES_PASSED++)) || true
    else
        ((SUITES_FAILED++)) || true
    fi
fi

# Agentic benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "agentic" ]]; then
    ((SUITES_RUN++)) || true
    if run_suite "agentic" "$SCRIPT_DIR/run_all_agentic_benchmarks.sh"; then
        ((SUITES_PASSED++)) || true
    else
        ((SUITES_FAILED++)) || true
    fi
fi

# Math benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "math" ]]; then
    ((SUITES_RUN++)) || true
    if run_suite "math" "$SCRIPT_DIR/run_all_math_benchmarks.sh"; then
        ((SUITES_PASSED++)) || true
    else
        ((SUITES_FAILED++)) || true
    fi
fi
process_queue  # Check for newly queued models

# Long Context benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "long_context" ]]; then
    ((SUITES_RUN++)) || true
    if run_suite "long_context" "$SCRIPT_DIR/run_all_long_context_benchmarks.sh"; then
        ((SUITES_PASSED++)) || true
    else
        ((SUITES_FAILED++)) || true
    fi
fi

# Instruction Precision benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "instruction_precision" ]]; then
    ((SUITES_RUN++)) || true
    if run_suite "instruction_precision" "$SCRIPT_DIR/run_all_instruction_precision_benchmarks.sh"; then
        ((SUITES_PASSED++)) || true
    else
        ((SUITES_FAILED++)) || true
    fi
fi

# Final queue check before speculative decoding
process_queue

# =============================================================================
# SPECULATIVE DECODING BENCHMARKS
# =============================================================================
if [[ "$SUITE" == "all" || "$SUITE" == "spec_decode" ]]; then
    log "=========================================="
    log "SPECULATIVE DECODING BENCHMARKS"
    log "=========================================="

    SPEC_DIR="$RUN_DIR/speculative_decoding"
    mkdir -p "$SPEC_DIR"

    # Define target/draft pairs
    DRAFT_MODEL="/mnt/raid0/llm/lmstudio/models/QuantFactory/Qwen2.5-0.5B-GGUF/Qwen2.5-0.5B.Q8_0.gguf"

    declare -a SPEC_PAIRS=(
        "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf|$DRAFT_MODEL|Qwen2.5-Coder-32B"
        "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf|$DRAFT_MODEL|Qwen2.5-72B-Instruct"
    )

    PROMPT="Write a Python function that implements binary search on a sorted list."

    for spec_pair in "${SPEC_PAIRS[@]}"; do
        IFS='|' read -r target draft name <<< "$spec_pair"

        if [[ ! -f "$target" || ! -f "$draft" ]]; then
            log "Skipping $name - model files not found"
            continue
        fi

        log "Testing speculative decoding: $name"

        if [[ "$DRY_RUN" == "true" ]]; then
            log "[DRY RUN] Would run spec decode test for $name"
            continue
        fi

        # Baseline
        log "  Running baseline..."
        timeout 300 OMP_NUM_THREADS=1 numactl --interleave=all \
            /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
            -m "$target" -t 96 -n 256 --temp 0.2 \
            -p "$PROMPT" \
            > "$SPEC_DIR/${name}_baseline.txt" 2>&1 || true

        # Speculative decode K=8
        log "  Running speculative decode (K=8)..."
        timeout 300 OMP_NUM_THREADS=1 numactl --interleave=all \
            /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
            -m "$target" -md "$draft" \
            --draft-max 8 -t 96 -n 256 --temp 0.2 \
            -p "$PROMPT" \
            > "$SPEC_DIR/${name}_spec_k8.txt" 2>&1 || true

        # Speculative decode K=16
        log "  Running speculative decode (K=16)..."
        timeout 300 OMP_NUM_THREADS=1 numactl --interleave=all \
            /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
            -m "$target" -md "$draft" \
            --draft-max 16 -t 96 -n 256 --temp 0.2 \
            -p "$PROMPT" \
            > "$SPEC_DIR/${name}_spec_k16.txt" 2>&1 || true

        # Extract and compare speeds
        baseline_speed=$(grep "eval time" "$SPEC_DIR/${name}_baseline.txt" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")
        spec8_speed=$(grep "eval time" "$SPEC_DIR/${name}_spec_k8.txt" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")
        spec16_speed=$(grep "eval time" "$SPEC_DIR/${name}_spec_k16.txt" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")

        log "  Results: baseline=${baseline_speed} t/s, K=8=${spec8_speed} t/s, K=16=${spec16_speed} t/s"
    done
fi

# =============================================================================
# LOOKUP DECODING BENCHMARKS (if available)
# =============================================================================
if [[ "$SUITE" == "all" || "$SUITE" == "lookup" ]]; then
    log "=========================================="
    log "LOOKUP DECODING BENCHMARKS"
    log "=========================================="

    LOOKUP_DIR="$RUN_DIR/lookup_decoding"
    mkdir -p "$LOOKUP_DIR"

    LLAMA_LOOKUP="/mnt/raid0/llm/llama.cpp/build/bin/llama-lookup"

    if [[ -x "$LLAMA_LOOKUP" ]]; then
        # Test models with lookup decoding
        declare -a LOOKUP_MODELS=(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf|Qwen2.5-Coder-32B"
        )

        # Prompt with repetitive patterns (good for lookup)
        LOOKUP_PROMPT='```python
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """Calculate the factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n-1)

# Test the functions
print(fibonacci(10))
print(factorial(5))
```

Write similar recursive functions for:
1. Sum of digits
2. Power function'

        for model_spec in "${LOOKUP_MODELS[@]}"; do
            IFS='|' read -r model_path model_name <<< "$model_spec"

            if [[ ! -f "$model_path" ]]; then
                log "Skipping $model_name - model file not found"
                continue
            fi

            log "Testing lookup decoding: $model_name"

            if [[ "$DRY_RUN" == "true" ]]; then
                log "[DRY RUN] Would run lookup decode test for $model_name"
                continue
            fi

            # With lookup
            timeout 300 OMP_NUM_THREADS=1 numactl --interleave=all \
                "$LLAMA_LOOKUP" \
                -m "$model_path" -t 96 -n 256 --temp 0.2 \
                -p "$LOOKUP_PROMPT" \
                > "$LOOKUP_DIR/${model_name}_lookup.txt" 2>&1 || true

            lookup_speed=$(grep "eval time" "$LOOKUP_DIR/${model_name}_lookup.txt" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "N/A")
            log "  Lookup speed: ${lookup_speed} t/s"
        done
    else
        log "llama-lookup not available - skipping lookup decoding tests"
    fi
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# One final queue check
process_queue

log ""
log "=============================================="
log "OVERNIGHT BENCHMARK COMPLETE"
log "=============================================="
log "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log "Suites run: $SUITES_RUN"
log "Suites passed: $SUITES_PASSED"
log "Suites failed: $SUITES_FAILED"
if [[ $QUEUED_MODELS_PROCESSED -gt 0 ]]; then
    log "Dynamically queued models processed: $QUEUED_MODELS_PROCESSED"
fi
log "Results: $RUN_DIR"
log "=============================================="

# Process results into permanent storage with JSONL index
log ""
log "Processing results into permanent storage..."
if "$SCRIPT_DIR/process_benchmark_results.sh" --run-id "$TIMESTAMP" >> "$LOG_FILE" 2>&1; then
    log "Results processed successfully"
    log "Permanent storage: /mnt/raid0/llm/claude/benchmarks/results/runs/$TIMESTAMP"
    log "Index: /mnt/raid0/llm/claude/benchmarks/results/index.jsonl"
else
    log "WARNING: Results processing failed"
fi

# Generate final report
{
    echo "=============================================="
    echo "OVERNIGHT BENCHMARK FINAL REPORT"
    echo "=============================================="
    echo ""
    echo "Run ID: $TIMESTAMP"
    echo "Started: $(cat "$RUN_DIR/system_info.txt" | grep Date | head -1)"
    echo "Completed: $(date)"
    echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "Suites: $SUITES_RUN run, $SUITES_PASSED passed, $SUITES_FAILED failed"
    echo ""
    echo "Output Directories:"
    echo "-------------------"
    ls -la "$RUN_DIR" 2>/dev/null || true
    echo ""
    echo "=============================================="
    echo "QUICK LINKS TO RESULTS"
    echo "=============================================="

    for suite in thinking coder vl general agentic math long_context instruction_precision; do
        suite_dir="$RUN_DIR/$suite"
        if [[ -d "$suite_dir" ]]; then
            echo ""
            echo "--- $suite ---"
            echo "Log: $suite_dir/output.log"
            # Find the most recent summary file
            latest_summary=$(find /tmp/claude/${suite}_rubric_results/ -name "benchmark_summary_*.txt" 2>/dev/null | sort | tail -1 || true)
            if [[ -n "$latest_summary" ]]; then
                echo "Summary: $latest_summary"
            fi
        fi
    done

    echo ""
    echo "=============================================="
    echo "SPECULATIVE DECODING RESULTS"
    echo "=============================================="
    if [[ -d "$RUN_DIR/speculative_decoding" ]]; then
        for f in "$RUN_DIR"/speculative_decoding/*.txt; do
            if [[ -f "$f" ]]; then
                name=$(basename "$f" .txt)
                speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
                echo "  $name: $speed"
            fi
        done
    else
        echo "  (not run)"
    fi

    echo ""
    echo "=============================================="
    echo "LOOKUP DECODING RESULTS"
    echo "=============================================="
    if [[ -d "$RUN_DIR/lookup_decoding" ]]; then
        for f in "$RUN_DIR"/lookup_decoding/*.txt; do
            if [[ -f "$f" ]]; then
                name=$(basename "$f" .txt)
                speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
                echo "  $name: $speed"
            fi
        done
    else
        echo "  (not run or llama-lookup not available)"
    fi

} > "$RUN_DIR/FINAL_REPORT.txt"

cat "$RUN_DIR/FINAL_REPORT.txt"

log ""
log "Full report: $RUN_DIR/FINAL_REPORT.txt"
log "Log file: $LOG_FILE"
