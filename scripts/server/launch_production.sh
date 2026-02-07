#!/bin/bash
set -euo pipefail

# Deterministic production orchestrator stack launcher
# Usage: ./launch_production.sh [--full | --minimal | --with-burst | --dev]
# Default: Full HOT tier with all architects (~510GB, 45% of 1130GB RAM)
#
# This script provides a simple, deterministic way to launch the orchestrator
# stack without requiring agent interpretation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

STACK_PY="$SCRIPT_DIR/orchestrator_stack.py"

# Validate script exists
if [[ ! -f "$STACK_PY" ]]; then
    echo "ERROR: orchestrator_stack.py not found at $STACK_PY"
    exit 1
fi

# Parse mode
MODE="${1:---full}"

echo "============================================================"
echo "PRODUCTION ORCHESTRATOR STACK LAUNCHER"
echo "============================================================"
echo ""

case "$MODE" in
    --full)
        echo "Mode: FULL production stack (~535GB)"
        echo ""
        echo "Components:"
        echo "  - frontdoor (8080): Qwen3-Coder-30B-A3B, MoE6, 18 t/s"
        echo "  - coder_escalation (8081): Qwen2.5-Coder-32B + spec + lookup, 39 t/s"
        echo "  - worker_explore (8082): Qwen2.5-7B-Instruct + spec + lookup, 44 t/s"
        echo "  - architect_general (8083): Qwen3-235B-A22B, MoE4, 6.75 t/s"
        echo "  - architect_coding (8084): Qwen3-Coder-480B-A35B, MoE3, 10.3 t/s"
        echo "  - ingest_long_context (8085): Qwen3-Next-80B-A3B, MoE4, 6.3 t/s"
        echo "  - worker_vision (8086): Qwen2.5-VL-7B + mmproj, ~15 t/s"
        echo "  - vision_escalation (8087): Qwen3-VL-30B-A3B + mmproj, MoE4, ~10 t/s"
        echo "  - embedder (8090): BGE-large-en-v1.5 (1024-dim) [fanout 8090-8095]"
        echo "  - orchestrator (8000): uvicorn API"
        echo "  - document_formalizer (9001): LightOnOCR-2-1B"
        echo ""
        python3 "$STACK_PY" start
        ;;
    --minimal)
        echo "Mode: MINIMAL stack (~45GB)"
        echo ""
        echo "Components:"
        echo "  - frontdoor (8080): Qwen3-Coder-30B-A3B"
        echo "  - coder_escalation (8081): Qwen2.5-Coder-32B + spec + lookup"
        echo "  - worker_explore (8082): Qwen2.5-7B-Instruct + spec"
        echo "  - embedder (8090): BGE-large-en-v1.5 (1024-dim) [fanout 8090-8095]"
        echo ""
        echo "WARNING: Architects excluded - for testing/development only"
        echo ""
        python3 "$STACK_PY" start --hot-only
        ;;
    --with-burst)
        echo "Mode: FULL + burst worker (~515GB)"
        echo ""
        echo "Additional burst capacity:"
        echo "  - worker_fast (8102): Qwen2.5-Coder-1.5B, 4 slots, 60 t/s"
        echo ""
        python3 "$STACK_PY" start --include-warm worker_fast
        ;;
    --dev)
        echo "Mode: DEV (single 0.5B model)"
        echo ""
        echo "Single model for testing/development"
        echo ""
        python3 "$STACK_PY" start --dev
        ;;
    --status)
        echo "Checking stack status..."
        echo ""
        python3 "$STACK_PY" status
        exit 0
        ;;
    --stop)
        echo "Stopping all components..."
        echo ""
        python3 "$STACK_PY" stop --all
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0 [--full | --minimal | --with-burst | --dev | --status | --stop]"
        echo ""
        echo "Modes:"
        echo "  --full (default)  Full HOT tier + architects (~535GB, 47% RAM)"
        echo "  --minimal         Core tier only, no architects (~45GB)"
        echo "  --with-burst      Full + fast workers for burst capacity (~515GB)"
        echo "  --dev             Single 0.5B model for testing"
        echo ""
        echo "Commands:"
        echo "  --status          Show current stack status"
        echo "  --stop            Stop all running components"
        echo ""
        echo "RAM breakdown (full mode):"
        echo "  - Base tier: ~45GB"
        echo "  - architect_general (235B): ~140GB"
        echo "  - architect_coding (480B): ~280GB"
        echo "  - ingest_long_context (80B): ~45GB"
        echo "  - worker_vision (7B VL): ~6GB"
        echo "  - vision_escalation (30B VL MoE): ~19GB"
        echo "  - Total: ~535GB (47% of 1130GB)"
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

# Verify health
echo ""
echo "============================================================"
echo "HEALTH CHECK"
echo "============================================================"
python3 "$STACK_PY" status
