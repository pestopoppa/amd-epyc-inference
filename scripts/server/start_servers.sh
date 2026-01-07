#!/bin/bash
# Start llama-server instances for the orchestrator
#
# This script launches llama-server with prefix caching enabled.
# Designed for the RadixAttention-style caching workflow.
#
# Usage:
#   ./start_servers.sh [model_path] [port] [slots]
#   ./start_servers.sh  # Uses defaults for test model
#
# Environment variables:
#   LLAMA_CPP_PATH  - Path to llama.cpp build (default: /mnt/raid0/llm/llama.cpp/build/bin)
#   MODELS_PATH     - Path to models directory (default: /mnt/raid0/llm/models)
#   LOG_DIR         - Directory for server logs (default: /mnt/raid0/llm/claude/logs)
#
# Examples:
#   # Start test server with Qwen2.5-Coder-0.5B
#   ./start_servers.sh
#
#   # Start production server with specific model
#   ./start_servers.sh /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf 8080 4
#
#   # Start on different port
#   PORT=8081 ./start_servers.sh

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

LLAMA_CPP_PATH="${LLAMA_CPP_PATH:-/mnt/raid0/llm/llama.cpp/build/bin}"
MODELS_PATH="${MODELS_PATH:-/mnt/raid0/llm/models}"
LOG_DIR="${LOG_DIR:-/mnt/raid0/llm/claude/logs}"

# Default test model (small, fast for development)
DEFAULT_MODEL="${MODELS_PATH}/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"
ALT_MODEL_PATH="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

# Server settings
DEFAULT_PORT="${PORT:-8080}"
DEFAULT_SLOTS="${SLOTS:-4}"
DEFAULT_CONTEXT="${CONTEXT:-4096}"
DEFAULT_THREADS="${THREADS:-16}"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

check_binary() {
    local binary="$1"
    if [[ ! -x "$binary" ]]; then
        log_error "Binary not found or not executable: $binary"
        log_error "Build llama.cpp first: cd /mnt/raid0/llm/llama.cpp && cmake -B build && cmake --build build -j"
        exit 1
    fi
}

check_model() {
    local model="$1"
    if [[ ! -f "$model" ]]; then
        log_error "Model not found: $model"
        exit 1
    fi
}

kill_existing_server() {
    local port="$1"
    local pid
    pid=$(lsof -t -i:"$port" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        log_info "Killing existing process on port $port (PID: $pid)"
        kill "$pid" 2>/dev/null || true
        sleep 1
    fi
}

wait_for_health() {
    local url="$1"
    local max_attempts="${2:-30}"
    local attempt=0

    log_info "Waiting for server health at $url..."

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s "$url/health" >/dev/null 2>&1; then
            log_info "Server is healthy!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    log_error "Server failed to become healthy after $max_attempts seconds"
    return 1
}

# =============================================================================
# Main
# =============================================================================

main() {
    local model="${1:-}"
    local port="${2:-$DEFAULT_PORT}"
    local slots="${3:-$DEFAULT_SLOTS}"

    # Determine model path
    if [[ -z "$model" ]]; then
        if [[ -f "$DEFAULT_MODEL" ]]; then
            model="$DEFAULT_MODEL"
        elif [[ -f "$ALT_MODEL_PATH" ]]; then
            model="$ALT_MODEL_PATH"
        else
            log_error "No model specified and default model not found"
            log_error "Expected: $DEFAULT_MODEL"
            log_error "Or: $ALT_MODEL_PATH"
            exit 1
        fi
    fi

    local server_binary="$LLAMA_CPP_PATH/llama-server"
    local log_file="$LOG_DIR/llama-server-$port.log"

    # Validate paths
    check_binary "$server_binary"
    check_model "$model"
    mkdir -p "$LOG_DIR"

    # Kill any existing server on this port
    kill_existing_server "$port"

    log_info "Starting llama-server:"
    log_info "  Model:   $model"
    log_info "  Port:    $port"
    log_info "  Slots:   $slots"
    log_info "  Context: $DEFAULT_CONTEXT"
    log_info "  Threads: $DEFAULT_THREADS"
    log_info "  Log:     $log_file"

    # Start server with prefix caching enabled
    # Key flags:
    #   -np N       : Number of parallel slots (for concurrent requests)
    #   -c N        : Context size per slot
    #   --host      : Bind address
    #   --port      : Listening port
    #   -t N        : Number of threads
    #
    # Prefix caching is automatic in llama-server when using the /completion
    # endpoint with cache_prompt=true in the request payload.

    env OMP_NUM_THREADS=1 \
    numactl --interleave=all \
    "$server_binary" \
        -m "$model" \
        --host 0.0.0.0 \
        --port "$port" \
        -np "$slots" \
        -c "$DEFAULT_CONTEXT" \
        -t "$DEFAULT_THREADS" \
        > "$log_file" 2>&1 &

    local server_pid=$!
    log_info "Server started with PID: $server_pid"

    # Wait for server to be ready
    if wait_for_health "http://localhost:$port"; then
        log_info "Server is ready for requests"
        log_info ""
        log_info "Test with:"
        log_info "  curl http://localhost:$port/health"
        log_info ""
        log_info "Completion with caching:"
        log_info "  curl http://localhost:$port/completion -d '{\"prompt\": \"Hello\", \"n_predict\": 10, \"cache_prompt\": true}'"
        log_info ""
        log_info "View logs:"
        log_info "  tail -f $log_file"
        log_info ""
        log_info "Stop server:"
        log_info "  kill $server_pid"
    else
        log_error "Server failed to start. Check logs: $log_file"
        tail -20 "$log_file" 2>/dev/null || true
        exit 1
    fi
}

# =============================================================================
# Additional Commands
# =============================================================================

case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [model_path] [port] [slots]"
        echo ""
        echo "Start llama-server with prefix caching enabled."
        echo ""
        echo "Arguments:"
        echo "  model_path  Path to GGUF model file (optional)"
        echo "  port        Server port (default: 8080)"
        echo "  slots       Number of parallel slots (default: 4)"
        echo ""
        echo "Environment variables:"
        echo "  LLAMA_CPP_PATH  Path to llama.cpp build directory"
        echo "  MODELS_PATH     Path to models directory"
        echo "  LOG_DIR         Directory for server logs"
        echo "  PORT            Default port"
        echo "  SLOTS           Default number of slots"
        echo "  CONTEXT         Context size per slot"
        echo "  THREADS         Number of threads"
        echo ""
        echo "Commands:"
        echo "  --help        Show this help"
        echo "  --status      Show running servers"
        echo "  --stop        Stop all llama-server processes"
        exit 0
        ;;
    --status)
        echo "Running llama-server processes:"
        pgrep -a llama-server || echo "No servers running"
        exit 0
        ;;
    --stop)
        echo "Stopping all llama-server processes..."
        pkill -f llama-server || echo "No servers to stop"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
