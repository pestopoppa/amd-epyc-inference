#!/bin/bash
# Agentic/Tool-Use Model Quality Rubric Test Script
# Runs all T1/T2/T3 tool-calling and agentic questions
# Supports multiple configurations: baseline, MoE reduction
set -euo pipefail

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}"  # dense, qwen3moe
OUTPUT_DIR="/tmp/claude/agentic_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TIMEOUT=120

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model.gguf> <model_name> [arch]"
    echo ""
    echo "Architecture types:"
    echo "  dense       - Standard dense model (no MoE optimization)"
    echo "  qwen3moe    - Qwen3-MoE model (tests baseline + MoE reduction)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/Qwen2.5-7B-Instruct.gguf Qwen2.5-7B dense"
    echo "  $0 /path/to/Qwen3-30B-A3B-Instruct.gguf Qwen3-30B qwen3moe"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Determine configurations to test based on architecture
declare -a CONFIGS
case "$MODEL_ARCH" in
    dense)
        CONFIGS=("baseline")
        ;;
    qwen3moe)
        CONFIGS=("baseline" "moe4")
        ;;
    *)
        echo "Unknown architecture: $MODEL_ARCH, using baseline only"
        CONFIGS=("baseline")
        ;;
esac

echo "=============================================="
echo "Agentic/Tool-Use Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Model file: $MODEL"
echo "Architecture: $MODEL_ARCH"
echo "Configurations to test: ${CONFIGS[*]}"
echo "Date: $(date)"
echo "=============================================="

# Function to get MoE override based on config and arch
get_moe_override() {
    local config="$1"
    local arch="$2"

    if [[ "$config" == "baseline" ]]; then
        echo ""
        return
    fi

    local expert_count="${config#moe}"

    case "$arch" in
        qwen3moe)
            echo "--override-kv qwen3moe.expert_used_count=int:$expert_count"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to run a test and extract timing
run_test() {
    local test_name="$1"
    local prompt="$2"
    local config="$3"
    local moe_override="$4"
    local output_file="$OUTPUT_DIR/${MODEL_NAME}_${config}_${test_name}.txt"

    echo ""
    echo "--- Running $test_name ($config) ---"

    # Write prompt to temp file
    echo "$prompt" > "/tmp/claude/agentic_prompt.txt"

    # Run model and capture output
    timeout "$TIMEOUT" "$LLAMA_COMPLETION" \
        -m "$MODEL" \
        -t 96 -n 512 --temp 0.2 \
        $moe_override \
        -f "/tmp/claude/agentic_prompt.txt" \
        > "$output_file" 2>&1 || true

    # Extract timing
    local speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

    echo "Speed: $speed"
    echo "Output saved to: $output_file"

    # Show the answer
    echo "--- Answer ---"
    grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:" "$output_file" | tail -30 | head -25
}

# Run tests for each configuration
for CONFIG in "${CONFIGS[@]}"; do
    MOE_OVERRIDE=$(get_moe_override "$CONFIG" "$MODEL_ARCH")

    echo ""
    echo "##############################################"
    echo "# Configuration: $CONFIG"
    [[ -n "$MOE_OVERRIDE" ]] && echo "# Override: $MOE_OVERRIDE"
    echo "##############################################"

    # T1: Baseline questions
    echo ""
    echo "========== TIER 1 (Baseline) =========="

    run_test "t1_q1_single_tool" \
'You have access to a tool:
{
  "name": "get_weather",
  "parameters": {
    "city": "string (required)",
    "units": "string (optional, celsius or fahrenheit, default celsius)"
  }
}

User asks: "What is the weather in Tokyo?"
Generate the tool call as JSON. Output only the JSON, nothing else.' "$CONFIG" "$MOE_OVERRIDE"

    run_test "t1_q2_multi_param" \
'Tool available:
{
  "name": "search_files",
  "parameters": {
    "pattern": "string (required) - glob pattern",
    "directory": "string (required) - path to search",
    "max_results": "integer (optional, default 10)"
  }
}

User asks: "Find all Python files in /src with max 5 results"
Generate the tool call as JSON. Output only the JSON.' "$CONFIG" "$MOE_OVERRIDE"

    run_test "t1_q3_choose_tool" \
'Available tools:
1. read_file: {"path": "string"} - Read file contents
2. write_file: {"path": "string", "content": "string"} - Write to file
3. list_directory: {"path": "string"} - List directory contents

User asks: "Show me what is in the config folder"
Which tool and what parameters? Output only the JSON tool call.' "$CONFIG" "$MOE_OVERRIDE"

    # T2: Medium-Hard questions
    echo ""
    echo "========== TIER 2 (Medium-Hard) =========="

    run_test "t2_q1_sequential" \
'Tools:
- read_file: {"path": "string"}
- grep_search: {"pattern": "string", "path": "string"}

User asks: "Find where ERROR is logged in /var/log/app.log, then show me that file"

Generate the tool calls in order as a JSON array. Output only the JSON array.' "$CONFIG" "$MOE_OVERRIDE"

    run_test "t2_q2_error_handling" \
'You called: {"name": "get_user", "parameters": {"id": 123}}

Tool returned: {"error": "User not found", "code": 404}

Tools available:
- get_user: {"id": "integer"}
- search_users: {"query": "string"} - Search by name/email
- create_user: {"name": "string", "email": "string"}

What should you do next? Either output a tool call as JSON, or a message to the user asking for clarification.' "$CONFIG" "$MOE_OVERRIDE"

    run_test "t2_q3_nested_params" \
'Tool:
{
  "name": "create_task",
  "parameters": {
    "title": "string (required)",
    "assignees": "array of strings (required)",
    "metadata": {
      "priority": "string (low/medium/high)",
      "tags": "array of strings",
      "due_date": "string (ISO date)"
    }
  }
}

User: "Create a high priority task called Fix login bug for Alice and Bob, tagged as bug and urgent, due 2024-03-15"

Output only the JSON tool call.' "$CONFIG" "$MOE_OVERRIDE"

    # T3: Hard questions
    echo ""
    echo "========== TIER 3 (Hard) =========="

    run_test "t3_q1_ambiguous" \
'Tools:
- send_email: {"to": "string", "subject": "string", "body": "string"}
- send_slack: {"channel": "string", "message": "string"}
- create_ticket: {"title": "string", "description": "string", "assignee": "string"}

User: "Let the team know about the outage"

This is ambiguous. Either ask a clarifying question, OR if you must act, choose the most appropriate tool and explain why in one sentence, then output the JSON.' "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q2_error_chain" \
'Conversation so far:

You: {"name": "deploy", "parameters": {"env": "prod", "version": "1.2.3"}}
Tool: {"error": "Deployment blocked: failing health check on staging"}

You: {"name": "get_health", "parameters": {"env": "staging"}}
Tool: {"status": "unhealthy", "failing_checks": ["database_connection"]}

You: {"name": "check_database", "parameters": {"env": "staging"}}
Tool: {"status": "connection_refused", "host": "db-staging.internal", "port": 5432}

What is your next action? Available tools:
- restart_service: {"service": "string", "env": "string"}
- get_logs: {"service": "string", "env": "string", "lines": "integer"}
- notify_oncall: {"message": "string", "severity": "string"}
- check_dns: {"hostname": "string"}

Explain your reasoning briefly, then output the JSON tool call.' "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q3_schema_edge" \
'Tool schema:
{
  "name": "query_api",
  "parameters": {
    "endpoint": "string (required) - must start with /",
    "method": "string (required) - GET/POST/PUT/DELETE",
    "body": "object (required for POST/PUT, must be null for GET/DELETE)",
    "headers": "object (optional)"
  }
}

User: "GET the users endpoint with an auth header Bearer token123"

Generate the correct tool call. Output only the JSON.' "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q4_orchestration" \
'You need to deploy a hotfix. Tools available:
- git_checkout: {"branch": "string"}
- run_tests: {"suite": "string"} - returns pass/fail
- build_image: {"tag": "string"}
- deploy: {"env": "string", "image": "string"}
- notify_slack: {"channel": "string", "message": "string"}
- rollback: {"env": "string", "to_version": "string"}

Current state: main branch, last deploy was v1.2.2, hotfix is on branch hotfix/auth-fix

Create a deployment plan as ordered tool calls. Include what to do if tests fail or deploy fails.
Output as JSON with structure: {"steps": [...], "on_test_fail": [...], "on_deploy_fail": [...]}' "$CONFIG" "$MOE_OVERRIDE"

    # Configuration Summary
    echo ""
    echo "=============================================="
    echo "CONFIG $CONFIG COMPLETE"
    echo "=============================================="
    echo "Speed summary for $MODEL_NAME ($CONFIG):"
    for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_*.txt; do
        if [[ -f "$f" ]]; then
            test_name=$(basename "$f" .txt | sed "s/${MODEL_NAME}_${CONFIG}_//")
            speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
            echo "  $test_name: $speed"
        fi
    done
done

# Final Summary
echo ""
echo "=============================================="
echo "AGENTIC RUBRIC TEST COMPLETE - ALL CONFIGURATIONS"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
