#!/bin/bash
# General/Instruct Model Quality Rubric Test Script
# Runs all T1/T2/T3 general instruction-following questions
# Supports multiple configurations: baseline, MoE reduction
set -euo pipefail

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}"  # dense, qwen3moe
OUTPUT_DIR="/tmp/claude/general_rubric_results"
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
echo "General/Instruct Model Quality Rubric"
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
    echo "$prompt" > "/tmp/claude/general_prompt.txt"

    # Run model and capture output
    timeout "$TIMEOUT" "$LLAMA_COMPLETION" \
        -m "$MODEL" \
        -t 96 -n 512 --temp 0.3 \
        $moe_override \
        -f "/tmp/claude/general_prompt.txt" \
        > "$output_file" 2>&1 || true

    # Extract timing
    local speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

    echo "Speed: $speed"
    echo "Output saved to: $output_file"

    # Show the answer
    echo "--- Answer ---"
    grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:" "$output_file" | tail -25 | head -20
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

    run_test "t1_q1_reformat" \
"Convert this to a bullet list:
The meeting covered three topics. First, we discussed the Q3 budget which is \$50K over.
Second, we reviewed hiring plans for two engineers. Third, we set the launch date for March 15." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t1_q2_summarize" \
"Summarize in one sentence:
The new caching layer reduced API latency from 200ms to 45ms, improving user experience
significantly. However, it increased memory usage by 2GB per server, requiring us to
upgrade our instance types from m5.large to m5.xlarge, adding \$500/month to our AWS bill." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t1_q3_extract" \
"Extract all email addresses from this text:
Contact John at john.doe@company.com for sales inquiries. For support, reach out to
support@company.com or help-desk@company.com. Press inquiries: press@external.org" "$CONFIG" "$MOE_OVERRIDE"

    # T2: Medium-Hard questions
    echo ""
    echo "========== TIER 2 (Medium-Hard) =========="

    run_test "t2_q1_json" \
"Parse this into JSON with fields: name, role, department, start_date
'Sarah Chen joined as Senior Engineer in the Platform team on 2024-03-15.'
Output only the JSON, no explanation." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t2_q2_multistep" \
"Process this list:
1. Remove duplicates (case-insensitive)
2. Sort alphabetically
3. Number each item
4. Add a count at the end

Items: banana, Apple, cherry, BANANA, apple, Date, cherry" "$CONFIG" "$MOE_OVERRIDE"

    run_test "t2_q3_compare" \
"Compare these two approaches in 2-3 sentences:
Approach A: Microservices - Each feature is a separate service with its own database.
Pros: Independent scaling, isolated failures. Cons: Network overhead, data consistency challenges.

Approach B: Monolith - Single application with shared database.
Pros: Simple deployment, easy data joins. Cons: Scaling limitations, coupled codebase." "$CONFIG" "$MOE_OVERRIDE"

    # T3: Hard questions
    echo ""
    echo "========== TIER 3 (Hard) =========="

    run_test "t3_q1_synthesis" \
"Synthesize these three perspectives into a unified recommendation:

Engineering: 'We need 3 months to refactor the auth system properly. Rushing will create tech debt.'
Product: 'Customers are churning due to login issues. We need a fix in 2 weeks.'
Finance: 'Q4 budget is tight. Any solution over \$20K needs board approval.'

Provide a concrete recommendation in 3-4 sentences." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q2_transform" \
"Transform this flat data into nested YAML grouped by department:

employees:
- name: Alice, dept: Engineering, level: Senior
- name: Bob, dept: Sales, level: Junior
- name: Carol, dept: Engineering, level: Junior
- name: Dave, dept: Sales, level: Senior" "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q3_schedule" \
"Schedule these meetings given constraints:
- Team sync (60min): Must include Alice, Bob, Carol
- 1:1 Alice-Dave (30min)
- 1:1 Bob-Dave (30min)
- Dave only available 9-11am and 2-4pm
- Alice unavailable 10-11am
- No back-to-back meetings for anyone

Available slots: 9am, 9:30am, 10am, 10:30am, 11am, 2pm, 2:30pm, 3pm, 3:30pm

Output a valid schedule or explain why impossible." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q4_inconsistency" \
"These 3 documents describe the same system. Find inconsistencies:

Doc A: 'The API accepts POST requests with JSON body. Rate limit is 100 req/min.'
Doc B: 'Send data via POST with form-encoded body. Rate limit is 100 requests per minute.'
Doc C: 'API endpoint accepts POST. JSON payload required. Rate limited to 1000 req/hour.'

List all inconsistencies found." "$CONFIG" "$MOE_OVERRIDE"

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
echo "GENERAL RUBRIC TEST COMPLETE - ALL CONFIGURATIONS"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
