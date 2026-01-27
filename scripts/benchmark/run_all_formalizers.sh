#!/bin/bash
# Run all formalizer model evaluations in background
# Usage: ./run_all_formalizers.sh
#
# Results will be saved to /mnt/raid0/llm/claude/logs/formalizer_eval/
# Check progress: tail -f logs/formalizer_eval/run.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/mnt/raid0/llm/claude/logs/formalizer_eval"
PROMPTS_DIR="/mnt/raid0/llm/claude/benchmarks/prompts/v1/formalizer"

mkdir -p "$LOG_DIR"

# Models to evaluate
MODELS=(
  "/mnt/raid0/llm/models/xLAM-2-1B-fc-r-Q4_K_M.gguf:xLAM-2-1B"
  "/mnt/raid0/llm/models/xLAM-1b-fc-r.Q4_K_M.gguf:xLAM-1B"
  "/mnt/raid0/llm/models/nexusraven-v2-13b.Q4_K_M.gguf:NexusRaven"
)

echo "=== Formalizer Evaluation Suite ===" | tee "$LOG_DIR/run.log"
echo "Started: $(date)" | tee -a "$LOG_DIR/run.log"
echo "" | tee -a "$LOG_DIR/run.log"

for entry in "${MODELS[@]}"; do
  model_path="${entry%%:*}"
  model_name="${entry##*:}"

  echo "[$model_name] Starting evaluation..." | tee -a "$LOG_DIR/run.log"

  "$SCRIPT_DIR/bench_formalizers.sh" \
    --model "$model_path" \
    --prompts "$PROMPTS_DIR" \
    --output "$LOG_DIR/$model_name" \
    2>&1 | tee -a "$LOG_DIR/run.log"

  echo "[$model_name] Complete!" | tee -a "$LOG_DIR/run.log"
  echo "" | tee -a "$LOG_DIR/run.log"
done

echo "=== All Formalizers Complete ===" | tee -a "$LOG_DIR/run.log"
echo "Finished: $(date)" | tee -a "$LOG_DIR/run.log"

# Generate comparison summary
echo "" | tee -a "$LOG_DIR/run.log"
echo "=== Comparison Summary ===" | tee -a "$LOG_DIR/run.log"

python3 <<'PYTHON' | tee -a "$LOG_DIR/run.log"
import csv
import os
import glob

log_dir = "/mnt/raid0/llm/claude/logs/formalizer_eval"
results = []

for csv_file in glob.glob(f"{log_dir}/*/results.csv"):
    model_name = os.path.basename(os.path.dirname(csv_file))
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        continue

    total = len(rows)
    parsable = sum(1 for r in rows if int(r["parsable"]) == 1)
    schema_valid = sum(1 for r in rows if int(r["schema_valid"]) == 1)
    avg_complete = sum(float(r["completeness"]) for r in rows) / total
    avg_tps = sum(float(r["tokens_per_sec"]) for r in rows) / total

    results.append({
        "model": model_name,
        "parsable": f"{parsable}/{total}",
        "schema_valid": f"{schema_valid}/{total}",
        "avg_complete": f"{avg_complete:.2f}",
        "avg_tps": f"{avg_tps:.1f}"
    })

if results:
    print(f"{'Model':<15} {'Parsable':<10} {'Schema Valid':<12} {'Completeness':<12} {'t/s':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<15} {r['parsable']:<10} {r['schema_valid']:<12} {r['avg_complete']:<12} {r['avg_tps']:<8}")
else:
    print("No results found")
PYTHON

echo ""
echo "Full results in: $LOG_DIR/"
