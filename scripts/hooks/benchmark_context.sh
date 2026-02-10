#!/bin/bash
set -euo pipefail
# Hook: PreToolUse → Write|Edit
# Injects reminder to use /benchmark skill when editing benchmark files

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

if [[ "$FILE_PATH" == *benchmarks/* ]] || [[ "$FILE_PATH" == *scripts/benchmark/* ]]; then
  cat <<'EOF'
{"additionalContext": "You are editing benchmark infrastructure. Run /benchmark for the full benchmarking workflow, scoring rubric, and eval analysis protocol."}
EOF
fi

exit 0
