#!/bin/bash
set -euo pipefail
# Hook: PreToolUse -> Write|Edit
# Enforce minimal schema for role files in agents/*.md

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

if [[ ! "$FILE_PATH" =~ (^|/)agents/[^/]+\.md$ ]]; then
  exit 0
fi

base=$(basename "$FILE_PATH")
if [[ "$base" == "README.md" || "$base" == "AGENT_INSTRUCTIONS.md" ]]; then
  exit 0
fi

if [[ ! -f "$FILE_PATH" ]]; then
  # New file - guard cannot inspect contents pre-write.
  exit 0
fi

required=(
  "## Mission"
  "## Use This Role When"
  "## Inputs Required"
  "## Outputs"
  "## Workflow"
  "## Guardrails"
)

missing=()
for marker in "${required[@]}"; do
  if ! rg -q "^${marker}$" "$FILE_PATH"; then
    missing+=("$marker")
  fi
done

if ((${#missing[@]} > 0)); then
  echo "BLOCKED: $FILE_PATH is missing required role sections:" >&2
  for m in "${missing[@]}"; do
    echo "  - $m" >&2
  done
  exit 2
fi

exit 0
