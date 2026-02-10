#!/bin/bash
set -euo pipefail
# Hook: PreToolUse → Bash
# Blocks pytest -n auto or -n N where N > 16 (192-thread machine safety)

INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [[ -z "$CMD" ]]; then
  exit 0
fi

if echo "$CMD" | grep -qP 'pytest.*-n\s*auto'; then
  echo "BLOCKED: pytest -n auto would spawn ~192 workers and crash this machine. Use -n 8 (default) or -n 4." >&2
  exit 2
fi

if N=$(echo "$CMD" | grep -oP 'pytest.*-n\s*\K[0-9]+'); then
  if [[ "$N" -gt 16 ]]; then
    echo "BLOCKED: pytest -n $N is too many workers for this 192-thread machine. Use -n 8 (default)." >&2
    exit 2
  fi
fi

exit 0
