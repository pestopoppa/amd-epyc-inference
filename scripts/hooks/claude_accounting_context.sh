#!/bin/bash
set -euo pipefail
# Hook: PreToolUse -> Write|Edit
# Reminder when editing CLAUDE policy files.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

if [[ "$FILE_PATH" == *"CLAUDE.md"* || "$FILE_PATH" == *"CLAUDE_GUIDE.md"* ]]; then
  cat <<'JSON'
{"additionalContext":"You are editing CLAUDE policy/docs. Keep docs/reference/agent-config/CLAUDE_MD_MATRIX.md and claude_md_matrix.json in sync if scope/governance changed."}
JSON
fi

exit 0
