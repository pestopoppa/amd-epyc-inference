#!/bin/bash
set -euo pipefail
# Hook: PreToolUse -> Write|Edit
# Reminder to keep command-skills and packaged-skills aligned.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

if [[ "$FILE_PATH" == *".claude/commands/"* || "$FILE_PATH" == *".claude/skills/"* ]]; then
  cat <<'JSON'
{"additionalContext":"Skill governance reminder: keep .claude/commands workflows aligned with packaged skills under .claude/skills, and update validation scripts if conventions change."}
JSON
fi

exit 0
