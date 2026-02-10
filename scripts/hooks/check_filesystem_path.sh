#!/bin/bash
set -euo pipefail
# Hook: PreToolUse → Write|Edit
# Blocks writes outside /mnt/raid0/ (except .claude/ config dir)

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

# Allow /mnt/raid0/* (project data)
if [[ "$FILE_PATH" == /mnt/raid0/* ]]; then
  exit 0
fi

# Allow .claude/ config paths (e.g. ~/.claude/, project .claude/)
if [[ "$FILE_PATH" == */.claude/* ]]; then
  exit 0
fi

echo "BLOCKED: Write to '$FILE_PATH' denied. All files must be on /mnt/raid0/. Root FS is a 120GB SSD." >&2
exit 2
