#!/bin/bash
set -euo pipefail
# Hook: PreToolUse -> Write|Edit
# Validate simple local markdown path references in agent/docs governance files.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
PROJECT_DIR=${CLAUDE_PROJECT_DIR:-$(pwd)}

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

case "$FILE_PATH" in
  agents/* | */agents/* | CLAUDE_GUIDE.md | */CLAUDE_GUIDE.md | README.md | */README.md | docs/guides/* | */docs/guides/* | docs/reference/agent-config/* | */docs/reference/agent-config/*) ;;
  *)
    exit 0
    ;;
esac

if [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

# Extract likely local markdown links from inline code spans.
mapfile -t refs < <(rg -o '`[^`]+\.md`' "$FILE_PATH" | tr -d '`' | sed 's/:.*$//' | sort -u)

missing=()
for ref in "${refs[@]}"; do
  [[ "$ref" =~ ^https?:// ]] && continue
  [[ "$ref" == *'*'* ]] && continue
  if [[ "$ref" == /* ]]; then
    target="$ref"
  else
    target="$PROJECT_DIR/$ref"
  fi
  [[ -f "$target" ]] || missing+=("$ref")
done

if ((${#missing[@]} > 0)); then
  echo "BLOCKED: unresolved local markdown references in $FILE_PATH:" >&2
  for m in "${missing[@]}"; do
    echo "  - $m" >&2
  done
  exit 2
fi

exit 0
