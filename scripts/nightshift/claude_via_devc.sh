#!/bin/bash
# claude_via_devc.sh — Claude Code proxy through devcontainer
#
# Passes all arguments to claude inside the project devcontainer,
# which has bypassPermissions mode enabled. Nightshift uses this
# as its claude binary to avoid host-side permission issues.
#
# The devcontainer:
#   - Has defaultMode: bypassPermissions
#   - Mounts /mnt/raid0/llm (shared git repo + worktree)
#   - Runs as user 'node'
#   - Can create branches, edit files, commit
#   - Cannot push to GitHub (no gh auth) — nightshift handles push on host
#
# Nightshift operates in a worktree at /mnt/raid0/llm/claude-nightshift
# to avoid disrupting agents working on main in /mnt/raid0/llm/claude.

set -euo pipefail

# Worktree path — nightshift works here, not in the main checkout
WORKTREE="/mnt/raid0/llm/claude-nightshift"

CONTAINER_ID="$(docker ps -q -f "ancestor=vsc-claude-e51ac396fef248434826b5406ff85b7fc60ec88212497878e70375898b83bdb6-uid" 2>/dev/null | head -1)"

if [[ -z "$CONTAINER_ID" ]]; then
  # Fallback: find any devcontainer for this project
  CONTAINER_ID="$(docker ps --format '{{.ID}} {{.Mounts}}' 2>/dev/null | grep "/mnt/raid0/llm" | awk '{print $1}' | head -1)"
fi

if [[ -z "$CONTAINER_ID" ]]; then
  echo "error: no devcontainer running for /mnt/raid0/llm/claude" >&2
  echo "hint: run 'devc /mnt/raid0/llm/claude' to start it" >&2
  exit 1
fi

# Forward all args to claude inside the container, working in the worktree
exec docker exec -u node -w "$WORKTREE" "$CONTAINER_ID" claude "$@"
