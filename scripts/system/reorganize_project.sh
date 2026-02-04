#!/bin/bash
# Project Directory Reorganization Script
# Works bottom-up (leaves to root) to reorganize the project structure
#
# Changes:
# 1. Move /mnt/raid0/llm/LOGS → /mnt/raid0/llm/claude/logs
# 2. Consolidate model_routing/ (8 files) → docs/model-routing.md
# 3. Consolidate RESEARCH_WRITER_*.md (3 files) → docs/research-writer.md
# 4. Move stray scripts from /mnt/raid0/llm/ → scripts/legacy/
# 5. Merge 00_START_HERE.md + INDEX.md → README.md
# 6. Update all path references
# 7. Clean up backups

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Use env vars (overridable)
PROJECT_ROOT="${PROJECT_ROOT}"
LLM_ROOT="${LLM_ROOT}"
BACKUP_DIR="$PROJECT_ROOT/backups/pre-reorg-$(date +%Y%m%d_%H%M%S)"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# Create backup
create_backup() {
  log "Creating backup at $BACKUP_DIR"
  mkdir -p "$BACKUP_DIR"

  # Backup key directories
  cp -r "$LLM_ROOT/LOGS" "$BACKUP_DIR/" 2>/dev/null || true
  cp -r "$PROJECT_ROOT/model_routing" "$BACKUP_DIR/" 2>/dev/null || true
  cp "$PROJECT_ROOT"/*.md "$BACKUP_DIR/" 2>/dev/null || true
  cp -r "$PROJECT_ROOT/agents" "$BACKUP_DIR/" 2>/dev/null || true
  cp -r "$PROJECT_ROOT/docs" "$BACKUP_DIR/" 2>/dev/null || true
}

# Step 1: Move LOGS to claude/logs
move_logs() {
  log "Step 1: Moving LOGS → claude/logs"

  if [ -d "$LLM_ROOT/LOGS" ] && [ ! -d "$PROJECT_ROOT/logs" ]; then
    mv "$LLM_ROOT/LOGS" "$PROJECT_ROOT/logs"
    log "  Moved LOGS to claude/logs"

    # Create symlink for backwards compatibility
    ln -sf "$PROJECT_ROOT/logs" "$LLM_ROOT/LOGS"
    log "  Created symlink LOGS → claude/logs"
  elif [ -d "$PROJECT_ROOT/logs" ]; then
    log "  logs/ already exists, skipping"
  fi
}

# Step 2: Consolidate model_routing documentation
consolidate_model_routing() {
  log "Step 2: Consolidating model_routing → docs/model-routing.md"

  local MODEL_ROUTING_DIR="$PROJECT_ROOT/model_routing"
  local OUTPUT="$PROJECT_ROOT/docs/model-routing.md"

  if [ -d "$MODEL_ROUTING_DIR" ]; then
    cat >"$OUTPUT" <<'HEADER'
# Model Routing Guide

This document consolidates the model routing strategy for the AMD EPYC 9655 inference project.

---

HEADER

    # Add quick reference first
    if [ -f "$MODEL_ROUTING_DIR/MODEL_ROUTING_QUICK_REFERENCE.md" ]; then
      echo "## Quick Reference" >>"$OUTPUT"
      echo "" >>"$OUTPUT"
      tail -n +2 "$MODEL_ROUTING_DIR/MODEL_ROUTING_QUICK_REFERENCE.md" >>"$OUTPUT"
      echo -e "\n---\n" >>"$OUTPUT"
    fi

    # Add implementation details
    if [ -f "$MODEL_ROUTING_DIR/MODEL_ROUTING_IMPLEMENTATION.md" ]; then
      echo "## Implementation Details" >>"$OUTPUT"
      echo "" >>"$OUTPUT"
      tail -n +2 "$MODEL_ROUTING_DIR/MODEL_ROUTING_IMPLEMENTATION.md" >>"$OUTPUT"
      echo -e "\n---\n" >>"$OUTPUT"
    fi

    # Archive the directory
    mv "$MODEL_ROUTING_DIR" "$BACKUP_DIR/model_routing_archived"
    log "  Consolidated 8 files → docs/model-routing.md"
  else
    log "  model_routing/ not found, skipping"
  fi
}

# Step 3: Consolidate RESEARCH_WRITER docs
consolidate_research_writer() {
  log "Step 3: Consolidating RESEARCH_WRITER_*.md → docs/research-writer.md"

  local OUTPUT="$PROJECT_ROOT/docs/research-writer.md"

  if [ -f "$PROJECT_ROOT/RESEARCH_WRITER_QUICK_REF.md" ]; then
    cat >"$OUTPUT" <<'HEADER'
# Research Writer Agent Guide

This document describes the research-writer agent's capabilities and workflows.

---

HEADER

    # Add quick reference
    echo "## Quick Reference" >>"$OUTPUT"
    echo "" >>"$OUTPUT"
    tail -n +2 "$PROJECT_ROOT/RESEARCH_WRITER_QUICK_REF.md" >>"$OUTPUT"
    echo -e "\n---\n" >>"$OUTPUT"

    # Add integration guide
    if [ -f "$PROJECT_ROOT/RESEARCH_WRITER_INTEGRATION.md" ]; then
      echo "## Integration Guide" >>"$OUTPUT"
      echo "" >>"$OUTPUT"
      tail -n +2 "$PROJECT_ROOT/RESEARCH_WRITER_INTEGRATION.md" >>"$OUTPUT"
      echo -e "\n---\n" >>"$OUTPUT"
    fi

    # Add summary
    if [ -f "$PROJECT_ROOT/RESEARCH_WRITER_SUMMARY.md" ]; then
      echo "## Summary" >>"$OUTPUT"
      echo "" >>"$OUTPUT"
      tail -n +2 "$PROJECT_ROOT/RESEARCH_WRITER_SUMMARY.md" >>"$OUTPUT"
    fi

    # Move originals to backup
    mv "$PROJECT_ROOT"/RESEARCH_WRITER_*.md "$BACKUP_DIR/" 2>/dev/null || true
    log "  Consolidated 3 files → docs/research-writer.md"
  else
    log "  RESEARCH_WRITER files not found, skipping"
  fi
}

# Step 4: Move stray scripts
move_stray_scripts() {
  log "Step 4: Moving stray scripts → scripts/legacy/"

  mkdir -p "$PROJECT_ROOT/scripts/legacy"

  local moved=0
  for script in "$LLM_ROOT"/*.sh "$LLM_ROOT"/*.py; do
    if [ -f "$script" ]; then
      local basename
      basename=$(basename "$script")
      # Don't move if it's a system script we want to keep at root
      if [[ "$basename" != "organize_claude_dir.sh" ]]; then
        mv "$script" "$PROJECT_ROOT/scripts/legacy/"
        ((moved++))
      fi
    fi
  done

  log "  Moved $moved scripts to scripts/legacy/"
}

# Step 5: Merge onboarding docs
merge_onboarding_docs() {
  log "Step 5: Merging onboarding docs → README.md"

  local README="$PROJECT_ROOT/README.md"

  if [ -f "$PROJECT_ROOT/00_START_HERE.md" ]; then
    # Use 00_START_HERE as base
    cp "$PROJECT_ROOT/00_START_HERE.md" "$README"

    # Add index content if different
    if [ -f "$PROJECT_ROOT/INDEX.md" ]; then
      echo -e "\n---\n" >>"$README"
      echo "## Project Index" >>"$README"
      echo "" >>"$README"
      tail -n +2 "$PROJECT_ROOT/INDEX.md" >>"$README"
    fi

    # Archive originals
    mv "$PROJECT_ROOT/00_START_HERE.md" "$BACKUP_DIR/" 2>/dev/null || true
    mv "$PROJECT_ROOT/INDEX.md" "$BACKUP_DIR/" 2>/dev/null || true

    log "  Created README.md from 00_START_HERE + INDEX"
  else
    log "  Onboarding docs not found, skipping"
  fi
}

# Step 6: Update path references in all files
update_path_references() {
  log "Step 6: Updating path references"

  # Define path mappings (old → new)
  declare -A PATH_MAPPINGS=(
    ["/mnt/raid0/llm/LOGS"]="/mnt/raid0/llm/claude/logs"
    ["/mnt/raid0/llm/claude/agent_log.sh"]="/mnt/raid0/llm/claude/scripts/utils/agent_log.sh"
    ["/mnt/raid0/llm/claude/agent_log_analyze.sh"]="/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh"
    ["/mnt/raid0/llm/claude/experiments/"]="/mnt/raid0/llm/claude/research/"
  )

  local updated=0

  # Update .md files
  for file in "$PROJECT_ROOT"/*.md "$PROJECT_ROOT"/**/*.md; do
    if [ -f "$file" ]; then
      for old_path in "${!PATH_MAPPINGS[@]}"; do
        new_path="${PATH_MAPPINGS[$old_path]}"
        if grep -q "$old_path" "$file" 2>/dev/null; then
          sed -i "s|$old_path|$new_path|g" "$file"
          ((updated++))
        fi
      done
    fi
  done

  # Update .sh files
  for file in "$PROJECT_ROOT"/scripts/**/*.sh; do
    if [ -f "$file" ]; then
      for old_path in "${!PATH_MAPPINGS[@]}"; do
        new_path="${PATH_MAPPINGS[$old_path]}"
        if grep -q "$old_path" "$file" 2>/dev/null; then
          sed -i "s|$old_path|$new_path|g" "$file"
          ((updated++))
        fi
      done
    fi
  done

  log "  Updated $updated path references"
}

# Step 7: Clean up old backups (keep only last 2)
cleanup_old_backups() {
  log "Step 7: Cleaning up old backups"

  local backup_count

  backup_count=$(ls -d "$PROJECT_ROOT/backups"/backup-* 2>/dev/null | wc -l)

  if [ "$backup_count" -gt 2 ]; then
    # Remove all but the 2 most recent
    ls -dt "$PROJECT_ROOT/backups"/backup-* | tail -n +3 | xargs rm -rf
    log "  Removed $((backup_count - 2)) old backups"
  else
    log "  No old backups to clean"
  fi
}

# Step 8: Update CLAUDE.md with new structure
update_claude_md() {
  log "Step 8: Updating CLAUDE.md directory structure section"

  # This will be done separately with more careful editing
  log "  CLAUDE.md needs manual review for structure updates"
}

# Main
main() {
  log "=== Project Directory Reorganization ==="
  log "Project root: $PROJECT_ROOT"
  log ""

  create_backup
  echo ""

  move_logs
  echo ""

  consolidate_model_routing
  echo ""

  consolidate_research_writer
  echo ""

  move_stray_scripts
  echo ""

  merge_onboarding_docs
  echo ""

  update_path_references
  echo ""

  cleanup_old_backups
  echo ""

  update_claude_md
  echo ""

  log "=== Reorganization Complete ==="
  log "Backup saved to: $BACKUP_DIR"
  log ""
  log "Next steps:"
  log "  1. Review docs/model-routing.md"
  log "  2. Review docs/research-writer.md"
  log "  3. Review README.md"
  log "  4. Update CLAUDE.md directory structure section"
  log "  5. Test that scripts still work with new paths"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
