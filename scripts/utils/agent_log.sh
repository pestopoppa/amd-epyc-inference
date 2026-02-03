#!/bin/bash
# agent_log.sh — Append-only audit logging for AI agent actions
# Source this file in scripts: source /path/to/scripts/utils/agent_log.sh
# All logs are append-only to ${LOG_DIR}/agent_audit.log

set -euo pipefail

# Source environment library for path variables
_AGENT_LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/env.sh
source "${_AGENT_LOG_DIR}/../lib/env.sh"
unset _AGENT_LOG_DIR

# Configuration (use LOG_DIR from env.sh)
AGENT_LOG_DIR="${LOG_DIR}"
AGENT_LOG_FILE="${AGENT_LOG_DIR}/agent_audit.log"
AGENT_SESSION_FILE="${AGENT_LOG_DIR}/.current_session"

# Ensure log directory exists
mkdir -p "$AGENT_LOG_DIR"

# Generate or retrieve session ID
if [[ -z "${AGENT_SESSION_ID:-}" ]]; then
  if [[ -f "$AGENT_SESSION_FILE" ]]; then
    # Check if session is stale (older than 4 hours)
    if [[ $(find "$AGENT_SESSION_FILE" -mmin +240 2>/dev/null) ]]; then
      AGENT_SESSION_ID="ses_$(date +%Y%m%d_%H%M%S)_$$"
      echo "$AGENT_SESSION_ID" >"$AGENT_SESSION_FILE"
    else
      AGENT_SESSION_ID=$(cat "$AGENT_SESSION_FILE")
    fi
  else
    AGENT_SESSION_ID="ses_$(date +%Y%m%d_%H%M%S)_$$"
    echo "$AGENT_SESSION_ID" >"$AGENT_SESSION_FILE"
  fi
  export AGENT_SESSION_ID
fi

# Core logging function (append-only)
_agent_log() {
  local level="$1"
  local category="$2"
  local message="$3"
  local details="${4:-}"

  local timestamp

  timestamp=$(date -Iseconds)
  local entry
  entry=$(
    cat <<EOF
{"ts":"$timestamp","session":"$AGENT_SESSION_ID","level":"$level","cat":"$category","msg":"$message","details":"$details"}
EOF
  )
  # Append-only write (>> never truncates)
  echo "$entry" >>"$AGENT_LOG_FILE"

  # Also echo to stderr for visibility (optional)
  if [[ "${AGENT_LOG_VERBOSE:-0}" == "1" ]]; then
    echo "[$level] $category: $message" >&2
  fi
}

# Public logging functions

# Log the start of a new task/goal
agent_task_start() {
  local task_description="$1"
  local reasoning="${2:-}"
  _agent_log "INFO" "TASK_START" "$task_description" "$reasoning"
  echo "TASK: $task_description"
}

# Log completion of a task
agent_task_end() {
  local task_description="$1"
  local outcome="${2:-success}"
  _agent_log "INFO" "TASK_END" "$task_description" "outcome=$outcome"
}

# Log a command BEFORE execution (intent)
agent_cmd_intent() {
  local command="$1"
  local reasoning="$2"
  _agent_log "INFO" "CMD_INTENT" "$command" "$reasoning"
}

# Log command result AFTER execution
agent_cmd_result() {
  local command="$1"
  local exit_code="$2"
  local output_summary="${3:-}"
  _agent_log "INFO" "CMD_RESULT" "$command" "exit=$exit_code; $output_summary"
}

# Log a file modification BEFORE it happens
agent_file_modify() {
  local filepath="$1"
  local action="$2" # create, edit, delete, move
  local reasoning="$3"
  _agent_log "INFO" "FILE_MODIFY" "$filepath" "action=$action; $reasoning"
}

# Log a decision or reasoning step
agent_decision() {
  local decision="$1"
  local reasoning="$2"
  _agent_log "INFO" "DECISION" "$decision" "$reasoning"
}

# Log a warning (potential issue noticed)
agent_warn() {
  local message="$1"
  local context="${2:-}"
  _agent_log "WARN" "WARNING" "$message" "$context"
}

# Log an error
agent_error() {
  local message="$1"
  local context="${2:-}"
  _agent_log "ERROR" "ERROR" "$message" "$context"
}

# Log state observation (system state, file contents, etc.)
agent_observe() {
  local what="$1"
  local value="$2"
  _agent_log "INFO" "OBSERVE" "$what" "$value"
}

# Log rollback information (how to undo an action)
agent_rollback_info() {
  local action="$1"
  local rollback_cmd="$2"
  _agent_log "INFO" "ROLLBACK" "$action" "undo_with: $rollback_cmd"
}

# Wrapper to execute command with full logging
agent_exec() {
  local reasoning="$1"
  shift
  local cmd="$*"

  agent_cmd_intent "$cmd" "$reasoning"

  # Capture output and exit code
  local output
  local exit_code
  output=$("$@" 2>&1) && exit_code=$? || exit_code=$?

  # Truncate output for log (first 500 chars)
  local output_summary="${output:0:500}"
  output_summary="${output_summary//$'\n'/ }" # Replace newlines

  agent_cmd_result "$cmd" "$exit_code" "$output_summary"

  # Echo output to stdout
  echo "$output"
  return $exit_code
}

# Start a new session explicitly
agent_session_start() {
  local purpose="$1"
  AGENT_SESSION_ID="ses_$(date +%Y%m%d_%H%M%S)_$$"
  echo "$AGENT_SESSION_ID" >"$AGENT_SESSION_FILE"
  export AGENT_SESSION_ID
  _agent_log "INFO" "SESSION_START" "$purpose" "new_session=$AGENT_SESSION_ID"
  echo "Session started: $AGENT_SESSION_ID"
}

# End session
agent_session_end() {
  local summary="${1:-}"
  _agent_log "INFO" "SESSION_END" "Session complete" "$summary"
  rm -f "$AGENT_SESSION_FILE"
}

# Helper to view recent logs
agent_log_tail() {
  local lines="${1:-50}"
  tail -n "$lines" "$AGENT_LOG_FILE" | jq -r '. | "\(.ts) [\(.level)] \(.cat): \(.msg)"' 2>/dev/null || tail -n "$lines" "$AGENT_LOG_FILE"
}

# Helper to view logs for current session
agent_log_session() {
  grep "\"session\":\"$AGENT_SESSION_ID\"" "$AGENT_LOG_FILE" | jq -r '. | "\(.ts) [\(.level)] \(.cat): \(.msg)"' 2>/dev/null || grep "\"session\":\"$AGENT_SESSION_ID\"" "$AGENT_LOG_FILE"
}

# Export functions
export -f _agent_log agent_task_start agent_task_end agent_cmd_intent agent_cmd_result
export -f agent_file_modify agent_decision agent_warn agent_error agent_observe
export -f agent_rollback_info agent_exec agent_session_start agent_session_end
export -f agent_log_tail agent_log_session

# Announce if sourced interactively
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Agent logging loaded. Session: $AGENT_SESSION_ID" >&2
fi
