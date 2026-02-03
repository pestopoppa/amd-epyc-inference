#!/bin/bash
# agent_log_analyze.sh — Analyze agent audit logs for issues
# Usage:
#   ./agent_log_analyze.sh              # Analyze recent activity
#   ./agent_log_analyze.sh --loops      # Detect potential loops
#   ./agent_log_analyze.sh --errors     # Show all errors
#   ./agent_log_analyze.sh --session ID # Analyze specific session
#   ./agent_log_analyze.sh --timeline   # Show chronological timeline

set -euo pipefail

# Source environment library for path variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

LOG_FILE="${LOG_DIR}/agent_audit.log"

if [[ ! -f "$LOG_FILE" ]]; then
  echo "No audit log found at $LOG_FILE"
  exit 1
fi

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
  echo "Agent Log Analyzer"
  echo ""
  echo "Usage: $0 [command]"
  echo ""
  echo "Commands:"
  echo "  --summary       Overview of recent activity (default)"
  echo "  --loops         Detect repeated commands (potential loops)"
  echo "  --errors        Show all errors and warnings"
  echo "  --sessions      List all sessions"
  echo "  --session ID    Show details for specific session"
  echo "  --timeline N    Show last N entries chronologically"
  echo "  --commands      Show command frequency analysis"
  echo "  --files         Show file modifications"
  echo "  --rollbacks     Show logged rollback commands"
  echo "  --tail N        Show last N raw log entries"
  echo ""
}

summary() {
  echo -e "${BLUE}=== Agent Activity Summary ===${NC}"
  echo ""

  local total_entries

  total_entries=$(wc -l <"$LOG_FILE")
  local sessions
  sessions=$(grep -o '"session":"[^"]*"' "$LOG_FILE" | sort -u | wc -l)
  local errors
  errors=$(grep -c '"level":"ERROR"' "$LOG_FILE" || echo 0)
  local warnings
  warnings=$(grep -c '"level":"WARN"' "$LOG_FILE" || echo 0)
  local commands
  commands=$(grep -c '"cat":"CMD_INTENT"' "$LOG_FILE" || echo 0)
  local tasks
  tasks=$(grep -c '"cat":"TASK_START"' "$LOG_FILE" || echo 0)

  echo "Total log entries:  $total_entries"
  echo "Unique sessions:    $sessions"
  echo "Tasks started:      $tasks"
  echo "Commands executed:  $commands"
  echo -e "Errors:             ${RED}$errors${NC}"
  echo -e "Warnings:           ${YELLOW}$warnings${NC}"
  echo ""

  echo -e "${BLUE}--- Recent Sessions ---${NC}"
  grep '"cat":"SESSION_START"' "$LOG_FILE" | tail -5 | while read -r line; do
    local ts
    ts=$(echo "$line" | jq -r '.ts' 2>/dev/null || echo "?")
    local session
    session=$(echo "$line" | jq -r '.session' 2>/dev/null || echo "?")
    local msg
    msg=$(echo "$line" | jq -r '.msg' 2>/dev/null || echo "?")
    echo "  $ts - $session: $msg"
  done
  echo ""

  if [[ $errors -gt 0 ]]; then
    echo -e "${RED}--- Recent Errors ---${NC}"
    grep '"level":"ERROR"' "$LOG_FILE" | tail -5 | while read -r line; do
      local ts
      ts=$(echo "$line" | jq -r '.ts' 2>/dev/null || echo "?")
      local msg
      msg=$(echo "$line" | jq -r '.msg' 2>/dev/null || echo "?")
      echo "  $ts: $msg"
    done
    echo ""
  fi
}

detect_loops() {
  echo -e "${BLUE}=== Loop Detection ===${NC}"
  echo ""

  # Find repeated commands in short time windows
  echo -e "${YELLOW}Commands executed more than 3 times:${NC}"
  grep '"cat":"CMD_INTENT"' "$LOG_FILE" |
    jq -r '.msg' 2>/dev/null |
    sort | uniq -c | sort -rn |
    awk '$1 > 3 {print "  " $1 "x: " substr($0, index($0,$2))}' | head -10
  echo ""

  # Find repeated commands within same session
  echo -e "${YELLOW}Repeated commands within sessions:${NC}"
  local sessions
  sessions=$(grep -o '"session":"[^"]*"' "$LOG_FILE" | sort -u | sed 's/"session":"//;s/"//')

  for session in $sessions; do
    local repeated
    repeated=$(grep "\"session\":\"$session\"" "$LOG_FILE" |
      grep '"cat":"CMD_INTENT"' |
      jq -r '.msg' 2>/dev/null |
      sort | uniq -c | sort -rn |
      awk '$1 > 2 {print $0}' | head -3)

    if [[ -n "$repeated" ]]; then
      echo -e "  ${BLUE}Session: $session${NC}"
      echo "$repeated" | while read -r line; do
        echo "    $line"
      done
    fi
  done
  echo ""

  # Check for rapid-fire commands (potential runaway)
  echo -e "${YELLOW}Rapid command sequences (>5 in 60 seconds):${NC}"
  grep '"cat":"CMD_INTENT"' "$LOG_FILE" |
    jq -r '.ts' 2>/dev/null |
    while read -r ts; do
      date -d "$ts" +%s 2>/dev/null || echo "0"
    done |
    awk 'NR>1 {diff=$1-prev; if(diff<60 && diff>=0) count++; else count=1; if(count>5) print "  Burst at line " NR ": " count " commands in <60s"} {prev=$1}' |
    head -5
  echo ""
}

show_errors() {
  echo -e "${BLUE}=== Errors and Warnings ===${NC}"
  echo ""

  echo -e "${RED}--- Errors ---${NC}"
  grep '"level":"ERROR"' "$LOG_FILE" | while read -r line; do
    local ts
    ts=$(echo "$line" | jq -r '.ts' 2>/dev/null || echo "?")
    local session
    session=$(echo "$line" | jq -r '.session' 2>/dev/null || echo "?")
    local msg
    msg=$(echo "$line" | jq -r '.msg' 2>/dev/null || echo "?")
    local details
    details=$(echo "$line" | jq -r '.details' 2>/dev/null || echo "")
    echo -e "${RED}[$ts]${NC} $msg"
    [[ -n "$details" && "$details" != "null" ]] && echo "         Details: $details"
    echo "         Session: $session"
  done
  echo ""

  echo -e "${YELLOW}--- Warnings ---${NC}"
  grep '"level":"WARN"' "$LOG_FILE" | while read -r line; do
    local ts
    ts=$(echo "$line" | jq -r '.ts' 2>/dev/null || echo "?")
    local msg
    msg=$(echo "$line" | jq -r '.msg' 2>/dev/null || echo "?")
    echo "  [$ts] $msg"
  done
  echo ""
}

list_sessions() {
  echo -e "${BLUE}=== Sessions ===${NC}"
  echo ""

  grep '"cat":"SESSION_START"' "$LOG_FILE" | while read -r line; do
    local ts
    ts=$(echo "$line" | jq -r '.ts' 2>/dev/null || echo "?")
    local session
    session=$(echo "$line" | jq -r '.session' 2>/dev/null || echo "?")
    local msg
    msg=$(echo "$line" | jq -r '.msg' 2>/dev/null || echo "?")

    local cmd_count

    cmd_count=$(grep "\"session\":\"$session\"" "$LOG_FILE" | grep -c '"cat":"CMD_INTENT"' || echo 0)
    local err_count
    err_count=$(grep "\"session\":\"$session\"" "$LOG_FILE" | grep -c '"level":"ERROR"' || echo 0)

    echo -e "${GREEN}$session${NC}"
    echo "  Started: $ts"
    echo "  Purpose: $msg"
    echo "  Commands: $cmd_count, Errors: $err_count"
    echo ""
  done
}

show_session() {
  local session_id="$1"
  echo -e "${BLUE}=== Session: $session_id ===${NC}"
  echo ""

  grep "\"session\":\"$session_id\"" "$LOG_FILE" | while read -r line; do
    local ts
    ts=$(echo "$line" | jq -r '.ts' 2>/dev/null | sed 's/T/ /' | cut -d'+' -f1)
    local level
    level=$(echo "$line" | jq -r '.level' 2>/dev/null || echo "?")
    local cat
    cat=$(echo "$line" | jq -r '.cat' 2>/dev/null || echo "?")
    local msg
    msg=$(echo "$line" | jq -r '.msg' 2>/dev/null || echo "?")
    local details
    details=$(echo "$line" | jq -r '.details' 2>/dev/null || echo "")

    case "$level" in
      ERROR) color=$RED ;;
      WARN) color=$YELLOW ;;
      *) color=$NC ;;
    esac

    printf "${color}%-20s${NC} %-15s %s\n" "$ts" "$cat" "$msg"
    [[ -n "$details" && "$details" != "null" && "$details" != "" ]] && echo "                                        └─ $details"
  done
}

show_timeline() {
  local count="${1:-50}"
  echo -e "${BLUE}=== Timeline (last $count entries) ===${NC}"
  echo ""

  tail -n "$count" "$LOG_FILE" | while read -r line; do
    local ts
    ts=$(echo "$line" | jq -r '.ts' 2>/dev/null | sed 's/T/ /' | cut -d'+' -f1)
    local level
    level=$(echo "$line" | jq -r '.level' 2>/dev/null || echo "?")
    local cat
    cat=$(echo "$line" | jq -r '.cat' 2>/dev/null || echo "?")
    local msg
    msg=$(echo "$line" | jq -r '.msg' 2>/dev/null || echo "?")

    case "$level" in
      ERROR) color=$RED ;;
      WARN) color=$YELLOW ;;
      *) color=$NC ;;
    esac

    printf "${color}%-20s${NC} [%-12s] %s\n" "$ts" "$cat" "${msg:0:80}"
  done
}

show_commands() {
  echo -e "${BLUE}=== Command Frequency ===${NC}"
  echo ""

  grep '"cat":"CMD_INTENT"' "$LOG_FILE" |
    jq -r '.msg' 2>/dev/null |
    sed 's/^[[:space:]]*//' |
    cut -d' ' -f1 |
    sort | uniq -c | sort -rn | head -20 |
    while read -r count cmd; do
      printf "  %4d  %s\n" "$count" "$cmd"
    done
}

show_files() {
  echo -e "${BLUE}=== File Modifications ===${NC}"
  echo ""

  grep '"cat":"FILE_MODIFY"' "$LOG_FILE" | while read -r line; do
    local ts
    ts=$(echo "$line" | jq -r '.ts' 2>/dev/null | sed 's/T/ /' | cut -d'+' -f1)
    local msg
    msg=$(echo "$line" | jq -r '.msg' 2>/dev/null || echo "?")
    local details
    details=$(echo "$line" | jq -r '.details' 2>/dev/null || echo "")
    echo "  [$ts] $msg"
    [[ -n "$details" && "$details" != "null" ]] && echo "           $details"
  done
}

show_rollbacks() {
  echo -e "${BLUE}=== Logged Rollback Commands ===${NC}"
  echo ""
  echo "Use these to undo agent actions if needed:"
  echo ""

  grep '"cat":"ROLLBACK"' "$LOG_FILE" | while read -r line; do
    local ts
    ts=$(echo "$line" | jq -r '.ts' 2>/dev/null | sed 's/T/ /' | cut -d'+' -f1)
    local msg
    msg=$(echo "$line" | jq -r '.msg' 2>/dev/null || echo "?")
    local details
    details=$(echo "$line" | jq -r '.details' 2>/dev/null || echo "")
    echo -e "  ${YELLOW}Action:${NC} $msg"
    echo -e "  ${GREEN}Undo:${NC}   ${details#undo_with: }"
    echo ""
  done
}

# Main
case "${1:-}" in
  --help | -h)
    show_help
    ;;
  --loops)
    detect_loops
    ;;
  --errors)
    show_errors
    ;;
  --sessions)
    list_sessions
    ;;
  --session)
    show_session "${2:-}"
    ;;
  --timeline)
    show_timeline "${2:-50}"
    ;;
  --commands)
    show_commands
    ;;
  --files)
    show_files
    ;;
  --rollbacks)
    show_rollbacks
    ;;
  --tail)
    tail -n "${2:-20}" "$LOG_FILE"
    ;;
  --summary | "")
    summary
    ;;
  *)
    echo "Unknown command: $1"
    show_help
    exit 1
    ;;
esac
