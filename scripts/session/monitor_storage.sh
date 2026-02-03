#!/bin/bash
# monitor_storage.sh - Monitor root FS and alert on high usage
# Usage: ./scripts/session/monitor_storage.sh
# Run in background: ./scripts/session/monitor_storage.sh &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

THRESHOLD_WARN=70
THRESHOLD_CRITICAL=85
CHECK_INTERVAL=30 # seconds

echo "Storage Monitor Started"
echo "Thresholds: WARN=${THRESHOLD_WARN}%, CRITICAL=${THRESHOLD_CRITICAL}%"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

while true; do
  # Get root FS usage percentage
  ROOT_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
  ROOT_AVAIL=$(df -h / | awk 'NR==2 {print $4}')

  # Get /tmp/claude usage if it exists
  if [ -d /tmp/claude ]; then
    TMP_CLAUDE_SIZE=$(du -sh /tmp/claude 2>/dev/null | cut -f1)
  else
    TMP_CLAUDE_SIZE="N/A"
  fi

  # Check thresholds
  if [ "$ROOT_USAGE" -ge "$THRESHOLD_CRITICAL" ]; then
    echo "🚨 CRITICAL: Root FS at ${ROOT_USAGE}% (available: ${ROOT_AVAIL})"
    echo "   /tmp/claude: $TMP_CLAUDE_SIZE"
    echo "   ACTION REQUIRED: Run emergency_cleanup.sh"

    # Log to agent audit
    if [ -f "${PROJECT_ROOT}/scripts/utils/agent_log.sh" ]; then
      source "${PROJECT_ROOT}/scripts/utils/agent_log.sh"
      agent_error "Root FS CRITICAL" "usage=${ROOT_USAGE}%, /tmp/claude=${TMP_CLAUDE_SIZE}"
    fi

    # Send system notification if possible
    notify-send -u critical "Root FS Critical" "Usage: ${ROOT_USAGE}%" 2>/dev/null || true

  elif [ "$ROOT_USAGE" -ge "$THRESHOLD_WARN" ]; then
    echo "⚠️  WARNING: Root FS at ${ROOT_USAGE}% (available: ${ROOT_AVAIL})"
    echo "   /tmp/claude: $TMP_CLAUDE_SIZE"

    # Log warning
    if [ -f "${PROJECT_ROOT}/scripts/utils/agent_log.sh" ]; then
      source "${PROJECT_ROOT}/scripts/utils/agent_log.sh"
      agent_warn "Root FS high usage" "usage=${ROOT_USAGE}%, /tmp/claude=${TMP_CLAUDE_SIZE}"
    fi
  else
    # Normal - just show status every 5 minutes
    TIMESTAMP=$(date +%H:%M:%S)
    echo "[$TIMESTAMP] OK - Root: ${ROOT_USAGE}% (${ROOT_AVAIL} free), /tmp/claude: $TMP_CLAUDE_SIZE"
  fi

  sleep "$CHECK_INTERVAL"
done
