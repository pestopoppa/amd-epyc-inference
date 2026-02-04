#!/bin/bash
# verify_llama_cpp.sh — Verify llama.cpp is on correct production branch
#
# Run this at session start to prevent accidentally using wrong branch.
# Returns non-zero if branch is wrong, allowing session_init.sh to warn/fail.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/env.sh"

# Configuration (derived from env.sh)
LLAMA_CPP_DIR="${LLM_ROOT}/llama.cpp"
EXPECTED_BRANCH="production-consolidated"
EXPERIMENTAL_DIR="${LLM_ROOT}/llama.cpp-experimental"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

verify_branch() {
  local dir="$1"
  local expected="$2"
  local label="$3"

  if [[ ! -d "$dir/.git" ]]; then
    echo -e "${YELLOW}⚠ $label not found at $dir${NC}"
    return 1
  fi

  local current_branch
  current_branch=$(cd "$dir" && git branch --show-current 2>/dev/null || echo "DETACHED")

  if [[ "$current_branch" == "$expected" ]]; then
    echo -e "${GREEN}✓ $label: $current_branch${NC}"
    return 0
  else
    echo -e "${RED}✗ $label: expected '$expected', got '$current_branch'${NC}"
    echo -e "${YELLOW}  Fix with: cd $dir && git checkout $expected${NC}"
    return 1
  fi
}

check_binary_exists() {
  local binary="$LLAMA_CPP_DIR/build/bin/llama-cli"
  if [[ -x "$binary" ]]; then
    echo -e "${GREEN}✓ Production binary exists: $binary${NC}"
    return 0
  else
    echo -e "${RED}✗ Production binary missing: $binary${NC}"
    echo -e "${YELLOW}  Rebuild with: cd $LLAMA_CPP_DIR && cmake -B build && cmake --build build -j$(nproc)${NC}"
    return 1
  fi
}

main() {
  echo "=== llama.cpp Branch Verification ==="
  echo ""

  local errors=0

  # Check production branch
  if ! verify_branch "$LLAMA_CPP_DIR" "$EXPECTED_BRANCH" "Production"; then
    ((errors++))
  fi

  # Check experimental is NOT production (should be on feature branch)
  if [[ -d "$EXPERIMENTAL_DIR/.git" ]]; then
    local exp_branch
    exp_branch=$(cd "$EXPERIMENTAL_DIR" && git branch --show-current 2>/dev/null || echo "DETACHED")
    if [[ "$exp_branch" == "$EXPECTED_BRANCH" ]]; then
      echo -e "${YELLOW}⚠ Experimental is also on $EXPECTED_BRANCH (expected feature branch)${NC}"
    else
      echo -e "${GREEN}✓ Experimental: $exp_branch (feature branch)${NC}"
    fi
  fi

  # Check binary exists
  if ! check_binary_exists; then
    ((errors++))
  fi

  echo ""

  if [[ $errors -gt 0 ]]; then
    echo -e "${RED}=== VERIFICATION FAILED ===${NC}"
    echo "Production llama.cpp is not correctly configured."
    echo "Fix the issues above before running inference."
    return 1
  else
    echo -e "${GREEN}=== VERIFICATION PASSED ===${NC}"
    return 0
  fi
}

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
