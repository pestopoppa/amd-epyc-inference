#!/bin/bash
# system_audit.sh — Zen 5 Optimization Reconnaissance
# Run this BEFORE making any changes to capture baseline state
# Usage: bash system_audit.sh
# Output: system_audit.log in LOGS directory + agent audit log

set -euo pipefail

# Script location and environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Source logging library
if [[ -f "${SCRIPT_DIR}/../utils/agent_log.sh" ]]; then
  source "${SCRIPT_DIR}/../utils/agent_log.sh"
else
  # Fallback: define no-op functions if logging not available
  agent_session_start() { echo "Session: $1"; }
  agent_task_start() { echo "Task: $1"; }
  agent_task_end() { :; }
  agent_observe() { :; }
  agent_error() { echo "ERROR: $1" >&2; }
fi

agent_session_start "System audit for Zen 5 optimization baseline"

LOG_FILE="${LOG_DIR}/system_audit_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

agent_task_start "Collect system state" "Capturing baseline before any optimization changes"

# Header
{
  echo "=============================================="
  echo "SYSTEM AUDIT: $(date)"
  echo "Host: $(hostname)"
  echo "User: $(whoami)"
  echo "=============================================="
} | tee "$LOG_FILE"

# 1. Hardware Verification
echo -e "\n--- CPU & Architecture ---" | tee -a "$LOG_FILE"
echo "Model:" | tee -a "$LOG_FILE"
lscpu | grep "Model name" | tee -a "$LOG_FILE"

echo -e "\nAVX-512 Flags:" | tee -a "$LOG_FILE"
grep -oE "avx512[^ ]+" /proc/cpuinfo | sort -u | head -n 20 | tee -a "$LOG_FILE"

echo -e "\nCores/Threads:" | tee -a "$LOG_FILE"
lscpu | grep -E "^CPU\(s\)|Thread|Core" | tee -a "$LOG_FILE"

# 2. NUMA Topology
echo -e "\n--- NUMA Configuration ---" | tee -a "$LOG_FILE"
numactl --hardware 2>/dev/null | tee -a "$LOG_FILE" || echo "numactl not available" | tee -a "$LOG_FILE"

# 3. Memory Configuration
echo -e "\n--- Memory ---" | tee -a "$LOG_FILE"
free -h | tee -a "$LOG_FILE"

echo -e "\nHugepages:" | tee -a "$LOG_FILE"
grep -E "Huge|AnonHuge" /proc/meminfo | tee -a "$LOG_FILE"

echo -e "\nTHP Status:" | tee -a "$LOG_FILE"
cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | tee -a "$LOG_FILE" || echo "THP not available" | tee -a "$LOG_FILE"

# 4. CPU Governor & Frequency
echo -e "\n--- CPU Power Management ---" | tee -a "$LOG_FILE"
echo "Governor:" | tee -a "$LOG_FILE"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null | tee -a "$LOG_FILE" || echo "Unknown" | tee -a "$LOG_FILE"

echo -e "\nCurrent Frequencies (sample of 4 cores):" | tee -a "$LOG_FILE"
for cpu in 0 24 48 72; do
  freq=$(cat /sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq 2>/dev/null || echo "N/A")
  echo "  CPU $cpu: $freq kHz" | tee -a "$LOG_FILE"
done

# 5. Compiler & Library Status
echo -e "\n--- Compilers ---" | tee -a "$LOG_FILE"
echo "GCC:" | tee -a "$LOG_FILE"
gcc --version 2>/dev/null | head -n 1 | tee -a "$LOG_FILE" || echo "Not found" | tee -a "$LOG_FILE"

echo "CMake:" | tee -a "$LOG_FILE"
cmake --version 2>/dev/null | head -n 1 | tee -a "$LOG_FILE" || echo "Not found" | tee -a "$LOG_FILE"

echo "Clang (AOCC check):" | tee -a "$LOG_FILE"
which clang 2>/dev/null | tee -a "$LOG_FILE" || echo "Not in PATH" | tee -a "$LOG_FILE"
clang --version 2>/dev/null | head -n 1 | tee -a "$LOG_FILE" || true

echo -e "\nAOCC Installation:" | tee -a "$LOG_FILE"
ls -d /opt/AMD/aocc* 2>/dev/null | tee -a "$LOG_FILE" || echo "Not installed in /opt/AMD" | tee -a "$LOG_FILE"

echo -e "\nAOCL Status:" | tee -a "$LOG_FILE"
ls -d /mnt/raid0/llm/aocl-linux-aocc-5.0.0 2>/dev/null | tee -a "$LOG_FILE" || echo "Not found" | tee -a "$LOG_FILE"

# 6. OpenBLAS
echo -e "\n--- OpenBLAS ---" | tee -a "$LOG_FILE"
ldconfig -p 2>/dev/null | grep -i openblas | tee -a "$LOG_FILE" || echo "Not found in ldconfig" | tee -a "$LOG_FILE"
dpkg -l 2>/dev/null | grep -i openblas | tee -a "$LOG_FILE" || true

# 7. Python Environment
echo -e "\n--- Python Environment ---" | tee -a "$LOG_FILE"
which python3 | tee -a "$LOG_FILE"
python3 --version | tee -a "$LOG_FILE"

echo "Conda environments:" | tee -a "$LOG_FILE"
conda env list 2>/dev/null | tee -a "$LOG_FILE" || echo "Conda not available" | tee -a "$LOG_FILE"

# 8. Existing llama.cpp Build Check
echo -e "\n--- llama.cpp Build Status ---" | tee -a "$LOG_FILE"
LLAMA_BUILD_DIR="/mnt/raid0/llm/llama.cpp/build"
if [ -f "$LLAMA_BUILD_DIR/CMakeCache.txt" ]; then
  echo "CMakeCache.txt found. Key flags:" | tee -a "$LOG_FILE"
  grep -E "LLAMA_AVX512|LLAMA_BLAS|LLAMA_NATIVE|CMAKE_C_FLAGS|CMAKE_CXX_FLAGS" "$LLAMA_BUILD_DIR/CMakeCache.txt" 2>/dev/null | tee -a "$LOG_FILE"

  echo -e "\nBuild binaries:" | tee -a "$LOG_FILE"
  ls -la "$LLAMA_BUILD_DIR/bin/" 2>/dev/null | head -n 10 | tee -a "$LOG_FILE" || echo "No bin directory" | tee -a "$LOG_FILE"
else
  echo "No CMakeCache.txt found in $LLAMA_BUILD_DIR" | tee -a "$LOG_FILE"
fi

# 9. Model Inventory
echo -e "\n--- Model Inventory ---" | tee -a "$LOG_FILE"

echo "HuggingFace models (/mnt/raid0/llm/hf/):" | tee -a "$LOG_FILE"
ls -1 /mnt/raid0/llm/hf/ 2>/dev/null | tee -a "$LOG_FILE" || echo "Directory not found" | tee -a "$LOG_FILE"

echo -e "\nGGUF models (/mnt/raid0/llm/models/):" | tee -a "$LOG_FILE"
find /mnt/raid0/llm/models -name "*.gguf" -exec ls -lh {} \; 2>/dev/null | tee -a "$LOG_FILE" || echo "No GGUF models found" | tee -a "$LOG_FILE"

echo -e "\nLM Studio GGUF models:" | tee -a "$LOG_FILE"
find /mnt/raid0/llm/lmstudio -name "*.gguf" 2>/dev/null | head -n 10 | tee -a "$LOG_FILE" || echo "None found" | tee -a "$LOG_FILE"

# 10. Disk Space
echo -e "\n--- Storage ---" | tee -a "$LOG_FILE"
df -h /mnt/raid0 | tee -a "$LOG_FILE"

# 11. Kernel
echo -e "\n--- Kernel ---" | tee -a "$LOG_FILE"
uname -r | tee -a "$LOG_FILE"

# Summary
echo -e "\n=============================================="
echo "AUDIT COMPLETE"
echo "Log saved to: $LOG_FILE"
echo "=============================================="

# Quick recommendations based on findings
echo -e "\n--- Quick Recommendations ---"

# Check GCC version
GCC_VER=$(gcc -dumpversion 2>/dev/null || echo "0")
if [[ "${GCC_VER%%.*}" -lt 13 ]]; then
  echo "⚠️  GCC $GCC_VER detected. GCC 13+ recommended for Zen 5 optimization."
fi

# Check governor
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [[ "$GOV" != "performance" ]]; then
  echo "⚠️  CPU governor is '$GOV'. Set to 'performance' for benchmarking."
fi

# Check THP
THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | grep -o '\[.*\]' | tr -d '[]')
if [[ "$THP" != "always" ]]; then
  echo "⚠️  THP is '$THP'. Consider 'always' for inference workloads."
fi

# Check for GGUF models
GGUF_COUNT=$(find /mnt/raid0/llm/models -name "*.gguf" 2>/dev/null | wc -l)
if [[ "$GGUF_COUNT" -eq 0 ]]; then
  echo "⚠️  No GGUF models in /mnt/raid0/llm/models/. Run model conversion."
fi

echo -e "\nDone."

agent_task_end "Collect system state" "success"
agent_observe "audit_log_location" "$LOG_FILE"
agent_session_end "System audit complete - baseline captured"
