#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Run two DAPO training scripts in two separate tmux sessions.
#   - train_exp_1.sh: Qwen3_8B_16K
#   - train_exp_2.sh: Qwen3_8B_8K
# ----------------------------------------------------------------------

# Project root (adjust if needed)
#cd /code/hongpaul-sandbox/temp/CudaForge_plus/

# Base name for sessions
SESSION="Qwen3_8B"

# Scripts (must exist in this directory)
SCRIPT_16K="./train_exp_1.sh"

# Session names
S16K="${SESSION}_16K_DAPO"
S8K="${SESSION}_8K_DAPO"

# Log directory
LOGDIR="./tmux_logs"
mkdir -p "${LOGDIR}"

# Preflight checks
command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux not found in PATH"; exit 1; }

[[ -f "${SCRIPT_16K}" ]] || { echo "ERROR: ${SCRIPT_16K} not found"; exit 1; }

# Ensure executable (optional but convenient)
chmod +x "${SCRIPT_16K}"

# Helper: (re)create a session and run a command
run_in_tmux() {
  local sess="$1"
  local cmd="$2"

  if tmux has-session -t "${sess}" 2>/dev/null; then
    echo "INFO: tmux session '${sess}' already exists; killing it..."
    tmux kill-session -t "${sess}"
  fi

  echo "INFO: starting tmux session '${sess}'"
  # Use bash -lc so that login env / conda env initialization can work if needed
  tmux new-session -d -s "${sess}" "bash -lc '${cmd}'"
}

# Commands (tee logs)
CMD_16K="cd $(pwd) && ${SCRIPT_16K} 2>&1 | tee -a ${LOGDIR}/${S16K}.log"


run_in_tmux "${S16K}" "${CMD_16K}"

echo "INFO: launched:"
echo "  - ${S16K}  (logs: ${LOGDIR}/${S16K}.log)"
echo
echo "Attach with:"
echo "  tmux attach -t ${S16K}"
echo
echo "List sessions:"
echo "  tmux ls"
