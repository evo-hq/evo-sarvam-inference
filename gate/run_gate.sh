#!/usr/bin/env bash
# evo gate entry point. Set at `evo init`:
#   bash {worktree}/evo_harness/gate/run_gate.sh {target} {worktree}
#
# Same parallel-safe prelude as run_bench.sh (blocking GPU lease, per-experiment
# JIT caches, worktree kernel resolution), then the accuracy check.
# Exit 0 = pass (keep), non-zero = fail (discard).
set -euo pipefail

TARGET="${1:?usage: run_gate.sh <target> <worktree>}"
WORKTREE="${2:?usage: run_gate.sh <target> <worktree>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS="${SARVAM_NUM_GPUS:-4}"
VLLM_BASE="${VLLM_BASE:-/home/ubuntu/vllm}"
VENV="${VENV:-/home/ubuntu/.venv}"
LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-900}"

GPU=""
deadline=$((SECONDS + LEASE_TIMEOUT))
while :; do
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    exec 9>"/tmp/sarvam_gpu_${i}.lock"
    if flock -n 9; then GPU="$i"; break; fi
  done
  [ -n "$GPU" ] && break
  [ $SECONDS -ge $deadline ] && { echo "run_gate: no free GPU after ${LEASE_TIMEOUT}s" >&2; exit 3; }
  sleep 5
done
export CUDA_VISIBLE_DEVICES="$GPU"

export TRITON_CACHE_DIR="$WORKTREE/.triton_cache"
export VLLM_CACHE_ROOT="$WORKTREE/.vllm_cache"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

if [ -d "$VLLM_BASE/vllm" ] && [ -d "$WORKTREE/vllm" ]; then
  while IFS= read -r so; do
    rel="${so#"$VLLM_BASE"/}"; dst="$WORKTREE/$rel"
    [ -e "$dst" ] || { mkdir -p "$(dirname "$dst")"; ln -s "$so" "$dst"; }
  done < <(find "$VLLM_BASE/vllm" -name '*.so')
  export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
fi

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
exec python "$SCRIPT_DIR/verify_quality.py" --target "$TARGET" --worktree "$WORKTREE"
