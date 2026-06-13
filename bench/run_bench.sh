#!/usr/bin/env bash
# evo benchmark entry point. evo invokes this (set at `evo init`):
#   bash {worktree}/evo_harness/bench/run_bench.sh {target} {worktree}
#
# Parallel-safe (width=N, one experiment per GPU):
#   1. Lease one GPU, BLOCKING (wait for a free card, don't fail on momentary
#      contention during bench<->gate handoffs).
#   2. Per-experiment Triton/vLLM JIT caches (concurrent compiles of the same
#      kernel must not race on a shared cache dir).
#   3. Worktree's edited Triton kernels win via PYTHONPATH; the precompiled vLLM
#      .so is reused from the base install (no rebuild).
set -euo pipefail

TARGET="${1:?usage: run_bench.sh <target> <worktree>}"
WORKTREE="${2:?usage: run_bench.sh <target> <worktree>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS="${SARVAM_NUM_GPUS:-4}"
VLLM_BASE="${VLLM_BASE:-/home/ubuntu/vllm}"
VENV="${VENV:-/home/ubuntu/.venv}"
LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-900}"

# --- 1. lease a GPU (block until one frees; fd 9 holds the lock for our lifetime) ---
GPU=""
deadline=$((SECONDS + LEASE_TIMEOUT))
while :; do
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    exec 9>"/tmp/sarvam_gpu_${i}.lock"
    if flock -n 9; then GPU="$i"; break; fi
  done
  [ -n "$GPU" ] && break
  [ $SECONDS -ge $deadline ] && { echo "run_bench: no free GPU after ${LEASE_TIMEOUT}s" >&2; exit 3; }
  sleep 5
done
export CUDA_VISIBLE_DEVICES="$GPU"

# --- 2. per-experiment JIT/compile caches (no cross-experiment races) ---
export TRITON_CACHE_DIR="$WORKTREE/.triton_cache"
export VLLM_CACHE_ROOT="$WORKTREE/.vllm_cache"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# --- 3. worktree kernels win, base .so reused ---
if [ -d "$VLLM_BASE/vllm" ] && [ -d "$WORKTREE/vllm" ]; then
  while IFS= read -r so; do
    rel="${so#"$VLLM_BASE"/}"; dst="$WORKTREE/$rel"
    [ -e "$dst" ] || { mkdir -p "$(dirname "$dst")"; ln -s "$so" "$dst"; }
  done < <(find "$VLLM_BASE/vllm" -name '*.so')
  export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
fi

# --- 4. run with the venv python (has vllm) ---
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
exec python "$SCRIPT_DIR/bench_decode.py" --target "$TARGET" --worktree "$WORKTREE"
