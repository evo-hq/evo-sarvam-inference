#!/usr/bin/env bash
# Generic GPU-locked command runner. The ONLY benchmark/gate infra the harness
# ships -- discover writes the actual benchmark + gate and invokes them through this.
#
# Usage (inside an evo benchmark or gate command):
#   bash {worktree}/evo_harness/gpu_locked.sh {worktree} -- <your command ...>
#
# What it guarantees (this is the reproducibility fix):
#   1. Leases ONE GPU, BLOCKING. With width=N subagents on a single GPU, only one
#      measurement runs at a time -> no cross-experiment contention -> scores
#      reproduce. (Parallel subagents still reason/edit concurrently; they queue
#      here only for the actual measurement.)
#   2. Pins CUDA_VISIBLE_DEVICES to the leased GPU.
#   3. Per-experiment Triton/vLLM JIT caches (no cross-experiment cache races).
#   4. Worktree's edited kernels win via PYTHONPATH; precompiled vLLM .so reused
#      from the base install (no rebuild).
#   5. Activates the venv, then exec's your command.
set -euo pipefail

WORKTREE="${1:?usage: gpu_locked.sh <worktree> -- <command>}"; shift
[ "${1:-}" = "--" ] && shift
[ "$#" -ge 1 ] || { echo "gpu_locked: no command given" >&2; exit 2; }

NUM_GPUS="${SARVAM_NUM_GPUS:-1}"
VLLM_BASE="${VLLM_BASE:-/home/ubuntu/vllm}"
VENV="${VENV:-/home/ubuntu/.venv}"
LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-1800}"

# --- 1. lease a GPU (block until one frees; fd 9 holds it for our lifetime) ---
GPU=""; deadline=$((SECONDS + LEASE_TIMEOUT))
while :; do
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    exec 9>"/tmp/sarvam_gpu_${i}.lock"
    if flock -n 9; then GPU="$i"; break; fi
  done
  [ -n "$GPU" ] && break
  [ $SECONDS -ge $deadline ] && { echo "gpu_locked: no free GPU after ${LEASE_TIMEOUT}s" >&2; exit 3; }
  sleep 5
done
export CUDA_VISIBLE_DEVICES="$GPU"

# --- 2. per-experiment JIT/compile caches ---
export TRITON_CACHE_DIR="$WORKTREE/.triton_cache" VLLM_CACHE_ROOT="$WORKTREE/.vllm_cache"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# --- 3. worktree kernels win, base .so reused (no rebuild) ---
if [ -d "$VLLM_BASE/vllm" ] && [ -d "$WORKTREE/vllm" ]; then
  while IFS= read -r so; do
    rel="${so#"$VLLM_BASE"/}"; dst="$WORKTREE/$rel"
    [ -e "$dst" ] || { mkdir -p "$(dirname "$dst")"; ln -s "$so" "$dst"; }
  done < <(find "$VLLM_BASE/vllm" -name '*.so')
  export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
fi

# --- 4. run under the venv ---
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
exec "$@"
