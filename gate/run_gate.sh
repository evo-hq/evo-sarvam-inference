#!/usr/bin/env bash
# evo gate entry point. Set at `evo init` as:
#   bash {worktree}/evo_harness/gate/run_gate.sh {target} {worktree}
#
# Same GPU-lease + worktree-kernel-resolution as run_bench.sh, then runs the
# accuracy check. Exit 0 = pass (keep), non-zero = fail (discard).
set -euo pipefail

TARGET="${1:?usage: run_gate.sh <target> <worktree>}"
WORKTREE="${2:?usage: run_gate.sh <target> <worktree>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS="${SARVAM_NUM_GPUS:-4}"
VLLM_BASE="${VLLM_BASE:-/home/ubuntu/vllm}"

GPU=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
  exec 9>"/tmp/sarvam_gpu_${i}.lock"
  if flock -n 9; then GPU="$i"; break; fi
done
[ -n "$GPU" ] || { echo "run_gate: no free GPU" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$GPU"

if [ -d "$VLLM_BASE/vllm" ] && [ -d "$WORKTREE/vllm" ]; then
  while IFS= read -r so; do
    rel="${so#"$VLLM_BASE"/}"; dst="$WORKTREE/$rel"
    [ -e "$dst" ] || { mkdir -p "$(dirname "$dst")"; ln -s "$so" "$dst"; }
  done < <(find "$VLLM_BASE/vllm" -name '*.so')
  export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
fi

exec python3 "$SCRIPT_DIR/verify_quality.py" --target "$TARGET" --worktree "$WORKTREE"
