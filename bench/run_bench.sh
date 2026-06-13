#!/usr/bin/env bash
# evo benchmark entry point.
#
# evo invokes this as the benchmark command, e.g. (set at `evo init`):
#   bash {worktree}/evo_harness/bench/run_bench.sh {target} {worktree}
#
# Responsibilities:
#   1. Lease one free GPU (width=N runs -> one experiment per GPU, no sharing).
#   2. Make the worktree's edited Triton kernels the ones Python imports, while
#      reusing the precompiled vLLM C-extension from the base install (no rebuild).
#   3. Run the decode-throughput benchmark, which writes the score to
#      $EVO_RESULT_PATH.
set -euo pipefail

TARGET="${1:?usage: run_bench.sh <target> <worktree>}"
WORKTREE="${2:?usage: run_bench.sh <target> <worktree>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS="${SARVAM_NUM_GPUS:-4}"
VLLM_BASE="${VLLM_BASE:-/home/ubuntu/vllm}"

# --- 1. GPU lease (held for the process lifetime via open fd 9) -------------
GPU=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
  exec 9>"/tmp/sarvam_gpu_${i}.lock"
  if flock -n 9; then GPU="$i"; break; fi
done
[ -n "$GPU" ] || { echo "run_bench: no free GPU (all ${NUM_GPUS} leased)" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$GPU"

# --- 2. Worktree kernels win, base .so reused ------------------------------
# The base install was built once with VLLM_USE_PRECOMPILED=1, so vllm/*.so
# exist only under $VLLM_BASE. Symlink them into the worktree's package dir so
# `import vllm` from the worktree resolves the compiled extension, while the
# edited Triton (.py) kernels in the worktree take effect with no recompile.
# VALIDATE ON BOX: confirm the .so set + that editable metadata doesn't shadow.
if [ -d "$VLLM_BASE/vllm" ] && [ -d "$WORKTREE/vllm" ]; then
  while IFS= read -r so; do
    rel="${so#"$VLLM_BASE"/}"
    dst="$WORKTREE/$rel"
    [ -e "$dst" ] || { mkdir -p "$(dirname "$dst")"; ln -s "$so" "$dst"; }
  done < <(find "$VLLM_BASE/vllm" -name '*.so')
  export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
fi

# --- 3. Run ----------------------------------------------------------------
exec python3 "$SCRIPT_DIR/bench_decode.py" --target "$TARGET" --worktree "$WORKTREE"
