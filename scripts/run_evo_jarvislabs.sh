#!/usr/bin/env bash
# Bootstrap + drive the Sarvam-30B inference-speed evo experiment on a JarvisLabs
# box (bare GPU VM: CUDA driver + system python, no torch/venv preinstalled).
#
# Everything lives under /home so it survives pause/resume. The evo workspace
# (.evo/) is created INSIDE the vLLM clone by /evo:discover; we do NOT hand-author
# evo init.
#
# Usage (on the box, after: cp .env.example .env && edit .env):
#   bash scripts/run_evo_jarvislabs.sh bootstrap   # one-time: venv + vLLM + weights + harness
#   bash scripts/run_evo_jarvislabs.sh evo-setup    # install evo CLI + Claude Code plugin (workflow driver)
#   bash scripts/run_evo_jarvislabs.sh reference     # capture baseline_gen.json (gate anchor) on the unmodified build
#   bash scripts/run_evo_jarvislabs.sh smoke         # DRY RUN: bench x3 (noise) + gate sanity, no agent
#   bash scripts/run_evo_jarvislabs.sh clocks        # lock GPU clocks on all GPUs (cuts benchmark noise)
#   bash scripts/run_evo_jarvislabs.sh run           # launch headless agent: /evo:discover then /evo:optimize
#   bash scripts/run_evo_jarvislabs.sh dashboard     # bridge dashboard 127.0.0.1:8080 -> 0.0.0.0:8090 (containers)
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HARNESS_DIR"
set -a; [ -f .env ] && . ./.env; set +a

WORK="${WORK:-/home/ubuntu}"
: "${VLLM_REPO:=https://github.com/vllm-project/vllm.git}"
: "${VLLM_PR:=33942}"
: "${VLLM_BASE:=$WORK/vllm}"
: "${VENV:=$WORK/.venv}"
: "${EVO_REPO:=https://github.com/evo-hq/evo.git}"
: "${EVO_REF:=main}"
: "${SARVAM_MODEL_PATH:=sarvamai/sarvam-30b}"
: "${SARVAM_NUM_GPUS:=4}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
pyvenv(){ . "$VENV/bin/activate"; }

bootstrap() {
  echo "== 1. system deps (bare VM has no python3-venv) =="
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv python3-pip git rsync util-linux tmux

  echo "== 2. venv + base pip =="
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  pyvenv
  pip install -q -U pip wheel setuptools hf_transfer huggingface_hub

  echo "== 3. vLLM + Sarvam PR #$VLLM_PR (VLLM_USE_PRECOMPILED: no CUDA build) =="
  [ -d "$VLLM_BASE" ] || git clone -q "$VLLM_REPO" "$VLLM_BASE"
  ( cd "$VLLM_BASE"
    git fetch -q origin "pull/${VLLM_PR}/head:sarvam-moe" || true
    git checkout -q sarvam-moe
    VLLM_USE_PRECOMPILED=1 pip install -e . )
  python -c "import vllm; print('vllm', vllm.__version__)"

  echo "== 4. weights =="
  hf download "$SARVAM_MODEL_PATH"

  echo "== 5. drop harness into the vLLM clone + commit (worktrees inherit it) =="
  rsync -a "$HARNESS_DIR/bench" "$HARNESS_DIR/gate" "$HARNESS_DIR/reference" "$VLLM_BASE/evo_harness/"
  ( cd "$VLLM_BASE" && git add -A && git commit -qm "add evo harness" || true )
  echo "bootstrap done. next: evo-setup, then reference, then run"
}

evo-setup() {
  # Install the evo CLI + the Claude Code plugin (carries the workflow/meta optimize
  # driver). --force repopulates the plugin cache; a stale cache silently drops it.
  pyvenv
  [ -d "$WORK/evo" ] || git clone -q --branch "$EVO_REF" "$EVO_REPO" "$WORK/evo"
  ( cd "$WORK/evo" && git fetch -q origin && git reset -q --hard "origin/$EVO_REF" )
  pip install -q -U uv
  ( cd "$WORK/evo" && uv tool install --force --editable ./plugins/evo )
  evo install claude-code --from-path "$WORK/evo" || true
  evo update claude-code --from-path "$WORK/evo" --force
  echo "evo + plugin installed. verify: evo --version"
}

reference() {
  pyvenv
  echo "== capture baseline reference (UNMODIFIED build, GPU 0) =="
  CUDA_VISIBLE_DEVICES=0 python "$VLLM_BASE/evo_harness/gate/capture_reference.py"
  ( cd "$VLLM_BASE" && git add -f evo_harness/reference/baseline_gen.json \
      && git commit -qm "capture baseline reference" )
  echo "committed baseline_gen.json. next: clocks, then run"
}

smoke() {
  pyvenv
  cd "$VLLM_BASE"
  for i in 1 2 3; do
    echo "== bench run $i =="
    EVO_RESULT_PATH="/tmp/smoke_$i.json" CUDA_VISIBLE_DEVICES=0 \
      python evo_harness/bench/bench_decode.py --target evo_harness/smoke --worktree "$VLLM_BASE"
  done
  echo "== gate (baseline vs baseline, expect PASS) =="
  CUDA_VISIBLE_DEVICES=0 python evo_harness/gate/verify_quality.py --target evo_harness/smoke --worktree "$VLLM_BASE" \
    && echo "GATE PASS" || echo "GATE FAIL (unexpected on baseline)"
}

clocks() {
  # Lock GPU clocks to remove thermal drift between experiments (tightens the
  # benchmark noise floor). Run once per session; persists until reset/-rgc.
  sudo nvidia-smi -pm 1 || true
  local maxclk
  maxclk=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
  sudo nvidia-smi -lgc "${maxclk},${maxclk}" || true
  nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm --format=csv
}

run() {
  pyvenv
  : "${CLAUDE_CODE_OAUTH_TOKEN:?set CLAUDE_CODE_OAUTH_TOKEN in .env}"
  : "${CLAUDE_MODEL:=claude-opus-4-8}"
  : "${CLAUDE_CODE_EFFORT_LEVEL:=max}"
  echo "== launch headless evo agent (model=$CLAUDE_MODEL effort=$CLAUDE_CODE_EFFORT_LEVEL; discover -> optimize) =="
  export ANTHROPIC_API_KEY="" CLAUDE_CODE_OAUTH_TOKEN CLAUDE_CODE_EFFORT_LEVEL
  export IS_SANDBOX=1                       # headless --dangerously-skip-permissions no-ops without this in containers
  export SARVAM_NUM_GPUS VLLM_BASE VENV SARVAM_MODEL_PATH SARVAM_QUANT SARVAM_MAX_MODEL_LEN HF_HOME
  mkdir -p "$HARNESS_DIR/runs"
  tmux new -d -s sarvam \
    "cd '$VLLM_BASE' && claude --print --model '$CLAUDE_MODEL' --effort '$CLAUDE_CODE_EFFORT_LEVEL' --dangerously-skip-permissions < '$HARNESS_DIR/evo/run_prompt.md' 2>&1 | tee '$HARNESS_DIR/runs/run_console.log'"
  echo "launched in tmux 'sarvam'. tail: tmux attach -t sarvam | console: runs/run_console.log"
}

dashboard() {
  # Container path: bridge the dashboard to a JL-exposed port. On a VM (SSH-only),
  # use an SSH local forward from your machine instead: ssh -L 8080:127.0.0.1:8080 ...
  command -v socat >/dev/null || sudo apt-get install -y -qq socat
  pkill -f 'TCP-LISTEN:8090' 2>/dev/null || true
  socat TCP-LISTEN:8090,fork,reuseaddr TCP:127.0.0.1:8080 &
  echo "bridged dashboard to 0.0.0.0:8090"
}

cmd="${1:-bootstrap}"; shift || true
case "$cmd" in
  bootstrap) bootstrap "$@" ;;
  evo-setup) evo-setup "$@" ;;
  reference) reference "$@" ;;
  smoke)     smoke "$@" ;;
  clocks)    clocks "$@" ;;
  run)       run "$@" ;;
  dashboard) dashboard "$@" ;;
  *) echo "usage: $0 {bootstrap|evo-setup|reference|smoke|clocks|run|dashboard}" >&2; exit 1 ;;
esac
