#!/usr/bin/env bash
# Bootstrap the Sarvam-30B inference-speed evo experiment on a JarvisLabs box.
#
# Run on a 4x H100 instance (VM or container). Everything lives under /home so it
# survives pause/resume; /root and system pip are wiped on pause.
#
# Usage (on the box):
#   cd /home/ubuntu/evo-sarvam-inference && cp .env.example .env && edit .env
#   bash scripts/run_evo_jarvislabs.sh bootstrap   # one-time: clone+build+weights+harness
#   bash scripts/run_evo_jarvislabs.sh reference    # capture baseline_gen.json on the unmodified build
#   bash scripts/run_evo_jarvislabs.sh smoke        # DRY RUN: bench baseline x3 (noise) + gate sanity, no agent
#   bash scripts/run_evo_jarvislabs.sh run          # launch the headless agent: /evo:discover then /evo:optimize
#   bash scripts/run_evo_jarvislabs.sh dashboard    # expose evo dashboard on 0.0.0.0:8090
#
# This is the replication harness. The actual evo workspace (.evo/) is created
# INSIDE the vLLM clone, because that clone is the codebase evo optimizes.
#
# STATUS: skeleton. Items marked VALIDATE were not run on a box yet; verify each
# before trusting the end-to-end flow (see docs/plan.md "Open validation items").
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HARNESS_DIR"
set -a; [ -f .env ] && . ./.env; set +a

WORK="${WORK:-/home/ubuntu}"
: "${VLLM_REPO:=https://github.com/vllm-project/vllm.git}"
: "${VLLM_PR:=33942}"
: "${VLLM_BASE:=$WORK/vllm}"
: "${EVO_REPO:=https://github.com/evo-hq/evo.git}"
: "${EVO_REF:=main}"
: "${SARVAM_MODEL_PATH:=sarvamai/sarvam-30b}"
: "${SARVAM_NUM_GPUS:=4}"
export HF_HOME="${HF_HOME:-$WORK/hf}"

bootstrap() {
  echo "== 1. system deps =="
  sudo apt-get update -y && sudo apt-get install -y git rsync flock || true

  echo "== 2. evo from source + Claude Code plugin =="
  [ -d "$WORK/evo" ] || git clone --branch "$EVO_REF" "$EVO_REPO" "$WORK/evo"
  ( cd "$WORK/evo" && git fetch origin && git reset --hard "origin/$EVO_REF" )
  pip install -U uv
  ( cd "$WORK/evo" && uv tool install --editable ./plugins/evo )
  evo install claude-code --from-path "$WORK/evo" || true
  evo update claude-code --from-path "$WORK/evo" --force    # --force wipes+repopulates cache; a stale cache silently drops the workflow/meta driver

  echo "== 3. vLLM + Sarvam PR #$VLLM_PR (precompiled: no CUDA build) =="
  [ -d "$VLLM_BASE" ] || git clone "$VLLM_REPO" "$VLLM_BASE"
  ( cd "$VLLM_BASE"
    git fetch origin "pull/${VLLM_PR}/head:sarvam-moe"
    git checkout sarvam-moe
    # VLLM_USE_PRECOMPILED downloads the matching prebuilt C-extension instead of
    # compiling CUDA, so editing Triton (.py) kernels needs no rebuild.
    VLLM_USE_PRECOMPILED=1 pip install -e . )   # VALIDATE: precompiled wheel matches the PR's base commit

  echo "== 4. weights =="
  pip install -U "huggingface_hub[cli]"
  hf download "$SARVAM_MODEL_PATH" --quiet   # VALIDATE: fp8 variant vs bf16; gated repo needs HF_TOKEN

  echo "== 5. drop the evo harness into the vLLM clone =="
  rsync -a --delete "$HARNESS_DIR/bench" "$HARNESS_DIR/gate" "$HARNESS_DIR/reference" \
        "$VLLM_BASE/evo_harness/"
  ( cd "$VLLM_BASE" && git add -A && git -c user.email=bot@evo -c user.name=evo commit -qm "add evo harness" || true )
  echo "bootstrap done. next: bash scripts/run_evo_jarvislabs.sh reference"
}

reference() {
  # Capture the baseline's greedy generations on the UNMODIFIED build, then commit
  # them into the vLLM repo so every experiment worktree inherits the gate anchor.
  echo "== capture baseline reference (UNMODIFIED build, GPU 0) =="
  CUDA_VISIBLE_DEVICES=0 python3 "$VLLM_BASE/evo_harness/gate/capture_reference.py"
  ( cd "$VLLM_BASE" && git add -f evo_harness/reference/baseline_gen.json \
      && git -c user.email=bot@evo -c user.name=evo commit -qm "capture baseline reference" )
  echo "committed baseline_gen.json. next: bash scripts/run_evo_jarvislabs.sh run"
}

run() {
  # Hand off to the evo skills exactly like the LawBench harness: the headless
  # agent runs /evo:discover (evo init + baseline + project.md) then /evo:optimize
  # under the WORKFLOW driver (meta/analyst). We do NOT hand-author evo init/new/run.
  : "${CLAUDE_CODE_OAUTH_TOKEN:?set CLAUDE_CODE_OAUTH_TOKEN in .env}"
  : "${CLAUDE_MODEL:=claude-opus-4-8}"
  : "${CLAUDE_CODE_EFFORT_LEVEL:=max}"
  echo "== launch headless evo agent (model=$CLAUDE_MODEL effort=$CLAUDE_CODE_EFFORT_LEVEL; discover -> optimize) =="
  export ANTHROPIC_API_KEY="" CLAUDE_CODE_OAUTH_TOKEN CLAUDE_CODE_EFFORT_LEVEL
  export IS_SANDBOX=1                       # headless --dangerously-skip-permissions no-ops without this in containers
  export SARVAM_NUM_GPUS VLLM_BASE SARVAM_MODEL_PATH SARVAM_QUANT SARVAM_MAX_MODEL_LEN HF_HOME
  mkdir -p "$HARNESS_DIR/runs"
  tmux new -d -s sarvam \
    "cd '$VLLM_BASE' && claude --print --model '$CLAUDE_MODEL' --effort '$CLAUDE_CODE_EFFORT_LEVEL' --dangerously-skip-permissions < '$HARNESS_DIR/evo/run_prompt.md' 2>&1 | tee '$HARNESS_DIR/runs/run_console.log'"
  echo "launched in tmux 'sarvam'. tail: tmux attach -t sarvam | console: runs/run_console.log"
}

dashboard() {
  # evo's dashboard binds 127.0.0.1; bridge to 0.0.0.0 so JarvisLabs can expose it.
  command -v socat >/dev/null || sudo apt-get install -y socat
  socat TCP-LISTEN:8090,fork,reuseaddr TCP:127.0.0.1:8080 &
  echo "dashboard bridged to 0.0.0.0:8090 (expose this port on the instance)"
}

cmd="${1:-bootstrap}"; shift || true
case "$cmd" in
  bootstrap) bootstrap "$@" ;;
  reference) reference "$@" ;;
  run)       run "$@" ;;
  dashboard) dashboard "$@" ;;
  *) echo "usage: $0 {bootstrap|reference|run|dashboard}" >&2; exit 1 ;;
esac
