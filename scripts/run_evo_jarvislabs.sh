#!/usr/bin/env bash
# Bootstrap + drive the Sarvam-30B inference-speed evo experiment on a JarvisLabs
# box (bare GPU VM: CUDA driver + system python, no torch/venv preinstalled).
#
# Everything lives under /home so it survives pause/resume. The evo workspace
# (.evo/) is created INSIDE the vLLM clone by /evo:discover; we do NOT hand-author
# evo init.
#
# Usage (on the box, after: cp .env.example .env && edit .env):
#   bash scripts/run_evo_jarvislabs.sh bootstrap   # one-time: venv + vLLM + weights + harness infra
#   bash scripts/run_evo_jarvislabs.sh evo-setup    # install Claude Code + evo CLI + plugin + auth check
#   bash scripts/run_evo_jarvislabs.sh clocks        # lock GPU clocks (cuts benchmark noise)
#   bash scripts/run_evo_jarvislabs.sh run           # launch headless agent: /evo:discover (builds its own benchmark) -> prose /evo:optimize
#   bash scripts/run_evo_jarvislabs.sh notify        # always-on Telegram/WhatsApp alert on each new best (tmux, reads .env)
#   bash scripts/run_evo_jarvislabs.sh dashboard     # publish a PUBLIC dashboard URL via cloudflared (anyone can watch)
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
: "${SARVAM_NUM_GPUS:=1}"   # 1x H100; the gpu_locked.sh lease serializes parallel subagents' measurements
export HF_HOME="${HF_HOME:-$WORK/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"     # uv tool installs the evo CLI here; not on PATH in non-login shells
pyvenv(){ . "$VENV/bin/activate"; }

bootstrap() {
  echo "== 1. system deps (bare VM has no python3-venv) =="
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv python3-pip git rsync util-linux tmux curl

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

  echo "== 5. drop harness INFRA into the vLLM clone + commit (worktrees inherit it) =="
  # Only the GPU-lock wrapper + the contract. discover writes the benchmark + gate itself.
  mkdir -p "$VLLM_BASE/evo_harness"
  rsync -a "$HARNESS_DIR/scripts/gpu_locked.sh" "$HARNESS_DIR/references" "$VLLM_BASE/evo_harness/"
  chmod +x "$VLLM_BASE/evo_harness/gpu_locked.sh"
  ( cd "$VLLM_BASE" && git add -A && git commit -qm "add evo harness infra (gpu_locked + contract)" || true )
  echo "bootstrap done. next: evo-setup, then clocks, then run"
}

evo-setup() {
  # Install the agent (Claude Code), the evo CLI, and the evo Claude Code plugin
  # (carries the workflow/meta optimize driver), then smoke-test auth.
  pyvenv
  echo "== Claude Code (the agent) =="
  command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash
  echo "claude: $(claude --version 2>&1 | head -1)"

  echo "== evo CLI (uv tool -> ~/.local/bin) =="
  [ -d "$WORK/evo" ] || git clone -q --branch "$EVO_REF" "$EVO_REPO" "$WORK/evo"
  ( cd "$WORK/evo" && git fetch -q origin && git reset -q --hard "origin/$EVO_REF" )
  pip install -q -U uv
  ( cd "$WORK/evo" && uv tool install --force --editable ./plugins/evo )
  echo "evo: $(evo --version)"

  echo "== evo Claude Code plugin (workflow driver) =="
  evo install claude-code --from-path "$WORK/evo" || true
  evo update claude-code --from-path "$WORK/evo" --force || true

  echo "== auth smoke (validates CLAUDE_CODE_OAUTH_TOKEN) =="
  ANTHROPIC_API_KEY="" IS_SANDBOX=1 timeout 150 claude --print \
    --model "${CLAUDE_MODEL:-claude-opus-4-8}" --dangerously-skip-permissions \
    "Reply with exactly: READY" \
    || echo "AUTH CHECK FAILED -- fix CLAUDE_CODE_OAUTH_TOKEN before run"
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
  : "${CLAUDE_CODE_OAUTH_TOKEN:?set CLAUDE_CODE_OAUTH_TOKEN in .env}"
  : "${CLAUDE_MODEL:=claude-opus-4-8}"
  : "${CLAUDE_CODE_EFFORT_LEVEL:=max}"
  echo "== launch headless evo agent (model=$CLAUDE_MODEL effort=$CLAUDE_CODE_EFFORT_LEVEL; discover -> optimize) =="
  mkdir -p "$HARNESS_DIR/runs"
  # tmux does NOT reliably inherit the launching shell's exported env, so the
  # agent's environment (PATH for claude/evo, OAuth, venv, IS_SANDBOX) is set
  # inside a self-contained inner runner. Path vars expand now; runtime vars
  # (\$HOME, \$CLAUDE_MODEL, \$?) resolve inside the runner after sourcing .env.
  cat > "$WORK/agent_run.sh" <<INNER
#!/usr/bin/env bash
cd "$HARNESS_DIR"
export PATH="\$HOME/.local/bin:\$PATH"
set -a; . ./.env; set +a
export ANTHROPIC_API_KEY=""
export IS_SANDBOX=1
: "\${CLAUDE_MODEL:=claude-opus-4-8}"
: "\${CLAUDE_CODE_EFFORT_LEVEL:=max}"
source "$VENV/bin/activate"
cd "$VLLM_BASE"
claude --print --output-format stream-json --verbose --model "\$CLAUDE_MODEL" --effort "\$CLAUDE_CODE_EFFORT_LEVEL" --dangerously-skip-permissions < "$HARNESS_DIR/evo/run_prompt.md"
echo "RUN_EXIT=\$?"
INNER
  chmod +x "$WORK/agent_run.sh"
  tmux kill-session -t sarvam 2>/dev/null || true
  tmux new -d -s sarvam "bash $WORK/agent_run.sh > $HARNESS_DIR/runs/run_console.log 2>&1"
  echo "launched in tmux 'sarvam'. tail: tmux attach -t sarvam | console: runs/run_console.log"
}

dashboard() {
  # The evo dashboard binds 127.0.0.1, and JarvisLabs VMs firewall inbound ports,
  # so http://<public-ip>:port never works. Publish a public URL via an OUTBOUND
  # cloudflared quick tunnel -- anyone with the link can watch, no key needed.
  # (Key holders can instead: ssh -i <key> -N -L 8080:127.0.0.1:8080 ubuntu@<ip>)
  mkdir -p "$HOME/.local/bin"
  cloudflared --version >/dev/null 2>&1 || {
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      -o "$HOME/.local/bin/cloudflared" && chmod +x "$HOME/.local/bin/cloudflared"; }
  tmux kill-session -t cf 2>/dev/null || true
  tmux new -d -s cf "$HOME/.local/bin/cloudflared tunnel --url http://localhost:8080 --no-autoupdate > $WORK/cf.log 2>&1"
  printf "public dashboard URL: "
  for _ in $(seq 1 20); do
    u=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$WORK/cf.log" 2>/dev/null | head -1)
    [ -n "$u" ] && { echo "$u"; return; }
    sleep 3
  done
  echo "(not ready; check $WORK/cf.log)"
}

notify() {
  # Always-on notifier ON THE BOX (survives your laptop sleeping). Reads
  # TELEGRAM_*/WHATSAPP_* from .env, polls evo locally, messages on each new best.
  local INT="${1:-20}"
  cat > "$WORK/notify_loop.sh" <<'INNER'
#!/usr/bin/env bash
cd /home/ubuntu/evo-sarvam-inference
set -a; [ -f .env ] && . ./.env; set +a
export PATH="$HOME/.local/bin:$PATH"
INT="${NOTIFY_INT:-20}"
send() {
  [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] && \
    curl -s -m 10 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
      --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" >/dev/null
  [ -n "${WHATSAPP_PHONE:-}" ] && [ -n "${CALLMEBOT_APIKEY:-}" ] && \
    curl -s -m 12 -G "https://api.callmebot.com/whatsapp.php" \
      --data-urlencode "phone=${WHATSAPP_PHONE}" --data-urlencode "text=$1" \
      --data-urlencode "apikey=${CALLMEBOT_APIKEY}" >/dev/null
}
send "evo: watching Sarvam-30B (box, poll ${INT}s)"
last=""
while true; do
  out=$(cd /home/ubuntu/vllm && evo status 2>/dev/null)
  best=$(printf '%s' "$out" | grep -oE 'best=[0-9.]+' | cut -d= -f2)
  exps=$(printf '%s' "$out" | grep -oE 'committed=[0-9]+' | cut -d= -f2)
  if [ -n "$best" ] && [ "$best" != "None" ]; then
    if [ -z "$last" ]; then last="$best"
    elif awk "BEGIN{exit !($best>$last)}"; then
      d=$(awk "BEGIN{printf \"%.2f\", ($best-$last)/$last*100}")
      send "evo: Sarvam-30B improved -> ${best} tok/s (+${d}%, committed=$exps)"
      last="$best"
    fi
  fi
  sleep "$INT"
done
INNER
  chmod +x "$WORK/notify_loop.sh"
  tmux kill-session -t notify 2>/dev/null || true
  tmux new -d -s notify "NOTIFY_INT=$INT bash $WORK/notify_loop.sh > $WORK/notify.log 2>&1"
  echo "box-side notifier in tmux 'notify' (set TELEGRAM_*/WHATSAPP_* in .env). log: $WORK/notify.log"
}

cmd="${1:-bootstrap}"; shift || true
case "$cmd" in
  bootstrap) bootstrap "$@" ;;
  evo-setup) evo-setup "$@" ;;
  clocks)    clocks "$@" ;;
  run)       run "$@" ;;
  notify)    notify "$@" ;;
  dashboard) dashboard "$@" ;;
  *) echo "usage: $0 {bootstrap|evo-setup|clocks|run|notify|dashboard}" >&2; exit 1 ;;
esac
