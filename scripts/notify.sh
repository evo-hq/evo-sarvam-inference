#!/usr/bin/env bash
# Notify (macOS) whenever evo's best score improves on the Sarvam-30B run.
#
#   scripts/notify.sh <box-ip> [poll-seconds]   (default 20s)
#
# Polls `evo status` on the box; on each new best it pops a macOS notification
# (Notification Center + sound) and prints the delta. Run it in its own tab.
# Needs ~/.ssh/jl_ed25519 (override JL_SSH_KEY).
set -uo pipefail
IP="${1:?usage: notify.sh <box-ip> [poll-seconds]}"
INT="${2:-20}"
KEY="${JL_SSH_KEY:-$HOME/.ssh/jl_ed25519}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 ubuntu@"$IP")

notify() {  # title, message
  osascript -e "display notification \"$2\" with title \"$1\" sound name \"Glass\"" 2>/dev/null
  printf '\a'  # terminal bell
}

last=""
echo "watching best score every ${INT}s  ($IP)  -- Ctrl-C to stop"
while true; do
  out=$("${SSH[@]}" 'export PATH=$HOME/.local/bin:$PATH; cd /home/ubuntu/vllm && evo status 2>/dev/null' 2>/dev/null)
  best=$(printf '%s' "$out" | grep -oE 'best=[0-9.]+' | cut -d= -f2)
  exps=$(printf '%s' "$out" | grep -oE 'committed=[0-9]+' | cut -d= -f2)
  ts=$(date +%T)
  if [ -z "$best" ]; then
    echo "[$ts] no scored experiment yet (committed=${exps:-0})"
  elif [ -z "$last" ]; then
    last="$best"; echo "[$ts] baseline best=$best (committed=$exps)"
  elif awk "BEGIN{exit !($best>$last)}"; then
    delta=$(awk "BEGIN{printf \"%.2f\", ($best-$last)/$last*100}")
    msg="new best ${best} tok/s  (+${delta}%, committed=$exps)"
    echo "[$ts] >>> IMPROVEMENT: $msg"
    notify "evo: Sarvam-30B improved" "$msg"
    last="$best"
  else
    echo "[$ts] best=$best (committed=$exps, no change)"
  fi
  sleep "$INT"
done
