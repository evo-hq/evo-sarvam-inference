#!/usr/bin/env bash
# Notify whenever evo's best score improves on the Sarvam-30B run.
#
#   scripts/notify.sh <box-ip> [poll-seconds]      (default 20s)
#
# Channels (set env vars for the ones you want; macOS notification is always on):
#   Telegram:  export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...
#   WhatsApp:  export WHATSAPP_PHONE=+9199...  CALLMEBOT_APIKEY=...   (via callmebot.com)
#
# Telegram setup:  message @BotFather -> /newbot -> token; then message your bot once and
#   open https://api.telegram.org/bot<TOKEN>/getUpdates to read chat.id (or ask @userinfobot).
# WhatsApp setup:  add +34 644 51 95 23, WhatsApp it "I allow callmebot to send me messages",
#   it replies with your apikey. (Third-party relay; Telegram is the cleaner option.)
#
# It also sends one "watching started" message so you can confirm the channel works.
set -uo pipefail
IP="${1:?usage: notify.sh <box-ip> [poll-seconds]}"
INT="${2:-20}"
KEY="${JL_SSH_KEY:-$HOME/.ssh/jl_ed25519}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 ubuntu@"$IP")

# Load channel tokens from a local .env if present (env vars still override).
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; [ -f "$HARNESS_DIR/.env" ] && . "$HARNESS_DIR/.env"; set +a

send_alert() {  # title, message
  local title="$1" msg="$2"
  osascript -e "display notification \"$msg\" with title \"$title\" sound name \"Glass\"" 2>/dev/null || true
  printf '\a'
  if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
    curl -s -m 10 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
      --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
      --data-urlencode "text=${title}
${msg}" >/dev/null || true
  fi
  if [ -n "${WHATSAPP_PHONE:-}" ] && [ -n "${CALLMEBOT_APIKEY:-}" ]; then
    curl -s -m 12 -G "https://api.callmebot.com/whatsapp.php" \
      --data-urlencode "phone=${WHATSAPP_PHONE}" \
      --data-urlencode "text=${title}: ${msg}" \
      --data-urlencode "apikey=${CALLMEBOT_APIKEY}" >/dev/null || true
  fi
}

chans="macOS"
[ -n "${TELEGRAM_BOT_TOKEN:-}" ] && chans="$chans, Telegram"
[ -n "${WHATSAPP_PHONE:-}" ] && chans="$chans, WhatsApp"
echo "watching best score every ${INT}s  ($IP)  channels: $chans  -- Ctrl-C to stop"
send_alert "evo: watching Sarvam-30B" "monitor started on $IP (poll ${INT}s)"

last=""
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
    send_alert "evo: Sarvam-30B improved" "$msg"
    last="$best"
  else
    echo "[$ts] best=$best (committed=$exps, no change)"
  fi
  sleep "$INT"
done
