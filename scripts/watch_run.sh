#!/usr/bin/env bash
# Local monitor for the Sarvam-30B evo run on the JarvisLabs box. Run from your Mac.
#
#   scripts/watch_run.sh <box-ip> [mode]
#     stream  (default)  live-follow the headless Claude session, parsed readable
#     raw                raw stream-json lines (pipe to jq yourself)
#     evo                refresh the evo experiment tree + status + frontier
#     gpu                live nvidia-smi across all 4 GPUs
#     dash               SSH tunnel -> evo dashboard at http://localhost:8080
#
# Needs the JarvisLabs SSH key (default ~/.ssh/jl_ed25519; override with JL_SSH_KEY).
# Current box IP:  jl get <machine_id> --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["public_ip"])'
#
# What shows where:
#   stream  = the orchestrator agent's live actions (discover, evo commands, kernel edits,
#             and when it launches the optimize WORKFLOW).
#   evo     = the experiments the workflow/subagents produce (tree, scores, best path).
#   dash    = the same, visually, plus per-subagent workflow progress.
set -uo pipefail
IP="${1:?usage: watch_run.sh <box-ip> [stream|raw|evo|gpu|dash]}"
MODE="${2:-stream}"
KEY="${JL_SSH_KEY:-$HOME/.ssh/jl_ed25519}"
LOG="/home/ubuntu/evo-sarvam-inference/runs/run_console.log"
VLLM="/home/ubuntu/vllm"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 ubuntu@"$IP")

case "$MODE" in
  stream)
    echo "# streaming Claude session from $IP  (Ctrl-C to stop)"
    "${SSH[@]}" "tail -n +1 -f $LOG" | python3 -u -c '
import sys, json
def sh(s, n=200):
    s = str(s).replace("\n", " ").strip()
    return s[:n] + (" ..." if len(s) > n else "")
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"): continue
    try: d = json.loads(line)
    except Exception: continue
    t = d.get("type")
    if t == "assistant":
        for b in d.get("message", {}).get("content", []):
            bt = b.get("type")
            if bt == "text" and b.get("text", "").strip():
                print("\n" + sh(b["text"], 1200))
            elif bt == "thinking" and b.get("thinking", "").strip():
                print("  (thinking) " + sh(b["thinking"], 180))
            elif bt == "tool_use":
                i = b.get("input", {})
                a = i.get("command") or i.get("file_path") or i.get("prompt") or i.get("pattern") or i.get("description") or ""
                print("  > " + str(b.get("name")) + ": " + sh(a, 180))
    elif t == "user":
        for b in d.get("message", {}).get("content", []):
            if b.get("type") == "tool_result":
                c = b.get("content", "")
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                print("      -> " + sh(c, 180))
    elif t == "result":
        print("\n=== RESULT: " + str(d.get("subtype")) + " ===")
'
    ;;
  raw)
    "${SSH[@]}" "tail -n +1 -f $LOG"
    ;;
  evo)
    while true; do
      clear; echo "== evo @ $(date +%T)  ($IP) =="
      "${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; cd $VLLM && evo tree 2>/dev/null; echo; evo status 2>/dev/null; echo '-- frontier --'; evo frontier 2>/dev/null | head"
      sleep 8
    done
    ;;
  gpu)
    "${SSH[@]}" "watch -n2 nvidia-smi"
    ;;
  dash)
    echo "# dashboard tunnel up -> open http://localhost:8080  (Ctrl-C to stop)"
    "${SSH[@]}" -N -L 8080:127.0.0.1:8080
    ;;
  *) echo "unknown mode: $MODE (use stream|raw|evo|gpu|dash)"; exit 1 ;;
esac
