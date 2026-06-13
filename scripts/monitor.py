#!/usr/bin/env python3
"""Single-window monitor for the Sarvam-30B evo run on JarvisLabs.

Top: live stats (claude up?, dashboard, GPU util/mem x4, evo best score / #experiments).
Bottom: the parsed Claude session streaming in.

Usage:  ./scripts/monitor.py <box-ip>
Needs the JarvisLabs key (default ~/.ssh/jl_ed25519; override with JL_SSH_KEY).
Get the current IP:  jl get <machine_id> --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["public_ip"])'
"""
import sys, os, re, json, time, shutil, threading, subprocess, collections

if len(sys.argv) < 2:
    print("usage: monitor.py <box-ip>"); sys.exit(1)
IP = sys.argv[1]
KEY = os.environ.get("JL_SSH_KEY", os.path.expanduser("~/.ssh/jl_ed25519"))
LOG = "/home/ubuntu/evo-sarvam-inference/runs/run_console.log"
VLLM = "/home/ubuntu/vllm"
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=accept-new",
       "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=12", "ubuntu@" + IP]

logbuf = collections.deque(maxlen=800)
stats = {"raw": "(loading stats...)"}
started = time.time()
dlock = threading.Lock()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def short(s, n):
    s = ANSI.sub("", str(s)).replace("\n", " ").replace("\t", " ").strip()
    return s[:n] + (" ..." if len(s) > n else "")


def parse(line):
    line = line.strip()
    if not line.startswith("{"):
        return []
    try:
        d = json.loads(line)
    except Exception:
        return []
    t = d.get("type"); out = []
    if t == "assistant":
        for b in d.get("message", {}).get("content", []):
            bt = b.get("type")
            if bt == "text" and b.get("text", "").strip():
                out.append("* " + short(b["text"], 400))
            elif bt == "tool_use":
                i = b.get("input", {})
                a = (i.get("command") or i.get("file_path") or i.get("prompt")
                     or i.get("pattern") or i.get("description") or "")
                out.append("  > " + str(b.get("name")) + ": " + short(a, 200))
    elif t == "user":
        for b in d.get("message", {}).get("content", []):
            if b.get("type") == "tool_result":
                c = b.get("content", "")
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                out.append("     -> " + short(c, 200))
    elif t == "result":
        out.append("=== RESULT: " + str(d.get("subtype")) + " ===")
    return out


def stat_loop():
    cmd = (
        "export PATH=$HOME/.local/bin:$PATH; "
        "echo \"claude: $(pgrep -f 'claude --print' >/dev/null && echo UP || echo DOWN)    "
        "dashboard: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080 2>/dev/null)    "
        "events: $(wc -l < " + LOG + " 2>/dev/null)\"; "
        "echo 'gpu  idx util% memMiB:'; "
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  /'; "
        "cd " + VLLM + " 2>/dev/null && evo status 2>/dev/null | head -4"
    )
    while True:
        try:
            out = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=30).stdout
            with dlock:
                stats["raw"] = out.strip() or "(no stats)"
        except Exception as e:
            with dlock:
                stats["raw"] = "(stat fetch error: %s)" % e
        time.sleep(6)


def stream_loop():
    while True:
        try:
            p = subprocess.Popen(SSH + ["tail -n 150 -f " + LOG],
                                 stdout=subprocess.PIPE, text=True, bufsize=1)
            for line in p.stdout:
                rows = parse(line)
                if rows:
                    with dlock:
                        logbuf.extend(rows)
        except Exception:
            time.sleep(3)


def render_loop():
    while True:
        cols, rows = shutil.get_terminal_size((100, 40))
        with dlock:
            raw = stats["raw"]; tail = list(logbuf)
        el = int(time.time() - started)
        hdr = ["\033[1;36m Sarvam-30B evo  |  %s  |  up %dh%02dm  |  %s \033[0m"
               % (IP, el // 3600, el % 3600 // 60, time.strftime("%H:%M:%S"))]
        for l in raw.splitlines():
            hdr.append(" " + l[:cols - 1])
        hdr.append("\033[2m" + "-" * min(cols, 140) + "\033[0m")
        body_rows = max(3, rows - len(hdr) - 1)
        body = [l[:cols - 1] for l in tail[-body_rows:]]
        out = "\033[H\033[J" + "\n".join(hdr) + "\n" + "\n".join(body)
        sys.stdout.write(out); sys.stdout.flush()
        time.sleep(0.5)


threading.Thread(target=stat_loop, daemon=True).start()
threading.Thread(target=stream_loop, daemon=True).start()
try:
    render_loop()
except KeyboardInterrupt:
    sys.stdout.write("\033[?25h\n")
