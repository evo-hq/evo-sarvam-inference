#!/usr/bin/env python3
"""Single-window monitor for the Sarvam-30B evo run on JarvisLabs.

Top  : live stats (claude up?, dashboard, GPU util/mem x4, evo best score / #experiments).
Below: the Claude session AND every workflow subagent, merged + color-tagged, streaming in.

It follows the whole Claude transcript dir on the box (main session + agent-*.jsonl),
so when the optimize workflow fans out to parallel subagents, each one shows up tagged
[sub:xxxxxx] in its own color.

Usage:  ./scripts/monitor.py <box-ip>
Key   : ~/.ssh/jl_ed25519 (override with JL_SSH_KEY).
IP    : jl get <machine_id> --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["public_ip"])'
"""
import sys, os, re, json, time, shutil, threading, subprocess, collections

if len(sys.argv) < 2:
    print("usage: monitor.py <box-ip>"); sys.exit(1)
IP = sys.argv[1]
KEY = os.environ.get("JL_SSH_KEY", os.path.expanduser("~/.ssh/jl_ed25519"))
VLLM = "/home/ubuntu/vllm"
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=accept-new",
       "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=12", "ubuntu@" + IP]

# --- box-side follower: tail every active *.jsonl under ~/.claude/projects, tagged ---
FOLLOWER = r'''
import sys, os, glob, time
ROOT = os.path.expanduser("~/.claude/projects")
pos = {}
def tag(fn):
    b = os.path.basename(fn).replace("agent-", "").replace(".jsonl", "")
    if "/subagents/workflows/" in fn:
        return "wf:" + b[:6]
    if "/subagents/" in fn:
        return "sub:" + b[:6]
    return "main:" + b[:6]
while True:
    for fn in glob.glob(os.path.join(ROOT, "**", "*.jsonl"), recursive=True):
        try:
            st = os.stat(fn)
            if fn not in pos:
                # seed recent context (~8KB) for active files; skip dead sessions
                if time.time() - st.st_mtime > 180:
                    pos[fn] = st.st_size
                    continue
                pos[fn] = max(0, st.st_size - 8000)
            if st.st_size < pos[fn]:
                pos[fn] = 0
            if st.st_size > pos[fn]:
                with open(fn, errors="replace") as f:
                    f.seek(pos[fn]); chunk = f.read(); pos[fn] = f.tell()
                for ln in chunk.splitlines():
                    if ln.strip():
                        sys.stdout.write(tag(fn) + "\t" + ln + "\n")
        except Exception:
            pass
    sys.stdout.flush(); time.sleep(0.4)
'''

C = {"r": "\033[0m", "b": "\033[1m", "d": "\033[2m", "red": "\033[91m",
     "grn": "\033[92m", "yel": "\033[93m", "cyn": "\033[96m", "wht": "\033[97m"}
SUBPAL = ["\033[95m", "\033[96m", "\033[94m", "\033[92m", "\033[93m", "\033[91m"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")

logbuf = collections.deque(maxlen=1000)
stats = {"raw": "(connecting...)"}
started = time.time()
dlock = threading.Lock()


def subcolor(tag):
    if tag == "main":
        return ""
    return SUBPAL[sum(ord(c) for c in tag) % len(SUBPAL)]


def clean(s, n):
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
                out.append(("text", clean(b["text"], 400)))
            elif bt == "thinking" and b.get("thinking", "").strip():
                out.append(("think", clean(b["thinking"], 220)))
            elif bt == "tool_use":
                i = b.get("input", {})
                a = (i.get("command") or i.get("file_path") or i.get("prompt")
                     or i.get("pattern") or i.get("description") or "")
                out.append(("tool", str(b.get("name")) + ": " + clean(a, 200)))
    elif t == "user":
        for b in d.get("message", {}).get("content", []):
            if b.get("type") == "tool_result":
                c = b.get("content", "")
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                out.append(("res", clean(c, 200)))
    elif t == "result":
        out.append(("done", "RESULT: " + str(d.get("subtype"))))
    return out


def colorize_header(raw):
    out = []
    for l in raw.splitlines():
        l = (l.replace("UP", C["grn"] + "UP" + C["r"])
              .replace("DOWN", C["red"] + "DOWN" + C["r"])
              .replace("dashboard: 200", "dashboard: " + C["grn"] + "200" + C["r"]))
        out.append(" " + l)
    return out


def render():
    cols, rows = shutil.get_terminal_size((100, 40))
    with dlock:
        raw = stats["raw"]; tail = list(logbuf)
    el = int(time.time() - started)
    head = [C["b"] + C["cyn"] + " Sarvam-30B evo  |  %s  |  up %dh%02dm  |  %s "
            % (IP, el // 3600, el % 3600 // 60, time.strftime("%H:%M:%S")) + C["r"]]
    head += colorize_header(raw)
    head.append(C["d"] + "-" * min(cols, 150) + C["r"])
    body_rows = max(3, rows - len(head) - 1)
    kind_col = {"text": C["wht"] + C["b"], "think": C["d"], "tool": C["yel"],
                "res": C["d"], "done": C["grn"] + C["b"]}
    body = []
    for tag, (kind, txt) in tail[-body_rows:]:
        scol = subcolor(tag)
        label = "" if tag == "main" else scol + "[" + tag + "] " + C["r"]
        lblw = 0 if tag == "main" else len(tag) + 3
        pfx = {"text": "", "think": "~ ", "tool": "> ", "res": "    -> ", "done": "=== "}[kind]
        budget = max(8, cols - lblw - len(pfx) - 1)
        body.append(label + kind_col[kind] + pfx + txt[:budget] + C["r"])
    sys.stdout.write("\033[H\033[J" + "\n".join(head) + "\n" + "\n".join(body))
    sys.stdout.flush()


def stat_loop():
    cmd = ("export PATH=$HOME/.local/bin:$PATH; "
           "echo \"claude: $(pgrep -f 'claude --print' >/dev/null && echo UP || echo DOWN)    "
           "dashboard: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080 2>/dev/null)    "
           "subagents: $(pgrep -fc 'claude' 2>/dev/null)\"; "
           "echo 'gpu  idx util% memMiB:'; "
           "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  /'; "
           "cd " + VLLM + " 2>/dev/null && evo status 2>/dev/null | head -4")
    while True:
        try:
            out = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=30).stdout
            with dlock:
                stats["raw"] = out.strip() or "(no stats)"
        except Exception as e:
            with dlock:
                stats["raw"] = "(stat error: %s)" % e
        time.sleep(6)


def stream_loop():
    while True:
        try:
            p = subprocess.Popen(SSH + ["python3 -"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True, bufsize=1)
            p.stdin.write(FOLLOWER); p.stdin.close()
            for line in p.stdout:
                tag, _, payload = line.partition("\t")
                for ev in parse(payload):
                    with dlock:
                        logbuf.append((tag, ev))
        except Exception:
            time.sleep(3)


def render_loop():
    while True:
        render(); time.sleep(0.5)


threading.Thread(target=stat_loop, daemon=True).start()
threading.Thread(target=stream_loop, daemon=True).start()
try:
    render_loop()
except KeyboardInterrupt:
    sys.stdout.write("\033[?25h\n")
