# evo-sarvam-inference

Point evo at vLLM serving [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b)
and let it make decode **faster at equal accuracy** by rewriting the MoE/attention
kernels. Throughput is the metric evo maximizes; an accuracy gate holds quality fixed.

- **Target model:** Sarvam-30B (MoE, 32B total / ~2.4B active, 128 experts top-6, GQA, Apache-2.0).
- **Engine:** vLLM at [PR #33942](https://github.com/vllm-project/vllm/pull/33942) (`SarvamMoEForCausalLM`, fused-MoE Triton path).
- **Metric:** geomean of median decode tokens/sec across fixed shapes (MAX). Single GPU, TP=1, FP8.
- **Gate:** greedy top-1 agreement + chosen-token logprob drift vs a captured baseline, within tolerance.
- **Compute:** 4x H100 on JarvisLabs. evo width=4, one experiment leased per GPU.

The evo workspace (`.evo/`) is created on the box **inside the vLLM clone**, because
that clone is the codebase evo optimizes. This repo is the replication harness: it
bootstraps the box, supplies the benchmark + gate, and hands the run off to the evo
skills. We do not hand-author `evo init`; `/evo:discover` does.

## Layout

```
bench/      decode-throughput benchmark (the evo metric)
  bench_decode.py   two-point decode isolation, median-of-N, geomean over shapes
  workloads.json    fixed operating points (short interactive + long context)
  run_bench.sh      evo benchmark entry: GPU lease + worktree-kernel resolution
gate/       accuracy rail (the evo gate)
  capture_reference.py  one-time baseline greedy generations
  verify_quality.py     top-1 agreement + logprob drift vs baseline
  run_gate.sh           evo gate entry
reference/  prompts.jsonl (fixed gate prompts); baseline_gen.json (captured on box)
evo/        run_prompt.md  the objective handed to the headless agent (discover -> optimize)
scripts/    run_evo_jarvislabs.sh  bootstrap | reference | run | dashboard
docs/       plan.md  metric/gate design, GPU layout, the .so-sharing trick, open validation items
```

## Run it

```bash
# 1. provision (jl CLI). 4x H100, region with availability.
jl gpus --json
jl create --gpu H100 --num-gpus 4 --storage 200 --yes --json

# 2. on the box, under /home (persists across pause/resume)
git clone <this-repo> /home/ubuntu/evo-sarvam-inference
cd /home/ubuntu/evo-sarvam-inference && cp .env.example .env   # fill in tokens

bash scripts/run_evo_jarvislabs.sh bootstrap    # clone+build+weights+harness (one-time)
bash scripts/run_evo_jarvislabs.sh reference     # capture the gate anchor
bash scripts/run_evo_jarvislabs.sh run           # launch the headless evo agent
bash scripts/run_evo_jarvislabs.sh dashboard     # optional: expose evo dashboard on :8090

# 3. monitor / clean up
tmux attach -t sarvam        # or: tail runs/run_console.log
jl pause <id> --yes --json   # stop GPU billing, keep /home
```

## Monitoring

From your machine (`cd ~/Work/evo-sarvam-inference`):

- `./scripts/monitor.py <ip>` — one window: pinned stats (claude, dashboard, GPUs, best
  score) on top; the live Claude session **and every workflow subagent** streaming below,
  tagged `main:`/`sub:`/`wf:` and color-coded.
- `./scripts/notify.sh <ip> [secs]` — alert only on a **new best** (macOS notification +
  sound; Telegram/WhatsApp too if configured below).
- `./scripts/watch_run.sh <ip> [stream|evo|gpu|dash]` — focused views: raw session,
  `evo tree`, live `nvidia-smi`, or an SSH tunnel to the dashboard.

Public dashboard (anyone, no key): run `bash scripts/run_evo_jarvislabs.sh dashboard` on
the box; it prints a cloudflared `https://...trycloudflare.com` URL. JarvisLabs VMs
firewall inbound ports, so `http://<public-ip>:port` never works, the dashboard binds
localhost and is reachable only via that tunnel (public) or an SSH local-forward (key holder).

Notifications: set `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` (or `WHATSAPP_PHONE`/`CALLMEBOT_APIKEY`)
in `.env`. `notify.sh` reads them locally; `run_evo_jarvislabs.sh notify` runs the same loop
on the box in tmux (always-on, survives your laptop sleeping).

## Status

Validated end-to-end on H100 (1x dry run + 4x run): vLLM @ PR #33942 builds with
`VLLM_USE_PRECOMPILED` (no CUDA rebuild), Sarvam-30B loads at FP8, the decode benchmark and
accuracy gate run, and the worktree kernel-swap (edit Triton, no rebuild) works. The 4x run
resizes the same box via `jl resume --num-gpus 4` (no re-download). See `docs/plan.md` for the
metric/gate design and the remaining gotchas.
