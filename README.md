# evo-sarvam-inference

Point evo at vLLM serving [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b)
and let it make inference **faster at equal accuracy** by editing the MoE/attention
kernels. The harness ships only infra: **`/evo:discover` designs and writes the
benchmark and the gate itself** (the metric is not pre-decided), runs every measurement
through an exclusive single-GPU lock so scores are reproducible, then a prose
`/evo:optimize` loop hill-climbs.

- **Target model:** Sarvam-30B (MoE, 32B total / ~2.4B active, 128 experts top-6, GQA, Apache-2.0).
- **Engine:** vLLM at [PR #33942](https://github.com/vllm-project/vllm/pull/33942) (`SarvamMoEForCausalLM`, fused-MoE Triton path).
- **Benchmark + gate:** discover-built, to the contract in `references/benchmark-contract.md`. Speed metric + an accuracy gate; reproducibility (median-of-N, no contention) is a hard requirement.
- **Compute:** 1x H100 on JarvisLabs. Subagents reason/edit in parallel; `gpu_locked.sh` serializes their measurements (one at a time → no cross-experiment contention → reproducible scores).
- **Orchestrator:** prose (not the workflow driver).

The evo workspace (`.evo/`) is created on the box **inside the vLLM clone**, because
that clone is the codebase evo optimizes. This repo is the replication harness: it
bootstraps the box, ships the GPU-lock + the benchmark contract, and hands the run to
the evo skills. We do not hand-author `evo init`; `/evo:discover` builds it.

## Layout

```
scripts/
  gpu_locked.sh           the only benchmark/gate infra: exclusive GPU lease + venv +
                          worktree-kernel resolution. discover invokes its benchmark
                          and gate THROUGH this.
  run_evo_jarvislabs.sh   bootstrap | evo-setup | clocks | run | notify | dashboard
  monitor.py / notify.sh / watch_run.sh   local monitoring (stream / alert / focused views)
references/
  benchmark-contract.md   what discover's benchmark + gate must satisfy (output, the
                          GPU lock, the reproducibility bar, the gate)
evo/
  run_prompt.md           objective handed to the headless agent: discover builds the
                          benchmark, prose optimize, single-GPU lock
docs/plan.md              design notes + the run-1 post-mortem (why the noise killed it)
```

There is no pre-built benchmark in this repo by design — discover writes it on the box.

## Run it

```bash
# 1. provision (jl CLI). 1x H100, region with availability.
jl create --gpu H100 --vm --num-gpus 1 --storage 200 --yes --json

# 2. on the box, under /home (persists across pause/resume)
git clone <this-repo> /home/ubuntu/evo-sarvam-inference
cd /home/ubuntu/evo-sarvam-inference && cp .env.example .env   # fill in tokens

bash scripts/run_evo_jarvislabs.sh bootstrap    # venv + vLLM + weights + harness infra (one-time)
bash scripts/run_evo_jarvislabs.sh evo-setup     # Claude Code + evo CLI + plugin + auth check
bash scripts/run_evo_jarvislabs.sh clocks        # lock GPU clocks (cuts noise)
bash scripts/run_evo_jarvislabs.sh run           # headless agent: discover builds benchmark -> prose optimize

# 3. monitor / clean up
bash scripts/run_evo_jarvislabs.sh dashboard     # public dashboard URL (cloudflared)
jl pause <id> --yes --json                        # stop GPU billing, keep /home
```

## Monitoring

From your machine (`cd ~/Work/evo-sarvam-inference`):

- `./scripts/monitor.py <ip>` — one window: pinned stats (claude, dashboard, GPU, best
  score) on top; the live Claude session **and every subagent** streaming below, tagged
  and color-coded.
- `./scripts/notify.sh <ip> [secs]` — alert only on a **new best** (macOS notification +
  sound; Telegram/WhatsApp too if configured).
- `./scripts/watch_run.sh <ip> [stream|evo|gpu|dash]` — focused views: raw session,
  `evo tree`, live `nvidia-smi`, or an SSH tunnel to the dashboard.

Public dashboard (anyone, no key): `bash scripts/run_evo_jarvislabs.sh dashboard` on the
box prints a cloudflared `https://...trycloudflare.com` URL. JarvisLabs VMs firewall
inbound ports, so `http://<public-ip>:port` never works; the dashboard binds localhost
and is reachable only via that tunnel (public) or an SSH local-forward (key holder).

Notifications: set `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` (or `WHATSAPP_PHONE`/`CALLMEBOT_APIKEY`)
in `.env`. `notify.sh` reads them locally; `run_evo_jarvislabs.sh notify` runs the loop on
the box in tmux (always-on, survives your laptop sleeping).

## Status

Run 1 (4x H100, workflow driver, hardcoded decode benchmark) found a +4.58% "win" that
turned out to be a **benchmark-noise artifact**: warm/cold bias + cross-experiment GPU
contention + a single-run anchor on a high-variance shape, so the loop committed a noise
high and discarded ~10 genuine improvements. See `docs/plan.md` for the post-mortem. This
redesign fixes the root cause: discover builds its own reproducible benchmark, and a
single GPU + exclusive lock removes the contention. The vLLM build / weights / venv carry
over on the same box (resized to 1x).
