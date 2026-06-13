# Design + open items

## Objective

Maximize Sarvam-30B decode throughput on vLLM, accuracy held within tolerance.
In evo terms: `metric = max(decode tok/s)`, `gate = accuracy within epsilon`.
Speed is never traded against accuracy by the optimizer; accuracy is a constraint
the gate enforces, not a second objective.

## Agent + orchestrator

- **Agent:** headless Claude Code on **Opus 4.8, `--effort max`**, OAuth subscription
  (`CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY` empty). Set in `run()` via
  `--model claude-opus-4-8` + `--effort max` (use `claude-opus-4-8[1m]` for 1M
  context on long loops). `CLAUDE_CODE_EFFORT_LEVEL=max` env also works and takes
  precedence.
- **Orchestrator = workflow driver** (the concurrent meta/analyst), not the prose
  loop. Two requirements: (1) the agent sets `evo config set default-orchestrator
  workflow` after discover (project config, recreated each run, so set in the run
  prompt); (2) the Claude Code plugin cache must carry the workflow driver, so
  bootstrap runs `evo update claude-code --from-path ... --force` (a stale cache
  silently falls back to prose with no analyst). The meta/analyst is read-only
  (nvidia-smi/logs), takes one concurrency slot, and never leases a GPU, so it does
  not compete with the four experiment lanes.

## The metric (bench/bench_decode.py)

`score = geomean over shapes of ( median over iters of decode_tok_s )`, MAX.

- **Decode isolation, two-point method.** For each shape, time generation at
  `gen_len = D` and `gen_len = 2D` on the same fixed batch. Prefill cost is
  identical, so `decode_tok_s = (batch * D) / (t_2D - t_D)` cancels prefill and
  isolates the per-token rate the fused-MoE / GQA kernels drive.
- **Median over `iters`** (default 5): GPU timings are noisy; never score one run.
- **Geomean over shapes** (short-interactive + long-context): stops a kernel
  config from overfitting one batch geometry (Goodhart guard).
- **Single GPU, TP=1.** Going multi-GPU lets NCCL comm mask kernel deltas and adds
  variance. The parallelism strategy is a separate later experiment, not this metric.
- **Lock GPU clocks** before running (`nvidia-smi -lgc`, fixed power cap) to remove
  thermal drift between experiments. Add to bootstrap once the box is up.

## The gate (gate/verify_quality.py)

Claim is "same accuracy", so a faster-but-divergent kernel must NOT pass.

- Re-run greedy on the fixed prompt set, compare vs `reference/baseline_gen.json`.
- `top1_agreement >= 0.995` (float reorder may flip a rare token; near-exact only).
- `max_logprob_drift <= 0.10` at agreeing positions (catches "same token, moved distribution").
- Tolerances are env-overridable (`GATE_MIN_TOP1`, `GATE_MAX_LOGPROB_DRIFT`) but
  loosening them is spending quality budget; do it with eyes open. The gate is the
  whole safety story for the kernel-rewrite surface.

## GPU layout

4x H100 -> evo `subagents=4`, one experiment per GPU. `run_bench.sh` / `run_gate.sh`
lease a free GPU via `flock` on `/tmp/sarvam_gpu_{0..3}.lock`, held for the process
lifetime, so two experiments never share a card (shared card = cross-contaminated
timings). Bench and gate for one experiment run sequentially, each leasing independently.

## The worktree kernel-resolution trick (KEY, validate first)

evo forks a git worktree per experiment. We want each worktree's edited Triton
kernels to take effect with no CUDA rebuild:

1. Base install once: `VLLM_USE_PRECOMPILED=1 pip install -e .` in `$VLLM_BASE`.
   This downloads the prebuilt C-extension (`vllm/*.so`) instead of compiling.
2. Per experiment, `run_bench.sh` symlinks `$VLLM_BASE/vllm/**/*.so` into the
   worktree and prepends the worktree to `PYTHONPATH`, so `import vllm` loads the
   worktree's edited `.py`/Triton kernels but resolves the compiled extension from
   base. Pure-Python/Triton edits need no rebuild.

This holds for **Triton** kernel edits (the fused-MoE `@triton.jit` path). Editing
CUDA (`.cu`) ops requires a real rebuild and breaks the no-recompile assumption;
keep run-1 on the Triton surface. The run prompt steers the agent toward the
fused-MoE Triton path accordingly without dictating the optimizations.

## Optimization surface (where the headroom is)

PR #33942 reuses vLLM's fused-MoE primitives with minimal Sarvam routing
extensions, so the integration is fresh and likely under-tuned for the
128-expert/top-6/GQA shape. Tractable, Triton-first edits:

- fused-MoE grouped-GEMM kernel tiling / block sizes / num_warps / num_stages /
  GROUP_SIZE_M for this expert shape.
- top-6 routing + expert-bias-normalization path the PR adds.
- GQA decode attention (4 KV heads), RMSNorm, RoPE (rope_theta 8e6) fused paths.

CUDA-level rewrites and quantization-recipe search are deliberately out of run-1
(rebuild cost / wider quality risk); revisit after the Triton surface is mined.

## Open validation items (NOT yet run on a box)

1. **Precompiled wheel vs PR base commit.** `VLLM_USE_PRECOMPILED=1` pulls a wheel
   for a recent main; confirm ABI matches PR #33942's base or pin accordingly.
2. **.so symlink + PYTHONPATH** actually makes worktree Triton edits win without
   shadowing the compiled extension. Smoke-test by editing a kernel and confirming
   the change is observed.
3. **Sarvam-30B load**: fp8 variant availability, gated-repo `HF_TOKEN`,
   `trust_remote_code`, `max_model_len` that fits KV cache on one H100 at FP8.
4. **`vllm/model_executor/layers/fused_moe/fused_moe.py`** is the right target path
   on the PR branch (confirm filename/location).
5. **Benchmark noise floor**: run the baseline 3x, confirm median decode tok/s is
   stable within a few percent with locked clocks before trusting score deltas.
6. **Gate sensitivity**: confirm an intentionally-wrong kernel fails the gate and a
   no-op edit passes, before the loop runs.
7. **discover constraints**: confirm the agent uses the existing bench/gate verbatim
   and does not redesign them (the run prompt forbids it; verify in practice).

Smoke-test one full experiment (new -> edit -> bench -> gate -> done/discard) before
launching the autonomous loop.
