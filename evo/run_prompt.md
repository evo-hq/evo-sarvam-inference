# Run prompt — Sarvam-30B inference-speed evo run

Handed to the headless agent on the box (`claude --print --dangerously-skip-permissions`).
The agent drives the evo skills; we do not hand-author `evo init`/`evo new`/`evo run`.
Keep this free of technique hints: state the objective and the rails, not the
kernel tricks to try. The whole point is that the loop discovers those.

---

You are optimizing inference speed for the Sarvam-30B MoE model served on vLLM.
The codebase is this vLLM clone (checked out at the SarvamMoE PR). Your working
directory is the vLLM repo root.

OBJECTIVE: maximize decode throughput (output tokens/sec) for Sarvam-30B, with
accuracy held within tolerance. Higher score is better.

The benchmark and the accuracy gate are ALREADY BUILT. Do not redesign or weaken
them, and do not edit the model weights or the harness:

- Benchmark: `bash {worktree}/evo_harness/bench/run_bench.sh {target} {worktree}`
  Score = geomean of median decode tok/s across fixed workload shapes. MAX metric.
- Gate: `bash {worktree}/evo_harness/gate/run_gate.sh {target} {worktree}`
  Fails (exit non-zero) if greedy top-1 agreement or chosen-token logprob drift
  vs the captured baseline exceeds tolerance. This is the accuracy rail.

Steps:

1. Run `/evo:discover`. During discover:
   - Explore the SarvamMoE model executor and the fused-MoE / attention path so
     you understand what is editable. The optimization target is
     `vllm/model_executor/layers/fused_moe/fused_moe.py`; you may edit any code in
     the worktree, but never the benchmark, the gate, or the weights.
   - Wire `evo init` to the EXACT benchmark and gate commands above, metric=max.
   - Confirm the baseline reference (`evo_harness/reference/baseline_gen.json`)
     exists; it is captured before this run on the unmodified build.
   - Run the baseline experiment (exp_0000) and commit it.
   - In `project.md`, document the resource profile (one H100 per experiment,
     TP=1, FP8) and the gaming risks specific to this run: (a) GPU-timing noise
     being committed as improvement, (b) a kernel that is faster because it is
     subtly wrong slipping through a too-loose gate. Note how the harness guards
     each (median-of-N + locked clocks; strict top-1/logprob tolerance).

2. Before optimizing, run `evo config set default-orchestrator workflow` so the
   loop runs under the workflow driver (the concurrent meta/analyst observes the
   run, not the prose loop). `default-orchestrator` is project config recreated by
   discover, so it must be set HERE, after discover.

3. Then run `/evo:optimize` with `subagents=4` (width = number of GPUs; each
   experiment leases one GPU). Drive the loop: branch from the frontier, let
   subagents form their own kernel hypotheses, keep what passes the gate AND
   improves throughput, discard the rest.

Do not touch the model weights, the benchmark, or the gate at any point. If you
believe the gate is wrong, stop and report rather than weakening it.
