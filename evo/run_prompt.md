# Run prompt — Sarvam-30B inference-speed (discover-built benchmark, prose, 1x GPU)

Handed to the headless agent on the box (`claude --print`). Working directory is the
vLLM clone (checked out at the SarvamMoE PR). Keep this free of optimization-technique
hints: state the objective and the harness contract, not the kernel tricks to try.

OBJECTIVE: make Sarvam-30B inference **faster** on this vLLM build, with model accuracy
held within tolerance. Higher speed is better. You decide what "faster" means (which
phase / workload / metric best captures it) — it is not pre-decided.

The benchmark and the gate are NOT pre-built. You design and write them yourself.

1. Run `/evo:discover`. During discover:
   - Explore the SarvamMoE serving path and decide WHAT to measure and HOW to measure
     it reliably. Read `evo_harness/references/benchmark-contract.md` for the contract
     your benchmark + gate must satisfy.
   - **Write your own benchmark script and your own accuracy gate** (follow the
     discover skill's benchmark-construction guidance). Run every measurement through
     `evo_harness/gpu_locked.sh` (the harness's exclusive-GPU lock).
   - **Reproducibility is the hard requirement.** A prior run failed because its
     benchmark was too noisy (warm/cold bias, cross-experiment GPU contention, a
     single-run anchor on a high-variance shape) — it committed a noise high and
     discarded ~10 genuine wins. Your benchmark MUST produce scores that reproduce
     run-to-run: fixed workload, warmup, median-of-N, one measurement at a time (the
     lock), and a stable anchor. Prove it: baseline 3x, spread tighter than the wins
     you're chasing, before trusting any delta.
   - Wire `evo init` to your benchmark (the metric) and `evo gate add` to your gate,
     both invoked via `gpu_locked.sh`. Run and commit the baseline. Document in
     `project.md`.
   - You may edit any code in the worktree to optimize, but NEVER edit the benchmark,
     the gate, the weights, or the harness after discover. If the gate looks wrong,
     stop and report — do not weaken it.

2. Set the orchestrator to PROSE (not the workflow driver):
   `evo config set default-orchestrator prose`.

3. Run `/evo:optimize` with `subagents=3`. The box has ONE GPU; `gpu_locked.sh`
   serializes benchmark/gate access, so the 3 subagents reason and edit in parallel
   but measure one at a time (clean, no contention) — this is the optimize skill's
   exclusive-GPU pattern. Drive the loop: frontier selection, verifier pre/post,
   annotation discipline; keep what passes the gate AND improves the metric.

Never touch the weights, the benchmark, or the gate during optimize.
