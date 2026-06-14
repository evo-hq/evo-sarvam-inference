# Run prompt — Sarvam-30B inference-speed (discover-built benchmark, prose + autonomous, 1x GPU)

Handed to the headless agent on the box (`claude --print`). Working directory is the
vLLM clone (checked out at the SarvamMoE PR). Keep this free of optimization-technique
hints: state the objective and the harness contract, not the kernel tricks to try.

## CRITICAL — headless execution rules (read first)
You are running HEADLESS, as ONE long-lived turn. If you end the turn, the run can DIE.
So:
- Run EVERY benchmark, gate, build, and long operation in the **FOREGROUND** and block
  on it in-line. NEVER launch work as a background task / background Bash / `&` and then
  "wait for a completion notification" — in headless mode that notification will NOT
  wake you; your turn ends and the run is dead. If something takes minutes, run it in
  the foreground and wait for it to return. (A previous run died exactly this way.)
- As soon as `evo init` has run, arm evo's keep-going so the loop survives turn
  boundaries: `evo autonomous on`, and `evo config set default-orchestrator prose`.
- Drive the whole loop yourself, continuously. Do not stop until the optimize budget is
  exhausted or you hit a real blocker (then report it).

OBJECTIVE: make Sarvam-30B inference **faster** on this vLLM build, with model accuracy
held within tolerance. Higher speed is better. You decide what "faster" means (which
phase / workload / metric best captures it) — it is not pre-decided.

The benchmark and the gate are NOT pre-built. You design and write them yourself.

1. Run `/evo:discover`. During discover:
   - Immediately after `evo init`: `evo autonomous on` and
     `evo config set default-orchestrator prose`.
   - Explore the SarvamMoE serving path; decide WHAT to measure and HOW to measure it
     reliably. Read `evo_harness/references/benchmark-contract.md`.
   - Write your own benchmark + gate. Run every measurement in the FOREGROUND through
     `evo_harness/gpu_locked.sh`.
   - Reproducibility is the hard requirement (the prior run died on benchmark noise:
     warm/cold bias, cross-experiment contention, single-run anchor on a high-variance
     shape). Fixed workload, warmup, median-of-N, one measurement at a time (the lock),
     stable anchor. Prove it: baseline 3x, spread tighter than the wins you chase.
   - Wire `evo init` to your benchmark and `evo gate add` to your gate (both via
     `gpu_locked.sh`). Run and commit the baseline. Document in `project.md`.
   - Never edit the benchmark, gate, weights, or harness after discover. If the gate
     looks wrong, stop and report — do not weaken it.

2. Run `/evo:optimize` with `subagents=3`. ONE GPU; `gpu_locked.sh` serializes
   benchmark/gate access, so the subagents reason and edit in parallel but measure one
   at a time (clean, no contention) — the optimize skill's exclusive-GPU pattern. Drive
   the loop continuously (autonomous is on): frontier selection, verifier pre/post,
   annotation discipline; keep what passes the gate AND improves the metric. Run all
   measurements in the foreground; spawn subagents synchronously (do not background them).

Never touch the weights, the benchmark, or the gate during optimize.
