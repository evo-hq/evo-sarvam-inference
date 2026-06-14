# Benchmark contract (what discover must build)

You (discover) design and **write** the benchmark and the accuracy gate from scratch.
The harness does not ship them. This is the contract they must satisfy so the evo
loop can run them and trust the scores.

## Instrumentation (output)
- The benchmark writes `{"score": <number>, "tasks": {"<name>": <number>, ...}}` to
  the path in `$EVO_RESULT_PATH` (write atomically). `score` is the single number evo
  optimizes; set `metric=max` or `metric=min` at `evo init` to match.
- Optionally write per-task detail to `$EVO_TRACES_DIR/task_<name>.json`.
- Exit 0 = measurement succeeded; non-zero = infrastructure failure (evo won't score it).

## GPU lock (REQUIRED — this is the reproducibility infra)
Run every GPU measurement through the wrapper the harness ships:

    bash {worktree}/evo_harness/gpu_locked.sh {worktree} -- <your command>

It leases ONE GPU (blocking), pins `CUDA_VISIBLE_DEVICES`, isolates per-experiment
JIT caches, makes the worktree's edited kernels win without a rebuild, and activates
the venv. Because the lease is exclusive, parallel subagents serialize on the GPU:
only one benchmark runs at a time, so no concurrent experiment perturbs your timing
or allocator state. Wire both the benchmark and the gate through it.

## Reproducibility bar (REQUIRED — the thing that broke last time)
A previous run's benchmark was too noisy: warm-vs-cold bias, cross-experiment GPU
contention, and a single-run anchor on a high-variance shape made it commit a noise
high and then discard ~10 genuine improvements as "regressions." Do not repeat that:
- Fixed, deterministic workload (pinned sizes; greedy/seeded).
- Warm up, then take the **median (or min) over several timed repeats** — never score
  a single run.
- One measurement at a time (the GPU lock guarantees this).
- **The run-to-run spread of `score` on identical code must be smaller than the
  improvements you intend to detect.** Verify it: run the baseline 3x and confirm the
  spread is tight (aim <1-2%). If it isn't, change the metric/workload/repeats (or
  lock GPU clocks) until it is. A metric noisier than the signal is unusable.
- Compare nodes under identical warm/cold conditions (a fixed warmup + the lock).

## Accuracy gate
- The benchmark measures speed only. Write a **separate gate** that holds model
  accuracy fixed: compare the candidate's outputs to a captured baseline within a
  tolerance you choose and justify; exit non-zero on regression. Register it via
  `evo gate add`, run through `gpu_locked.sh`.
- A single-run correctness check can pass a kernel that is correct only by allocator
  luck (correct on a free GPU, wrong under load). Make the gate as robust as you can
  (re-check, or run under the same exclusive lock).

## Don't
- Don't read the gate's reference answers inside the benchmark (no leakage).
- Don't subset or narrow the eval to make it easier.
- After discover, the benchmark + gate are **frozen** — the optimize loop must never
  edit them, the weights, or this contract. If you believe the gate is wrong, stop
  and report rather than weaken it.
