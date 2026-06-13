#!/usr/bin/env python3
"""Decode-throughput benchmark for Sarvam-30B on vLLM, wired for evo.

Score (MAX metric, higher = faster): geomean across workload shapes of the
median decode tokens/sec.

Decode isolation (two-point method): for each shape we time generation at
gen_len = D and gen_len = 2D on the SAME fixed batch. The prefill cost is
identical in both, so

    decode_tok_s = (batch * D) / (t_2D - t_D)

cancels prefill and isolates the per-token decode rate, which is what the
fused-MoE / GQA kernels drive. We take the median over `iters` timed repeats
(GPU timings are noisy) and the geomean over shapes (so no single batch
geometry dominates the score).

Determinism / low noise:
  - greedy (temperature 0), ignore_eos so every request emits exactly gen_len.
  - fixed dummy prompt token ids so prompt length is exact and identical.
  - warmup runs discarded (CUDA graph capture, Triton autotune, expert cache).
  - LOCK GPU CLOCKS before running (see run_bench.sh) to kill thermal drift.

Output contract: writes {"score": float, ...} to $EVO_RESULT_PATH and a
per-shape trace to $EVO_TRACES_DIR/task_<shape>.json. Exit 0 on success;
non-zero is an infrastructure failure (evo discards, does not score).
"""
import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

DUMMY_TOKEN_ID = 100  # arbitrary in-vocab id; content is irrelevant to decode rate


def atomic_write_json(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, str(p))


def time_generate(llm, token_prompts, gen_len):
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=gen_len, min_tokens=gen_len,
                        ignore_eos=True, temperature=0.0)
    t0 = time.perf_counter()
    llm.generate(token_prompts, sp, use_tqdm=False)
    return time.perf_counter() - t0


def build_token_prompts(prompt_len, batch):
    # vLLM accepts {"prompt_token_ids": [...]} dicts for exact-length prompts.
    return [{"prompt_token_ids": [DUMMY_TOKEN_ID] * prompt_len} for _ in range(batch)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="", help="kernel file under optimization (informational)")
    ap.add_argument("--worktree", default="", help="experiment worktree path (informational)")
    ap.add_argument("--model", default=os.environ.get("SARVAM_MODEL_PATH", "sarvamai/sarvam-30b"))
    ap.add_argument("--quantization", default=os.environ.get("SARVAM_QUANT", "fp8"))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("SARVAM_MAX_MODEL_LEN", "8192")))
    ap.add_argument("--workloads", default=str(Path(__file__).with_name("workloads.json")))
    args = ap.parse_args()

    cfg = json.loads(Path(args.workloads).read_text())
    warmup, iters, shapes = cfg["warmup"], cfg["iters"], cfg["shapes"]

    from vllm import LLM

    quant = None if args.quantization in ("none", "bf16", "") else args.quantization
    llm = LLM(
        model=args.model,
        quantization=quant,
        tensor_parallel_size=1,           # single GPU: isolates kernel time, no NCCL noise
        trust_remote_code=True,
        enforce_eager=False,              # keep CUDA graphs (production-realistic)
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
    )

    traces_dir = os.environ.get("EVO_TRACES_DIR")
    per_shape_median = {}
    for shape in shapes:
        prompts = build_token_prompts(shape["prompt_len"], shape["batch"])
        D = shape["gen_len"]

        for _ in range(warmup):
            time_generate(llm, prompts, D)

        samples = []
        for _ in range(iters):
            tD = time_generate(llm, prompts, D)
            t2D = time_generate(llm, prompts, 2 * D)
            denom = t2D - tD
            if denom <= 0:
                # timing inversion (noise / too-short gen). Skip this sample.
                continue
            samples.append((shape["batch"] * D) / denom)

        if not samples:
            print(f"[bench] shape {shape['name']}: no valid samples", file=sys.stderr)
            sys.exit(2)

        med = statistics.median(samples)
        per_shape_median[shape["name"]] = med
        if traces_dir:
            atomic_write_json(
                os.path.join(traces_dir, f"task_{shape['name']}.json"),
                {"score": med, "samples": samples, "shape": shape, "unit": "decode_tok_s"},
            )
        print(f"[bench] {shape['name']}: median decode {med:.1f} tok/s "
              f"(n={len(samples)})", file=sys.stderr)

    score = math.prod(per_shape_median.values()) ** (1.0 / len(per_shape_median))

    result_path = os.environ.get("EVO_RESULT_PATH")
    payload = {
        "score": score,
        "tasks": per_shape_median,
        "metric": "decode_tok_s_geomean",
        "target": args.target,
    }
    if result_path:
        atomic_write_json(result_path, payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
