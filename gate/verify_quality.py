#!/usr/bin/env python3
"""Accuracy gate: does the candidate build still say the same thing as baseline?

Re-runs the candidate (edited-kernel) build greedily on the fixed prompt set and
compares against reference/baseline_gen.json:

  - top1_agreement: fraction of positions where the candidate's greedy token
    matches the baseline's. A correct kernel rewrite should be ~1.0; float
    reordering may flip a rare token, hence a tolerance below 1.0.
  - max_logprob_drift: largest absolute chosen-token logprob difference at
    agreeing positions. Catches "same token, but the distribution moved."

Exit 0 => gate PASS (within tolerance, experiment may be kept).
Exit 1 => gate FAIL (quality regressed, evo discards even if it got faster).
Exit >1 => infrastructure error.

Tolerances are deliberately strict for the kernel-rewrite surface: we are
claiming "same accuracy", so a divergent-but-faster kernel must NOT pass. Loosen
only with eyes open; the gate is the whole safety story.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="", help="kernel file under optimization (informational)")
    ap.add_argument("--worktree", default="")
    ap.add_argument("--model", default=os.environ.get("SARVAM_MODEL_PATH", "sarvamai/sarvam-30b"))
    ap.add_argument("--quantization", default=os.environ.get("SARVAM_QUANT", "fp8"))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("SARVAM_MAX_MODEL_LEN", "8192")))
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] / "reference" / "prompts.jsonl"))
    ap.add_argument("--reference", default=str(Path(__file__).resolve().parents[1] / "reference" / "baseline_gen.json"))
    ap.add_argument("--min-top1-agreement", type=float,
                    default=float(os.environ.get("GATE_MIN_TOP1", "0.995")))
    ap.add_argument("--max-logprob-drift", type=float,
                    default=float(os.environ.get("GATE_MAX_LOGPROB_DRIFT", "0.10")))
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    prompts = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    gen_len = ref["gen_len"]

    from vllm import LLM, SamplingParams

    quant = None if args.quantization in ("none", "bf16", "") else args.quantization
    llm = LLM(model=args.model, quantization=quant, tensor_parallel_size=1,
              trust_remote_code=True, enforce_eager=True,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              disable_log_stats=True)

    sp = SamplingParams(max_tokens=gen_len, min_tokens=gen_len,
                        ignore_eos=True, temperature=0.0, logprobs=1)
    outs = llm.generate([p["prompt"] for p in prompts], sp, use_tqdm=False)

    total, agree = 0, 0
    max_drift = 0.0
    for p, o in zip(prompts, outs):
        r = ref["items"].get(p["id"])
        if r is None:
            continue
        comp = o.outputs[0]
        cand_ids = list(comp.token_ids)
        cand_lps = []
        for tok_id, lp_dict in zip(comp.token_ids, comp.logprobs or []):
            lp = lp_dict.get(tok_id)
            cand_lps.append(getattr(lp, "logprob", None) if lp is not None else None)
        for i, rid in enumerate(r["token_ids"]):
            if i >= len(cand_ids):
                break
            total += 1
            if cand_ids[i] == rid:
                agree += 1
                rl, cl = r["logprobs"][i], cand_lps[i]
                if rl is not None and cl is not None:
                    max_drift = max(max_drift, abs(rl - cl))

    top1 = agree / total if total else 0.0
    passed = top1 >= args.min_top1_agreement and max_drift <= args.max_logprob_drift
    report = {
        "top1_agreement": round(top1, 5),
        "max_logprob_drift": round(max_drift, 5),
        "positions": total,
        "thresholds": {"min_top1": args.min_top1_agreement, "max_drift": args.max_logprob_drift},
        "pass": passed,
    }
    print(json.dumps(report))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
