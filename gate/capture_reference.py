#!/usr/bin/env python3
"""Capture the baseline's greedy generations once, to anchor the accuracy gate.

Run this ONCE on the baseline (unmodified) build, inside the baseline worktree,
before any optimization. It records, for each fixed prompt, the greedy token
sequence and the chosen-token logprob at each position. verify_quality.py later
re-runs the candidate build on the same prompts and checks it stayed within
tolerance of this reference.

Why generations + logprobs rather than raw logits: vLLM exposes logprobs
cleanly but not full logit tensors. Top-1 token agreement plus chosen-token
logprob drift is a faithful, engine-native proxy for "the kernel rewrite did
not change what the model says."
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SARVAM_MODEL_PATH", "sarvamai/sarvam-30b"))
    ap.add_argument("--quantization", default=os.environ.get("SARVAM_QUANT", "fp8"))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("SARVAM_MAX_MODEL_LEN", "8192")))
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] / "reference" / "prompts.jsonl"))
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "reference" / "baseline_gen.json"))
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    prompts = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    quant = None if args.quantization in ("none", "bf16", "") else args.quantization
    llm = LLM(model=args.model, quantization=quant, tensor_parallel_size=1,
              trust_remote_code=True, enforce_eager=True,  # eager = most reproducible
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              disable_log_stats=True)

    sp = SamplingParams(max_tokens=args.gen_len, min_tokens=args.gen_len,
                        ignore_eos=True, temperature=0.0, logprobs=1)
    outs = llm.generate([p["prompt"] for p in prompts], sp, use_tqdm=False)

    ref = {"model": args.model, "quantization": args.quantization, "gen_len": args.gen_len, "items": {}}
    for p, o in zip(prompts, outs):
        comp = o.outputs[0]
        chosen_logprobs = []
        for tok_id, lp_dict in zip(comp.token_ids, comp.logprobs or []):
            lp = lp_dict.get(tok_id)
            chosen_logprobs.append(getattr(lp, "logprob", None) if lp is not None else None)
        ref["items"][p["id"]] = {"token_ids": list(comp.token_ids), "logprobs": chosen_logprobs}

    Path(args.out).write_text(json.dumps(ref))
    print(f"[capture] wrote {len(ref['items'])} reference generations to {args.out}")


if __name__ == "__main__":
    main()
