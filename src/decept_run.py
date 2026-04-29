"""Canonical DECEPT-Bench runner. Backend dispatched by model name.

  - claude-*               → Anthropic API
  - gpt-* / o-*            → OpenAI API (auto-detects reasoning models)
  - any HF org/repo path   → vLLM (GPU) by default, or HF transformers
                              fallback if vLLM cannot serve the architecture

Runs all 9 multi-mechanism tests (M1-M7, M9, M10). M8 has its own
companion runner at decept_m8.py because it interleaves Python execution
with model calls.

Usage:
  ANTHROPIC_API_KEY=… python src/decept_run.py --model claude-haiku-4-5 \\
      --out_dir data/haiku
  OPENAI_API_KEY=…    python src/decept_run.py --model gpt-5.4-mini \\
      --out_dir data/gpt54mini --reasoning_effort high
  python src/decept_run.py --model Qwen/Qwen2.5-7B-Instruct \\
      --out_dir data/qwen7b
  python src/decept_run.py --model google/gemma-4-E4B-it \\
      --out_dir data/gemma4 --backend hf
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from decept_mechanisms import (
    run_m1, run_m2, run_m3, run_m4, run_m5, run_m6, run_m7, run_m9, run_m10,
    load_bbh, apply_paraphrases,
)


def detect_backend(model: str) -> str:
    """Default backend dispatch — overridable via --backend."""
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o3", "o4", "o-")):
        return "openai"
    return "vllm"


def make_client(args):
    backend = args.backend or detect_backend(args.model)
    if backend == "anthropic":
        from decept_api_client import AnthropicBatchClient
        return AnthropicBatchClient(model=args.model,
                                       max_concurrency=args.concurrency)
    if backend == "openai":
        from decept_openai_client import OpenAIBatchClient
        return OpenAIBatchClient(model=args.model,
                                    max_concurrency=args.concurrency,
                                    reasoning_effort=args.reasoning_effort)
    if backend == "vllm":
        from decept_vllm_client import VLLMBatchClient, clean_dsr1
        post = clean_dsr1 if "deepseek-r1" in args.model.lower() else None
        return VLLMBatchClient(model=args.model, gpu_mem=args.gpu_mem,
                                  post_process=post)
    if backend == "hf":
        from decept_hf_client import HFBatchClient
        return HFBatchClient(model=args.model, batch_size=args.hf_batch_size)
    raise ValueError(f"unknown backend: {backend}")


async def amain(args):
    rng = random.Random(args.seed)
    client = make_client(args)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace(".", "-").replace("/", "_").lower()

    bbh_qs = load_bbh(["logical_deduction_three_objects",
                        "logical_deduction_five_objects",
                        "logical_deduction_seven_objects",
                        "tracking_shuffled_objects_three_objects",
                        "tracking_shuffled_objects_five_objects",
                        "causal_judgement",
                        "formal_fallacies"], per_task=80)
    print(f"BBH questions loaded: {len(bbh_qs)}")
    # Paraphrase ON by default when the bank exists. Submitters who want
    # the original-prompt-only condition can pass --paraphrase_prob 0.
    paraphrase_path = args.paraphrase_path
    if paraphrase_path is None:
        default_bank = (Path(__file__).resolve().parent.parent
                          / "data" / "paraphrases.jsonl")
        if default_bank.exists():
            paraphrase_path = str(default_bank)
    if paraphrase_path and args.paraphrase_prob > 0.0:
        bbh_qs = apply_paraphrases(bbh_qs, paraphrase_path, rng,
                                     prob=args.paraphrase_prob)

    for mech, fn, kwargs in [
        ("m9", run_m9, dict(rng=rng, repeat=args.repeat_small)),
        ("m6", run_m6, dict(rng=rng, repeat=args.repeat_small)),
        ("m10", run_m10, dict(rng=rng, repeat=args.repeat_small)),
        ("m3", run_m3, dict(rng=rng, repeat=args.repeat_small)),
        ("m4", run_m4, dict(rng=rng, repeat=args.repeat_small)),
        ("m7", run_m7, dict(rng=rng, repeat=args.repeat_small)),
        ("m5", run_m5, dict(qs=bbh_qs, rng=rng, per_task=args.per_task_m5)),
        ("m2", run_m2, dict(qs=bbh_qs, rng=rng, per_task=args.per_task_m2)),
        ("m1", run_m1, dict(qs=bbh_qs, rng=rng, per_task=args.per_task_m1)),
    ]:
        out_file = out_dir / f"{slug}__{mech}.jsonl"
        if out_file.exists() and not args.overwrite:
            print(f"  {mech}: already done — skipping")
            continue
        print(f"\n=== {mech} ===")
        t0 = time.time()
        rows = await fn(client, **kwargs)
        elapsed = time.time() - t0
        with open(out_file, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        cost_now = client.cost_estimate_usd()
        print(f"  rows={len(rows)}  elapsed={elapsed:.1f}s  "
              f"cumulative_cost=${cost_now:.4f}  "
              f"in_tok={client.total_input_tok:,}  "
              f"out_tok={client.total_output_tok:,}")

    print(f"\n=== TOTAL COST: ${client.cost_estimate_usd():.4f} ===")
    print(f"=== input tokens: {client.total_input_tok:,} ===")
    print(f"=== output tokens: {client.total_output_tok:,} ===")

    # Auto-score: run the canonical scorer over out_dir so users get
    # scores.json without a second invocation. Idempotent — safe to re-run.
    if not args.skip_score:
        print("\n=== running scorer ===")
        import subprocess
        subprocess.run(
            ["python", str(Path(__file__).parent / "decept_score_v1.py"),
             "--data_dir", str(out_dir), "--out_dir", str(out_dir)],
            check=False,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backend", default=None,
                    choices=[None, "anthropic", "openai", "vllm", "hf"],
                    help="override backend detection (default: pick by model name)")
    ap.add_argument("--per_task_m1", type=int, default=40)
    ap.add_argument("--per_task_m2", type=int, default=40)
    ap.add_argument("--per_task_m5", type=int, default=60)
    ap.add_argument("--repeat_small", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    # Backend-specific (ignored when irrelevant):
    ap.add_argument("--reasoning_effort", default=None,
                    choices=[None, "minimal", "low", "medium", "high", "xhigh"],
                    help="OpenAI reasoning models only")
    ap.add_argument("--gpu_mem", type=float, default=0.85,
                    help="vLLM GPU memory fraction")
    ap.add_argument("--hf_batch_size", type=int, default=4,
                    help="HF transformers batch size (when --backend=hf)")
    # Adversarial:
    ap.add_argument("--paraphrase_path", default=None,
                    help="path to paraphrases.jsonl; if set, BBH inputs "
                          "(M1/M2/M5) get paraphrased per --paraphrase_prob")
    ap.add_argument("--paraphrase_prob", type=float, default=1.0)
    ap.add_argument("--skip_score", action="store_true",
                    help="skip the auto-scoring step at the end")
    args = ap.parse_args()
    asyncio.run(amain(args))
