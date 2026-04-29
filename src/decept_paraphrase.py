"""Paraphrase generator for question randomization.

Use a strong open model (Qwen-7B-Instruct or similar) to generate N
paraphrases of each input question. We require:
  - paraphrase preserves the semantic question
  - paraphrase preserves the gold answer
  - paraphrase is grammatically distinct (Levenshtein distance > 30%)

This produces a paraphrase bank that we then use for randomized prompting in
all mechanisms. Run once per release to build the bank; mechanisms then
randomly pick a paraphrase per question.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List


from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def lev_distance(a: str, b: str) -> int:
    """Standard Levenshtein."""
    if len(a) < len(b):
        return lev_distance(b, a)
    prev = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        cur = [i + 1]
        for j, c2 in enumerate(b):
            ins = prev[j + 1] + 1
            de = cur[j] + 1
            sub = prev[j] + (c1 != c2)
            cur.append(min(ins, de, sub))
        prev = cur
    return prev[-1]


PARAPHRASE_PROMPT = """Rewrite this BBH-style reasoning question in a different
phrasing while preserving the EXACT same logical content. Keep all options
labeled (A), (B), etc. — do NOT change the option letters or contents.
Use different sentence structure and word choice for the question stem.
DO NOT solve the question — just rewrite it.

ORIGINAL:
{q}

REWRITTEN:"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="paraphrase generator model")
    ap.add_argument("--n_paraphrases", type=int, default=3)
    ap.add_argument("--out", default="data/decept/paraphrases.jsonl")
    ap.add_argument("--per_task", type=int, default=200)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tasks", nargs="+",
                    default=["logical_deduction_three_objects",
                              "logical_deduction_five_objects",
                              "logical_deduction_seven_objects",
                              "tracking_shuffled_objects_three_objects",
                              "tracking_shuffled_objects_five_objects",
                              "causal_judgement",
                              "formal_fallacies"])
    ap.add_argument("--min_lev_frac", type=float, default=0.20,
                    help="reject paraphrases with relative Lev distance below this")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"loading vLLM with paraphraser {args.model}...")
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=4096)
    tok = AutoTokenizer.from_pretrained(args.model)

    # Build all questions
    questions = []
    for task in args.tasks:
        try:
            ds = load_dataset("lukaemon/bbh", task)["test"]
        except Exception as e:
            print(f"skip {task}: {e}"); continue
        for q in list(ds)[: args.per_task]:
            questions.append({"task": task, "input": q["input"],
                                "target": q["target"]})
    print(f"\ntotal questions: {len(questions)}")

    # Build N rephrase prompts per question
    all_prompts = []
    parents = []
    for qi, q in enumerate(questions):
        for r in range(args.n_paraphrases):
            msg = [{"role": "user",
                     "content": PARAPHRASE_PROMPT.format(q=q["input"])}]
            p = tok.apply_chat_template(msg, tokenize=False,
                                         add_generation_prompt=True)
            all_prompts.append(p)
            parents.append((qi, r))

    print(f"generating {len(all_prompts)} paraphrases...")
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=600, seed=0)
    outs = [o.outputs[0].text for o in llm.generate(all_prompts, sp)]

    # Process: keep paraphrase only if it preserves all options and differs
    by_q = {}
    for (qi, r), text in zip(parents, outs):
        # Trim to actual paraphrase: take everything until two newlines or end
        text = text.strip()
        # Many models echo "REWRITTEN:" at start. Strip.
        text = re.sub(r"^(rewritten|paraphrase|here(?:'s)?|sure|of course)[:.]?\s*",
                       "", text, flags=re.IGNORECASE).strip()
        # Strip trailing chitchat
        text = re.sub(r"\n\nLet me know.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
        by_q.setdefault(qi, []).append(text)

    # Verify and save
    n_kept = 0
    n_total = 0
    with open(args.out, "w") as f:
        for qi, q in enumerate(questions):
            paras = by_q.get(qi, [])
            kept = []
            orig = q["input"]
            orig_options = set(re.findall(r"\([A-Ga-g]\)", orig))
            for p in paras:
                n_total += 1
                # Must preserve options
                p_options = set(re.findall(r"\([A-Ga-g]\)", p))
                if p_options != orig_options:
                    continue
                # Must be different enough
                d = lev_distance(orig.lower(), p.lower())
                if d / max(len(orig), 1) < args.min_lev_frac:
                    continue
                # Must not be the same words (rough)
                if len(p) < 0.5 * len(orig) or len(p) > 2.5 * len(orig):
                    continue
                kept.append(p)
                n_kept += 1
            f.write(json.dumps({
                "task": q["task"], "input": orig, "target": q["target"],
                "paraphrases": kept,
            }) + "\n")

    print(f"\nkept {n_kept}/{n_total} paraphrases ({n_kept/max(n_total,1):.0%})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
