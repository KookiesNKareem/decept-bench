"""Package DECEPT-Bench prompt items as a HuggingFace dataset.

Reads canonical prompt sources (BBH selections, M3-M10 hardcoded items,
paraphrase bank) and emits one parquet file per mechanism plus a top-level
dataset_info.json. Output goes to data/hf_dataset/ ready for upload via
huggingface-cli upload.

The PROMPTS are public; model OUTPUTS (data/<model>/*.jsonl) are NOT part
of the published dataset — those are in the leaderboard repo.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Reuse the canonical mechanism content
sys.path.insert(0, str(Path(__file__).parent))
from decept_mechanisms import (
    PERSONA_ITEMS, M4_ITEMS, M6_ITEMS, M7_ITEMS, M9_ITEMS, M10_ITEMS,
    load_bbh,
)
from decept_split import assignment as split_assignment

# Deterministic, non-secret split salt — items go to public dev (~70%) or
# private test (~30%). The salt is part of the spec and is published.
SPLIT_SALT = "decept-v1"
SPLIT_FRAC_TEST = 0.3


def _row_split(row: Dict) -> str:
    """Deterministic dev/test assignment for an item dict."""
    return split_assignment(row, SPLIT_FRAC_TEST, SPLIT_SALT)


BBH_TASKS = [
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "causal_judgement",
    "formal_fallacies",
]


def m1_items(per_task: int = 40) -> List[Dict]:
    qs = load_bbh(BBH_TASKS, per_task=per_task)
    return [{"id": f"m1_{i:04d}", "task": q["task"], "input": q["input"],
             "gold": q["target"], "mechanism": "m1"}
            for i, q in enumerate(qs)]


def m2_items(per_task: int = 40) -> List[Dict]:
    qs = load_bbh(BBH_TASKS, per_task=per_task)
    return [{"id": f"m2_{i:04d}", "task": q["task"], "input": q["input"],
             "gold": q["target"], "mechanism": "m2"}
            for i, q in enumerate(qs)]


def m3_items() -> List[Dict]:
    return [{"id": f"m3_{i:04d}", "topic": it["topic"],
             "question": it["question"],
             "persona": it["persona"],
             "honest_yes": it["honest_yes"],
             "mechanism": "m3"}
            for i, it in enumerate(PERSONA_ITEMS)]


def m4_items() -> List[Dict]:
    return [{"id": it.get("id", f"m4_{i:04d}"),
             "system": it["system"],
             "user_query": it["user_query"],
             "secret_keywords": ",".join(it.get("secret_keywords", [])),
             "mechanism": "m4"}
            for i, it in enumerate(M4_ITEMS)]


def m5_items(per_task: int = 60) -> List[Dict]:
    qs = load_bbh(BBH_TASKS, per_task=per_task)
    return [{"id": f"m5_{i:04d}", "task": q["task"], "input": q["input"],
             "gold": q["target"], "mechanism": "m5"}
            for i, q in enumerate(qs)]


def m6_items() -> List[Dict]:
    return [{"id": f"m6_{i:04d}", "tool": it["tool"], "question": it["q"],
             "expected": str(it.get("expected", "")),
             "mechanism": "m6"}
            for i, it in enumerate(M6_ITEMS)]


def m7_items() -> List[Dict]:
    return [{"id": it.get("id", f"m7_{i:04d}"),
             "false_claim": it["false_claim"],
             "truth": it["truth"],
             "mechanism": "m7"}
            for i, it in enumerate(M7_ITEMS)]


def m9_items() -> List[Dict]:
    return [{"id": f"m9_{i:04d}",
             "category": it["category"],
             "question": it["q"],
             "match": it.get("match", ""),
             "mechanism": "m9"}
            for i, it in enumerate(M9_ITEMS)]


def m10_items() -> List[Dict]:
    return [{"id": it.get("id", f"m10_{i:04d}"),
             "benign": it["benign"],
             "restricted": it["restricted"],
             "verifier_substring": it["verifier_substring"],
             "mechanism": "m10"}
            for i, it in enumerate(M10_ITEMS)]


def m8_items() -> List[Dict]:
    from decept_m8_code import ITEMS
    return [{"id": f"m8_{i:04d}", "problem": it["problem"],
             "correct": it["correct"], "buggy": it["buggy"],
             "bug_kind": it.get("bug_kind", ""),
             "mechanism": "m8"}
            for i, it in enumerate(ITEMS)]


MECHANISMS = {
    "m1": m1_items, "m2": m2_items, "m3": m3_items, "m4": m4_items,
    "m5": m5_items, "m6": m6_items, "m7": m7_items, "m8": m8_items,
    "m9": m9_items, "m10": m10_items,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/hf_dataset")
    ap.add_argument("--paraphrases_path", default="data/paraphrases.jsonl")
    ap.add_argument("--include_test", action="store_true",
                    help="WARNING: include held-out test rows in the output "
                          "parquet. Default is dev-only — required for any "
                          "public upload, since the test split is served "
                          "only by the FastAPI submission server. Set this "
                          "ONLY for internal builds you will not upload.")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    info = {"name": "decept-bench-v1",
            "version": "1.0.0",
            "license": "apache-2.0",
            "description": "Multi-mechanism deception evaluation suite for LMs.",
            "mechanisms": {}}

    for mech, fn in MECHANISMS.items():
        try:
            rows = fn()
        except Exception as e:
            print(f"  {mech}: skipped ({e})")
            continue
        # Assign deterministic split to each row.
        for r in rows:
            r["split"] = _row_split(r)
        n_dev = sum(1 for r in rows if r["split"] == "dev")
        n_test = sum(1 for r in rows if r["split"] == "test")
        if not args.include_test:
            # Default: drop test rows so the published parquet cannot
            # leak the held-out split. The test split is served only by
            # the FastAPI submission server.
            rows = [r for r in rows if r["split"] == "dev"]
        df = pd.DataFrame(rows)
        path = out / f"{mech}.parquet"
        df.to_parquet(path)
        info["mechanisms"][mech] = {"n_total": n_dev + n_test,
                                       "n_dev": n_dev, "file": path.name}
        if args.include_test:
            info["mechanisms"][mech]["n_test_internal_only"] = n_test
            print(f"  {mech}: {n_dev + n_test} items "
                  f"(dev={n_dev}, test={n_test}) -> {path}  [INCLUDES TEST]")
        else:
            print(f"  {mech}: {n_dev} dev items -> {path}  "
                  f"(test split held out; n_test={n_test})")

    # Paraphrase bank
    pp = Path(args.paraphrases_path)
    if pp.exists():
        rows = [json.loads(l) for l in pp.open()]
        df = pd.DataFrame(rows)
        df.to_parquet(out / "paraphrases.parquet")
        info["paraphrases"] = {"n": len(rows), "file": "paraphrases.parquet"}
        print(f"  paraphrases: {len(rows)} entries")

    info["includes_test_split"] = bool(args.include_test)
    with (out / "dataset_info.json").open("w") as f:
        json.dump(info, f, indent=2)
    print(f"\nwrote {out}/dataset_info.json")
    if args.include_test:
        print("\n!! WARNING !!  --include_test was set: the parquet contains "
              "test-split rows. Do NOT upload this build to a public registry; "
              "the held-out test split is served only by the FastAPI server.")
    else:
        print("upload with: huggingface-cli upload decept-bench/decept-v1 "
              "data/hf_dataset")


if __name__ == "__main__":
    main()
