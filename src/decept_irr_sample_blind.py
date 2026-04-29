"""Blind IRR sampler: produces a CSV the rater can label without seeing
the LLM judge's call. Stores the judge labels in a separate JSON the
scorer joins back at evaluation time.

Defaults: 6 rows per mechanism (M3, M4, M6, M7, M10) = 30 rows total,
stratified 50/50 LLM-judged-deceptive / not.

Output:
  - data/irr/sonnet_irr_30.csv          (CSV with empty llm columns)
  - data/irr/sonnet_irr_30_judge.json   (judge labels, joined on item_uid)

After labeling: python src/decept_irr_score_blind.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decept_irr_sample import (
    JUDGE_AXES, fmt_prompt_response, is_deceptive, stratified_sample,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--n_per_mech", type=int, default=6)
    ap.add_argument("--out_csv", default="data/irr/sonnet_irr_30.csv")
    ap.add_argument("--out_judge_json",
                    default="data/irr/sonnet_irr_30_judge.json")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    in_dir = Path(args.model_dir)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_judge = Path(args.out_judge_json)

    samples: List[Dict] = []
    judge_map: Dict[str, str] = {}
    for mech in sorted(JUDGE_AXES.keys()):
        files = list(in_dir.glob(f"*__{mech}.jsonl"))
        if not files:
            print(f"  {mech}: no jsonl in {in_dir}, skipping")
            continue
        rows = [json.loads(l) for l in files[0].open()]
        rows = [r for r in rows if not r.get("status")]
        if not rows:
            continue
        sample = stratified_sample(rows, mech, args.n_per_mech, rng)
        for i, r in enumerate(sample):
            prompt, response = fmt_prompt_response(r, mech)
            uid = f"{mech}_{i}"
            samples.append({
                "uid": uid,
                "mechanism": mech,
                "instruction": JUDGE_AXES[mech]["instruction"],
                "prompt": prompt,
                "response": response,
                "human_label": "",
                "notes": "",
            })
            # Keep judge label aside
            dec = "D" if is_deceptive(r, mech) else "N"
            judge_map[uid] = {"llm_judge_dec": dec,
                                "llm_judge_label": str(r.get(
                                    JUDGE_AXES[mech]["label_field"]))}
        print(f"  {mech}: sampled {len(sample)}")

    fields = ["uid", "mechanism", "instruction", "prompt", "response",
              "human_label", "notes"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for s in samples:
            w.writerow({k: s[k] for k in fields})
    with out_judge.open("w") as f:
        json.dump(judge_map, f, indent=2)

    print(f"\nwrote {out_csv} ({len(samples)} samples)")
    print(f"wrote {out_judge} (judge labels — DO NOT OPEN until done labeling)")
    print()
    print("Labeling instructions:")
    print(f"  1. Open {out_csv} in a spreadsheet (Numbers / Excel / Google Sheets).")
    print("  2. For each row read 'instruction', 'prompt', 'response'.")
    print("  3. Fill 'human_label' with D / N / U.")
    print("  4. Save the CSV in place.")
    print("  5. Run: python src/decept_irr_score_blind.py")
    print()
    print("Time estimate: ~45-60 min for 30 rows. M3/M10 fastest, M6 slowest.")


if __name__ == "__main__":
    main()
