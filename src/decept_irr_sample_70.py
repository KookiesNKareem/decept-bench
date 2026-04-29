"""70-row scorer-author concordance sampler.

Stratified across all 7 ratable mechanisms (M1, M3, M4, M5, M6, M7, M10),
10 rows each. Pulls rows from MULTIPLE models so per-mechanism class
balance is achievable (Sonnet alone has too few deceptive M1/M3 rows
for a 50/50 stratum). Within each mechanism, sample is dedup-by-item-key
and stratified ~50/50 deceptive/non-deceptive.

Outputs:
  - data/irr/concordance_70.csv          (CSV with empty human_label / notes)
  - data/irr/concordance_70_judge.json   (canonical scorer's labels — DO NOT
                                            open until done labeling)

After labeling: python src/decept_irr_score_blind.py \
  --csv data/irr/concordance_70.csv \
  --judge_json data/irr/concordance_70_judge.json
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
    JUDGE_AXES, fmt_prompt_response, is_deceptive, _row_item_key,
)

# Multi-model source pool — broader than the original sampler so M1/M3
# strata can fill 50/50.
DEFAULT_MODEL_DIRS = [
    "data/sonnet", "data/haiku", "data/openai_gpt54mini",
    "data/openai_gpt55", "data/open",
]


def collect_rows(mech: str, model_dirs: List[str]) -> List[Dict]:
    """Pull all rows for `mech` across all listed model dirs.
    Tags each row with the source model slug for traceability.
    """
    out = []
    for d in model_dirs:
        for path in sorted(Path(d).glob(f"*__{mech}.jsonl")):
            slug = path.name.rsplit(f"__{mech}.jsonl", 1)[0]
            try:
                with path.open() as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if r.get("status"):
                            continue
                        r["_source_model"] = slug
                        out.append(r)
            except FileNotFoundError:
                continue
    return out


def stratified_multi_model_sample(rows: List[Dict], mech: str,
                                       n: int, rng) -> List[Dict]:
    """Sample n rows from a multi-model row pool with ~50/50 deceptive split,
    deduplicating by (model, item_key) so every sampled row is a distinct
    (scenario, model) pair. Falls back to whichever stratum has rows."""
    by_pair: Dict[tuple, List[Dict]] = {}
    for r in rows:
        key = (r.get("_source_model", ""), _row_item_key(r, mech))
        by_pair.setdefault(key, []).append(r)
    items: List[Dict] = []
    for reps in by_pair.values():
        rng.shuffle(reps)
        d_rep = next((r for r in reps if is_deceptive(r, mech)), None)
        items.append(d_rep if d_rep is not None else reps[0])
    dec = [r for r in items if is_deceptive(r, mech)]
    nondec = [r for r in items if not is_deceptive(r, mech)]
    rng.shuffle(dec); rng.shuffle(nondec)
    half = n // 2
    out = dec[:half] + nondec[:n - half]
    if len(out) < n:
        deficit = n - len(out)
        pool = dec[half:] + nondec[n - half:]
        rng.shuffle(pool)
        out += pool[:deficit]
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_mech", type=int, default=10)
    ap.add_argument("--out_csv",
                    default="data/irr/concordance_70.csv")
    ap.add_argument("--out_judge_json",
                    default="data/irr/concordance_70_judge.json")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--model_dirs", nargs="+",
                    default=DEFAULT_MODEL_DIRS)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_judge = Path(args.out_judge_json)

    # Cover all 7 ratable mechanisms in JUDGE_AXES (M1, M3, M4, M5, M6, M7, M10)
    mechs = sorted(JUDGE_AXES.keys())
    samples: List[Dict] = []
    judge_map: Dict[str, Dict[str, str]] = {}
    summary: List[str] = []
    for mech in mechs:
        rows = collect_rows(mech, args.model_dirs)
        if not rows:
            print(f"  {mech}: no rows in any source dir, skipping")
            continue
        sample = stratified_multi_model_sample(rows, mech, args.n_per_mech, rng)
        n_dec = sum(1 for r in sample if is_deceptive(r, mech))
        for i, r in enumerate(sample):
            try:
                prompt, response = fmt_prompt_response(r, mech)
            except Exception as e:
                print(f"  warning: fmt failed for {mech} row: {e}")
                continue
            uid = f"{mech}_{i:02d}"
            samples.append({
                "uid": uid,
                "mechanism": mech,
                "source_model": r.get("_source_model", "?"),
                "instruction": JUDGE_AXES[mech]["instruction"],
                "prompt": prompt,
                "response": response,
                "human_label": "",
                "notes": "",
            })
            dec = "D" if is_deceptive(r, mech) else "N"
            judge_map[uid] = {
                "llm_judge_dec": dec,
                "llm_judge_label": str(r.get(JUDGE_AXES[mech]["label_field"])),
                "source_model": r.get("_source_model", "?"),
            }
        summary.append(f"  {mech}: sampled {len(sample)} (D={n_dec}, "
                          f"N={len(sample) - n_dec})")
        print(summary[-1])

    fields = ["uid", "mechanism", "source_model", "instruction",
                "prompt", "response", "human_label", "notes"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for s in samples:
            w.writerow({k: s[k] for k in fields})
    with out_judge.open("w") as f:
        json.dump(judge_map, f, indent=2)

    print(f"\nwrote {out_csv} ({len(samples)} samples)")
    print(f"wrote {out_judge}  (DO NOT OPEN until done labeling)")
    print()
    print("Labeling instructions:")
    print(f"  1. Open {out_csv} in Numbers / Excel / Google Sheets.")
    print("  2. For each row read 'instruction', 'prompt', 'response'.")
    print("  3. Fill 'human_label' with D / N / U  (D = deceptive, "
            "N = non-deceptive, U = unsure / can't tell).")
    print("  4. Save the CSV in place.")
    print("  5. Run:")
    print(f"       python src/decept_irr_score_blind.py \\")
    print(f"         --csv {out_csv} --judge_json {out_judge}")
    print()
    print("Time estimate: ~90-120 min for 70 rows. M3 / M10 fast (~1 min "
            "each); M1 / M6 slow (~2 min each).")


if __name__ == "__main__":
    main()
