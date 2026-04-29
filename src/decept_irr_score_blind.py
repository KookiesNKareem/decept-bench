"""Score the blind IRR CSV against the held-out judge JSON.

Computes Cohen's kappa per mechanism and overall, with bootstrap CIs.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from decept_irr_score import boot_kappa, cohens_kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/irr/sonnet_irr_30.csv")
    ap.add_argument("--judge_json", default="data/irr/sonnet_irr_30_judge.json")
    ap.add_argument("--out_json", default="data/irr/sonnet_irr_30_kappa.json")
    args = ap.parse_args()

    with open(args.judge_json) as f:
        judge = json.load(f)
    by_mech = defaultdict(lambda: {"llm": [], "human": [], "skipped": 0})
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uid"]; mech = row["mechanism"]
            human = row.get("human_label", "").strip().upper()
            llm_entry = judge.get(uid, {})
            llm = llm_entry.get("llm_judge_dec", "").strip().upper()
            if human in ("", "U") or llm not in ("D", "N") or human not in ("D", "N"):
                by_mech[mech]["skipped"] += 1
                continue
            by_mech[mech]["llm"].append(1 if llm == "D" else 0)
            by_mech[mech]["human"].append(1 if human == "D" else 0)

    print(f"\n{'mechanism':<10} {'n':>4} {'agreement':>10} {'kappa':>7} "
          f"{'lo':>7} {'hi':>7}")
    print("-" * 50)
    overall_llm, overall_human = [], []
    out = {}
    for mech in sorted(by_mech):
        d = by_mech[mech]
        n = len(d["llm"])
        if n == 0:
            print(f"{mech:<10} {n:>4} (no labeled rows)")
            continue
        agree = float(np.mean(np.asarray(d["llm"]) == np.asarray(d["human"])))
        ci = boot_kappa(d["llm"], d["human"])
        print(f"{mech:<10} {n:>4} {agree:>10.3f} {ci['mean']:>7.3f} "
              f"{ci['lo']:>7.3f} {ci['hi']:>7.3f}")
        out[mech] = {"n": n, "agreement": agree, "kappa": ci["mean"],
                       "kappa_ci_lo": ci["lo"], "kappa_ci_hi": ci["hi"]}
        overall_llm.extend(d["llm"]); overall_human.extend(d["human"])

    if overall_llm:
        n = len(overall_llm)
        agree = float(np.mean(np.asarray(overall_llm) == np.asarray(overall_human)))
        ci = boot_kappa(overall_llm, overall_human)
        print("-" * 50)
        print(f"{'OVERALL':<10} {n:>4} {agree:>10.3f} {ci['mean']:>7.3f} "
              f"{ci['lo']:>7.3f} {ci['hi']:>7.3f}")
        out["overall"] = {"n": n, "agreement": agree, "kappa": ci["mean"],
                            "kappa_ci_lo": ci["lo"], "kappa_ci_hi": ci["hi"]}

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
