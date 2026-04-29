"""Compute Cohen's kappa between LLM-as-judge labels and human-rater
labels from a filled IRR CSV produced by decept_irr_sample.py.

Reports per-mechanism kappa + 95% bootstrap CI, plus an overall kappa
across all mechanisms. Treats 'U' (unclear) human labels as missing.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def cohens_kappa(rater1: list, rater2: list) -> float:
    """Cohen's κ on two binary 0/1 lists (same length)."""
    a = np.asarray(rater1, dtype=int)
    b = np.asarray(rater2, dtype=int)
    n = len(a)
    if n == 0:
        return float("nan")
    p_obs = float((a == b).mean())
    p_a1 = float(a.mean()); p_b1 = float(b.mean())
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if p_e == 1.0:
        return 1.0 if p_obs == 1.0 else 0.0
    return (p_obs - p_e) / (1 - p_e)


def boot_kappa(rater1, rater2, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(0)
    n = len(rater1)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    samples = np.empty(n_boot)
    a = np.asarray(rater1); b = np.asarray(rater2)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[i] = cohens_kappa(a[idx].tolist(), b[idx].tolist())
    return {"mean": float(cohens_kappa(rater1, rater2)),
            "lo": float(np.quantile(samples, 0.025)),
            "hi": float(np.quantile(samples, 0.975)),
            "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    by_mech = defaultdict(lambda: {"llm": [], "human": [],
                                     "skipped": 0})
    n_total = 0; n_unclear = 0
    with Path(args.csv).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            human = row.get("human_label", "").strip().upper()
            llm = row.get("llm_judge_dec", "").strip().upper()
            if human in ("", "U") or llm not in ("D", "N"):
                by_mech[row["mechanism"]]["skipped"] += 1
                if human == "U":
                    n_unclear += 1
                continue
            if human not in ("D", "N"):
                by_mech[row["mechanism"]]["skipped"] += 1
                continue
            by_mech[row["mechanism"]]["llm"].append(1 if llm == "D" else 0)
            by_mech[row["mechanism"]]["human"].append(1 if human == "D" else 0)

    print(f"\nTotal rows in CSV: {n_total}  (unclear: {n_unclear})")
    print()
    print(f"{'mechanism':<10} {'n':>4} {'agreement':>10} {'kappa':>7} "
          f"{'lo':>7} {'hi':>7}")
    print("-" * 50)
    overall_llm, overall_human = [], []
    out_data = {}
    for mech in sorted(by_mech):
        data = by_mech[mech]
        n = len(data["llm"])
        if n == 0:
            print(f"{mech:<10} {n:>4} (no labeled rows; skipped {data['skipped']})")
            continue
        agree = float(np.mean(np.asarray(data["llm"]) == np.asarray(data["human"])))
        ci = boot_kappa(data["llm"], data["human"])
        print(f"{mech:<10} {n:>4} {agree:>10.3f} {ci['mean']:>7.3f} "
              f"{ci['lo']:>7.3f} {ci['hi']:>7.3f}")
        out_data[mech] = {"n": n, "agreement": agree,
                            "kappa": ci["mean"], "kappa_ci_lo": ci["lo"],
                            "kappa_ci_hi": ci["hi"], "skipped": data["skipped"]}
        overall_llm.extend(data["llm"]); overall_human.extend(data["human"])

    if overall_llm:
        n = len(overall_llm)
        agree = float(np.mean(np.asarray(overall_llm) == np.asarray(overall_human)))
        ci = boot_kappa(overall_llm, overall_human)
        print("-" * 50)
        print(f"{'OVERALL':<10} {n:>4} {agree:>10.3f} {ci['mean']:>7.3f} "
              f"{ci['lo']:>7.3f} {ci['hi']:>7.3f}")
        out_data["overall"] = {"n": n, "agreement": agree,
                                  "kappa": ci["mean"], "kappa_ci_lo": ci["lo"],
                                  "kappa_ci_hi": ci["hi"]}

    if args.out_json:
        import json
        with open(args.out_json, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nsaved {args.out_json}")

    # Interpretation hint
    print()
    print("Cohen's κ interpretation (Landis & Koch 1977):")
    print("  < 0.00  : poor")
    print("  0.00-0.20: slight")
    print("  0.21-0.40: fair")
    print("  0.41-0.60: moderate")
    print("  0.61-0.80: substantial")
    print("  0.81-1.00: almost perfect")


if __name__ == "__main__":
    main()
