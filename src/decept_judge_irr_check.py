"""Validate the LLM judge against human labels on the 70-row sample.

Computes Cohen's kappa between the LLM judge labels (produced by
``decept_judge_score.py --csv``) and the human labels in the
scorer-author concordance CSV. Reports per-mechanism + overall κ with
bootstrap 95% CI.

Used as a calibration gate: if the LLM judge does not substantially
agree with held-out human labels, the rubric or judge model needs work
before running on the full panel.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def cohens_kappa(human: list, judge: list) -> float:
    """Cohen's kappa for binary D/N labels."""
    if len(human) < 2:
        return float("nan")
    h = np.array([1 if x == "D" else 0 for x in human])
    j = np.array([1 if x == "D" else 0 for x in judge])
    n = len(h)
    po = float((h == j).mean())
    p_h_d = float(h.mean()); p_j_d = float(j.mean())
    pe = p_h_d * p_j_d + (1 - p_h_d) * (1 - p_j_d)
    if abs(pe - 1.0) < 1e-9:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def bootstrap_kappa_ci(human: list, judge: list, n_boot: int = 2000,
                          seed: int = 0) -> tuple:
    if len(human) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(human)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        h = [human[i] for i in idx]
        j = [judge[i] for i in idx]
        try:
            k = cohens_kappa(h, j)
            if not np.isnan(k):
                ks.append(k)
        except Exception:
            continue
    if not ks:
        return (float("nan"), float("nan"))
    return (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/irr/concordance_70.csv")
    ap.add_argument("--judge_json", default="data/judge/irr_70_judge.json")
    ap.add_argument("--out_json", default="data/judge/irr_70_kappa.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    judge = json.load(open(args.judge_json))

    by_mech = defaultdict(lambda: ([], []))  # mech -> (human, judge)
    overall_h = []; overall_j = []
    n_unparseable = 0
    for r in rows:
        uid = r["uid"]; h = r["human_label"].strip().upper()
        if h not in ("D", "N"):  # skip U / unlabeled
            continue
        if uid not in judge:
            continue
        jl = judge[uid]["judge_label"]
        if jl not in ("D", "N"):
            n_unparseable += 1
            continue
        by_mech[r["mechanism"]][0].append(h)
        by_mech[r["mechanism"]][1].append(jl)
        overall_h.append(h); overall_j.append(jl)

    out = {}
    print()
    print(f"{'mechanism':<10} {'n':>4} {'agree':>8} {'kappa':>7} {'lo':>7} {'hi':>7}")
    print("-" * 50)
    for mech in sorted(by_mech.keys()):
        h, j = by_mech[mech]
        n = len(h)
        agreement = sum(1 for a, b in zip(h, j) if a == b) / n if n else 0.0
        kappa = cohens_kappa(h, j)
        lo, hi = bootstrap_kappa_ci(h, j)
        print(f"{mech:<10} {n:>4} {agreement:>8.3f} {kappa:>7.3f} {lo:>7.3f} {hi:>7.3f}")
        out[mech] = {"n": n, "agreement": agreement,
                       "kappa": kappa, "kappa_ci_lo": lo, "kappa_ci_hi": hi}
    n = len(overall_h)
    agreement = sum(1 for a, b in zip(overall_h, overall_j) if a == b) / n if n else 0.0
    kappa = cohens_kappa(overall_h, overall_j)
    lo, hi = bootstrap_kappa_ci(overall_h, overall_j)
    print("-" * 50)
    print(f"{'OVERALL':<10} {n:>4} {agreement:>8.3f} {kappa:>7.3f} {lo:>7.3f} {hi:>7.3f}")
    out["overall"] = {"n": n, "agreement": agreement,
                       "kappa": kappa, "kappa_ci_lo": lo, "kappa_ci_hi": hi,
                       "n_unparseable": n_unparseable}
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"saved {args.out_json}")
    print()
    print("Gate: overall kappa >= 0.70 means the LLM judge is good enough")
    print("to replace the regex classifier on M3/M4/M6/M7/M10.")
    if kappa >= 0.70:
        print(f"  PASS (kappa={kappa:.3f}). Proceed to full panel.")
    else:
        print(f"  FAIL (kappa={kappa:.3f}). Tighten the rubric and rerun.")


if __name__ == "__main__":
    main()
