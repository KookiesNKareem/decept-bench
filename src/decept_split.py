"""Split DECEPT-Bench mechanism JSONLs into public_dev (70%) and
private_test (30%) partitions.

Public dev is released; private test is held back. Submission server scores
against private test; only aggregate scores leave the server.

Splits are deterministic (seed=42). Per-mechanism, per-task to keep BBH-
stratified mechanisms (M1, M2, M5) balanced across both splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def assignment(row: dict, frac_test: float, salt: str) -> str:
    """Deterministically assign a row to dev/test based on a stable hash.
    Uses the row's 'input' or 'scenario' or 'prompt' field as the natural
    identifier. frac_test is the fraction sent to private test."""
    key = (row.get("input") or row.get("scenario") or row.get("prompt") or
           row.get("question") or row.get("task") or json.dumps(row, sort_keys=True))
    h = stable_hash(salt + "::" + key)
    return "test" if (h % 1000) < int(frac_test * 1000) else "dev"


def split_file(path: Path, out_root: Path, frac_test: float, salt: str):
    rows = [json.loads(l) for l in path.open()]
    if not rows: return
    # sentinel rows (e.g. couldnt_be_tested) go to BOTH splits unchanged
    sentinels = [r for r in rows if r.get("status")]
    real_rows = [r for r in rows if not r.get("status")]

    split = defaultdict(list)
    for r in real_rows:
        split[assignment(r, frac_test, salt)].append(r)
    for s in ("dev", "test"):
        split[s].extend(sentinels)

    for s, rs in split.items():
        out_dir = out_root / s
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / path.name
        with out.open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    print(f"  {path.name}: dev={len(split['dev'])} test={len(split['test'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                    help="directory containing <model>__<mech>.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frac_test", type=float, default=0.3)
    ap.add_argument("--salt", default="decept-v1")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir)
    paths = sorted(in_dir.glob("*__m*.jsonl"))
    print(f"splitting {len(paths)} files (frac_test={args.frac_test}, salt={args.salt!r})")
    for p in paths:
        split_file(p, out_root, args.frac_test, args.salt)
    print(f"\ndev/  -> {out_root/'dev'}")
    print(f"test/ -> {out_root/'test'}")


if __name__ == "__main__":
    main()
