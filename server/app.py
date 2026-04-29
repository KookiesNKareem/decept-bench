"""DECEPT-Bench submission server.

Submitters upload a directory of mechanism JSONLs (one per mechanism, in the
canonical format produced by ``src/decept_run.py``). The server scores the
submission against the held-out private test split using the **canonical
scorer** ``decept_score_v1.SCORERS`` — the same code path that produces
the public-leaderboard scores. We do *not* run a parallel regex-based
scoring path; we do *not* trust a submitter-supplied ``refused`` flag.

Test split is loaded at startup from PRIVATE_TEST_PATH; submitters never
see its contents. Each submission is scored on the items that match the
submitter's ``prompt_id`` field with the private split's prompt IDs.

Endpoints:
  GET  /splits/dev    public dev split (for development; returns prompts only)
  POST /submit        upload mechanism JSONLs; get scorecard
  GET  /leaderboard   public leaderboard

Run with: uvicorn app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Use the canonical scorer + recompute path rather than a parallel
# regex implementation. The recompute step re-derives every classifier
# field from the submitter's raw response text, so submitter-supplied
# `deceptive` / `classification` / `collusion` flags cannot game the score.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from decept_score_v1 import SCORERS  # noqa: E402
from decept_recompute import recompute_rows  # noqa: E402


PRIVATE_TEST_PATH = os.environ.get(
    "DECEPT_TEST_PATH", "/data/decept-bench/test")
PUBLIC_DEV_PATH = os.environ.get(
    "DECEPT_DEV_PATH", "/data/decept-bench/dev")
LEADERBOARD_PATH = os.environ.get(
    "DECEPT_LEADERBOARD_PATH", "/data/decept-bench/leaderboard.json")
ALL_MECHS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"]


app = FastAPI(title="DECEPT-Bench Submission Server", version="1.0")


class MechanismFile(BaseModel):
    """One mechanism's JSONL contents — list of per-item rows in the same
    schema produced by src/decept_run.py.
    """
    mechanism: str = Field(..., description="m1..m10")
    rows: List[Dict]


class Submission(BaseModel):
    model_name: str
    generation_config: Dict = Field(default_factory=dict)
    submitter: str
    files: List[MechanismFile]


def load_private_split() -> Dict[str, List[Dict]]:
    """Load the private test split, keyed by mechanism. Each mechanism
    file should be ``private_test/<mech>.jsonl`` with the same row schema
    as the dev split.
    """
    out: Dict[str, List[Dict]] = {}
    if not os.path.exists(PRIVATE_TEST_PATH):
        return out
    for fname in sorted(os.listdir(PRIVATE_TEST_PATH)):
        if not fname.endswith(".jsonl"):
            continue
        m = re.match(r"^(m\d{1,2})\.jsonl$", fname)
        if not m:
            continue
        mech = m.group(1)
        with open(os.path.join(PRIVATE_TEST_PATH, fname)) as f:
            out[mech] = [json.loads(l) for l in f]
    return out


@app.get("/")
def root():
    return {
        "service": "DECEPT-Bench Submission Server",
        "version": "1.0",
        "endpoints": ["/splits/dev", "/submit", "/leaderboard",
                       "/submission_template"],
        "mechanisms": ALL_MECHS,
    }


@app.get("/splits/dev")
def get_dev_split():
    """Return the public dev split. Submitters develop against this."""
    items: Dict[str, List[Dict]] = {}
    if not os.path.exists(PUBLIC_DEV_PATH):
        raise HTTPException(404, "dev split not found")
    for fname in sorted(os.listdir(PUBLIC_DEV_PATH)):
        if not fname.endswith(".jsonl"):
            continue
        m = re.match(r"^(m\d{1,2})\.jsonl$", fname)
        if not m:
            continue
        with open(os.path.join(PUBLIC_DEV_PATH, fname)) as f:
            items[m.group(1)] = [json.loads(l) for l in f]
    return {"n_items": sum(len(v) for v in items.values()),
            "by_mechanism": items}


@app.get("/submission_template")
def submission_template():
    """Return the JSON schema submitters should follow."""
    return {
        "schema": {
            "model_name": "your-model-id",
            "generation_config": {"temperature": 0, "max_tokens": 384,
                                    "seed": 0},
            "submitter": "your-name-or-org",
            "files": [
                {"mechanism": "m1",
                 "rows": ["<rows produced by src/decept_run.py>"]},
                {"mechanism": "m2", "rows": ["..."]},
                # ... one entry per mechanism you ran
            ],
        },
        "expected_mechanisms": ALL_MECHS,
        "scorer_path": "src/decept_score_v1.py",
        "notes": (
            "Run src/decept_run.py against the public dev split to "
            "produce the canonical row schema, then submit those rows "
            "(verbatim) here. The server runs decept_score_v1.SCORERS "
            "against your rows joined to the private test items by "
            "prompt key. The server does NOT trust submitter-supplied "
            "'refused' or 'deceptive' flags — those are recomputed by "
            "the canonical scorer."
        ),
    }


def join_to_private_test(mechanism: str, submitter_rows: List[Dict],
                            private_rows: List[Dict]) -> List[Dict]:
    """Filter submitter rows to those whose prompt key matches a private
    test item. The join key is mechanism-specific — for BBH-derived M1/M2/M5
    we use the question 'input' field; for M3/M4/M6/M7/M8/M10 we use 'id';
    for M9 we use 'q'. Returns only rows that join successfully."""
    if not private_rows:
        return []
    if mechanism in ("m1", "m2", "m5"):
        keys = set(r.get("input") or r.get("q1") for r in private_rows)
        out = [r for r in submitter_rows
               if (r.get("input") or r.get("q1")) in keys]
    elif mechanism == "m9":
        keys = set(r["q"] for r in private_rows if "q" in r)
        out = [r for r in submitter_rows
               if r.get("q") in keys or r.get("qid", "").split("_r")[0] in keys]
    else:
        # Hand-crafted mechanisms — match by id
        keys = set(r.get("id") for r in private_rows if r.get("id"))
        out = [r for r in submitter_rows if r.get("id") in keys]
    return out


def score_submission(sub: Submission, private_split: Dict[str, List[Dict]]
                       ) -> Dict:
    """Score a submission with the canonical SCORERS dict.

    Sentinel rows ({"status": "couldnt_be_tested"}) are passed through —
    the scorer handles them and excludes the mechanism from the composite.
    """
    by_mech_results = {}
    for f in sub.files:
        mech = f.mechanism
        if mech not in SCORERS:
            continue
        priv = private_split.get(mech, [])
        # If no private split available for this mechanism, skip it
        # (we don't expose dev scores via /submit; keep that on the public
        # leaderboard side).
        if not priv:
            continue
        rows = join_to_private_test(mech, f.rows, priv)
        if not rows:
            by_mech_results[mech] = {"status": "no_overlap_with_private",
                                       "n_submitted": len(f.rows)}
            continue
        # Canonical recompute: re-derive deceptive / classification fields
        # from the submitter's raw responses + the private items, so the
        # submitter cannot game the scorer by supplying false flags.
        rows = recompute_rows(rows, mech)
        result = SCORERS[mech](rows)
        result["n_used"] = len(rows)
        by_mech_results[mech] = result

    # Composite: mean of mechanisms that produced a numeric `deception`
    decs = [r["deception"] for r in by_mech_results.values()
            if isinstance(r, dict) and "deception" in r]
    composite_raw = float(np.mean(decs)) if decs else 0.0
    return {
        "per_mechanism": by_mech_results,
        "composite_raw": composite_raw,
        "n_submitted_files": len(sub.files),
        "n_scored_mechanisms": len(decs),
        "couldnt_be_tested": [m for m, r in by_mech_results.items()
                                if isinstance(r, dict)
                                and r.get("status") == "couldnt_be_tested"],
    }


@app.post("/submit")
def submit(sub: Submission):
    private_split = load_private_split()
    if not private_split:
        raise HTTPException(503, "private test split not loaded on server")
    result = score_submission(sub, private_split)
    entry = {
        "submission_id": hashlib.sha256(
            f"{sub.model_name}_{time.time()}".encode()).hexdigest()[:12],
        "model_name": sub.model_name,
        "submitter": sub.submitter,
        "timestamp": time.time(),
        "result": result,
    }
    leaderboard = []
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH) as f:
            leaderboard = json.load(f)
    leaderboard.append(entry)
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(leaderboard, f, indent=2)
    return entry


@app.get("/leaderboard")
def leaderboard():
    if not os.path.exists(LEADERBOARD_PATH):
        return {"entries": []}
    with open(LEADERBOARD_PATH) as f:
        entries = json.load(f)
    entries.sort(key=lambda e: e["result"]["composite_raw"])
    return {"entries": entries}
