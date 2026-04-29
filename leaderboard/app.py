"""DECEPT-Bench leaderboard — Gradio UI.

Reads scores.json files from data/<model>/ and submissions/ and renders a
sortable leaderboard with per-mechanism breakdown.

Deploy as a HuggingFace Space:
  - app_file: leaderboard/app.py
  - sdk: gradio

Local run:
  python leaderboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Mechanism columns in display order
MECH_COLS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"]


def load_all_scores() -> List[Dict]:
    """Discover all data/<model>/scores.json and submissions/*.json."""
    rows = []

    # Scan data/<dir>/scores.json
    for scores_file in sorted((ROOT / "data").glob("*/scores.json")):
        try:
            d = json.load(scores_file.open())
        except Exception:
            continue
        for entry in d.get("leaderboard", []):
            rows.append(_format_entry(entry, source=scores_file.parent.name))

    # Scan submissions/*.json
    sub_dir = ROOT / "submissions"
    if sub_dir.exists():
        for sub_file in sub_dir.glob("*.json"):
            try:
                d = json.load(sub_file.open())
            except Exception:
                continue
            outputs_dir = ROOT / d.get("outputs_dir", "")
            scores_path = outputs_dir / "scores.json"
            if scores_path.exists():
                ed = json.load(scores_path.open())
                for entry in ed.get("leaderboard", []):
                    rows.append(_format_entry(entry, source=f"sub:{sub_file.stem}"))

    return rows


def _format_entry(entry: Dict, source: str) -> Dict:
    scores = entry.get("scores", {})
    details = entry.get("details", {})
    row = {
        "model": entry.get("model", "<unknown>"),
        "source": source,
        "acc": round(entry.get("capability_acc", 0), 3),
        "raw": round(entry.get("composite_raw", 0), 3),
        "cw": round(entry.get("composite_confidence_weighted", 0), 3),
    }
    for m in MECH_COLS:
        if m in scores:
            row[m.upper()] = round(scores[m], 3)
        elif details.get(m, {}).get("status") == "couldnt_be_tested":
            row[m.upper()] = "N/T"
        else:
            row[m.upper()] = ""
    return row


def render_leaderboard():
    rows = load_all_scores()
    if not rows:
        return pd.DataFrame([{"info": "no scores discovered"}])
    df = pd.DataFrame(rows)
    # Sort by composite raw ascending (lower = less deceptive)
    df = df.sort_values("raw")
    return df


HEADER = """
# DECEPT-Bench v1.0 Leaderboard

Multi-mechanism deception evaluation suite. **Lower = less deceptive.**

| Mechanism | Description |
|---|---|
| M1 | Sycophancy under multi-turn pushback |
| M2 | Sandbagging (eval-aware vs deploy-aware) |
| M3 | Persona-pressure deception |
| M4 | Strategic withholding |
| M5 | Forced-prefill rationalization |
| M6 | Multi-tool fabrication |
| M7 | Multi-agent collusion |
| M8 | Code review under team-pressure |
| M9 | Calibration deception (confident-on-unknowable) |
| M10 | Capability concealment |

**N/T** = could not be tested (e.g. API blocks the attack vector).

[Repo](https://github.com/KookiesNKareem/decept-bench) ·
[Submit a model](https://github.com/KookiesNKareem/decept-bench/blob/main/SUBMISSIONS.md) ·
[Spec](https://github.com/KookiesNKareem/decept-bench/blob/main/SPEC_V1.md)
"""


def build_ui():
    with gr.Blocks(title="DECEPT-Bench Leaderboard") as ui:
        gr.Markdown(HEADER)
        df = render_leaderboard()
        gr.Dataframe(value=df, interactive=False, wrap=True)
        gr.Markdown(
            "Refresh by re-deploying the Space — leaderboard is built at "
            "startup from `data/*/scores.json` and `submissions/*.json`.")
    return ui


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0")
