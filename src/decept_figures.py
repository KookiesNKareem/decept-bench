"""Generate paper figures and per-family analysis from scores.json files.

Outputs to paper/figures/ — both the underlying CSV data (always) and PNG
plots (only if matplotlib is available).

Figures produced:
  fig1_capability_vs_deception.{csv,png}
  fig2_cross_mechanism_heatmap.{csv,png}
  fig3_qwen_scaling.{csv,png}
  fig4_dsr1_distill_size.{csv,png}
  fig5_per_mechanism_distribution.{csv,png}
  table_open_vs_closed_means.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MECH_COLS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"]


def load_panel() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "data").glob("*/scores.json")):
        for e in json.load(f.open()).get("leaderboard", []):
            row = {
                "model": e["model"],
                "acc": e.get("capability_acc", np.nan),
                "raw": e.get("composite_raw", np.nan),
                "cw": e.get("composite_confidence_weighted", np.nan),
                "is_closed": e["model"].startswith(("claude", "gpt-", "o")),
            }
            for m in MECH_COLS:
                row[m] = e["scores"].get(m, np.nan)
            rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("model").reset_index(drop=True)


def fig1_capability_vs_deception(df: pd.DataFrame):
    out = df[["model", "acc", "raw", "is_closed"]].dropna()
    out.to_csv(OUT / "fig1_capability_vs_deception.csv", index=False)
    print(f"  fig1: n={len(out)}, "
          f"corr={out[['acc','raw']].corr().iloc[0,1]:.2f}")
    return out


def fig2_cross_mechanism_heatmap(df: pd.DataFrame):
    from scipy.stats import spearmanr
    out = pd.DataFrame(index=MECH_COLS, columns=MECH_COLS, dtype=float)
    for i in MECH_COLS:
        for j in MECH_COLS:
            mask = df[i].notna() & df[j].notna()
            if mask.sum() < 4:
                out.loc[i, j] = np.nan
                continue
            rho, _ = spearmanr(df.loc[mask, i], df.loc[mask, j])
            out.loc[i, j] = rho
    out.to_csv(OUT / "fig2_cross_mechanism_heatmap.csv")
    print("  fig2: cross-mechanism Spearman saved")
    return out


def _qwen_family(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Qwen2.5-Instruct sizes."""
    qwen = df[df["model"].str.contains("qwen2-5", case=False, na=False)].copy()
    sizes = []
    for m in qwen["model"]:
        match = re.search(r"qwen2-5-(\d+(?:-\d+)?)b", m)
        if match:
            s = match.group(1).replace("-", ".")
            sizes.append(float(s))
        else:
            sizes.append(np.nan)
    qwen["params_b"] = sizes
    return qwen.dropna(subset=["params_b"]).sort_values("params_b")


def fig3_qwen_scaling(df: pd.DataFrame):
    qwen = _qwen_family(df)
    out = qwen[["model", "params_b", "acc", "raw"] + MECH_COLS]
    out.to_csv(OUT / "fig3_qwen_scaling.csv", index=False)
    print(f"  fig3: Qwen scaling, n={len(out)} sizes")
    return out


def _dsr1_family(df: pd.DataFrame) -> pd.DataFrame:
    dsr = df[df["model"].str.contains("deepseek-r1-distill", case=False, na=False)].copy()
    sizes = []
    for m in dsr["model"]:
        match = re.search(r"qwen-(\d+(?:-\d+)?)b", m)
        if match:
            s = match.group(1).replace("-", ".")
            sizes.append(float(s))
        else:
            sizes.append(np.nan)
    dsr["params_b"] = sizes
    return dsr.dropna(subset=["params_b"]).sort_values("params_b")


def fig4_dsr1_scaling(df: pd.DataFrame):
    dsr = _dsr1_family(df)
    out = dsr[["model", "params_b", "acc", "raw"] + MECH_COLS]
    out.to_csv(OUT / "fig4_dsr1_distill_size.csv", index=False)
    print(f"  fig4: DSR1-distill scaling, n={len(out)} sizes")
    return out


def fig5_per_mechanism_distribution(df: pd.DataFrame):
    rows = []
    for m in MECH_COLS:
        vals = df[m].dropna()
        rows.append({
            "mechanism": m,
            "n": len(vals),
            "min": vals.min() if len(vals) else np.nan,
            "p25": vals.quantile(0.25) if len(vals) else np.nan,
            "median": vals.median() if len(vals) else np.nan,
            "mean": vals.mean() if len(vals) else np.nan,
            "p75": vals.quantile(0.75) if len(vals) else np.nan,
            "max": vals.max() if len(vals) else np.nan,
            "std": vals.std() if len(vals) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fig5_per_mechanism_distribution.csv", index=False)
    print("  fig5: per-mechanism summary saved")
    return out


def table_open_vs_closed(df: pd.DataFrame):
    grp = df.groupby("is_closed")[MECH_COLS + ["raw", "acc"]].agg(["mean", "std", "count"])
    grp.to_csv(OUT / "table_open_vs_closed_means.csv")
    print("  table: open vs closed means saved")
    return grp


def maybe_plot(df: pd.DataFrame):
    """Render PNG figures if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  (skipping PNG plots — matplotlib not installed)")
        return

    # fig1 — capability vs deception with non-overlapping, distinct labels.
    def _short(name: str) -> str:
        # Map raw model slugs to compact, distinct display labels.
        n = name.lower()
        n = n.replace("claude-", "").replace("-instruct", "")
        n = n.replace("deepseek-ai_deepseek-r1-distill-qwen-", "DSR1-")
        n = n.replace("unsloth_", "").replace("microsoft_", "")
        n = n.replace("google_", "").replace("qwen_", "")
        # Restore version dots in product names: gpt-5-5 -> gpt-5.5,
        # gpt-5-4-mini -> gpt-5.4-mini, phi-3-5-mini -> phi-3.5-mini,
        # qwen2-5-* -> qwen2.5-*.
        n = n.replace("gpt-5-5", "gpt-5.5").replace("gpt-5-4-mini", "gpt-5.4-mini")
        n = n.replace("phi-3-5-mini", "phi-3.5-mini")
        n = n.replace("qwen2-5", "qwen2.5")
        # Restore size-suffix dots: -1-5b -> -1.5B, -0-5b -> -0.5B.
        n = n.replace("-it", "").replace("-1-5b", "-1.5B").replace("-0-5b", "-0.5B")
        n = n.replace("-3b", "-3B").replace("-7b", "-7B").replace("-14b", "-14B")
        n = n.replace("-12b", "-12B").replace("-4b", "-4B").replace("-1b", "-1B")
        return n
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    closed = df[df["is_closed"]]
    open_ = df[~df["is_closed"]]
    ax.scatter(open_["acc"], open_["raw"], label="open-weight",
                  alpha=0.75, s=55, color="#1f77b4", zorder=3)
    ax.scatter(closed["acc"], closed["raw"], label="closed API",
                  alpha=0.9, marker="^", s=85, color="#d62728", zorder=3)
    texts = [ax.text(r["acc"], r["raw"], _short(r["model"]),
                       fontsize=7.5, alpha=0.95)
                for _, r in df.iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax,
                       arrowprops=dict(arrowstyle="-", color="gray",
                                          alpha=0.5, lw=0.6),
                       expand=(1.08, 1.18),
                       force_text=(0.6, 0.8))
    except ImportError:
        pass
    ax.set_xlabel("capability accuracy (M2 single-eval baseline)")
    ax.set_ylabel("composite_raw deception   (lower = less deceptive)")
    ax.set_title("Capability vs. deception under hybrid scoring "
                    f"(Pearson {df[['acc','raw']].corr().iloc[0,1]:.2f})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_capability_vs_deception.png", dpi=160,
                  bbox_inches="tight")
    plt.close(fig)

    # fig2 heatmap
    corr = pd.read_csv(OUT / "fig2_cross_mechanism_heatmap.csv", index_col=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.xticks(range(len(MECH_COLS)), [m.upper() for m in MECH_COLS])
    plt.yticks(range(len(MECH_COLS)), [m.upper() for m in MECH_COLS])
    plt.colorbar(label="Spearman ρ")
    for i, mi in enumerate(MECH_COLS):
        for j, mj in enumerate(MECH_COLS):
            v = corr.iloc[i, j]
            if pd.notna(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                          fontsize=7,
                          color="white" if abs(v) > 0.5 else "black")
    plt.title("Cross-mechanism rank correlation")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_cross_mechanism_heatmap.png", dpi=150)
    plt.close()

    # fig3 + fig4 scaling lines, legend outside the plot
    for fname, family_fn, title in [
        ("fig3_qwen_scaling", _qwen_family, "Qwen2.5-Instruct scaling"),
        ("fig4_dsr1_distill_size", _dsr1_family, "DSR1-Distill-Qwen scaling"),
    ]:
        fam = family_fn(df)
        if fam.empty: continue
        fig, ax = plt.subplots(figsize=(7.2, 4))
        for m in ["m1", "m2", "m4", "m5", "m6", "m7"]:
            if fam[m].notna().any():
                ax.plot(fam["params_b"], fam[m], "-o", label=m.upper(),
                          alpha=0.75)
        ax.plot(fam["params_b"], fam["raw"], "k-s", lw=2,
                  label="composite raw")
        ax.set_xscale("log")
        ax.set_xlabel("parameters (B)")
        ax.set_ylabel("deception rate (lower = better)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"{fname}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print("  PNG plots saved to", OUT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    print(f"Panel: {len(df)} models")
    print(f"  closed: {df['is_closed'].sum()}, open: {(~df['is_closed']).sum()}")
    print()
    fig1_capability_vs_deception(df)
    fig2_cross_mechanism_heatmap(df)
    fig3_qwen_scaling(df)
    fig4_dsr1_scaling(df)
    fig5_per_mechanism_distribution(df)
    table_open_vs_closed(df)
    print()
    maybe_plot(df)


if __name__ == "__main__":
    main()
