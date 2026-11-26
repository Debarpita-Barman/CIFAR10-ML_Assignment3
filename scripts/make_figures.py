#!/usr/bin/env python3
"""Redraw report figures from the saved results (no training, no GPU, no dataset needed).

The notebooks draw every figure during the run. Two of them had overlapping labels, so this script redraws
them from the saved tables with clearer labels. The data are identical; only the layout changes:

  results/q1_outputs/fig_a3_capacity_and_factorial_report.png   (from table_a3_validation_summary.csv)
  results/q2_outputs/fig_q2_5_tile_curve_report.png              (from q2 results_summary.json)

Usage: python scripts/make_figures.py [--results results]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
GREY = "#7a7a7a"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.prop_cycle": plt.cycler(color=PALETTE), "axes.grid": True,
    "grid.color": "#e7e7e7", "grid.linewidth": 0.8, "axes.edgecolor": "#c2c2c2",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "font.size": 10,
})


def fig_a3(q1: Path):
    S = pd.read_csv(q1 / "table_a3_validation_summary.csv", index_col="model")
    mlps = ["MLP-S", "MLP-M", "MLP-L", "MLP-XL"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    P = S.loc[mlps, "parameters"].values.astype(float)
    ax1.errorbar(P, S.loc[mlps, "val_acc_mean"], yerr=S.loc[mlps, "val_acc_sd"], marker="o", ms=7, capsize=3,
                 color=PALETTE[0], label="MLP validation (best epoch)")
    ax1.plot(P, S.loc[mlps, "train_acc_at_best"], marker="o", ms=7, ls="--", color=PALETTE[0], alpha=0.55,
             label="MLP train (eval mode) at that epoch")
    for nm, mk, c in [("CNN", "*", PALETTE[1]), ("CNN-perm", "X", PALETTE[2])]:
        ax1.errorbar([S.loc[nm, "parameters"]], [S.loc[nm, "val_acc_mean"]], yerr=[S.loc[nm, "val_acc_sd"]],
                     marker=mk, ms=13, ls="none", capsize=3, color=c, label=f"{nm} validation")
    for nm, x in zip(mlps, P):
        ax1.annotate(nm, (x, S.loc[nm, "val_acc_mean"]), textcoords="offset points",
                     xytext=(0, 9) if nm == "MLP-M" else (0, -16), ha="center", fontsize=8)
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("parameters (log scale)")
    ax1.set_ylabel("accuracy")
    ax1.set_title("Capacity: MLP width sweep vs. CNN at MLP-M's size", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")

    bw = 0.36
    for gi, (orig, perm) in enumerate([("MLP-M", "MLP-M-perm"), ("CNN", "CNN-perm")]):
        for k, (nm, lab, c) in enumerate([(orig, "original pixels", PALETTE[0]), (perm, "permuted pixels", PALETTE[1])]):
            x = gi + (k - 0.5) * (bw + 0.02)
            m, s = S.loc[nm, "val_acc_mean"], S.loc[nm, "val_acc_sd"]
            ax2.bar(x, m, bw, yerr=s, capsize=4, color=c, edgecolor="white", linewidth=2,
                    label=lab if gi == 0 else None)
            ax2.text(x, m + s + 0.015, f"{m:.3f}", ha="center", fontsize=8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"MLP-M ({S.loc['MLP-M', 'parameters'] / 1e6:.2f}M)",
                         f"CNN ({S.loc['CNN', 'parameters'] / 1e6:.2f}M)"])
    ax2.set_ylim(0, 1.15)
    ax2.set_yticks(np.arange(0, 1.01, 0.2))
    ax2.set_ylabel("validation accuracy (best epoch)")
    ax2.set_title("Permutation control (error bars: ±1 sd over 3 seeds)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper center", ncol=2)
    fig.suptitle("Fig. A3: separating spatial inductive bias from capacity (validation)")
    fig.tight_layout()
    out = q1 / "fig_a3_capacity_and_factorial_report.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_q2_5(q2: Path):
    s = json.loads((q2 / "results_summary.json").read_text())
    acc = {k: np.asarray(v) for k, v in s["test_acc_per_seed"].items()}
    tiles = s["tile_sizes"]
    conds = [f"tile{t}" for t in tiles]
    R = s["retention_R"]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = ["clean\n(t = 32)"] + [f"t = {t}" for t in tiles]
    for k, arch in enumerate(["MLP-M", "CNN"]):
        vals = [acc[f"{arch}|clean"]] + [acc[f"{arch}|{c}"] for c in conds]
        ax.errorbar(range(len(xs)), [v.mean() for v in vals], yerr=[v.std() for v in vals], marker="o", ms=7,
                    capsize=3, color=PALETTE[k], label=f"{arch}: tile shuffle")
        ax.axhline(acc[f"{arch}|perm"].mean(), color=PALETTE[k], ls="--", lw=1.2, alpha=0.8,
                   label=f"{arch}: fixed permutation (b)")
    for i, c in enumerate(conds, start=1):
        ax.annotate(f"R = {R[c]:.2f}", (i, acc[f"CNN|{c}"].mean()), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tile size t (local structure kept up to t px; global arrangement random)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Fig. Q2-5: test accuracy as global arrangement is removed\n"
                 "(R = share of the CNN's spatial benefit retained)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    out = q2 / "fig_q2_5_tile_curve_report.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    res = Path(ap.parse_args().results)
    for fn, sub in [(fig_a3, "q1_outputs"), (fig_q2_5, "q2_outputs")]:
        d = res / sub
        if d.exists():
            print("wrote", fn(d))
        else:
            print("skipped", fn.__name__, "-", d, "not found")


if __name__ == "__main__":
    main()
