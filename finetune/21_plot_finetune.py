"""
W7 step 5 (figure) -- MACE fine-tuning convergence. Plots the validation RMSE of forces and
energy per epoch (from mace_run_train's JSON-lines log) against the foundation (pre-fine-tune)
baseline, so the "234 -> 62 meV/A" improvement is a figure, not just a table row.

    finetune/results/<name>_train.txt  --(this)-->  figures/04_finetune_convergence.{png,svg}
                                                     + source_data/04_finetune_convergence.csv

Usage:  python finetune/21_plot_finetune.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src import plotstyle as ps  # noqa: E402


def load_evals(path):
    """Return (foundation_eval, [per-epoch evals]) from a mace_run_train *_train.txt log."""
    evals = [json.loads(l) for l in open(path) if l.strip() and json.loads(l).get("mode") == "eval"]
    foundation = next((e for e in evals if e.get("epoch") is None), None)   # pre-fine-tune baseline
    curve = sorted((e for e in evals if e.get("epoch") is not None), key=lambda e: e["epoch"])
    return foundation, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=os.path.join(HERE, "results", "li6ps5cl_ft_run-0_train.txt"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    args = ap.parse_args()

    ps.apply_publication_style()
    foundation, curve = load_evals(args.log)
    ep = [e["epoch"] for e in curve]
    rf = [e["rmse_f"] * 1000.0 for e in curve]            # eV/A -> meV/A
    re = [e["rmse_e_per_atom"] * 1000.0 for e in curve]   # eV/atom -> meV/atom
    f0 = foundation["rmse_f"] * 1000.0 if foundation else None
    e0 = foundation["rmse_e_per_atom"] * 1000.0 if foundation else None
    c = ps.PALETTE["blue_main"]

    fig, axes = plt.subplots(1, 2, figsize=(ps.COL_DOUBLE_IN, 2.9))
    panels = [(axes[0], rf, f0, "$F$", "meV Å$^{-1}$"),
              (axes[1], re, e0, "$E$", "meV atom$^{-1}$")]
    for ax, y, y0, sym, unit in panels:
        ax.plot(ep, y, color=c, lw=1.0, zorder=3)
        if y0 is not None:   # foundation (un-fine-tuned) baseline as a dashed reference
            ax.axhline(y0, ls="--", lw=0.8, color=ps.PALETTE["neutral_mid"])
            ax.text(0.98, y0, f"foundation {y0:.0f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=ps.FS_ANNOT, color=ps.PALETTE["neutral_mid"])
        ax.annotate(f"{y[-1]:.0f}", (ep[-1], y[-1]), textcoords="offset points", xytext=(3, 0),
                    ha="left", va="center", fontsize=ps.FS_ANNOT, color=c)
        ax.set(xlabel="epoch", ylabel=f"valid. RMSE {sym}  ({unit})")
        ax.set_ylim(bottom=0)
    for ax, ltr in zip(axes, "ab"):
        ps.add_panel_label(ax, ltr)

    out = os.path.join(args.fig_dir, "04_finetune_convergence.png")
    rows = ([["foundation", f"{f0:.2f}", f"{e0:.4f}"]] if foundation else []) + \
        [[e, f"{v_f:.4f}", f"{v_e:.5f}"] for e, v_f, v_e in zip(ep, rf, re)]
    ps.save_source_data(out, ["epoch", "valid_rmse_f_meV_A", "valid_rmse_e_meV_atom"], rows)
    print("[21] figure ->", ps.finalize_figure(fig, out)[0])


if __name__ == "__main__":
    main()
