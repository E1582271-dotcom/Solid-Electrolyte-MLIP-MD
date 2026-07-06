"""
Project 2 -- W9: Cl-excess doping trend, one plot.

Reads data/metrics_dope_cl{100,125,150,175}.json (produced by 03_analyze_transport.py
--traj-tag _dope_cl<NNN> --no-expt, one call per composition) plus data/doped/doped.json
for the exact supercell size of each composition, and renders a two-panel comparison:
  a. sigma(300 K) vs Cl content (log y -- the four values span ~29x).
  b. Ea vs Cl content (linear y).

Series is Li6-x PS5-x Cl1+x (x = 0/0.25/0.5/0.75), built by 05_build_doped.py: free-anion
S2- -> Cl- substitution with Li-vacancy charge compensation, same MACE-MP-0 (small) /
NVT Langevin / 600-800-1000 K / 150 ps protocol as the W11 funnel leads (04_prepare_leads.py)
-- i.e. the "screening-grade" (400-atom-class) tier, NOT the W8 production tier (416-atom /
200 ps). The Cl=1.0 anchor (Li6PS5Cl itself) therefore reads ~7.3 mS/cm here, not the 5.57
mS/cm of the production baseline in metrics_prod.json -- same order of magnitude, different
convergence tier, not a discrepancy (see the in-figure caveat + REPORT.md).

Run after all four `03_analyze_transport.py --traj-tag _dope_cl<NNN> --no-expt` calls:
    python 07_doping_trend.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, HERE)
from src import plotstyle as ps  # noqa: E402

# (cltag, Cl content) in ascending order -- x = cl_content - 1
COMPOSITIONS = [("cl100", 1.00), ("cl125", 1.25), ("cl150", 1.50), ("cl175", 1.75)]
# Cl content is a *sequential* physical variable -> cool->warm perceptual ramp,
# same convention as TEMP_COLORS (temperature) elsewhere in this repo.
_RAMP = [ps.PALETTE["blue_main"], ps.PALETTE["teal"], ps.PALETTE["violet"], ps.PALETTE["red_strong"]]
DOPE_COLORS = {tag: _RAMP[i] for i, (tag, _) in enumerate(COMPOSITIONS)}


def main():
    ps.apply_publication_style()
    doped_meta = json.load(open(os.path.join(DATA, "doped", "doped.json")))["compositions"]

    rows = []
    for tag, cl in COMPOSITIONS:
        path = os.path.join(DATA, f"metrics_dope_{tag}.json")
        if not os.path.exists(path):
            sys.exit(f"missing {path} -- run 03_analyze_transport.py --traj-tag _dope_{tag} "
                     f"--no-expt first")
        d = json.load(open(path))
        fit = d["arrhenius"]["mace"]
        per_run = d["per_run"]
        rel = np.array([(r.get("kinisi_sigma_std_mS_cm") or 0.0) / r["kinisi_sigma_mS_cm"]
                        if r.get("kinisi_sigma_mS_cm") else 0.0 for r in per_run])
        sig300_rel_err = float(np.nanmean(rel)) if len(rel) else 0.0
        meta = doped_meta[tag]
        rows.append({
            "tag": tag, "cl_content": cl, "formula": meta["formula"],
            "n_atoms": meta["n_atoms"], "sigma300_mS_cm": fit["sigma300_mS_cm"],
            "Ea_eV": fit["Ea_eV"], "R2": fit["R2"], "sigma300_rel_err": sig300_rel_err,
        })

    xs = np.array([r["cl_content"] for r in rows])
    sig = np.array([r["sigma300_mS_cm"] for r in rows])
    ea = np.array([r["Ea_eV"] for r in rows])
    sig_err = sig * np.array([r["sigma300_rel_err"] for r in rows])
    colors = [DOPE_COLORS[r["tag"]] for r in rows]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(ps.COL_DOUBLE_IN, 3.0))

    # --- panel a: sigma(300K) vs Cl content, log y (spans ~29x) ---------------
    axA.plot(xs, sig, color=ps.PALETTE["neutral_mid"], lw=0.8, zorder=1)
    axA.errorbar(xs, sig, yerr=sig_err, fmt="none", ecolor=ps.PALETTE["neutral_dark"],
                elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=2)
    axA.scatter(xs, sig, c=colors, s=34, edgecolor="white", linewidth=0.5, zorder=3)
    axA.set_yscale("log")
    axA.set_ylim(sig.min() * 0.5, sig.max() * 4.0)   # headroom for the in-axes annotation
    axA.set_xlabel("Cl content in Li$_{6-x}$PS$_{5-x}$Cl$_{1+x}$")
    axA.set_ylabel("$\\sigma$(300 K)  (mS cm$^{-1}$, log)")
    ratio = sig[-1] / sig[0]
    axA.text(0.05, 0.93, f"~{ratio:.0f}× monotonic increase\n(Cl 1.0 → 1.75)",
             transform=axA.transAxes, ha="left", va="top", fontsize=ps.FS_ANNOT,
             color=ps.PALETTE["neutral_dark"], fontweight="bold")

    # --- panel b: Ea vs Cl content, linear y -----------------------------------
    axB.plot(xs, ea, color=ps.PALETTE["neutral_mid"], lw=0.8, zorder=1)
    axB.scatter(xs, ea, c=colors, s=34, edgecolor="white", linewidth=0.5, zorder=3)
    axB.set_xlabel("Cl content in Li$_{6-x}$PS$_{5-x}$Cl$_{1+x}$")
    axB.set_ylabel("$E_a$  (eV)")
    axB.text(0.95, 0.93, "more Cl disorder +\nmore Li vacancies →\nlower migration barrier",
             transform=axB.transAxes, ha="right", va="top", fontsize=ps.FS_ANNOT,
             color=ps.PALETTE["neutral_dark"], style="italic")

    ps.add_panel_label(axA, "a")
    ps.add_panel_label(axB, "b")

    src_rows = [[r["tag"], r["cl_content"], r["formula"], r["n_atoms"], r["sigma300_mS_cm"],
                r["Ea_eV"], r["R2"]] for r in rows]
    ps.save_source_data(os.path.join(FIG, "07_doping_trend.png"),
                        ["tag", "cl_content", "formula", "n_atoms", "sigma300_mS_cm",
                         "Ea_eV", "R2"], src_rows)
    ps.finalize_figure(fig, os.path.join(FIG, "07_doping_trend.png"), w_pad=3.0)
    print("Saved figures/07_doping_trend.png, source_data/07_doping_trend.csv")


if __name__ == "__main__":
    main()
