"""
Project 2 -- W11 capstone: the four screening-funnel leads, same MD protocol, one plot.

Reads data/metrics_lead_<key>.json (produced by 03_analyze_transport.py --traj-tag
_lead_<key> --no-expt, one call per lead) plus the Li6PS5Cl production baseline
(data/metrics_prod.json) for scale, and renders a single comparison figure:
  a. Arrhenius (log10(sigma*T) vs 1000/T) for all 4 leads, MACE-MP-0 baseline,
     same protocol as Li6PS5Cl -- absolute sigma inherits the potential's own bias
     (baseline overpredicts Li6PS5Cl by 1.8x), but the CROSS-LEAD ranking is
     internally consistent.
  b. sigma(300 K) bar, log scale, annotated with the upstream screening-funnel
     prior rank so the reader sees where the coarse prior held up and where it
     didn't -- this is the funnel's own report card, not just a results table.

Two of the four leads (Li20Si3P3S23Cl, Li8TiS6) came from Project 1's MP screen;
the other two (Li3PS4 gen016, LiPS3 gen021) came from Project 3's MatterGen
generation + self-consistent MLIP hull. See 04_prepare_leads.py for provenance.

Run after all four `03_analyze_transport.py --traj-tag _lead_<key> --no-expt` calls:
    python 06_compare_leads.py
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
SRC = os.path.join(HERE, "source_data")
sys.path.insert(0, HERE)
from src import plotstyle as ps  # noqa: E402
from src import outpaths as op  # noqa: E402

# (key, display name, source project, upstream screening-funnel prior log10(sigma))
LEADS = [
    ("li20si3p3s23cl", "Li$_{20}$Si$_3$P$_3$S$_{23}$Cl", "P1", -4.63),
    ("li8tis6",        "Li$_8$TiS$_6$",                   "P1", -4.77),
    ("li3ps4_gen016",  "Li$_3$PS$_4$ (gen016)",           "P3", -6.83),
    ("lips3_gen021",   "LiPS$_3$ (gen021)",               "P3", -7.12),
]
# distinct hue per lead (not a lightness ramp -- these are 4 different materials,
# not tiers of one family); P1 leads get cool colours, P3 leads get warm ones,
# echoing the purple/coral provenance split used in the portfolio-level pipeline
# diagram earlier in this session
LEAD_COLORS = {
    "li20si3p3s23cl": ps.PALETTE["blue_main"],
    "li8tis6":        ps.PALETTE["teal"],
    "li3ps4_gen016":  ps.PALETTE["violet"],
    "lips3_gen021":   ps.PALETTE["red_strong"],
}
BASELINE_COLOR = ps.PALETTE["neutral_mid"]


def _sigma_model(T, fit):
    return np.exp(fit["slope"] / T + fit["intercept"]) / T


def main():
    ps.apply_publication_style()
    baseline = json.load(open(os.path.join(DATA, "metrics_prod.json")))["arrhenius"]["mace"]

    results = {}
    for key, name, src, prior in LEADS:
        path = os.path.join(DATA, f"metrics_lead_{key}.json")
        if not os.path.exists(path):
            sys.exit(f"missing {path} -- run 03_analyze_transport.py --traj-tag _lead_{key} "
                     f"--no-expt first")
        d = json.load(open(path))
        results[key] = {"name": name, "src": src, "prior": prior,
                        "per_run": d["per_run"], "fit": d["arrhenius"]["mace"]}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(ps.COL_DOUBLE_IN, 3.6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # --- panel a: Arrhenius, all 4 leads + Li6PS5Cl baseline for scale --------
    # Baseline's fit nearly coincides with two of the four leads for most of the
    # range (they share the same host lattice family and protocol) -- an in-plot
    # text label would sit on top of overlapping curves wherever placed, so the
    # baseline is identified via the legend instead (5th handle, dashed gray).
    Tgrid = np.linspace(290, 1050, 200)
    axA.plot(1000.0 / Tgrid, np.log10(_sigma_model(Tgrid, baseline) * Tgrid),
             color=BASELINE_COLOR, lw=0.9, ls="--", zorder=1)

    for key, name, src, prior in LEADS:
        r = results[key]
        c = LEAD_COLORS[key]
        T = np.array([x["T"] for x in r["per_run"]])
        sig = np.array([x["sigma_mS_cm"] for x in r["per_run"]])
        rel = np.array([(x.get("kinisi_sigma_std_mS_cm") or 0.0) / x["kinisi_sigma_mS_cm"]
                       if x.get("kinisi_sigma_mS_cm") else 0.0 for x in r["per_run"]])
        yT = np.log10(sig * T)
        axA.errorbar(1000.0 / T, yT, yerr=rel / np.log(10.0), fmt="o", ms=3.6,
                    color=c, mec="white", mew=0.4, elinewidth=0.7, capsize=1.8, zorder=3)
        fit = r["fit"]
        if np.isfinite(fit.get("Ea_eV", np.nan)):
            axA.plot(1000.0 / Tgrid, np.log10(_sigma_model(Tgrid, fit) * Tgrid),
                    color=c, lw=1.0, zorder=2)
            y300 = np.log10(fit["sigma300_mS_cm"] * 300)
            lo, hi = fit.get("sigma300_min_mS_cm"), fit.get("sigma300_max_mS_cm")
            if lo and hi and np.isfinite(lo) and np.isfinite(hi):
                axA.errorbar([1000.0 / 300], [y300],
                             yerr=[[y300 - np.log10(lo * 300)], [np.log10(hi * 300) - y300]],
                             fmt="none", ecolor=c, elinewidth=0.8, capsize=2, capthick=0.8, zorder=3)
            axA.scatter([1000.0 / 300], [y300],
                       color=c, marker="*", s=60, edgecolor=ps.PALETTE["neutral_black"],
                       linewidth=0.4, zorder=4)
    axA.axvline(1000.0 / 300, ls=":", color=ps.PALETTE["neutral_mid"], lw=0.6)
    axA.text(1000.0 / 300 - 0.03, 0.97, "300 K", transform=axA.get_xaxis_transform(),
             color=ps.PALETTE["neutral_mid"], fontsize=ps.FS_ANNOT, ha="right", va="top")
    axA.set_xlabel("1000 / $T$  (K$^{-1}$)")
    axA.set_ylabel("log$_{10}$ $\\sigma T$  (mS cm$^{-1}$ K)")
    handles = [plt.Line2D([0], [0], color=LEAD_COLORS[k], lw=1.5, marker="o", ms=4)
              for k, *_ in LEADS]
    handles.append(plt.Line2D([0], [0], color=BASELINE_COLOR, lw=1.2, ls="--"))
    axA.legend(handles, [name for _, name, *_ in LEADS] + ["Li$_6$PS$_5$Cl (baseline)"],
              loc="lower left", fontsize=ps.FS_LEGEND, handlelength=1.4)

    # --- panel b: sigma(300K) bar, annotated with upstream prior rank --------
    order = sorted(LEADS, key=lambda x: results[x[0]]["fit"]["sigma300_mS_cm"], reverse=True)
    prior_rank = {k: i + 1 for i, (k, *_) in
                 enumerate(sorted(LEADS, key=lambda x: x[3], reverse=True))}
    labels = [name for key, name, *_ in order]
    vals = [results[key]["fit"]["sigma300_mS_cm"] for key, *_ in order]
    colors = [LEAD_COLORS[key] for key, *_ in order]
    ypos = range(len(order))
    # asymmetric weighted-fit sigma(300 K) interval per lead (0 when a fit lacks bounds)
    xe_lo, xe_hi = [], []
    for key, *_ in order:
        f = results[key]["fit"]; v = f["sigma300_mS_cm"]
        lo, hi = f.get("sigma300_min_mS_cm"), f.get("sigma300_max_mS_cm")
        xe_lo.append(v - lo if lo and np.isfinite(lo) else 0.0)
        xe_hi.append(hi - v if hi and np.isfinite(hi) else 0.0)
    xerr = [xe_lo, xe_hi] if any(xe_lo) or any(xe_hi) else None
    axB.barh(ypos, vals, color=colors, edgecolor=ps.PALETTE["neutral_black"], linewidth=0.5,
             xerr=xerr, error_kw=dict(elinewidth=0.7, capsize=2, capthick=0.7,
                                      ecolor=ps.PALETTE["neutral_black"]))
    axB.set_xscale("log")
    axB.axvline(baseline["sigma300_mS_cm"], color=BASELINE_COLOR, ls="--", lw=0.8, zorder=1)
    axB.text(baseline["sigma300_mS_cm"], len(order) - 0.4, " Li$_6$PS$_5$Cl",
             color=BASELINE_COLOR, fontsize=ps.FS_ANNOT, ha="left", va="bottom", rotation=0)
    for y, (key, *_rest) in zip(ypos, order):
        v = results[key]["fit"]["sigma300_mS_cm"]
        rank = prior_rank[key]
        axB.annotate(f"{v:.2g}  (prior rank {rank})", (v, y), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=ps.FS_ANNOT)
    axB.set_yticks(ypos)
    axB.set_yticklabels(labels)
    axB.set_xlabel("$\\sigma$(300 K)  (mS cm$^{-1}$, log)")
    axB.set_xlim(1e-8, 3e3)

    ps.add_panel_label(axA, "a")
    ps.add_panel_label(axB, "b", x=-0.42)
    src_tier = op.tier("06_compare_leads")
    os.makedirs(os.path.join(SRC, src_tier), exist_ok=True)
    rows = []
    for key, name, src, prior in LEADS:
        fit = results[key]["fit"]
        rows.append([name, src, prior, prior_rank[key], fit["sigma300_mS_cm"],
                    fit["Ea_eV"], fit["R2"]])
    import pandas as pd
    pd.DataFrame(rows, columns=["lead", "source_project", "prior_log10_sigma",
                                "prior_rank", "sigma300_mS_cm", "Ea_eV", "R2"]).to_csv(
        os.path.join(SRC, src_tier, "06_compare_leads.csv"), index=False)
    ps.finalize_figure(fig, op.fig(FIG, "06_compare_leads.png"), w_pad=3.0)
    print(f"Saved figures/{src_tier}/06_compare_leads.png, "
          f"source_data/{src_tier}/06_compare_leads.csv")


if __name__ == "__main__":
    main()
