"""
03 / pipeline -- trajectories -> D -> sigma(T) -> Arrhenius -> compare vs experiment
(Project 2, W6/W9).

Scans data/traj/<mlip>_<T>K.traj, computes the Li diffusivity with pymatgen (backbone) and
kinisi (error bar), builds the Nernst-Einstein conductivity sigma(T), fits Arrhenius per
MLIP to get Ea and the extrapolated sigma(300 K), and plots MACE vs MatterSim vs the
experimental Li6PS5Cl value (~3.15 mS/cm). Writes metrics.json with honest caveats.

This is a BASELINE read-out (un-fine-tuned MLIPs, thin MD) -- numbers are order-of-magnitude
indicative, not converged. CPU-only; safe to run anywhere.

Usage:
    python 03_analyze_transport.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import transport as tr  # noqa: E402
from src import plotstyle as ps  # noqa: E402

MLIP_COLORS = ps.MLIP_COLORS
EXPT_COLOR = ps.EXPT_COLOR
# experimental Li6PS5Cl room-T ionic conductivity references (mS/cm)
EXPT = {"Li6PS5Cl (sintered)": 3.15, "Li6PS5Cl (mechanochem.)": 1.33}
EXPT_MAIN = 3.15
TRAJ_RE = re.compile(r"(?P<mlip>[a-zA-Z]+)_(?P<T>\d+)K\.traj$")


def _load_run_params(data_dir, tag=""):
    """timestep_fs + log_every per (mlip,T) from md_runs{tag}.json (fallback handled by caller)."""
    path = os.path.join(data_dir, f"md_runs{tag}.json")
    params = {}
    if os.path.exists(path):
        for r in json.load(open(path)).get("runs", []):
            params[(r["mlip"], int(r["temperature_K"]))] = (
                r.get("timestep_fs", 1.0), r.get("log_every", 50))
    return params


def _sigma_model(T, fit):
    """Arrhenius model sigma(T) in mS/cm from the ln(sigma*T) fit."""
    return np.exp(fit["slope"] / T + fit["intercept"]) / T


def _arrhenius_panel(ax, per_mlip, fits, yfunc):
    """Draw MD points + Arrhenius fit curve + 300 K star on `ax`, mapping each
    (sigma_mS_cm, T) to a y-value via yfunc(sigma, T). The caller owns the y-limits,
    the experiment reference, axis labels and legend (which differ per y-axis choice)."""
    Tgrid = np.linspace(290, 1100, 200)
    for mlip, rows in per_mlip.items():
        c = MLIP_COLORS.get(mlip, ps.PALETTE["neutral_dark"])
        T = np.array([r["T"] for r in rows])
        sig = np.array([r["sigma_mS_cm"] for r in rows])
        # kinisi bootstrap uncertainty (relative) -> error bar in the log y-axis. Since both
        # panels plot log10 of something proportional to sigma and T is exact, the log-space
        # half-height is the same: d(log10 x) = (1/ln10)*dx/x.
        rel = np.array([(r.get("kinisi_sigma_std_mS_cm") or 0.0) / r["kinisi_sigma_mS_cm"]
                        if r.get("kinisi_sigma_mS_cm") else 0.0 for r in rows])
        ax.errorbar(1000.0 / T, yfunc(sig, T), yerr=rel / np.log(10.0), fmt="o", ms=3.6,
                    color=c, mec="white", mew=0.4, elinewidth=0.7, capsize=1.8, capthick=0.7,
                    zorder=3, label=f"{mlip} (MD)")
        fit = fits.get(mlip)
        if fit and np.isfinite(fit["Ea_eV"]):
            ax.plot(1000.0 / Tgrid, yfunc(_sigma_model(Tgrid, fit), Tgrid), color=c, lw=1.0,
                    label=f"{mlip} fit")
            ax.scatter([1000.0 / 300], [yfunc(fit["sigma300_mS_cm"], 300.0)], color=c,
                       marker="*", s=70, edgecolor=ps.PALETTE["neutral_black"],
                       linewidth=0.4, zorder=4)
    ax.axvline(1000.0 / 300, ls=":", color=ps.PALETTE["neutral_mid"], lw=0.6)


def _plot_arrhenius(per_mlip, fits, path):
    """Side-by-side: the SAME Arrhenius fit shown on two y-axes.
      (a) log10 sigma   -> visibly curved, because sigma = (sigma*T)/T drops a log T term;
      (b) log10(sigma*T) -> a straight line, the textbook Arrhenius form fitted in
          ln(sigma*T) vs 1/T."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(ps.COL_DOUBLE_IN, 3.2))

    # -- (a) log10 sigma : curved by the sigma = sigma*T / T factor --
    _arrhenius_panel(axL, per_mlip, fits, lambda s, T: np.log10(s))
    yloL = np.log10(EXPT_MAIN) - 0.25
    yhiL = max(np.log10(r["sigma_mS_cm"]) for rows in per_mlip.values() for r in rows) + 0.35
    axL.set_ylim(yloL, yhiL)
    axL.text(1000.0 / 300, 1.01, "300 K", transform=axL.get_xaxis_transform(),
             color=ps.PALETTE["neutral_mid"], fontsize=ps.FS_ANNOT, ha="center", va="bottom")
    axL.axhline(np.log10(EXPT_MAIN), ls="--", color=EXPT_COLOR, lw=0.8)
    axL.text(0.015, np.log10(EXPT_MAIN) - 0.04, f"expt {EXPT_MAIN} mS cm$^{{-1}}$",
             color=EXPT_COLOR, fontsize=ps.FS_ANNOT, ha="left", va="top",
             transform=axL.get_yaxis_transform())
    axL.set(xlabel="1000 / $T$  (K$^{-1}$)", ylabel="log$_{10}$ $\\sigma$  (mS cm$^{-1}$)")
    axL.legend(loc="upper right")
    ps.add_panel_label(axL, "a")

    # -- (b) log10(sigma*T) : the straight line actually being fitted --
    yT = lambda s, T: np.log10(s * T)
    _arrhenius_panel(axR, per_mlip, fits, yT)
    data_yT = [yT(r["sigma_mS_cm"], r["T"]) for rows in per_mlip.values() for r in rows]
    ref_yT = [yT(EXPT_MAIN, 300.0)] + [
        yT(f["sigma300_mS_cm"], 300.0) for f in fits.values()
        if f and np.isfinite(f.get("sigma300_mS_cm", np.nan))]
    yhiR = max(data_yT) + 0.35
    axR.set_ylim(min(ref_yT) - 0.25, yhiR)
    axR.text(1000.0 / 300, 1.01, "300 K", transform=axR.get_xaxis_transform(),
             color=ps.PALETTE["neutral_mid"], fontsize=ps.FS_ANNOT, ha="center", va="bottom")
    # experiment is measured at room T only -> a single reference point at 300 K, not a line
    axR.scatter([1000.0 / 300], [yT(EXPT_MAIN, 300.0)], marker="X", s=40, color=EXPT_COLOR,
                zorder=5, label="expt @300 K")
    # per-MLIP fit stats compacted into one corner box (kept out of the legend)
    stats = [f"{m}: $E_a$ {fits[m]['Ea_eV']:.2f} eV · $\\sigma_{{300}}$ "
             f"{fits[m]['sigma300_mS_cm']:.1f} · $R^2$ {fits[m]['R2']:.3f}"
             for m in per_mlip if fits.get(m) and np.isfinite(fits[m].get("R2", np.nan))]
    if stats:
        axR.text(0.03, 0.03, "\n".join(stats), transform=axR.transAxes,
                 fontsize=ps.FS_ANNOT, ha="left", va="bottom", color=ps.PALETTE["neutral_dark"])
    axR.set(xlabel="1000 / $T$  (K$^{-1}$)", ylabel="log$_{10}$ $\\sigma T$  (mS cm$^{-1}$ K)")
    axR.legend(loc="upper right")
    ps.add_panel_label(axR, "b")
    return ps.finalize_figure(fig, path)[0]


def _plot_sigma300_bar(fits, path):
    """Extrapolated sigma(300 K) vs experiment (single-column, log y). No chart title;
    the material + fit provenance belong in the caption."""
    fig, ax = plt.subplots(figsize=(ps.COL_SINGLE_IN, 3.0))
    labels, vals, colors = [], [], []
    for mlip, fit in fits.items():
        if fit and np.isfinite(fit.get("sigma300_mS_cm", np.nan)):
            labels.append(f"{mlip}\n(baseline)")
            vals.append(fit["sigma300_mS_cm"])
            colors.append(MLIP_COLORS.get(mlip, ps.PALETTE["neutral_dark"]))
    for name, v in EXPT.items():
        labels.append("sintered\n(expt)" if "sinter" in name else "mech.chem\n(expt)")
        vals.append(v)
        colors.append(ps.PALETTE["neutral_mid"])   # experiment = neutral reference
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=ps.FS_TICK)
    ax.set_ylabel("$\\sigma$(300 K)  (mS cm$^{-1}$, log)")
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.2g}", ha="center", va="bottom", fontsize=ps.FS_ANNOT)
    return ps.finalize_figure(fig, path)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--specie", default="Li")
    ap.add_argument("--timestep", type=float, default=1.0, help="fallback fs if no md_runs.json")
    ap.add_argument("--log-every", type=int, default=50, help="fallback frame spacing")
    ap.add_argument("--traj-tag", default="", help="analyse data/traj{tag}/ + md_runs{tag}.json "
                    "and write 03_*{tag}/metrics{tag} (match 02's --traj-tag)")
    args = ap.parse_args()
    ps.apply_publication_style()

    traj_files = sorted(glob.glob(os.path.join(args.data_dir, f"traj{args.traj_tag}", "*.traj")))
    if not traj_files:
        sys.exit(f"no trajectories in {args.data_dir}/traj{args.traj_tag} -- run 02_baseline_md.py first")
    run_params = _load_run_params(args.data_dir, args.traj_tag)

    per_mlip, all_rows = {}, []
    for tf in traj_files:
        m = TRAJ_RE.search(os.path.basename(tf))
        if not m:
            continue
        mlip, T = m["mlip"], float(m["T"])
        ts, skip = run_params.get((mlip, int(T)), (args.timestep, args.log_every))
        print(f"[03] {mlip} @ {int(T)}K  (dt={ts}fs, every {skip}) ...")

        structures, _ = tr.load_structures(tf)
        n = tr.carrier_density(structures[0], args.specie)
        pmg = tr.diffusivity_pymatgen(structures, args.specie, T, ts, skip)
        kin = tr.diffusivity_kinisi(tf, args.specie, ts, skip, n_per_cm3=n, T=T)

        row = {
            "mlip": mlip, "T": T,
            "D_cm2_s": pmg["D_cm2_s"],
            "sigma_mS_cm": pmg["sigma_mS_cm"],          # pymatgen backbone -> Arrhenius
            "kinisi_D_cm2_s": kin.get("D_cm2_s"),
            "kinisi_D_std_cm2_s": kin.get("D_std_cm2_s"),
            "kinisi_sigma_mS_cm": kin.get("sigma_mS_cm"),
            "kinisi_sigma_std_mS_cm": kin.get("sigma_std_mS_cm"),
            "n_frames": pmg["n_frames"],
        }
        if "error" in kin:
            row["kinisi_error"] = kin["error"]
        print(f"     D(pmg)={pmg['D_cm2_s']:.2e} cm2/s  sigma={pmg['sigma_mS_cm']:.2e} mS/cm"
              f"  | kinisi D={kin.get('D_cm2_s')!r}")
        per_mlip.setdefault(mlip, []).append(row)
        all_rows.append(row)

    fits = {}
    for mlip, rows in per_mlip.items():
        rows.sort(key=lambda r: r["T"])
        fit = tr.arrhenius_fit([r["T"] for r in rows], [r["sigma_mS_cm"] for r in rows])
        fits[mlip] = fit
        if np.isfinite(fit["Ea_eV"]):
            ratio = fit["sigma300_mS_cm"] / EXPT_MAIN
            print(f"[03] {mlip}: Ea={fit['Ea_eV']:.3f} eV  sigma300={fit['sigma300_mS_cm']:.2e} "
                  f"mS/cm  (expt {EXPT_MAIN}; ratio {ratio:.2g}x)  R2={fit['R2']:.3f}")
        else:
            print(f"[03] {mlip}: need >=2 temperatures for Arrhenius (have {fit['n_points']})")

    os.makedirs(args.fig_dir, exist_ok=True)
    figs = {}
    if any(len(r) >= 1 for r in per_mlip.values()):
        figs["arrhenius"] = _plot_arrhenius(per_mlip, fits,
                                            os.path.join(args.fig_dir, f"03_arrhenius{args.traj_tag}.png"))
    if any(np.isfinite(f.get("sigma300_mS_cm", np.nan)) for f in fits.values()):
        figs["sigma300"] = _plot_sigma300_bar(fits,
                                              os.path.join(args.fig_dir, f"03_sigma300_vs_expt{args.traj_tag}.png"))
    for k, p in figs.items():
        print(f"[03] figure[{k}] -> {os.path.relpath(p, HERE)}")

    metrics = {
        "system": "Li6PS5Cl",
        "baseline": True,
        "fine_tuned": False,
        "experiment_mS_cm": EXPT,
        "per_run": all_rows,
        "arrhenius": fits,
        "caveats": [
            "Un-fine-tuned universal MLIP baseline: expect 2-40% bias in volume/D/sigma "
            "vs experiment; fine-tuning is W7.",
            "Thin MD (short trajectory): MSD statistics NOT converged -> sigma is "
            "order-of-magnitude indicative only.",
            "Nernst-Einstein ignores ion correlation (no Haven ratio).",
            "Single-crystal upper bound: no grain boundaries (real polycrystals are lower).",
            "Argyrodite shows non-Arrhenius transport; linear extrapolation to 300 K adds error.",
            "S/Cl disorder sampled by a few representative orderings, not the full ensemble.",
        ],
    }
    with open(os.path.join(args.data_dir, f"metrics{args.traj_tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[03] wrote metrics.json ({len(all_rows)} runs). Baseline read-out complete.")


if __name__ == "__main__":
    main()
