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
    """Draw MD points + Arrhenius fit + 300 K star, with a direct in-colour label beside each fit
    line (no legend). yfunc maps (sigma, T) to the panel's y-value; the caller owns the y-limits,
    the experiment reference and axis labels."""
    label = {"mace": "MACE", "mattersim": "MatterSim"}
    side = {"mace": 1, "mattersim": -1}          # +1 = label above its line, -1 = below
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
                    zorder=3)
        fit = fits.get(mlip)
        if fit and np.isfinite(fit["Ea_eV"]):
            ax.plot(1000.0 / Tgrid, yfunc(_sigma_model(Tgrid, fit), Tgrid), color=c, lw=1.0)
            ax.scatter([1000.0 / 300], [yfunc(fit["sigma300_mS_cm"], 300.0)], color=c,
                       marker="*", s=70, edgecolor=ps.PALETTE["neutral_black"],
                       linewidth=0.4, zorder=4)
            s = side.get(mlip, 1)                # direct label beside the fit line (replaces legend)
            x_lab, T_lab = 1.4, 1000.0 / 1.4
            ax.annotate(label.get(mlip, mlip), (x_lab, yfunc(_sigma_model(T_lab, fit), T_lab)),
                        textcoords="offset points", xytext=(0, 7 * s), ha="center",
                        va="bottom" if s > 0 else "top", color=c, fontsize=ps.FS_ANNOT,
                        fontweight="bold", zorder=5)
    ax.axvline(1000.0 / 300, ls=":", color=ps.PALETTE["neutral_mid"], lw=0.6)


def _plot_arrhenius(per_mlip, fits, path, expt_main=EXPT_MAIN):
    """Arrhenius plot in the linear form that is actually fitted: log10(sigma*T) vs 1000/T
    (ln(sigma*T) = -Ea/kT + c). Points = MD (kinisi error bars), line = fit, star = 300 K
    extrapolation, X = experiment (omitted when expt_main is None, e.g. literature-blank
    W11 leads); corner box gives Ea, sigma300, R^2. Single panel."""
    fig, ax = plt.subplots(figsize=(ps.COL_SINGLE_IN, 3.2))
    yT = lambda s, T: np.log10(s * T)
    _arrhenius_panel(ax, per_mlip, fits, yT)
    data_yT = [yT(r["sigma_mS_cm"], r["T"]) for rows in per_mlip.values() for r in rows]
    # y-floor must clear the LOWEST 300 K reference (expt or a fit star, e.g. fine-tuned)
    ref_yT = ([yT(expt_main, 300.0)] if expt_main else []) + [
        yT(f["sigma300_mS_cm"], 300.0) for f in fits.values()
        if f and np.isfinite(f.get("sigma300_mS_cm", np.nan))]
    ax.set_ylim(min(ref_yT) - 0.25, max(data_yT) + 0.35)
    ax.text(1000.0 / 300 - 0.03, 0.97, "300 K", transform=ax.get_xaxis_transform(),
            color=ps.PALETTE["neutral_mid"], fontsize=ps.FS_ANNOT, ha="right", va="top")
    if expt_main:
        # experiment is measured at room T only -> a single reference point at 300 K
        ax.scatter([1000.0 / 300], [yT(expt_main, 300.0)], marker="X", s=40, color=EXPT_COLOR, zorder=5)
        ax.annotate("expt", (1000.0 / 300, yT(expt_main, 300.0)), textcoords="offset points",
                    xytext=(-5, 0), ha="right", va="center", color=EXPT_COLOR, fontsize=ps.FS_ANNOT)
    # per-MLIP fit stats in one corner box
    stats = [f"{m}: $E_a$ {fits[m]['Ea_eV']:.2f} eV · $\\sigma_{{300}}$ "
             f"{fits[m]['sigma300_mS_cm']:.1f} · $R^2$ {fits[m]['R2']:.3f}"
             for m in per_mlip if fits.get(m) and np.isfinite(fits[m].get("R2", np.nan))]
    if stats:
        ax.text(0.03, 0.03, "\n".join(stats), transform=ax.transAxes,
                fontsize=ps.FS_ANNOT, ha="left", va="bottom", color=ps.PALETTE["neutral_dark"])
    ax.set(xlabel="1000 / $T$  (K$^{-1}$)", ylabel="log$_{10}$ $\\sigma T$  (mS cm$^{-1}$ K)")
    src_rows = []
    for m, rows in per_mlip.items():
        f = fits.get(m, {}) or {}
        for r in rows:
            src_rows.append([m, r["T"], r["sigma_mS_cm"], r.get("kinisi_sigma_std_mS_cm", ""),
                             f.get("Ea_eV", ""), f.get("sigma300_mS_cm", ""), f.get("R2", "")])
    ps.save_source_data(path, ["mlip", "T_K", "sigma_mS_cm", "kinisi_sigma_std_mS_cm",
                               "fit_Ea_eV", "fit_sigma300_mS_cm", "fit_R2"], src_rows)
    return ps.finalize_figure(fig, path)[0]


def _plot_sigma300_bar(fits, path, expt=EXPT):
    """Extrapolated sigma(300 K) vs experiment (single-column, log y; expt bars omitted
    when expt is falsy). No chart title; material + fit provenance belong in the caption."""
    fig, ax = plt.subplots(figsize=(ps.COL_SINGLE_IN, 3.0))
    labels, vals, colors, cats = [], [], [], []
    pretty = {"mace": "MACE", "mattersim": "MatterSim"}   # run type is in the caption/filename
    for mlip, fit in fits.items():
        if fit and np.isfinite(fit.get("sigma300_mS_cm", np.nan)):
            labels.append(pretty.get(mlip, mlip))
            vals.append(fit["sigma300_mS_cm"])
            colors.append(MLIP_COLORS.get(mlip, ps.PALETTE["neutral_dark"]))
            cats.append(pretty.get(mlip, mlip))
    for name, v in (expt or {}).items():
        lab = "sintered\n(expt)" if "sinter" in name else "mech.chem\n(expt)"
        labels.append(lab)
        vals.append(v)
        colors.append(ps.PALETTE["neutral_mid"])   # experiment = neutral reference
        cats.append(lab.replace("\n", " "))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=ps.FS_TICK)
    ax.set_ylabel("$\\sigma$(300 K)  (mS cm$^{-1}$, log)")
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.2g}", ha="center", va="bottom", fontsize=ps.FS_ANNOT)
    ps.save_source_data(path, ["category", "sigma300_mS_cm"], list(zip(cats, vals)))
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
    ap.add_argument("--system", default="Li6PS5Cl", help="material label written to metrics "
                    "(W11 leads: the lead formula)")
    ap.add_argument("--no-expt", action="store_true", help="omit the Li6PS5Cl experimental "
                    "references (W11 leads are literature-blank -- no expt sigma exists)")
    args = ap.parse_args()
    expt, expt_main = (None, None) if args.no_expt else (EXPT, EXPT_MAIN)
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
            vs = (f"(expt {expt_main}; ratio {fit['sigma300_mS_cm'] / expt_main:.2g}x)"
                  if expt_main else "(no expt reference)")
            print(f"[03] {mlip}: Ea={fit['Ea_eV']:.3f} eV  sigma300={fit['sigma300_mS_cm']:.2e} "
                  f"mS/cm  {vs}  R2={fit['R2']:.3f}")
        else:
            print(f"[03] {mlip}: need >=2 temperatures for Arrhenius (have {fit['n_points']})")

    os.makedirs(args.fig_dir, exist_ok=True)
    figs = {}
    if any(len(r) >= 1 for r in per_mlip.values()):
        figs["arrhenius"] = _plot_arrhenius(per_mlip, fits,
                                            os.path.join(args.fig_dir, f"03_arrhenius{args.traj_tag}.png"),
                                            expt_main=expt_main)
    if any(np.isfinite(f.get("sigma300_mS_cm", np.nan)) for f in fits.values()):
        name = f"03_sigma300{'_vs_expt' if expt else ''}{args.traj_tag}.png"
        figs["sigma300"] = _plot_sigma300_bar(fits, os.path.join(args.fig_dir, name), expt=expt)
    for k, p in figs.items():
        print(f"[03] figure[{k}] -> {os.path.relpath(p, HERE)}")

    caveats = [
        "Un-fine-tuned universal MLIP baseline: expect a systematic bias in volume/D/sigma "
        "(this project measured 0.16-9.4x vs experiment for Li6PS5Cl depending on protocol).",
        "Check MSD convergence per run (the 50->150->200 ps ladder): short trajectories "
        "give order-of-magnitude sigma only.",
        "Nernst-Einstein ignores ion correlation (no Haven ratio).",
        "Single-crystal upper bound: no grain boundaries (real polycrystals are lower).",
        "High-T -> 300 K Arrhenius extrapolation adds error if transport is non-Arrhenius.",
    ]
    if args.system == "Li6PS5Cl":
        caveats.append("S/Cl disorder sampled by a few representative orderings, "
                       "not the full ensemble.")
    metrics = {
        "system": args.system,
        "baseline": True,
        "fine_tuned": False,
        "experiment_mS_cm": expt,
        "per_run": all_rows,
        "arrhenius": fits,
        "caveats": caveats,
    }
    with open(os.path.join(args.data_dir, f"metrics{args.traj_tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[03] wrote metrics.json ({len(all_rows)} runs). Baseline read-out complete.")


if __name__ == "__main__":
    main()
