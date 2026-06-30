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

NAVY = "#1F4E79"
MLIP_COLORS = {"mace": "#2E75B6", "mattersim": "#C00000"}
# experimental Li6PS5Cl room-T ionic conductivity references (mS/cm)
EXPT = {"Li6PS5Cl (sintered)": 3.15, "Li6PS5Cl (mechanochem.)": 1.33}
EXPT_MAIN = 3.15
TRAJ_RE = re.compile(r"(?P<mlip>[a-zA-Z]+)_(?P<T>\d+)K\.traj$")


def _load_run_params(data_dir):
    """timestep_fs + log_every per (mlip,T) from md_runs.json (fallback handled by caller)."""
    path = os.path.join(data_dir, "md_runs.json")
    params = {}
    if os.path.exists(path):
        for r in json.load(open(path)).get("runs", []):
            params[(r["mlip"], int(r["temperature_K"]))] = (
                r.get("timestep_fs", 1.0), r.get("log_every", 50))
    return params


def _sigma_model(T, fit):
    """Arrhenius model sigma(T) in mS/cm from the ln(sigma*T) fit."""
    return np.exp(fit["slope"] / T + fit["intercept"]) / T


def _plot_arrhenius(per_mlip, fits, path):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    Tgrid = np.linspace(290, 1100, 200)
    for mlip, rows in per_mlip.items():
        c = MLIP_COLORS.get(mlip, NAVY)
        T = np.array([r["T"] for r in rows])
        sig = np.array([r["sigma_mS_cm"] for r in rows])
        ax.scatter(1000.0 / T, np.log10(sig), color=c, s=45, zorder=3,
                   label=f"{mlip} (MD points)")
        fit = fits.get(mlip)
        if fit and np.isfinite(fit["Ea_eV"]):
            ax.plot(1000.0 / Tgrid, np.log10(_sigma_model(Tgrid, fit)), color=c, lw=1.5,
                    label=f"{mlip} fit: Ea={fit['Ea_eV']:.2f} eV, "
                          f"sigma300={fit['sigma300_mS_cm']:.2e} mS/cm")
            ax.scatter([1000.0 / 300], [np.log10(fit["sigma300_mS_cm"])], color=c,
                       marker="*", s=180, edgecolor="k", zorder=4)
    ax.axvline(1000.0 / 300, ls=":", color="grey", lw=1)
    ax.text(1000.0 / 300 + 0.02, ax.get_ylim()[0] + 0.2, "300 K", color="grey", fontsize=9)
    ax.axhline(np.log10(EXPT_MAIN), ls="--", color="green", lw=1.2)
    ax.text(0.98, 0.02, f"expt (sintered) {EXPT_MAIN} mS/cm", color="green", fontsize=9,
            ha="right", va="bottom", transform=ax.transAxes)
    ax.set(xlabel="1000 / T  (1/K)", ylabel="log$_{10}$  sigma  (mS/cm)",
           title="Arrhenius: baseline MLIP-MD vs experiment (Li$_6$PS$_5$Cl)")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_sigma300_bar(fits, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    labels, vals, colors = [], [], []
    for mlip, fit in fits.items():
        if fit and np.isfinite(fit.get("sigma300_mS_cm", np.nan)):
            labels.append(f"{mlip}\n(baseline)")
            vals.append(fit["sigma300_mS_cm"])
            colors.append(MLIP_COLORS.get(mlip, NAVY))
    for name, v in EXPT.items():
        labels.append(name)
        vals.append(v)
        colors.append("#70AD47")
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("sigma(300 K)  (mS/cm, log scale)")
    ax.set_title("Extrapolated room-T conductivity vs experiment")
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.2g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--specie", default="Li")
    ap.add_argument("--timestep", type=float, default=1.0, help="fallback fs if no md_runs.json")
    ap.add_argument("--log-every", type=int, default=50, help="fallback frame spacing")
    args = ap.parse_args()

    traj_files = sorted(glob.glob(os.path.join(args.data_dir, "traj", "*.traj")))
    if not traj_files:
        sys.exit(f"no trajectories in {args.data_dir}/traj -- run 02_baseline_md.py first")
    run_params = _load_run_params(args.data_dir)

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
                                            os.path.join(args.fig_dir, "03_arrhenius.png"))
    if any(np.isfinite(f.get("sigma300_mS_cm", np.nan)) for f in fits.values()):
        figs["sigma300"] = _plot_sigma300_bar(fits,
                                              os.path.join(args.fig_dir, "03_sigma300_bar.png"))
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
    with open(os.path.join(args.data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[03] wrote metrics.json ({len(all_rows)} runs). Baseline read-out complete.")


if __name__ == "__main__":
    main()
