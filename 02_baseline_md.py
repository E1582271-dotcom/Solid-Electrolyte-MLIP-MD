"""
02 / pipeline -- baseline multi-temperature NVT MD for Li6PS5Cl (Project 2, W6).

For each MLIP in {mace, mattersim} x each temperature, load a config from 01, run Langevin
NVT MD, and save the production trajectory to data/traj/<mlip>_<T>K.traj plus run metadata.
These are **un-fine-tuned** baselines on purpose -- 03 turns the trajectories into sigma(T)
and we read the bias off against experiment before fine-tuning (W7).

THIN baseline defaults (the user's "validate the pipeline first" choice): 50 ps production
after 5 ps equilibration, 1 fs steps. Statistics are NOT converged at this length -- scale
--steps up (and --supercell in 01) for the W8 production run on a rented GPU.

Run this on a GPU (Colab T4). Heavy MLIP packages are intentionally not installed locally.

Usage (Colab):
    python 02_baseline_md.py                                  # both MLIPs, 600/800/1000 K
    python 02_baseline_md.py --mlip mace --temps 600,1000     # quick smoke subset
    python 02_baseline_md.py --steps 200000                   # longer (toward convergence)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import md as mdrun  # noqa: E402
from src import plotstyle as pstyle  # noqa: E402

TEMP_COLORS = pstyle.TEMP_COLORS


def _plot_md(records, mlip, fig_dir, tag=""):
    """Temperature + energy traces per temperature for one MLIP (sanity: stable, no blow-up).
    Nature journal-final: double-column, thin traces, bold a/b letters, no chart titles/suptitle
    (the MLIP + system belong in the caption)."""
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(pstyle.COL_DOUBLE_IN, 2.8))
    rows = []
    for rec in records:
        s = rec["series"]
        t_ps = [stp * rec["timestep_fs"] / 1000.0 for stp in s["step"]]
        T = rec["temperature_K"]
        c = TEMP_COLORS.get(int(T), pstyle.PALETTE["neutral_dark"])
        axes[0].plot(t_ps, s["T"], color=c, lw=0.7, label=f"{int(T)} K")
        axes[1].plot(t_ps, s["E_per_atom"], color=c, lw=0.7, label=f"{int(T)} K")
        Ta, Ea, tp = np.asarray(s["T"], float), np.asarray(s["E_per_atom"], float), np.asarray(t_ps, float)
        drift = (np.polyfit(tp, Ea, 1)[0] * 1000.0) if len(tp) > 1 else 0.0   # meV/atom/ps
        rows.append([int(T), f"{Ta.mean():.2f}", f"{Ta.std():.2f}", f"{Ea.mean():.5f}", f"{drift:.4f}"])
    for T in {int(r["temperature_K"]) for r in records}:
        axes[0].axhline(T, ls="--", lw=0.5, color=TEMP_COLORS.get(T, pstyle.PALETTE["neutral_mid"]))
    axes[0].set(xlabel="time (ps)", ylabel="temperature (K)")
    axes[1].set(xlabel="time (ps)", ylabel="potential energy (eV/atom)")
    for ax, ltr in zip(axes, "ab"):
        pstyle.add_panel_label(ax, ltr)
    # traces fill the frame -> put the T legend BELOW both panels so it never overlaps a curve
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="target $T$", title_fontsize=pstyle.FS_LEGEND,
               fontsize=pstyle.FS_LEGEND, loc="lower center", ncol=len(labels),
               frameon=False, columnspacing=1.6, handletextpad=0.4, bbox_to_anchor=(0.5, -0.02))
    out = os.path.join(fig_dir, f"02_md_stability_{mlip}{tag}.png")
    pstyle.save_source_data(out, ["temperature_K", "mean_T_K", "std_T_K",
                                  "mean_E_per_atom_eV", "drift_meV_atom_ps"], rows)
    return pstyle.finalize_figure(fig, out)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mlip", default="both", choices=["mace", "mattersim", "both"])
    ap.add_argument("--temps", default="600,800,1000")
    ap.add_argument("--steps", type=int, default=50000, help="production MD steps (1 fs each)")
    ap.add_argument("--equilib", type=int, default=5000, help="equilibration steps before recording")
    ap.add_argument("--config", type=int, default=0, help="which config{N}.cif from 01")
    ap.add_argument("--supercell-tag", default="", help="e.g. '_sc222' to match a 01 supercell run")
    ap.add_argument("--timestep", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50, help="steps between recorded frames")
    ap.add_argument("--mace-model", default="small",
                    help="'small'|'medium'|'large' or a path to a fine-tuned .model (W7)")
    ap.add_argument("--mattersim-model", default=None)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (auto if unset)")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--traj-tag", default="", help="suffix for an isolated run, e.g. '_long' "
                    "-> data/traj_long/ + md_runs_long.json (won't clobber the 50ps baseline)")
    args = ap.parse_args()
    pstyle.apply_publication_style()

    from ase.io import read

    temps = [float(t) for t in args.temps.split(",")]
    mlips = ["mace", "mattersim"] if args.mlip == "both" else [args.mlip]
    traj_dir = os.path.join(args.data_dir, f"traj{args.traj_tag}")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    cif = os.path.join(args.data_dir, f"config{args.config}{args.supercell_tag}.cif")
    if not os.path.exists(cif):
        sys.exit(f"missing {cif} -- run 01_build_structure.py first")
    atoms0 = read(cif)
    print(f"[02] cell: {atoms0.get_chemical_formula()}  {len(atoms0)} atoms  "
          f"a={atoms0.cell.lengths()[0]:.3f} A  (config {args.config})")
    print(f"[02] BASELINE (un-fine-tuned) | {args.steps*args.timestep/1000:.0f} ps prod "
          f"+ {args.equilib*args.timestep/1000:.0f} ps equil | MLIPs={mlips} | T={temps}")

    runs_meta = []
    for mlip in mlips:
        print(f"\n[02] === {mlip} ===  loading potential (first run downloads weights) ...")
        calc = mdrun.load_calculator(
            mlip, device=args.device, mace_model=args.mace_model,
            mattersim_model=args.mattersim_model,
        )
        records = []
        for T in temps:
            traj_path = os.path.join(traj_dir, f"{mlip}_{int(T)}K.traj")
            print(f"[02] {mlip} @ {int(T)} K -> {os.path.relpath(traj_path, HERE)}")
            summ = mdrun.run_nvt(
                atoms0, calc, temperature_K=T, n_steps=args.steps,
                timestep_fs=args.timestep, equilib_steps=args.equilib,
                traj_path=traj_path, log_every=args.log_every,
            )
            print(f"     mean T={summ['mean_T']}K  E/atom={summ['mean_E_per_atom']}eV  "
                  f"drift={summ['drift_meV_atom_ps']} meV/atom/ps  "
                  f"{summ['n_frames']} frames  ({summ['wall_seconds']}s)")
            records.append({"mlip": mlip, "config": args.config, **summ})
        fig = _plot_md(records, mlip, args.fig_dir, args.traj_tag)
        print(f"[02] {mlip} health figure -> {os.path.relpath(fig, HERE)}")
        # drop the bulky per-frame series before persisting metadata
        for r in records:
            r.pop("series", None)
        runs_meta.extend(records)

    meta_path = os.path.join(args.data_dir, f"md_runs{args.traj_tag}.json")
    with open(meta_path, "w") as f:
        json.dump({"baseline": True, "fine_tuned": False, "runs": runs_meta}, f, indent=2)
    print(f"\n[02] wrote {len(runs_meta)} run records -> {os.path.relpath(meta_path, HERE)}")
    print("[02] done. Next: python 03_analyze_transport.py")


if __name__ == "__main__":
    main()
