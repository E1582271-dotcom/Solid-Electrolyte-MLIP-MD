"""
01 / pipeline -- build Li6PS5Cl argyrodite cells for baseline MD (Project 2, W5).

Pulls the real Materials Project structure (mp-985592, cached as CIF), reconstructs the
S/Cl 'free anion' site disorder, enumerates a few Ewald-ranked ordered approximants,
optionally supercells them, writes each as a CIF in data/, and saves a structure
overview figure + structures.json.

Runs on a plain CPU (pymatgen only) -- verify this step locally before touching a GPU.

Usage:
    python 01_build_structure.py                      # 2 configs, 52-atom conv. cell
    python 01_build_structure.py --supercell 2,2,2    # production-size cells (416 atoms)
    python 01_build_structure.py --n-configs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# pymatgen warns when it serialises the disordered intermediate to CIF for hashing;
# the structures we actually write are ordered (verified), so silence that one message.
warnings.filterwarnings("ignore", message="Site labels are not unique")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import structure as st  # noqa: E402

NAVY = "#1F4E79"


def _plot_configs(configs, summaries, energies, path):
    """One structural view per config (ASE plot_atoms), titled with key stats."""
    from ase.visualize.plot import plot_atoms

    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
    for j, (cfg, summ, en) in enumerate(zip(configs, summaries, energies)):
        ax = axes[0][j]
        atoms = st.to_ase(cfg)
        plot_atoms(atoms, ax, radii=0.45, rotation="20x,20y,0z")
        ax.set_axis_off()
        tag = "ground-state S/Cl" if j == 0 else f"disorder variant {j}"
        ax.set_title(
            f"config {j}  ({tag})\n{summ['formula']}  ·  {summ['n_atoms']} atoms"
            f"  ·  {summ['n_Li']} Li\nEwald rank {j}  (E={en:.1f})",
            fontsize=10, color=NAVY,
        )
    fig.suptitle(
        "Li$_6$PS$_5$Cl baseline cells (MP mp-985592, S/Cl disorder enumerated)",
        fontsize=12, color=NAVY, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--n-configs", type=int, default=2,
                    help="number of symmetry-distinct S/Cl orderings to keep")
    ap.add_argument("--supercell", default="1,1,1",
                    help="diagonal supercell, e.g. '2,2,2' (52-atom conv. cell -> 416)")
    ap.add_argument("--api-key", default=None, help="MP API key (else $MP_API_KEY / shared file)")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    reps = tuple(int(x) for x in args.supercell.split(","))

    print("[01] loading base Li6PS5Cl (MP mp-985592, cached) ...")
    base = st.load_base(args.data_dir, api_key=args.api_key)
    base_summ = st.summary(base)
    print(f"     base conv. cell: {base_summ['formula']}  {base_summ['n_atoms']} atoms  "
          f"a={base_summ['abc'][0]} A  (+{base_summ['a_vs_expt_pct']}% vs expt {st.EXPT_A} A)")

    print(f"[01] enumerating up to {args.n_configs} S/Cl orderings (Ewald-ranked) ...")
    configs, energies = st.enumerate_orderings(base, n_configs=args.n_configs)
    print(f"     kept {len(configs)} symmetry-distinct ordering(s)")

    records, fig_cells, fig_summ = [], [], []
    for j, cfg in enumerate(configs):
        cell = st.make_supercell(cfg, reps) if reps != (1, 1, 1) else cfg
        summ = st.summary(cell)
        sc_tag = "" if reps == (1, 1, 1) else f"_sc{reps[0]}{reps[1]}{reps[2]}"
        name = f"config{j}{sc_tag}.cif"
        st.write_structure(cell, os.path.join(args.data_dir, name))
        rec = {"config": j, "ewald_rank": j, "ewald_energy": round(float(energies[j]), 3),
               "cif": name, **summ}
        records.append(rec)
        fig_cells.append(cfg)        # plot the (smaller) primitive-ish conv cell, clearer
        fig_summ.append(st.summary(cfg))
        print(f"     -> {name}: {summ['n_atoms']} atoms, {summ['n_Li']} Li, "
              f"rho={summ['density_g_cm3']} g/cm3")

    fig_path = os.path.join(args.fig_dir, "01_structures.png")
    _plot_configs(fig_cells, fig_summ, energies, fig_path)
    print(f"[01] figure -> {os.path.relpath(fig_path, HERE)}")

    meta = {
        "source": f"Materials Project {st.MP_ID} (Li6PS5Cl, F-43m, ordered DFT approximant)",
        "supercell": list(reps),
        "expt_a_angstrom": st.EXPT_A,
        "note": "S/Cl free-anion disorder reconstructed from the ordered cell and "
                "Ewald-enumerated; MP/PBE lattice ~4% larger than experiment.",
        "configs": records,
    }
    with open(os.path.join(args.data_dir, "structures.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[01] wrote {len(records)} CIF(s) + structures.json to {os.path.relpath(args.data_dir, HERE)}/")
    print("[01] done. Next: run the baseline MD on Colab (02_baseline_md.py / notebook).")


if __name__ == "__main__":
    main()
