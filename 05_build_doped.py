"""
05 / pipeline -- W9 doping trend: build the Cl-excess argyrodite series for MLIP-MD.

Reproduces the experimental "more Cl -> higher sigma" trend in Li6-x PS5-x Cl1+x. On the
free-anion (4a/4c) sublattice we replace S2- by Cl-, and compensate the charge by removing
Li+ (Li vacancies) -- the accepted argyrodite conductivity-enhancement mechanism (extra
site disorder + mobile-Li vacancies both lower the effective migration barrier).

Built with the SAME machinery as the Li6PS5Cl baseline (01_build_structure.py): start from
the real MP cell (mp-985592), set the target fractional occupancies on the anion + Li
sublattices, Ewald-enumerate the lowest-energy ordered approximant, verify charge neutrality
and stoichiometry, then supercell to the production 2x2x2 (~400 atoms). No hand-placed atoms.

Series (conventional cell has Z=4 formula units; 8 anion sites, 24 Li sites):
    x=0.00  Li6PS5Cl        Cl 1.00   (anchor; 4 free-S / 4 Cl / 24 Li)   -> literature 3.15 mS/cm
    x=0.25  Li5.75PS4.75Cl1.25  Cl 1.25   (3 free-S / 5 Cl / 23 Li)
    x=0.50  Li5.5PS4.5Cl1.5     Cl 1.50   (2 free-S / 6 Cl / 22 Li)       -> literature 9.4 mS/cm
    x=0.75  Li5.25PS4.25Cl1.75  Cl 1.75   (1 free-S / 7 Cl / 21 Li)

CPU-only (pymatgen); run locally, then rsync data/doped/ to Vanda for MD (hpc/run_doped.pbs).

Usage:
    python 05_build_doped.py                       # x = 0, 0.25, 0.5, 0.75, 2x2x2 supercells
    python 05_build_doped.py --supercell 1,1,1     # 52-atom cells (fast smoke)
    python 05_build_doped.py --x 0.5               # a single composition
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="Site labels are not unique")

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import structure as st  # noqa: E402  (reuse load_base / free_anion_indices / OXI / ...)

DOPED_DIR = os.path.join(HERE, "data", "doped")

# literature Cl-excess targets for the report (Yu 2018 ACS AMI 10.1021/acsami.8b07476 and the
# Cl-rich argyrodite series in the project README) -- for the trend overlay, NOT a per-x fit.
LIT_TREND = {1.00: 3.15, 1.50: 9.4, 1.30: 6.4}


def make_cl_excess_disordered(base, x):
    """Li6-x PS5-x Cl1+x as a pymatgen *disordered* Structure: set the free-anion sublattice to
    the target S2-/Cl- ratio and the Li sublattice to the (vacancy-reduced) occupancy, with
    oxidation states so the Ewald enumerator can rank orderings. Returns (disordered, counts)."""
    free_s, cl = st.free_anion_indices(base)
    anion_sites = sorted(free_s + cl)
    li_sites = [i for i, s in enumerate(base) if s.specie.symbol == "Li"]
    n_anion, n_li = len(anion_sites), len(li_sites)
    Z = n_anion // 2                       # formula units per cell (2 anion sites per formula)
    n_cl = round((1.0 + x) * Z)            # Cl on the anion sublattice
    n_free_s = n_anion - n_cl              # remaining free S2-
    n_li_t = round((6.0 - x) * Z)          # Li after vacancy compensation
    if not (0 <= n_free_s <= n_anion and 0 < n_li_t <= n_li):
        raise ValueError(f"x={x} gives non-physical counts (Cl={n_cl}, freeS={n_free_s}, Li={n_li_t})")

    dis = base.copy()
    dis.add_oxidation_state_by_element(st.OXI)
    for idx in anion_sites:
        dis.replace(idx, {"S2-": n_free_s / n_anion, "Cl-": n_cl / n_anion})
    for idx in li_sites:
        dis.replace(idx, {"Li+": n_li_t / n_li})
    if abs(dis.charge) > 1e-6:
        raise ValueError(f"x={x} disordered cell not neutral (charge={dis.charge:.4f})")
    counts = {"Z": Z, "n_Cl": n_cl, "n_free_S": n_free_s, "n_Li": n_li_t,
              "cl_content": round(n_cl / Z, 4)}
    return dis, counts


def enumerate_doped(base, x, pool=24):
    """Lowest-Ewald ordered approximant of the Cl-excess disordered cell (MD-ready, no oxi states).
    Returns (structure, ewald_energy, counts)."""
    dis, counts = make_cl_excess_disordered(base, x)
    odt = OrderDisorderedStructureTransformation(algo=2)   # fast heuristic, Ewald-ranked
    ranked = odt.apply_transformation(dis, return_ranked_list=pool)
    best = ranked[0] if isinstance(ranked, list) else {"structure": ranked, "energy": float("nan")}
    s = best["structure"].copy()
    s.remove_oxidation_states()
    return s, float(best.get("energy", float("nan"))), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--x", type=float, default=None, help="single composition (else the full series)")
    ap.add_argument("--supercell", default="2,2,2", help="diagonal reps (52-atom conv -> 416)")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    xs = [args.x] if args.x is not None else [0.0, 0.25, 0.5, 0.75]
    reps = tuple(int(v) for v in args.supercell.split(","))
    os.makedirs(DOPED_DIR, exist_ok=True)

    print("[05] loading base Li6PS5Cl (MP mp-985592, cached) ...")
    base = st.load_base(args.data_dir, api_key=args.api_key)
    base_summ = st.summary(base)
    print(f"     base conv. cell: {base_summ['n_atoms']} atoms, a={base_summ['abc'][0]} A")

    manifest = {
        "source": f"Materials Project {st.MP_ID} (Li6PS5Cl, F-43m, ordered DFT approximant)",
        "method": "Cl-excess Li6-x PS5-x Cl1+x: free-anion S2-->Cl- substitution with Li-vacancy "
                  "charge compensation; Ewald-enumerated lowest-energy ordering; 2x2x2 supercell. "
                  "Same protocol/potential as the Li6PS5Cl baseline so the trend is internally "
                  "consistent. MP/PBE lattice ~4% larger than experiment (bias carried by all x).",
        "md_protocol": "MACE-MP-0 (small), NVT Langevin 1 fs friction 0.01/fs, 600/800/1000 K, "
                       "150 ps + 10 ps equil (un-fine-tuned baseline; the Li6PS5Cl fine-tuned "
                       "model is composition-specific and deliberately NOT used).",
        "supercell": list(reps),
        "lit_trend_mS_cm": LIT_TREND,
        "compositions": {},
    }

    for x in xs:
        s, energy, counts = enumerate_doped(base, x)
        cell = st.make_supercell(s, reps) if reps != (1, 1, 1) else s
        summ = st.summary(cell)
        cl_tag = f"cl{int(round(counts['cl_content'] * 100)):03d}"   # cl100 / cl125 / cl150 / cl175
        cif = os.path.join(DOPED_DIR, f"dope_{cl_tag}.cif")
        st.write_structure(cell, cif)
        manifest["compositions"][cl_tag] = {
            "x": x, "formula": summ["formula"], "cl_content": counts["cl_content"],
            "conv_counts": counts, "ewald_energy_eV": round(energy, 3),
            "unit_cell_atoms": len(s), "supercell": list(reps), "n_atoms": summ["n_atoms"],
            "n_Li": summ["n_Li"], "abc_A": summ["abc"], "volume_A3": summ["volume"],
            "density_g_cm3": summ["density_g_cm3"], "cif": os.path.relpath(cif, HERE),
        }
        lit = LIT_TREND.get(counts["cl_content"])
        print(f"[05] x={x:<4} {summ['formula']:<16} Cl={counts['cl_content']:<4} "
              f"| {summ['n_atoms']} atoms ({summ['n_Li']} Li) rho={summ['density_g_cm3']} "
              f"| Ewald={energy:.2f} eV" + (f"  [expt ~{lit} mS/cm]" if lit else ""))

    with open(os.path.join(DOPED_DIR, "doped.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[05] wrote {len(manifest['compositions'])} cells + doped.json -> "
          f"{os.path.relpath(DOPED_DIR, HERE)}/")
    print("[05] next: rsync data/doped -> Vanda, then qsub hpc/run_doped.pbs per (comp, temp); "
          "analyse with 03_analyze_transport.py --traj-tag _dope_<cltag> --system <formula> --no-expt")


if __name__ == "__main__":
    main()
