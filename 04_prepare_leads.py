"""
04 / pipeline -- W11 hand-off: prepare the screening-funnel LEADS for MLIP-MD.

This is the step that closes the portfolio loop: the leads surfaced by Project 1
(MP screen + adversarial audit) and Project 3 (MatterGen -> self-consistent MLIP hull
-> S.U.N.) are turned into MD-ready supercells so 02/03 can run the SAME baseline
protocol used for Li6PS5Cl (MACE-MP-0 small, NVT Langevin, multi-T Arrhenius).

Four leads, two per source project:
  - li20si3p3s23cl : P1 lead (LGPS family), MP mp-1097035 (lowest e_hull polymorph of 3)
  - li8tis6        : P1 lead, MP mp-753546
  - li3ps4_gen016  : P3 lead, MatterGen NOVEL Li3PS4 polymorph (MACE hull e_hull=0.006)
  - lips3_gen021   : P3 lead, MatterGen LiPS3, MLIP-predicted NEW ground state (e_hull=-0.031)

Structure sources (all scripted, no hand-made files):
  - MP leads   : fetched live via mp-api (key: $MP_API_KEY or ../project1_screening/
                 mp_api_key.txt, same convention as P1/P3) and cached in data/leads/raw/.
  - P3 leads   : the MACE-relaxed cells from the Vanda production run; pull them once with
                 rsync vanda:~/AI4SSB/project3_generative/data/relaxed/relaxed_gen_0{16,21}.cif \
                       data/leads/raw/

Supercell policy (deterministic, recorded in the manifest): replicate each axis to reach
>= --min-len Angstrom, then grow the shortest axes until >= --min-atoms. This puts every
lead in the same statistics tier as the CONVERGED 52-atom/150 ps Li6PS5Cl baseline
(box >= ~10 A), not the 416-atom production tier -- honest screening-grade MD.

Outputs: data/leads/lead_<key>.cif + data/leads/leads.json (provenance manifest).
Next:    qsub -v LEAD=<key>,TEMPS=<T> run_leads.pbs   (see that file's header)
         python 03_analyze_transport.py --traj-tag _lead_<key> --system <formula> --no-expt

CPU-only; safe to run locally.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LEADS_DIR = os.path.join(HERE, "data", "leads")
RAW_DIR = os.path.join(LEADS_DIR, "raw")

# Provenance numbers below are copied from the source projects' shipped artifacts
# (P1 screen_mp_results.csv / P3 candidates_final.csv) so the manifest is self-contained.
LEADS = {
    "li20si3p3s23cl": {
        "kind": "mp",
        "mp_id": "mp-1097035",
        "from_project": "project1_screening",
        "provenance": "P1 MP screen + 15-hit adversarial audit (screen_audit.md); LGPS-family "
                      "literature-blank lead. 3 polymorphs rank top-3 of the whole screen; this "
                      "mp-id has the lowest MP e_above_hull (0.0305 eV/atom, band gap 2.31 eV). "
                      "P1 predicted log10(sigma)=-4.63 (coarse prior).",
    },
    "li8tis6": {
        "kind": "mp",
        "mp_id": "mp-753546",
        "from_project": "project1_screening",
        "provenance": "P1 MP screen lead #2 (literature-blank). MP e_above_hull=0.0035 eV/atom, "
                      "band gap 2.29 eV, spacegroup 185. P1 predicted log10(sigma)=-4.77.",
    },
    "li3ps4_gen016": {
        "kind": "p3",
        "raw_cif": "relaxed_gen_016.cif",
        "from_project": "project3_generative",
        "provenance": "MatterGen Li-P-S conditional generation (Vanda job 1215079), label gen_016: "
                      "NOVEL Li3PS4 polymorph (beta-Li3PS4 is a known fast-ion conductor), "
                      "MACE self-consistent hull e_above_hull=0.006 eV/atom (essentially on-hull), "
                      "S.U.N.=True. P3/P1 predicted log10(sigma)=-6.83.",
    },
    "lips3_gen021": {
        "kind": "p3",
        "raw_cif": "relaxed_gen_021.cif",
        "from_project": "project3_generative",
        "provenance": "MatterGen label gen_021: LiPS3, MLIP-predicted NEW ground state "
                      "(e_above_hull=-0.031 eV/atom on the MACE self-consistent hull), S.U.N.=True. "
                      "P3/P1 predicted log10(sigma)=-7.12.",
    },
}


def _find_api_key(explicit=None):
    """$MP_API_KEY, or the P1 key file (same convention as P1's 04_screen_mp.py)."""
    if explicit:
        return explicit
    if os.environ.get("MP_API_KEY"):
        return os.environ["MP_API_KEY"]
    p1_key = os.path.join(HERE, "..", "project1_screening", "mp_api_key.txt")
    if os.path.exists(p1_key):
        return open(p1_key).read().strip()
    return None


def _fetch_mp(mp_id: str, api_key=None):
    """Conventional cell from Materials Project, cached as data/leads/raw/<mp_id>.cif."""
    from pymatgen.core import Structure

    cache = os.path.join(RAW_DIR, f"{mp_id}.cif")
    if os.path.exists(cache):
        return Structure.from_file(cache)
    key = _find_api_key(api_key)
    if not key:
        sys.exit("no MP API key ($MP_API_KEY or ../project1_screening/mp_api_key.txt)")
    from mp_api.client import MPRester

    with MPRester(key) as mpr:
        s = mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=True)
    s.to(filename=cache)
    return s


def choose_supercell(structure, min_len=10.0, min_atoms=96, max_atoms=700):
    """Smallest reps with every lattice vector >= min_len; then grow the shortest axis
    until >= min_atoms. Errors out (instead of silently shrinking) past max_atoms."""
    abc = structure.lattice.abc
    reps = [max(1, math.ceil(min_len / L)) for L in abc]
    while len(structure) * reps[0] * reps[1] * reps[2] < min_atoms:
        i = min(range(3), key=lambda k: reps[k] * abc[k])
        reps[i] += 1
    n = len(structure) * reps[0] * reps[1] * reps[2]
    if n > max_atoms:
        sys.exit(f"supercell {reps} = {n} atoms exceeds --max-atoms {max_atoms}; "
                 f"tune --min-len/--min-atoms for this cell (abc={abc})")
    return reps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--min-len", type=float, default=10.0, help="min supercell edge (Angstrom)")
    ap.add_argument("--min-atoms", type=int, default=96)
    ap.add_argument("--max-atoms", type=int, default=700)
    ap.add_argument("--only", default=None, help="comma-separated lead keys (default: all)")
    args = ap.parse_args()

    from pymatgen.core import Structure

    os.makedirs(RAW_DIR, exist_ok=True)
    keys = args.only.split(",") if args.only else list(LEADS)
    manifest = {"protocol": {
        "mlip": "MACE-MP-0 (small) baseline, un-fine-tuned -- same potential/protocol as the "
                "Li6PS5Cl baseline so the cross-lead ranking is internally consistent",
        "md": "NVT Langevin, 1 fs, friction 0.01/fs, 600/800/1000 K, 150 ps + 10 ps equil "
              "(the tier at which the 52-atom Li6PS5Cl baseline was convergence-verified)",
        "note": "sigma is screening-grade (order of magnitude + ranking), NOT production-grade; "
                "the Li6PS5Cl fine-tuned model is system-specific and deliberately NOT used here.",
        "supercell_policy": f"min edge {args.min_len} A, min {args.min_atoms} atoms",
    }, "leads": {}}

    for key in keys:
        spec = LEADS[key]
        if spec["kind"] == "mp":
            s = _fetch_mp(spec["mp_id"], args.api_key)
            src = spec["mp_id"]
        else:
            raw = os.path.join(RAW_DIR, spec["raw_cif"])
            if not os.path.exists(raw):
                sys.exit(f"missing {raw} -- pull it from Vanda first:\n"
                         f"  rsync vanda:~/AI4SSB/project3_generative/data/relaxed/"
                         f"{spec['raw_cif']} {RAW_DIR}/")
            s = Structure.from_file(raw)
            src = spec["raw_cif"]

        reps = choose_supercell(s, args.min_len, args.min_atoms, args.max_atoms)
        sc = s * reps
        n_li = sum(1 for site in sc if site.specie.symbol == "Li")
        out_cif = os.path.join(LEADS_DIR, f"lead_{key}.cif")
        sc.to(filename=out_cif)

        manifest["leads"][key] = {
            "formula": sc.composition.reduced_formula,
            "source": src,
            "from_project": spec["from_project"],
            "provenance": spec["provenance"],
            "unit_cell_atoms": len(s),
            "supercell": reps,
            "n_atoms": len(sc),
            "n_Li": n_li,
            "abc_A": [round(x, 3) for x in sc.lattice.abc],
            "cif": os.path.relpath(out_cif, HERE),
        }
        print(f"[04] {key:>16s}  {sc.composition.reduced_formula:>14s}  "
              f"{len(s):>3d} atoms x {reps} = {len(sc):>3d} ({n_li} Li)  "
              f"abc={['%.2f' % x for x in sc.lattice.abc]}")

    with open(os.path.join(LEADS_DIR, "leads.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[04] wrote {len(manifest['leads'])} leads -> data/leads/leads.json")
    print("[04] next: qsub per lead+temp (see run_leads.pbs header), "
          "then 03_analyze_transport.py --traj-tag _lead_<key> --system <formula> --no-expt")


if __name__ == "__main__":
    main()
