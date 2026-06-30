"""
Li6PS5Cl argyrodite structure builder for the MLIP-MD conductivity pipeline (Project 2).

Honest provenance (surface this in the README):
- Base structure  = Materials Project **mp-985592** (Li6PS5Cl, F-43m), an *ordered
  DFT approximant* -- MP does not store the experimental site disorder. The
  conventional CIF is cached to ``data/Li6PS5Cl_mp-985592.cif`` on first fetch so all
  later runs are offline and reproducible.
- Argyrodite's defining **S2- / Cl- site disorder** on the "free anion" (4a/4c)
  sublattice is RECONSTRUCTED from the ordered cell -- the free anions are found by
  P-S distance (the 16 PS4 sulfurs sit ~2 A from P; the 4 free S sit ~4.5 A away),
  then set to 50/50 S/Cl and enumerated as ordered approximants ranked by Ewald
  electrostatic energy (``OrderDisorderedStructureTransformation``). We never
  hard-code Wyckoff coordinates, so the geometry stays as faithful as the MP source.

Known limitations (baseline-honest, repeat in README):
- MP/PBE lattice (a ~= 10.28 A) is ~4% larger than experiment (~9.86 A) -> volume bias
  that propagates into D and sigma. Un-fine-tuned MLIPs add more on top.
- The enumerated orderings are a small *representative* set, NOT the full S/Cl ensemble.
- The Li sublattice is the single MP ordering; real Li is partially occupied over the
  cage (24g/48h) sites.

This module is import-only: callers get ready-to-model pymatgen Structures / ASE Atoms.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
)

# --- target system constants -------------------------------------------------
MP_ID = "mp-985592"                 # Li6PS5Cl, F-43m, ordered DFT approximant
CACHE_CIF = "Li6PS5Cl_mp-985592.cif"
EXPT_A = 9.859                      # experimental cubic lattice parameter (A), for the bias note
FREE_ANION_CUTOFF = 2.6            # P-S distance (A) splitting PS4 sulfurs from free anions
OXI = {"Li": 1, "P": 5, "S": -2, "Cl": -1}


# --- API key + MP fetch ------------------------------------------------------
def _find_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """MP API key from (1) arg, (2) $MP_API_KEY, (3) the shared project-1 key file."""
    if explicit:
        return explicit.strip()
    if os.environ.get("MP_API_KEY"):
        return os.environ["MP_API_KEY"].strip()
    here = os.path.dirname(os.path.abspath(__file__))
    shared = os.path.normpath(
        os.path.join(here, "..", "..", "project1_screening", "mp_api_key.txt")
    )
    if os.path.exists(shared):
        return open(shared).read().strip()
    return None


def fetch_from_mp(api_key: Optional[str] = None) -> Structure:
    """Pull mp-985592 and return its conventional cubic cell (needs network + key)."""
    key = _find_api_key(api_key)
    if not key:
        raise RuntimeError(
            "No Materials Project API key found. Set $MP_API_KEY, pass api_key=..., "
            "or drop the cached CIF into data/ (see load_base)."
        )
    from mp_api.client import MPRester  # imported lazily; not needed once CIF is cached

    with MPRester(key) as mpr:
        prim = mpr.get_structure_by_material_id(MP_ID)
    return SpacegroupAnalyzer(prim).get_conventional_standard_structure()


def load_base(data_dir: str, api_key: Optional[str] = None) -> Structure:
    """Conventional Li6PS5Cl cell, cached. Loads ``data/<CACHE_CIF>`` if present,
    otherwise fetches from MP and writes the cache (so subsequent runs are offline)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, CACHE_CIF)
    if os.path.exists(path):
        return Structure.from_file(path)
    struct = fetch_from_mp(api_key)
    struct.to(filename=path)
    return struct


def from_cif(path: str) -> Structure:
    """Load any structure from a CIF (e.g. an ICSD disordered argyrodite, or LGPS)."""
    return Structure.from_file(path)


# --- disorder reconstruction + enumeration -----------------------------------
def free_anion_indices(struct: Structure, cutoff: float = FREE_ANION_CUTOFF):
    """Return (free_S_indices, Cl_indices): the 4a/4c 'free anion' sublattice.

    Free S are the sulfurs NOT bonded to any P (PS4 sulfurs are within ``cutoff``)."""
    p_idx = [i for i, s in enumerate(struct) if s.specie.symbol == "P"]
    if not p_idx:
        raise ValueError("No P found -- is this an argyrodite cell?")
    dm = struct.distance_matrix
    free_s, cl = [], []
    for i, site in enumerate(struct):
        sym = site.specie.symbol
        if sym == "Cl":
            cl.append(i)
        elif sym == "S" and min(dm[i][p] for p in p_idx) >= cutoff:
            free_s.append(i)
    return free_s, cl


def make_disordered(struct: Structure) -> Structure:
    """Add oxidation states and set the free-anion sites to 50/50 S2-/Cl-
    (the argyrodite site disorder), returning a charge-neutral disordered Structure."""
    free_s, cl = free_anion_indices(struct)
    dis = struct.copy()
    dis.add_oxidation_state_by_element(OXI)
    for idx in free_s + cl:
        dis.replace(idx, {"S2-": 0.5, "Cl-": 0.5})
    if abs(dis.charge) > 1e-6:
        raise ValueError(f"Disordered cell not neutral (charge={dis.charge:.3f})")
    return dis


def enumerate_orderings(struct: Structure, n_configs: int = 2, pool: int = 16):
    """Return up to ``n_configs`` symmetry-distinct ordered approximants of the S/Cl
    disorder, lowest Ewald energy first (config 0 = the ground-state arrangement).

    Oxidation states are stripped from the returned structures so they are MD-ready."""
    dis = make_disordered(struct)
    odt = OrderDisorderedStructureTransformation(algo=2)
    ranked = odt.apply_transformation(dis, return_ranked_list=pool)
    matcher = StructureMatcher()
    distinct, energies = [], []
    for item in ranked:
        s = item["structure"]
        if any(matcher.fit(s, kept) for kept in distinct):
            continue
        distinct.append(s)
        energies.append(item["energy"])
        if len(distinct) >= n_configs:
            break
    out = []
    for s in distinct:
        s = s.copy()
        s.remove_oxidation_states()
        out.append(s)
    return out, energies[: len(out)]


# --- supercell + export helpers ----------------------------------------------
def make_supercell(struct: Structure, reps=(1, 1, 1)) -> Structure:
    """Diagonal supercell. reps=(2,2,2) on the 52-atom conv. cell -> 416 atoms."""
    sc = struct.copy()
    sc.make_supercell(list(reps))
    return sc


def to_ase(struct: Structure):
    """pymatgen Structure -> ASE Atoms (oxidation states stripped first)."""
    s = struct.copy()
    s.remove_oxidation_states()
    return s.to_ase_atoms()


def write_structure(struct: Structure, path: str) -> str:
    """Write CIF or POSCAR (by extension). Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    struct.to(filename=path)
    return path


def summary(struct: Structure) -> dict:
    """Compact, JSON-friendly description used by the driver + metrics."""
    n_li = sum(1 for s in struct if s.specie.symbol == "Li")
    a = struct.lattice.abc[0]
    return {
        "formula": struct.composition.reduced_formula,
        "n_atoms": len(struct),
        "n_Li": n_li,
        "abc": [round(x, 4) for x in struct.lattice.abc],
        "angles": [round(x, 2) for x in struct.lattice.angles],
        "volume": round(struct.volume, 2),
        "density_g_cm3": round(float(struct.density), 4),
        "a_vs_expt_pct": round(100 * (a - EXPT_A) / EXPT_A, 2),  # PBE/MP lattice bias
    }
