"""
Calculator-agnostic NVT molecular dynamics for the Li6PS5Cl baseline (Project 2, W6).

Two **un-fine-tuned universal MLIPs** are supported so we can compare them on the same
cell (the user's W5-W6 decision): MACE-MP-0 and MatterSim. Running the same trajectory
through both is the honest "see the baseline bias" step -- neither is fine-tuned to
sulfides yet (that is W7), so expect a few-to-tens-of-percent error in volume / D / sigma.

Heavy deps (torch / mace-torch / mattersim) are imported lazily inside ``load_calculator``
so this module imports fine on a laptop with only pymatgen/ASE -- the MD itself runs on a
GPU (Colab T4 for the thin baseline; AutoDL RTX 5090 for W8 production).

import-only.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

SUPPORTED = ("mace", "mattersim", "lj")


def pick_device(prefer: Optional[str] = None) -> str:
    """'cuda' if available, else Apple 'mps', else 'cpu'. ``prefer`` overrides."""
    if prefer:
        return prefer
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_calculator(
    mlip: str,
    device: Optional[str] = None,
    mace_model: str = "small",
    mattersim_model: Optional[str] = None,
    dtype: str = "float32",
):
    """Return an ASE calculator for ``mlip`` in {'mace','mattersim','lj'}.

    - mace      : MACE-MP-0 (``mace_model`` 'small'|'medium'|'large'); float32 on T4.
    - mattersim : MatterSim universal potential (``mattersim_model`` optional load path).
    - lj        : ASE Lennard-Jones -- a CPU plumbing fixture ONLY (same convention as
                  project 3's stability.py): energies are meaningless, never a physics result.
    """
    mlip = mlip.lower()
    if mlip == "lj":                      # before pick_device: no torch on the laptop
        from ase.calculators.lj import LennardJones

        return LennardJones()
    device = pick_device(device)
    if mlip == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model=mace_model, dispersion=False,
                       default_dtype=dtype, device=device)
    if mlip == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        if mattersim_model:
            return MatterSimCalculator(load_path=mattersim_model, device=device)
        return MatterSimCalculator(device=device)
    raise ValueError(f"unknown mlip {mlip!r}; supported: {SUPPORTED}")


def run_nvt(
    atoms: Atoms,
    calc,
    temperature_K: float,
    n_steps: int,
    timestep_fs: float = 1.0,
    friction: float = 0.01,
    equilib_steps: int = 5000,
    traj_path: Optional[str] = None,
    log_every: int = 50,
    seed: int = 0,
) -> dict:
    """Run Langevin NVT MD at fixed cell. Equilibrate first, then record a production
    trajectory (one frame every ``log_every`` steps) to ``traj_path``.

    Returns a summary dict (per-frame T / E series, energy drift, wall time, traj path).
    NVT is run at the input (MP/PBE) cell volume -- no NPT relaxation -- which is a
    documented baseline choice, not an oversight.
    """
    np.random.seed(seed)
    atoms = atoms.copy()
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)      # zero net linear momentum (else COM drift pollutes the MSD)
    ZeroRotation(atoms)

    dyn = Langevin(atoms, timestep=timestep_fs * units.fs,
                   temperature_K=temperature_K, friction=friction)

    t0 = time.time()
    if equilib_steps:
        dyn.run(equilib_steps)

    steps, temps, epa = [], [], []

    def _log():
        s = dyn.get_number_of_steps()
        steps.append(int(s))
        temps.append(float(atoms.get_temperature()))
        epa.append(float(atoms.get_potential_energy() / len(atoms)))

    dyn.attach(_log, interval=log_every)
    traj = None
    if traj_path:
        traj = Trajectory(traj_path, "w", atoms)
        dyn.attach(traj.write, interval=log_every)

    _log()  # record the production starting frame
    dyn.run(n_steps)
    if traj:
        traj.close()
    wall = time.time() - t0

    temps_arr = np.array(temps)
    epa_arr = np.array(epa)
    # energy drift: linear slope of E/atom over the production window (meV/atom/ps)
    if len(epa_arr) > 2:
        ps = np.array(steps) * timestep_fs / 1000.0
        drift = float(np.polyfit(ps, epa_arr, 1)[0] * 1000.0)
    else:
        drift = float("nan")

    return {
        "temperature_K": temperature_K,
        "n_steps": n_steps,
        "timestep_fs": timestep_fs,
        "production_ps": n_steps * timestep_fs / 1000.0,
        "log_every": log_every,
        "n_frames": len(steps),
        "mean_T": round(float(temps_arr.mean()), 1) if len(temps_arr) else None,
        "std_T": round(float(temps_arr.std()), 1) if len(temps_arr) else None,
        "mean_E_per_atom": round(float(epa_arr.mean()), 4) if len(epa_arr) else None,
        "drift_meV_atom_ps": round(drift, 3),
        "wall_seconds": round(wall, 1),
        "traj_path": traj_path,
        "series": {"step": steps, "T": temps, "E_per_atom": epa},
    }
