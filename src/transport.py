"""
Transport analysis: MD trajectory -> MSD -> D -> Nernst-Einstein sigma -> Arrhenius
(Project 2, W6/W8).

Two diffusion estimators, by design:
- **pymatgen** ``DiffusionAnalyzer`` -- stable backbone, returns D (cm^2/s) and the
  Nernst-Einstein conductivity (mS/cm) directly. Carries the pipeline.
- **kinisi** ``DiffusionAnalyzer.from_ase`` -- the same D but with a Bayesian
  **error bar** (the whole point of the W6 checkpoint: "why does my D have an error
  bar and where does it come from"). Best-effort: if the kinisi call fails, the
  pymatgen number still stands and the error bar is reported as NaN.

Arrhenius: fit ln(sigma*T) vs 1/T (slope = -Ea/kB), giving the activation energy and the
extrapolated room-temperature sigma(300 K).

import-only.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

# physical constants (SI unless noted)
_Q = 1.602176634e-19         # C
_KB_J = 1.380649e-23         # J/K
_KB_EV = 8.617333262e-5      # eV/K


def load_structures(traj_path: str):
    """ASE trajectory file -> (list[pymatgen Structure], ase Atoms list)."""
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor

    frames = read(traj_path, index=":")
    adaptor = AseAtomsAdaptor()
    structures = [adaptor.get_structure(f) for f in frames]
    return structures, frames


def nernst_einstein_sigma(D_cm2_s: float, n_per_cm3: float, T: float, z: int = 1) -> float:
    """Nernst-Einstein conductivity (S/cm) from a tracer D. Ignores ion correlation
    (no Haven ratio) -- a documented baseline approximation."""
    return (n_per_cm3 * (z * _Q) ** 2 / (_KB_J * T)) * D_cm2_s


def carrier_density(structure, specie: str = "Li") -> float:
    """Number density of ``specie`` (1/cm^3) from a pymatgen Structure."""
    n = sum(1 for s in structure if s.specie.symbol == specie)
    vol_cm3 = structure.volume * 1e-24  # A^3 -> cm^3
    return n / vol_cm3


def diffusivity_pymatgen(structures, specie, T, time_step_fs, step_skip, start_frac=0.3):
    """pymatgen DiffusionAnalyzer -> dict(D_cm2_s, sigma_mS_cm). ``time_step`` is the MD
    integrator step (fs) and ``step_skip`` the frames-per-step spacing of ``structures``."""
    from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer

    da = DiffusionAnalyzer.from_structures(
        structures, specie=specie, temperature=T,
        time_step=time_step_fs, step_skip=step_skip, smoothed="max",
    )
    return {
        "D_cm2_s": float(da.diffusivity),
        "sigma_mS_cm": float(da.conductivity),
        "n_frames": len(structures),
    }


def diffusivity_kinisi(traj_path, specie, time_step_fs, step_skip,
                       start_dt_fs: Optional[float] = None, n_per_cm3=None, T=None):
    """kinisi from_ase -> dict(D_cm2_s, D_std_cm2_s, sigma_mS_cm, sigma_std_mS_cm).
    Returns NaNs (and an 'error' key) if kinisi/scipp choke, so callers can fall back."""
    try:
        import scipp as sc
        from ase.io.trajectory import Trajectory
        from kinisi.analyze import DiffusionAnalyzer

        traj = Trajectory(traj_path)
        n_frames = len(traj)
        frame_dt_fs = time_step_fs * step_skip
        if start_dt_fs is None:
            start_dt_fs = 0.3 * n_frames * frame_dt_fs  # diffusive regime ~ last 70%

        kda = DiffusionAnalyzer.from_ase(
            traj, specie=specie,
            time_step=sc.scalar(time_step_fs, unit="fs"),
            step_skip=sc.scalar(int(step_skip), unit=sc.units.dimensionless),
            progress=False,
        )
        kda.diffusion(sc.scalar(start_dt_fs, unit="fs"), progress=False)

        D = kda.D  # scipp variable in (distance_unit)^2 / (time_step unit) = A^2/fs
        try:
            D_cm2s = float(D.to(unit=sc.Unit("cm^2/s")).values.mean())
            D_std = float(D.to(unit=sc.Unit("cm^2/s")).values.std())
        except Exception:
            samples = np.atleast_1d(np.asarray(D.values, dtype=float))
            A2fs_to_cm2s = 0.1  # 1 A^2/fs = 0.1 cm^2/s
            D_cm2s = float(samples.mean()) * A2fs_to_cm2s
            D_std = float(samples.std()) * A2fs_to_cm2s

        out = {"D_cm2_s": D_cm2s, "D_std_cm2_s": D_std}
        if n_per_cm3 is not None and T is not None:
            s = nernst_einstein_sigma(D_cm2s, n_per_cm3, T) * 1e3   # S/cm -> mS/cm
            s_std = nernst_einstein_sigma(D_std, n_per_cm3, T) * 1e3
            out.update({"sigma_mS_cm": s, "sigma_std_mS_cm": s_std})
        return out
    except Exception as e:  # noqa: BLE001
        return {"D_cm2_s": float("nan"), "D_std_cm2_s": float("nan"),
                "sigma_mS_cm": float("nan"), "sigma_std_mS_cm": float("nan"),
                "error": f"{type(e).__name__}: {e}"}


def arrhenius_fit(temps, sigmas_mS_cm):
    """Fit ln(sigma*T) vs 1/T. Returns Ea (eV), sigma300 (mS/cm), R^2, and fit line.
    Needs >= 2 temperatures with positive sigma."""
    T = np.asarray(temps, float)
    s = np.asarray(sigmas_mS_cm, float)
    ok = np.isfinite(s) & (s > 0) & np.isfinite(T)
    T, s = T[ok], s[ok]
    if len(T) < 2:
        return {"Ea_eV": float("nan"), "sigma300_mS_cm": float("nan"),
                "R2": float("nan"), "n_points": int(len(T))}
    x = 1.0 / T
    y = np.log(s * T)                      # ln(sigma*T)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    Ea = -slope * _KB_EV                   # slope = -Ea/kB
    sigma300 = math.exp(slope / 300.0 + intercept) / 300.0
    return {
        "Ea_eV": float(Ea),
        "sigma300_mS_cm": float(sigma300),
        "R2": float(r2),
        "n_points": int(len(T)),
        "slope": float(slope),
        "intercept": float(intercept),
    }
