"""Nernst-Einstein conversion, MSD upper bound, Arrhenius fit and the protocol's sigma(300 K) bounds."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

from .protocol import BOUND_LAG_PS, E_CHARGE_C, K_B_EV_K, K_B_J_K, T_REF_K


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def ne_factor_mS_cm_per_cm2_s(n_li: float, volume_A3: float, temperature_K: float) -> float:
    """Nernst-Einstein factor: sigma[mS/cm] = factor * D[cm^2/s] (monovalent, uncorrelated Li)."""
    density_cm3 = n_li / (volume_A3 * 1e-24)
    return 1e3 * density_cm3 * E_CHARGE_C ** 2 / (K_B_J_K * temperature_K)


def d_bound_from_msd(li_msd_A2: float, lag_ps: float = BOUND_LAG_PS) -> float:
    """Protocol D upper bound MSD_Li(lag)/(6 lag) in cm^2/s (1 A^2/ps = 1e-4 cm^2/s)."""
    return li_msd_A2 / (6.0 * lag_ps) * 1e-4


def arrhenius_fit(temperatures: Sequence[float], diffusivities: Sequence[float],
                  t_ref: float = T_REF_K) -> dict[str, Any]:
    """OLS of ln D on 1/T. Needs >= 2 distinct temperatures.

    Returns Ea (eV), D0 (cm^2/s), R^2, D at ``t_ref`` and, with >= 3 points, the
    95 % confidence interval of the mean line at ``t_ref`` (Student t, n-2 dof).
    """
    t = np.asarray(temperatures, dtype=float)
    d = np.asarray(diffusivities, dtype=float)
    if t.size != d.size or len(set(t.tolist())) < 2:
        raise ValueError("Arrhenius fit needs at least two distinct temperatures")
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        raise ValueError("Arrhenius fit needs positive finite D values")
    x, y = 1.0 / t, np.log(d)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    x0 = 1.0 / t_ref
    y0 = intercept + slope * x0
    ci = None
    if t.size >= 3:
        dof = t.size - 2
        s2 = ss_res / dof
        sxx = float(np.sum((x - x.mean()) ** 2))
        se = math.sqrt(s2 * (1.0 / t.size + (x0 - x.mean()) ** 2 / sxx))
        q = float(stats.t.ppf(0.975, dof))
        ci = (math.exp(y0 - q * se), math.exp(y0 + q * se))
    return {"Ea_eV": float(-slope * K_B_EV_K), "D0_cm2_s": float(math.exp(intercept)), "R2": float(r2),
            "D_ref_cm2_s": float(math.exp(y0)), "D_ref_ci95_cm2_s": ci, "n_points": int(t.size)}


def prefactor_bound(temperature_K: float, d_upper_cm2_s: float, d0_min_cm2_s: float,
                    t_ref: float = T_REF_K) -> tuple[float, float]:
    """Prefactor rule: D(T) <= d_upper and D0 >= d0_min give Ea >= kT ln(d0_min/d_upper).

    Returns (Ea lower bound in eV, upper bound of D at ``t_ref``). Ea is clamped at
    >= 0, so the bound is never looser than ``d_upper`` itself.
    """
    if not (t_ref < temperature_K and d_upper_cm2_s > 0 and d0_min_cm2_s > 0):
        raise ValueError("prefactor bound needs T > T_ref and positive D values")
    ea_min = max(0.0, K_B_EV_K * temperature_K * math.log(d0_min_cm2_s / d_upper_cm2_s))
    d_ref = d_upper_cm2_s * math.exp(-ea_min / K_B_EV_K * (1.0 / t_ref - 1.0 / temperature_K))
    return ea_min, d_ref


def pair_bound(t_high_K: float, d_high_lower_cm2_s: float, t_low_K: float, d_low_upper_cm2_s: float,
               t_ref: float = T_REF_K) -> tuple[float, float]:
    """Two-point slope bound: D(T_high) >= d_high_lower and D(T_low) <= d_low_upper.

    Ea >= k ln(d_high_lower/d_low_upper) / (1/T_low - 1/T_high); the bound on D at
    ``t_ref`` is extrapolated from the low-temperature upper bound (Ea clamped >= 0).
    """
    if not (t_ref < t_low_K < t_high_K and d_high_lower_cm2_s > 0 and d_low_upper_cm2_s > 0):
        raise ValueError("pair bound needs T_ref < T_low < T_high and positive D values")
    ea_min = max(0.0, K_B_EV_K * math.log(d_high_lower_cm2_s / d_low_upper_cm2_s)
                 / (1.0 / t_low_K - 1.0 / t_high_K))
    d_ref = d_low_upper_cm2_s * math.exp(-ea_min / K_B_EV_K * (1.0 / t_ref - 1.0 / t_low_K))
    return ea_min, d_ref
