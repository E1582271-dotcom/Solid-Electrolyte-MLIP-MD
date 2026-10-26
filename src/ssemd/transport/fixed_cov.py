"""Fixed-covariance estimator used for the seed 0/1/2 replicates of the six census leads.

Every replicate trajectory (and the seed-0 census trajectory of each lead) was re-analysed with
one estimator: periodic unwrapping at the original 50 fs spacing, removal of the non-Li mean
drift, time origins every 1 ps, the full 200 ps trajectory, lags 60-100 ps on a 1 ps grid, the
kinisi covariance matrix at condition limit 1e8, and the analytic conditional posterior below
(generalised least squares with a free intercept and a non-negative slope). These are the
``fixed_cov_analytic`` records of data/runs.csv.

Reference code: the published values are not recomputed here (see docs/methods.md).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import truncnorm

SETTINGS = {
    "trajectory": "full 200 ps",
    "origin_spacing_ps": 1.0,
    "lag_window_ps": [60.0, 100.0],
    "lag_grid_ps": 1.0,
    "covariance_cond_max": 1e8,
    "posterior": "fixed covariance, free intercept, slope >= 0 (truncated normal)",
}


def _linear_model(t_ps, msd_A2, covariance_A4):
    t, y, c = (np.asarray(a, float) for a in (t_ps, msd_A2, covariance_A4))
    if t.ndim != 1 or y.shape != t.shape or len(t) < 3 or c.shape != (len(t), len(t)):
        raise ValueError("Invalid time/MSD/covariance dimensions")
    if not all(np.isfinite(v).all() for v in (t, y, c)):
        raise ValueError("Non-finite input")
    if not np.allclose(c, c.T, rtol=1e-10, atol=0):
        raise ValueError("Covariance is not symmetric")
    try:
        lower = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance is not positive definite") from exc
    a = solve_triangular(lower, np.column_stack((t, np.ones_like(t))), lower=True)
    b = solve_triangular(lower, y, lower=True)
    beta, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank != 2:
        raise ValueError("Unidentifiable slope/intercept")
    # Only the 2x2 parameter precision is solved; C is never explicitly inverted.
    v = np.linalg.solve(a.T @ a, np.eye(2))
    if not np.isfinite(v).all() or v[0, 0] <= 0:
        raise ValueError("Invalid parameter covariance")
    return t, y, c, lower, beta, v


def fixed_covariance_reference(t_ps, msd_A2, covariance_A4) -> dict:
    """Marginal slope posterior with a flat free intercept and a flat slope >= 0 prior.

    Positive quantiles imposed by the prior do not demonstrate physical MSD growth; the
    replicate analysis also required MSD growth >= 1 A^2 and a log-log slope of 0.5-1.5.
    """
    _, _, _, _, beta, v = _linear_model(t_ps, msd_A2, covariance_A4)
    sd = np.sqrt(v[0, 0])
    q = truncnorm.ppf([0.025, 0.5, 0.975], -beta[0] / sd, np.inf, loc=beta[0], scale=sd)
    d = q / 6 * 1e-4
    if not np.isfinite(d).all() or not d[2] > d[0]:
        raise ValueError("Invalid conditional posterior reference")
    return {"GLS_slope_A2_ps": float(beta[0]), "GLS_intercept_A2": float(beta[1]),
            "GLS_slope_sd_A2_ps": float(sd), "GLS_D_cm2_s": float(beta[0] / 6 * 1e-4),
            "D_lo_cm2_s": float(d[0]), "D_cm2_s": float(d[1]), "D_hi_cm2_s": float(d[2]),
            "prior": "gradient >= 0; intercept free; covariance fixed"}


def covariance_for_group(group, lo_ps: float, cond_max: float):
    """kinisi's covariance construction for a scipp MSD group, independent of its sampler."""
    import scipp as sc
    from kinisi.diffusion import Diffusion

    diff = Diffusion(group)
    diff._start_dt = sc.scalar(lo_ps, unit="ps")
    diff._cond_max = cond_max
    diff._recondition = False
    diff.diff_regime = np.flatnonzero(group["da"].coords["time interval"].values >= lo_ps)[0]
    return np.asarray(diff.compute_covariance_matrix().values)
