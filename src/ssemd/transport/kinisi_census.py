"""Li transport analysis of one MLIP-MD trajectory: the estimator behind every kinisi value here.

The census (186 trajectories) and the three definitive campaigns (98 trajectories) were
analysed with these settings. This module generalises the census script, which hard-coded
4000 frames, 200 ps and the 600/800/1000 K grid, to any temperature set, production length and
MD seed. Every hard-coded quantity is derived from the run's ``md_run.json``:

* frame interval   = log_every x timestep_fs; frames = n_steps // log_every; L = frames x interval;
* kinisi windows   = fractions of L (the census 100/150/200 ps and second-100 ps windows are
  0.5/0.75/1.0 and the second half of a 200 ps run), plus a fixed ``first200ps`` window whenever
  L >= 200 ps so that longer runs stay comparable with the census;
* framework and Li QC at fixed times (100 ps lag, 200 ps elapsed), independent of L.

``CONFIG`` is unchanged from the analysis that produced the published numbers; its sha256 is
recorded in each campaign's analysis ``run.json`` and checked by tests/test_transport.py.
Reference code: the published tables are not recomputed from trajectories here (the
trajectories are not in this repository; see docs/data.md). CPU only. Needs numpy, ase,
kinisi and scipp (emcee via kinisi); pymatgen and pymatgen-analysis-diffusion only for the
optional pymatgen cross-check.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np

SCHEMA = 'p2_definitive_v1/transport_generic/1'  # schema of the published result.json files
KB_EV = 8.617333262145e-5          # census analyze.py KB
LI_Z = 3
# Census analysis script (path in the original working repository) recorded in every result.json.
CENSUS_SOURCE = 'campaigns/p1_20260908_large200/analysis/analyze.py'
CENSUS_SOURCE_SHA256 = 'a4fdea48a7a59bc655e5b1e3079697ad74fc734a901b32877fa5914029b73b43'
COMPLETED_STATUSES = ('COMPLETED', 'COMPLETED_RAW_VALIDATED')

# Values copied from census analyze.py CONFIG (line 13) and its kinisi_fit/analyze_task code.
# Census window -> fraction mapping: 100ps=[0,0.5], 150ps=[0,0.75], 200ps=[0,1], second100ps=[0.5,1].
CONFIG = {
    'seed': 20260910,
    'lag_count': 60,
    'fit_start_fraction': 0.30,
    'max_lag_fraction': 0.50,
    'covariance_cond_max': 1e8,
    'covariance_sensitivity': [1e6, 1e10],
    'kinisi_samples': 1000,
    'kinisi_walkers': 32,
    'kinisi_burn': 500,
    'kinisi_thin': 10,
    'kinisi_min_posterior': 100,
    'split_rhat_max': 1.1,
    'interval': 'equal-tail 95% posterior; conditional on estimator and fit',
    'sensitivity_factor': 2,
    'framework_review_msd_A2': 1.0,
    'framework_fraction_max': 0.10,
    'displacement_threshold_A': 2.0,
    'temperature_review_relative': 0.05,
    'min_msd_growth_A2': 1.0,
    'alpha_range': [0.5, 1.5],
    'drift_reference': 'arithmetic mean non-Li displacement, matching kinisi/pymatgen',
    'unwrap': 'minimum-image fractional frame differences, cumulative sum, non-Li mean drift removed',
    'fraction_windows': {'first_half': [0.0, 0.5], 'first_three_quarters': [0.0, 0.75],
                         'full': [0.0, 1.0], 'second_half': [0.5, 1.0]},
    'fixed_windows_ps': {'first200ps': [0.0, 200.0]},
    'primary_window': 'full',
    'length_windows': ['first_half', 'first_three_quarters', 'full'],
    'half_windows': ['first_half', 'second_half'],
    # census seed = SEED + 10*task_index + position of the window in its CONFIG['windows'].
    'window_seed_index': {'first_half': 0, 'first_three_quarters': 1, 'full': 2,
                          'second_half': 3, 'first200ps': 4},
    'fixed_lag_ps': 100.0,
    'fixed_time_ps': 200.0,
    'msd_table_points': 100,
    # Census survey QC (p1_all62_analysis/analyze.py), source of trajectory_summary_186 columns
    # D_direct_MSD_cm2_s / Li_msd_100ps_A2 / Li_msd_loglog_alpha; recorded for cross-reference only.
    'survey_lags_ps': [60.0, 100.0, 5.0],
    'survey_origin_stride_frames': 20,
}

CSV_COLUMNS = [
    'p1_rank', 'formula', 'candidate_key', 'temperature_K', 'md_seed', 'production_ps', 'n_atoms',
    'n_Li', 'cell_volume_A3', 'temperature_mean_K', 'temperature_bias_fraction', 'D_kinisi_cm2_s',
    'D_kinisi_lo_cm2_s', 'D_kinisi_hi_cm2_s', 'D_direct_MSD_cm2_s', 'Li_msd_100ps_A2',
    'D_bound_msd_cm2_s', 'framework_msd_100ps_A2', 'framework_fraction_gt2A_200ps',
    'diffusion_resolved', 'framework_intact', 'eligible', 'review_flags',
    'ne_factor_mS_cm_per_cm2_s', 'trajectory_sha256', 'source_run_relpath',
]

WINDOW_COLUMNS = [
    'candidate_key', 'temperature_K', 'md_seed', 'production_ps', 'window', 'start_frame', 'end_frame',
    'start_ps', 'end_ps', 'length_ps', 'alias_of', 'kinisi_seed', 'cond_max', 'D_cm2_s', 'D_lo', 'D_hi',
    'n_posterior', 'split_rhat', 'covariance_condition', 'start_lag_ps', 'end_lag_ps',
    'diagnostic_D_cm2_s', 'diagnostic_r2', 'msd_growth_fit_A2', 'loglog_alpha', 'error',
]


# ----------------------------------------------------------------------------- small helpers
def clean(x):
    """JSON-safe copy (census analyze.py clean())."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def _csv_value(value):
    value = clean(value)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return ''
    return value


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})
    tmp.replace(path)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_sha256():
    return file_sha256(Path(__file__))


def config_sha256(settings=None):
    return hashlib.sha256(json.dumps(clean(settings or CONFIG), sort_keys=True).encode()).hexdigest()


_ABS_PATH = re.compile(r'(?:[A-Za-z]:)?/(?:Users|home|scratch|private|tmp|var|data|mnt|opt|hpctmp|app|apps|gpfs|lustre)[^\s\'"<>:,)]*')


def sanitize_text(text, roots=()):
    """Remove absolute paths (and with them user names) from free text such as tracebacks."""
    text = str(text)
    for label, root in roots:
        if root:
            text = text.replace(str(root), label)
    return _ABS_PATH.sub(lambda m: '<abs>/' + Path(m.group(0)).name, text)


def relpath(path, base):
    """Path of ``path`` relative to ``base`` without resolving symlinks; never absolute."""
    path, base = os.path.abspath(path), os.path.abspath(base)
    try:
        rel = os.path.relpath(path, base)
    except ValueError:
        return Path(path).name
    if rel.startswith('..'):
        return Path(path).name
    return Path(rel).as_posix()


def ratio(values):
    """max/min of positive values (census length/half/regularization ratios); None otherwise."""
    if not values or any(v is None or not np.isfinite(v) or v <= 0 for v in values):
        return None
    return float(max(values) / min(values))


# ----------------------------------------------------------------------------- core numerics
def unwrap(frac, cell, framework):
    """Copied from census analyze.py unwrap(): minimum-image frame differences, cumulative sum,
    arithmetic-mean non-Li drift removed. Returns (drift-corrected, raw unwrapped, drift)."""
    delta = np.diff(frac, axis=0)
    delta -= np.rint(delta)
    pos = np.concatenate([np.zeros_like(frac[:1]), np.cumsum(delta @ cell, axis=0)])
    drift = pos[:, framework].mean(axis=1, keepdims=True)
    return pos - drift, pos, drift[:, 0]


def msd(pos, mask, lags):
    """Copied from census analyze.py msd(): all-origin MSD per Cartesian axis, shape (lags, 3)."""
    arr = pos[:, mask]
    return np.array([np.mean((arr[k:] - arr[:-k]) ** 2, axis=(0, 1)) for k in lags])


def msd_at_lag_strided(positions, mask, lag, stride):
    """Copied from p1_all62_analysis/analyze.py msd_at_lag() (census survey QC, origins every 20 frames)."""
    origins = np.arange(0, len(positions) - lag, stride)
    delta = positions[origins + lag][:, mask] - positions[origins][:, mask]
    return float(np.mean(np.sum(delta * delta, axis=2)))


def ne_factor(n, vol, T):
    """Copied from census analyze.py: Nernst-Einstein factor, mS/cm per cm^2/s (vol in A^3)."""
    return n / (vol * 1e-24) * (1.602176634e-19) ** 2 / (1.380649e-23 * T) * 1e3


def frames_for_ps(t_ps, frame_dt_ps):
    return int(round(t_ps / frame_dt_ps))


def window_msd_diagnostics(pos_window, li, frame_dt_ps, settings=None):
    """Direct-MSD diagnostics of one window, copied from census analyze_task(): MSD at lag_count
    linear lags up to max_lag_fraction of the window, OLS over lags >= fit_start_fraction * L."""
    s = settings or CONFIG
    n = len(pos_window)
    L = n * frame_dt_ps
    lag = np.unique(np.linspace(1, int(n * s['max_lag_fraction']), s['lag_count']).astype(int))
    mm = msd(pos_window, li, lag).sum(axis=1)
    x = lag * frame_dt_ps
    sel = x >= s['fit_start_fraction'] * L
    slope, intercept = np.polyfit(x[sel], mm[sel], 1)
    pred = slope * x[sel] + intercept
    den = np.sum((mm[sel] - mm[sel].mean()) ** 2)
    r2 = 1 - np.sum((mm[sel] - pred) ** 2) / den if den > 0 else 0
    return {'diagnostic_D_cm2_s': float(slope / 6 * 1e-4), 'diagnostic_r2': float(r2),
            'msd_growth_fit_A2': float(mm[sel][-1] - mm[sel][0]),
            'loglog_alpha': float(np.polyfit(np.log(x[sel]), np.log(np.maximum(mm[sel], 1e-30)), 1)[0])}


def kinisi_fit(frames, L, seed, cond_max=None, timestep_fs=1.0, log_every=50, settings=None):
    """Census analyze.py kinisi_fit() with the 1 fs / 50-step clock taken from md_run.json.

    Same lag grid (lag_count linear lags up to max_lag_fraction of the window), fit start
    fit_start_fraction * L, MCMC settings, covariance check and split-chain R-hat gate."""
    import scipp as sc
    from kinisi.analyze import DiffusionAnalyzer
    s = settings or CONFIG
    cond_max = s['covariance_cond_max'] if cond_max is None else cond_max
    walkers = s['kinisi_walkers']
    lags = np.unique(np.linspace(1, int(len(frames) * s['max_lag_fraction']), s['lag_count']).astype(int))
    start = s['fit_start_fraction'] * L
    k = DiffusionAnalyzer.from_ase(
        frames, specie='Li', time_step=sc.scalar(float(timestep_fs), unit='fs'),
        step_skip=sc.scalar(int(log_every), unit='dimensionless'),
        dt=sc.array(dims=['time interval'], values=lags * float(log_every * timestep_fs), unit='fs'),
        progress=False)
    k.diffusion(sc.scalar(start * 1000, unit='fs'), cond_max=cond_max, n_samples=s['kinisi_samples'],
                n_walkers=walkers, n_burn=s['kinisi_burn'], n_thin=s['kinisi_thin'], progress=False,
                random_state=np.random.RandomState(seed))
    samples = np.asarray(k.D.to(unit=sc.Unit('cm^2/s')).values).reshape(-1)
    assert np.isfinite(samples).all() and len(samples) >= s['kinisi_min_posterior']
    eig = np.linalg.eigvalsh(k.diff._covariance_matrix.values)
    if eig[0] <= 0:
        raise ValueError('Non-positive covariance after regularization')
    chain = samples.reshape(-1, walkers)
    n = len(chain) // 2
    split = np.concatenate([chain[:n], chain[-n:]], axis=1)
    W = np.mean(np.var(split, axis=0, ddof=1))
    B = n * np.var(np.mean(split, axis=0), ddof=1)
    rhat = float(np.sqrt(((n - 1) * W / n + B / n) / W))
    if not np.isfinite(rhat) or rhat > s['split_rhat_max']:
        raise ValueError(f'Split-chain diagnostic exceeds {s["split_rhat_max"]}: {rhat}')
    frame_dt_ps = log_every * timestep_fs / 1000
    result = {'covariance_condition': float(eig[-1] / eig[0]), 'split_rhat': rhat, 'cond_max': cond_max,
              'D_cm2_s': float(np.median(samples)), 'D_lo': float(np.quantile(samples, .025)),
              'D_hi': float(np.quantile(samples, .975)), 'n_posterior': len(samples), 'start_lag_ps': start,
              'end_lag_ps': float(lags[-1] * frame_dt_ps), 'D_unit': str(k.D.unit), 'error': None}
    return result, samples


# ----------------------------------------------------------------------------- Arrhenius / bounds
def _ne_times_T(temperatures, ne):
    T = np.asarray(temperatures, dtype=float)
    ne = np.broadcast_to(np.asarray(ne, dtype=float), T.shape)
    c = ne * T          # n e^2 / (V kB) * 1e3: temperature-independent for one composition and cell
    if not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError('Nernst-Einstein factors must be positive and finite')
    if not np.allclose(c, c.mean(), rtol=1e-6, atol=0):
        raise ValueError('NE factor x T differs between points; fit one composition/cell at a time')
    return float(c.mean())


def arrhenius_fit(temperatures, D, ne, T_ref=300.0):
    """OLS Arrhenius fit ln D = ln D0 - Ea/(kB T) over any >= 2 distinct temperatures.

    ``ne`` is the Nernst-Einstein factor (mS/cm per cm^2/s) per point or one value per T, as given
    by ``ne_factor(n_Li, V, T)``; sigma(T_ref) = D(T_ref) * ne(T_ref) with ne(T) proportional to 1/T.
    Repeated temperatures (several MD seeds) are separate points. R2 is None with fewer than
    3 distinct temperatures (a two-temperature fit is exact).
    """
    T = np.asarray(temperatures, dtype=float)
    D = np.asarray(D, dtype=float)
    if T.shape != D.shape or T.ndim != 1:
        raise ValueError('temperatures and D must be 1-D arrays of equal length')
    if len(np.unique(T)) < 2:
        raise ValueError('Need at least two distinct temperatures')
    if not np.all(np.isfinite(D)) or np.any(D <= 0) or np.any(T <= 0):
        raise ValueError('D and T must be positive and finite')
    c = _ne_times_T(T, ne)
    x, y = 1 / T, np.log(D)
    slope, intercept = np.polyfit(x, y, 1)
    r2 = None
    if len(np.unique(T)) >= 3:
        den = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1 - np.sum((y - (slope * x + intercept)) ** 2) / den) if den > 0 else None
    D_ref = float(np.exp(intercept + slope / T_ref))
    return {'Ea_eV': float(-slope * KB_EV), 'D0_cm2_s': float(np.exp(intercept)),
            'sigma300_mS_cm': D_ref * c / T_ref, 'D300_cm2_s': D_ref, 'R2': r2,
            'n_points': int(len(T)), 'n_temperatures': int(len(np.unique(T))),
            'temperatures_K': sorted(set(float(t) for t in T)), 'slope_K': float(slope),
            'intercept_lnD': float(intercept), 'T_ref_K': T_ref}


def arrhenius_fit_posterior(temperatures, samples, ne, seed, draws=10000, T_ref=300.0):
    """Census analyze.py fit_posterior() generalised from exactly 3 temperatures to any >= 2
    distinct temperatures (repeated temperatures = pooled MD seeds). Weighted LS of
    ln(D * ne * T) on 1/T with per-point weights from the posterior spread; returns the
    median and 95% interval of sigma(T_ref) and Ea. R2 (of the centre fit) is None with
    fewer than 3 distinct temperatures."""
    T = np.asarray(temperatures, dtype=float)
    if len(samples) != len(T) or len(np.unique(T)) < 2:
        raise ValueError('Need one posterior per point and at least two distinct temperatures')
    if any(len(s) < 100 or not np.isfinite(s).all() or np.any(np.asarray(s) <= 0) for s in samples):
        raise ValueError('Missing or invalid posterior; no manufactured interval')
    ne = np.broadcast_to(np.asarray(ne, dtype=float), T.shape)
    rng = np.random.default_rng(seed)
    X = np.column_stack([1 / T, np.ones(len(T))])
    Y = np.array([np.log(rng.choice(np.asarray(s), draws) * f * t) for t, s, f in zip(T, samples, ne)])
    center = np.array([np.log(np.median(s) * f * t) for t, s, f in zip(T, samples, ne)])
    err = np.std(Y, axis=1, ddof=1)
    if not np.all(err > 0):
        raise ValueError('Degenerate posterior spread')
    A = np.linalg.inv(X.T @ np.diag(err ** -2) @ X) @ X.T @ np.diag(err ** -2)
    beta = A @ Y
    point = A @ center
    yhat = X @ point
    den = ((center - center.mean()) ** 2).sum()
    y_ref = beta[0] / T_ref + beta[1] - np.log(T_ref)
    sigma = np.exp(np.clip(y_ref, -700, 700))
    ea = -beta[0] * KB_EV
    r2 = float(1 - ((center - yhat) ** 2).sum() / den) if den > 0 and len(np.unique(T)) >= 3 else None
    return {'sigma300': float(np.median(sigma)), 'sigma300_lo': float(np.quantile(sigma, .025)),
            'sigma300_hi': float(np.quantile(sigma, .975)), 'Ea_eV': float(np.median(ea)),
            'Ea_lo': float(np.quantile(ea, .025)), 'Ea_hi': float(np.quantile(ea, .975)), 'R2': r2,
            'slope': float(point[0]), 'intercept': float(point[1]), 'draws': draws,
            'n_points': int(len(T)), 'n_temperatures': int(len(np.unique(T)))}


def msd_upper_bound_D(li_msd_100ps_A2, lag_ps=100.0):
    """Protocol D bound for an unresolved run: MSD_Li(lag)/(6 lag); A^2/ps -> cm^2/s (x 1e-4)."""
    if li_msd_100ps_A2 is None or not np.isfinite(li_msd_100ps_A2):
        return None
    return float(li_msd_100ps_A2 / (6 * lag_ps) * 1e-4)


def prefactor_sigma300_upper_bound(temperature_K, D_bound, ne, D0_min=1e-4, T_ref=300.0):
    """Protocol prefactor rule for one unresolved temperature (docs/CLASSIFICATION_PROTOCOL.md).

    With D0 >= D0_min and D(T) <= D_bound: Ea >= kB T ln(D0_min / D_bound) (floored at 0), hence
    D(T_ref) <= D_bound * exp(-Ea_min/kB * (1/T_ref - 1/T)). Returns the bound on sigma(T_ref) in
    mS/cm with ``ne`` the NE factor at ``temperature_K``. Several unresolved temperatures each give a
    valid bound; the smallest applies."""
    T = float(temperature_K)
    if D_bound is None or D_bound <= 0 or T <= T_ref:
        raise ValueError('Need a positive D bound at a temperature above T_ref')
    ea_min = max(0.0, KB_EV * T * np.log(D0_min / D_bound))
    D_ref = D_bound * np.exp(-ea_min / KB_EV * (1 / T_ref - 1 / T))
    return {'Ea_min_eV': float(ea_min), 'D300_upper_cm2_s': float(D_ref),
            'sigma300_upper_mS_cm': float(D_ref * float(ne) * T / T_ref), 'D0_min_cm2_s': D0_min}


# ----------------------------------------------------------------------------- run discovery
def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _number(value):
    value = float(value)
    return int(value) if value.is_integer() else value


def load_manifest_candidates(manifest_path):
    if not manifest_path or not Path(manifest_path).is_file():
        return {}
    manifest = _read_json(manifest_path)
    return {c['key']: c for c in manifest.get('candidates', []) if 'key' in c}


def spec_from_run_dir(run_dir):
    """Run specification (identity, layout, trajectory path, records) of one run folder."""
    run_dir = Path(run_dir)
    md_run = _read_json(run_dir / 'md_run.json')
    status_path = run_dir / 'status.json'
    status = _read_json(status_path) if status_path.is_file() else {}
    task = md_run.get('task') or status.get('task') or {}
    T = _number(md_run.get('temperature_K', task.get('temperature_K')))
    if run_dir.parent.name == 'runs':
        layout = 'census_run_task'
        key_dir = run_dir.parent.parent
        traj = key_dir / 'data' / 'traj' / f'mace_{T}K.traj'
    else:
        layout = 'v7_run_one'
        key_dir = run_dir.parent
        traj = run_dir / 'md.traj'
        if not traj.exists():
            candidates = sorted(run_dir.glob('*.traj'))
            traj = candidates[0] if len(candidates) == 1 else traj
    key = task.get('candidate_key') or key_dir.name
    seed = task.get('md_seed', md_run.get('seed', status.get('md_seed')))
    if seed is None:
        match = re.search(r'seed(\d+)$', run_dir.name)
        seed = int(match.group(1)) if match else 0
    return {'run_dir': run_dir, 'traj_path': traj, 'md_run': md_run, 'status': status,
            'series_path': run_dir / 'series.json', 'candidate_key': key, 'temperature_K': T,
            'md_seed': int(seed), 'layout': layout, 'run_id': f'{key}/{T}K_seed{int(seed)}'}


def discover_runs(results_dir, include_smoke=False):
    """Find completed runs under ``results_dir`` in either runner layout.

    * census runner (run_task.py): results/<key>/runs/<T>K/{md_run,status,series}.json with the
      trajectory at results/<key>/data/traj/mace_<T>K.traj;
    * v7 / pool runner (run_one.py): results/<key>/<T>K_seed<n>/{md_run,status,series}.json + md.traj.

    Returns (runs, skipped); a run is used only if its status is COMPLETED[_RAW_VALIDATED]."""
    results_dir = Path(results_dir)
    runs, skipped = [], []
    for md_run_path in sorted(results_dir.rglob('md_run.json')):
        run_dir = md_run_path.parent
        if '.attempt_' in run_dir.name:
            continue
        spec = spec_from_run_dir(run_dir)
        reason = None
        if spec['status'].get('status') not in COMPLETED_STATUSES:
            reason = f'status {spec["status"].get("status")!r} not completed'
        elif spec['status'].get('smoke') and not include_smoke:
            reason = 'smoke run'
        elif not Path(spec['traj_path']).is_file():
            reason = 'trajectory missing'
        (skipped if reason else runs).append({**spec, 'skip_reason': reason} if reason else spec)
    ids = [r['run_id'] for r in runs]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f'Duplicate run identities (key, T, seed): {duplicates}')
    return runs, skipped


def protocol_from_md_run(md_run, status=None):
    """Frame count, frame interval, equilibration and production length from md_run.json."""
    status = status or {}
    n_steps = int(md_run['n_steps'])
    log_every = int(md_run['log_every'])
    timestep_fs = float(md_run['timestep_fs'])
    equil = md_run.get('equilib_steps', status.get('equilibration_steps'))
    if equil is None:
        raise ValueError('md_run.json lacks equilib_steps')
    n_frames = n_steps // log_every
    frame_dt_ps = log_every * timestep_fs / 1000
    recorded = md_run.get('n_frames')
    if recorded is not None and int(recorded) != n_frames:
        raise ValueError(f'md_run.json n_frames {recorded} != n_steps // log_every = {n_frames}')
    return {'n_steps': n_steps, 'log_every': log_every, 'timestep_fs': timestep_fs,
            'equilib_steps': int(equil), 'equilibration_ps': int(equil) * timestep_fs / 1000,
            'n_frames': n_frames, 'frame_dt_ps': frame_dt_ps, 'production_ps': n_frames * frame_dt_ps,
            'temperature_K': _number(md_run['temperature_K'])}


def run_seed_offset(candidate_key, temperature_K, md_seed):
    """Deterministic per-run kinisi seed offset (census used 10 x task index)."""
    text = f'{candidate_key}|{float(temperature_K):g}|{int(md_seed)}'
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 100000


def kinisi_seed(offset, window, settings=None):
    s = settings or CONFIG
    return int(s['seed'] + 10 * offset + s['window_seed_index'][window])


# ----------------------------------------------------------------------------- trajectory input
def load_trajectory(path, n_expected):
    """Read every frame once; validate identity, cell and finiteness (census validate_frame)."""
    from ase import Atoms
    from ase.io.trajectory import Trajectory
    frac, temps, energies, light = [], [], [], []
    with Trajectory(str(path), 'r') as traj:
        if len(traj) != n_expected:
            raise ValueError(f'Expected {n_expected} production frames, found {len(traj)}')
        ref = None
        for i, a in enumerate(traj):
            if ref is None:
                ref = a.copy()
            if len(a) != len(ref) or not np.array_equal(a.numbers, ref.numbers):
                raise ValueError(f'Atom identity mismatch at frame {i}')
            if not a.pbc.all() or not np.allclose(a.cell, ref.cell, atol=1e-7, rtol=0):
                raise ValueError(f'Periodic cell mismatch at frame {i}')
            energy = a.get_potential_energy()
            if not np.isfinite(a.positions).all() or not np.isfinite(energy):
                raise ValueError(f'Non-finite frame {i}')
            frac.append(a.get_scaled_positions(wrap=False))
            energies.append(energy / len(a))
            temps.append(a.get_temperature())
            light.append(Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc))
    return {'frac': np.asarray(frac), 'temps': np.asarray(temps), 'energies': np.asarray(energies),
            'frames': light, 'numbers': np.asarray(ref.numbers), 'cell': np.asarray(ref.cell),
            'volume': float(ref.get_volume()), 'symbols': ref.get_chemical_symbols(),
            'formula': ref.get_chemical_formula(mode='metal', empirical=True)}


def check_series(series, protocol, temps, energies):
    """Census frame/log alignment: series has the equilibration sample then one per frame."""
    steps = np.asarray(series['step'])
    n, e0, every = protocol['n_frames'], protocol['equilib_steps'], protocol['log_every']
    expected = np.r_[e0, e0 + every * np.arange(1, n + 1)]
    if not np.array_equal(steps, expected):
        raise ValueError('Invalid logged time grid in series.json')
    if not np.allclose(energies, series['E_per_atom'][1:], atol=1e-8, rtol=1e-8):
        raise ValueError('Frame/log energy alignment mismatch')
    if not np.allclose(temps, series['T'][1:], atol=1e-6, rtol=1e-7):
        raise ValueError('Frame/log temperature alignment mismatch')
    return True


# ----------------------------------------------------------------------------- per-run analysis
def window_bounds(n_frames, frame_dt_ps, settings=None):
    """Frame ranges [a, b) of all kinisi windows for a run of n_frames frames."""
    s = settings or CONFIG
    out = {}
    for name, (f0, f1) in s['fraction_windows'].items():
        out[name] = {'start_frame': int(round(f0 * n_frames)), 'end_frame': int(round(f1 * n_frames)),
                     'definition': f'fraction [{f0}, {f1}] of production'}
    for name, (t0, t1) in s['fixed_windows_ps'].items():
        a, b = frames_for_ps(t0, frame_dt_ps), frames_for_ps(t1, frame_dt_ps)
        if b <= n_frames:
            out[name] = {'start_frame': a, 'end_frame': b, 'definition': f'fixed [{t0}, {t1}] ps'}
    for w in out.values():
        w.update(start_ps=w['start_frame'] * frame_dt_ps, end_ps=w['end_frame'] * frame_dt_ps,
                 length_ps=(w['end_frame'] - w['start_frame']) * frame_dt_ps)
    return out


def _formula_and_rank(spec, manifest_candidates, structure_formula):
    md_run, status = spec['md_run'], spec['status']
    cand = manifest_candidates.get(spec['candidate_key'], {})
    formula = md_run.get('p1_formula') or status.get('p1_formula') or cand.get('p1_formula')
    rank = md_run.get('p1_rank', status.get('p1_rank', cand.get('p1_rank')))
    source = 'run_record' if (md_run.get('p1_formula') or status.get('p1_formula')) else 'manifest'
    if rank is None:
        match = re.match(r'p1r(\d+)_', spec['candidate_key'])
        rank = int(match.group(1)) if match else None
    if not formula:
        formula, source = structure_formula, 'structure'
    return formula, (int(rank) if rank is not None else None), source


def analyze_run(spec, out_dir, *, rel_base, manifest_candidates=None, seed_offset=None,
                settings=None, pymatgen=False, roots=()):
    """Analyse one run; write result.json, msd.csv and posterior_<window>.npy into ``out_dir``."""
    s = settings or CONFIG
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_from_md_run(spec['md_run'], spec['status'])
    T = protocol['temperature_K']
    dt = protocol['frame_dt_ps']
    n = protocol['n_frames']
    L = protocol['production_ps']
    flags = []

    traj_sha = file_sha256(spec['traj_path'])
    recorded = (spec['status'].get('outputs_sha256') or {}).get('md.traj')
    if recorded and recorded != traj_sha:
        raise ValueError('Trajectory sha256 differs from status.json outputs_sha256')
    data = load_trajectory(spec['traj_path'], n)
    series_checked = False
    if Path(spec['series_path']).is_file():
        series_checked = check_series(_read_json(spec['series_path']), protocol, data['temps'], data['energies'])
    else:
        flags.append('series_missing')

    numbers, cell = data['numbers'], data['cell']
    li = numbers == LI_Z
    framework = ~li
    if not li.any() or not framework.any():
        raise ValueError('Need both Li and non-Li atoms')
    pos, raw, drift = unwrap(data['frac'], cell, framework)
    del raw
    formula, rank, formula_source = _formula_and_rank(spec, manifest_candidates or {}, data['formula'])
    if formula_source == 'structure':
        flags.append('formula_from_structure')

    # --- temperature / energy QC (census)
    temps, energies = data['temps'], data['energies']
    T_mean = float(np.mean(temps))
    T_bias = float(T_mean / T - 1)
    temperature_ok = abs(T_bias) <= s['temperature_review_relative']
    qc = {'temperature_mean_K': T_mean, 'temperature_bias_fraction': T_bias, 'temperature_ok': temperature_ok,
          'temperature_std_K': float(np.std(temps)),
          'energy_slope_meV_atom_ps': float(np.polyfit(np.arange(1, n + 1) * dt, energies, 1)[0] * 1000),
          'max_raw_fractional_step': float(np.abs(np.diff(data['frac'], axis=0)).max()),
          'max_framework_drift_A': float(np.linalg.norm(drift, axis=1).max()),
          'series_alignment_checked': series_checked, 'identity_pass': True}

    # --- fixed-time QC: 100 ps lag MSDs, 200 ps displacement fractions
    lag_fixed = frames_for_ps(s['fixed_lag_ps'], dt)
    n_fixed = frames_for_ps(s['fixed_time_ps'], dt)
    thr = s['displacement_threshold_A']
    fixed = {'fixed_lag_ps': s['fixed_lag_ps'], 'fixed_time_ps': s['fixed_time_ps'],
             'framework_msd_100ps_A2': None, 'Li_msd_100ps_A2': None,
             'framework_msd_100ps_A2_first200ps': None, 'Li_msd_100ps_A2_first200ps': None}
    if lag_fixed < n:
        fixed['framework_msd_100ps_A2'] = float(msd(pos, framework, [lag_fixed]).sum())
        fixed['Li_msd_100ps_A2'] = float(msd(pos, li, [lag_fixed]).sum())
        if n_fixed <= n:
            fixed['framework_msd_100ps_A2_first200ps'] = float(msd(pos[:n_fixed], framework, [lag_fixed]).sum())
            fixed['Li_msd_100ps_A2_first200ps'] = float(msd(pos[:n_fixed], li, [lag_fixed]).sum())
    else:
        flags.append('shorter_than_fixed_lag')
    idx_fixed = min(n_fixed, n) - 1
    if n_fixed > n:
        flags.append('fraction_evaluated_before_200ps')
    disp_fixed = np.linalg.norm(pos[idx_fixed] - pos[0], axis=1)
    disp_end = np.linalg.norm(pos[-1] - pos[0], axis=1)
    fixed.update(fraction_frame_index=int(idx_fixed), fraction_time_ps=float((idx_fixed + 1) * dt),
                 framework_fraction_gt2A_200ps=float(np.mean(disp_fixed[framework] > thr)),
                 Li_fraction_gt2A_200ps=float(np.mean(disp_fixed[li] > thr)),
                 framework_fraction_gt2A_end=float(np.mean(disp_end[framework] > thr)),
                 Li_fraction_gt2A_end=float(np.mean(disp_end[li] > thr)))
    fixed['D_bound_msd_cm2_s'] = msd_upper_bound_D(fixed['Li_msd_100ps_A2'], s['fixed_lag_ps'])
    framework_intact = bool(fixed['framework_msd_100ps_A2'] is not None
                            and fixed['framework_msd_100ps_A2'] <= s['framework_review_msd_A2']
                            and fixed['framework_fraction_gt2A_200ps'] <= s['framework_fraction_max'])
    if framework_intact and fixed['framework_fraction_gt2A_end'] > s['framework_fraction_max']:
        flags.append('framework_late_motion')

    # --- census survey-style QC (fixed 60-100 ps lags, origins every 20 frames), cross-reference only
    survey = None
    lo, hi, step = s['survey_lags_ps']
    survey_lags = np.array([frames_for_ps(t, dt) for t in np.arange(lo, hi + step / 2, step)])
    if survey_lags[-1] < n:
        stride = s['survey_origin_stride_frames']
        li_s = np.array([msd_at_lag_strided(pos, li, lag, stride) for lag in survey_lags])
        x = survey_lags * dt
        slope, intercept = np.polyfit(x, li_s, 1)
        den = float(np.sum((li_s - li_s.mean()) ** 2))
        survey = {'D_direct_MSD_cm2_s': float(slope / 6 * 1e-4),
                  'Li_msd_60ps_A2': float(li_s[0]), 'Li_msd_100ps_A2': float(li_s[-1]),
                  'framework_msd_100ps_A2': msd_at_lag_strided(pos, framework, survey_lags[-1], stride),
                  'Li_msd_fit_R2': float(1 - np.sum((li_s - (slope * x + intercept)) ** 2) / den) if den else None,
                  'Li_msd_loglog_alpha': float(np.polyfit(np.log(x), np.log(np.maximum(li_s, 1e-30)), 1)[0]),
                  'method': 'p1_all62_analysis/analyze.py survey: origins every 20 frames, 60-100 ps lags'}

    # --- MSD table (census msd.csv columns; lags up to max_lag_fraction of L, fixed lags included)
    table_lags = np.geomspace(1, max(1, int(n * s['max_lag_fraction'])), s['msd_table_points']).astype(int)
    extra = [lag for lag in (frames_for_ps(60.0, dt), lag_fixed) if lag < n]
    table_lags = np.unique(np.concatenate([table_lags, extra]).astype(int))
    li_tab, fw_tab = msd(pos, li, table_lags), msd(pos, framework, table_lags)
    write_csv(out_dir / 'msd.csv', [{'lag_ps': k * dt, 'Li_x_A2': l[0], 'Li_y_A2': l[1], 'Li_z_A2': l[2],
                                     'Li_total_A2': l.sum(), 'framework_A2': f.sum()}
                                    for k, l, f in zip(table_lags, li_tab, fw_tab)],
              ['lag_ps', 'Li_x_A2', 'Li_y_A2', 'Li_z_A2', 'Li_total_A2', 'framework_A2'])

    # --- kinisi windows (census windows as fractions of L, plus fixed first 200 ps)
    offset = run_seed_offset(spec['candidate_key'], T, spec['md_seed']) if seed_offset is None else int(seed_offset)
    bounds = window_bounds(n, dt, s)
    windows, posteriors = {}, {}
    for name, b in bounds.items():
        a, e = b['start_frame'], b['end_frame']
        if name in s['fixed_windows_ps'] and (a, e) == (bounds['full']['start_frame'], bounds['full']['end_frame']):
            windows[name] = {**windows['full'], 'window': name, **b, 'alias_of': 'full'}
            continue
        w = {'window': name, **b, 'alias_of': None, 'kinisi_seed': kinisi_seed(offset, name, s)}
        w.update(window_msd_diagnostics(pos[a:e], li, dt, s))
        try:
            kin, samples = kinisi_fit(data['frames'][a:e], b['length_ps'], w['kinisi_seed'],
                                      timestep_fs=protocol['timestep_fs'], log_every=protocol['log_every'],
                                      settings=s)
            w.update(kin)
            np.save(out_dir / f'posterior_{name}.npy', samples)
            posteriors[name] = f'posterior_{name}.npy'
        except Exception as exc:  # census: record the failure, no D for this window
            w.update({'error': sanitize_text(f'{type(exc).__name__}: {exc}', roots),
                      'D_cm2_s': None, 'D_lo': None, 'D_hi': None})
        windows[name] = w
    primary = windows[s['primary_window']]

    def resolved(w):
        lo_a, hi_a = s['alpha_range']
        return bool(w.get('D_lo') is not None and w['D_lo'] > 0 and w['diagnostic_D_cm2_s'] > 0
                    and w['msd_growth_fit_A2'] >= s['min_msd_growth_A2'] and lo_a <= w['loglog_alpha'] <= hi_a)

    diffusion_resolved = resolved(primary)
    criteria = {'primary_window': s['primary_window'], 'diffusion_resolved': diffusion_resolved,
                'diffusion_resolved_first200ps': resolved(windows['first200ps']) if 'first200ps' in windows else None,
                'framework_intact': framework_intact, 'temperature_ok': temperature_ok,
                'eligible': bool(diffusion_resolved and framework_intact and temperature_ok)}

    # --- census sensitivity ratios on fractional windows
    sens = {'length_ratio': ratio([windows[k].get('D_cm2_s') for k in s['length_windows']]),
            'half_ratio': ratio([windows[k].get('D_cm2_s') for k in s['half_windows']])}
    reg = []
    full = bounds['full']
    for cond in s['covariance_sensitivity']:
        try:
            result, _ = kinisi_fit(data['frames'][full['start_frame']:full['end_frame']], full['length_ps'],
                                   kinisi_seed(offset, 'full', s), cond_max=cond,
                                   timestep_fs=protocol['timestep_fs'], log_every=protocol['log_every'],
                                   settings=s)
            reg.append({'cond_max': cond, **result})
        except Exception as exc:
            reg.append({'cond_max': cond, 'D_cm2_s': None, 'error': sanitize_text(str(exc), roots)})
    sens['covariance_sensitivity'] = reg
    sens['regularization_ratio'] = ratio([primary.get('D_cm2_s')] + [r.get('D_cm2_s') for r in reg])

    if not framework_intact:
        flags.append('framework_motion_review')
    if not temperature_ok:
        flags.append('temperature_review')
    if not diffusion_resolved:
        flags.append('diffusion_unresolved')
    for key, flag in (('length_ratio', 'length_sensitive'), ('half_ratio', 'half_sensitive'),
                      ('regularization_ratio', 'regularization_sensitive')):
        if sens[key] is None or sens[key] > s['sensitivity_factor']:
            flags.append(flag)

    pmg = None
    if pymatgen:
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer as PMG
            pm = PMG.from_structures([AseAtomsAdaptor.get_structure(a) for a in data['frames']], specie='Li',
                                     temperature=T, time_step=protocol['timestep_fs'],
                                     step_skip=protocol['log_every'], smoothed='max')
            pmg = {'D_cm2_s': float(pm.diffusivity), 'error': None}
        except Exception as exc:
            pmg = {'D_cm2_s': None, 'error': sanitize_text(str(exc), roots)}

    nef = ne_factor(int(li.sum()), data['volume'], T)
    result = {
        'schema': SCHEMA, 'run_id': spec['run_id'],
        'identity': {'candidate_key': spec['candidate_key'], 'p1_rank': rank, 'formula': formula,
                     'formula_source': formula_source, 'temperature_K': T, 'md_seed': spec['md_seed'],
                     'runner_layout': spec['layout']},
        'source': {'source_run_relpath': relpath(spec['run_dir'], rel_base),
                   'trajectory_relpath': relpath(spec['traj_path'], rel_base), 'trajectory_sha256': traj_sha,
                   'status': spec['status'].get('status'),
                   'status_trajectory_sha256_match': (recorded == traj_sha) if recorded else None},
        'protocol': protocol,
        'structure': {'n_atoms': int(len(numbers)), 'n_Li': int(li.sum()), 'n_framework': int(framework.sum()),
                      'cell_volume_A3': data['volume'], 'structure_formula': data['formula'],
                      'cell_A': cell.tolist()},
        'qc': qc, 'fixed_time': fixed, 'survey_style': survey, 'windows': windows, 'criteria': criteria,
        'sensitivity': sens, 'pymatgen': pmg,
        'ne_factor_mS_cm_per_cm2_s': nef,
        'sigma_NE_kinisi_mS_cm': primary['D_cm2_s'] * nef if primary.get('D_cm2_s') else None,
        'review_flags': flags,
        'analysis': {'seed_offset': offset, 'code_sha256': code_sha256(), 'config_sha256': config_sha256(s),
                     'census_source': CENSUS_SOURCE, 'census_source_sha256': CENSUS_SOURCE_SHA256,
                     'wall_seconds': time.perf_counter() - started},
        'outputs': {'msd_table': 'msd.csv', 'posteriors': posteriors},
    }
    write_json(out_dir / 'result.json', result)
    return result


def trajectory_row(result):
    """One trajectory_results.csv row (S2 contract, CSV_COLUMNS)."""
    ident, struct, fixed = result['identity'], result['structure'], result['fixed_time']
    primary = result['windows'][result['criteria']['primary_window']]
    return {
        'p1_rank': ident['p1_rank'], 'formula': ident['formula'], 'candidate_key': ident['candidate_key'],
        'temperature_K': ident['temperature_K'], 'md_seed': ident['md_seed'],
        'production_ps': result['protocol']['production_ps'], 'n_atoms': struct['n_atoms'], 'n_Li': struct['n_Li'],
        'cell_volume_A3': struct['cell_volume_A3'], 'temperature_mean_K': result['qc']['temperature_mean_K'],
        'temperature_bias_fraction': result['qc']['temperature_bias_fraction'],
        'D_kinisi_cm2_s': primary.get('D_cm2_s'), 'D_kinisi_lo_cm2_s': primary.get('D_lo'),
        'D_kinisi_hi_cm2_s': primary.get('D_hi'), 'D_direct_MSD_cm2_s': primary.get('diagnostic_D_cm2_s'),
        'Li_msd_100ps_A2': fixed['Li_msd_100ps_A2'], 'D_bound_msd_cm2_s': fixed['D_bound_msd_cm2_s'],
        'framework_msd_100ps_A2': fixed['framework_msd_100ps_A2'],
        'framework_fraction_gt2A_200ps': fixed['framework_fraction_gt2A_200ps'],
        'diffusion_resolved': result['criteria']['diffusion_resolved'],
        'framework_intact': result['criteria']['framework_intact'], 'eligible': result['criteria']['eligible'],
        'review_flags': ';'.join(result['review_flags']),
        'ne_factor_mS_cm_per_cm2_s': result['ne_factor_mS_cm_per_cm2_s'],
        'trajectory_sha256': result['source']['trajectory_sha256'],
        'source_run_relpath': result['source']['source_run_relpath'],
    }


def window_rows(result):
    ident = result['identity']
    rows = []
    for name, w in result['windows'].items():
        rows.append({'candidate_key': ident['candidate_key'], 'temperature_K': ident['temperature_K'],
                     'md_seed': ident['md_seed'], 'production_ps': result['protocol']['production_ps'],
                     **{k: w.get(k) for k in WINDOW_COLUMNS if k in w}, 'window': name})
    return rows


def sort_key(result):
    ident = result['identity']
    return (ident['p1_rank'] if ident['p1_rank'] is not None else 10 ** 6, ident['candidate_key'],
            float(ident['temperature_K']), ident['md_seed'])


def write_campaign_tables(out_root, results):
    results = sorted(results, key=sort_key)
    write_csv(Path(out_root) / 'trajectory_results.csv', [trajectory_row(r) for r in results], CSV_COLUMNS)
    write_csv(Path(out_root) / 'window_sensitivity.csv', [row for r in results for row in window_rows(r)],
              WINDOW_COLUMNS)
    return results


def environment_versions():
    import importlib.metadata
    import platform
    import sys
    packages = {}
    for name in ('numpy', 'scipy', 'ase', 'kinisi', 'scipp', 'emcee', 'pymatgen',
                 'pymatgen-analysis-diffusion', 'matplotlib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version.split()[0], 'implementation': platform.python_implementation(),
            'system': platform.system(), 'machine': platform.machine(), 'packages': packages}


def run_error_record(spec, exc, rel_base, roots):
    return {'schema': SCHEMA, 'run_id': spec['run_id'],
            'source_run_relpath': relpath(spec['run_dir'], rel_base),
            'error': sanitize_text(f'{type(exc).__name__}: {exc}', roots),
            'traceback': sanitize_text(traceback.format_exc(), roots)}
