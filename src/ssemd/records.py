"""Read the committed data tables into the run records the classifier works on."""
from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .arrhenius import d_bound_from_msd, ne_factor_mS_cm_per_cm2_s
from .paths import COMPOSITIONS, LEAD_DECISIONS, LEGACY_LEAD_DECISIONS, RUNS, relpath
from .protocol import FRAMEWORK_FRACTION_MAX, FRAMEWORK_MSD_MAX_A2, REPLICATE_PASS_DECISIONS, T_BIAS_MAX

# Run-record contract of the classifier. ``eligible`` is the census eligibility criterion
# (resolved diffusion, intact framework, |T bias| <= 5 %); amendment A3.2 is applied by the
# classifier itself.
RUN_COLUMNS = (
    "p1_rank", "formula", "candidate_key", "temperature_K", "md_seed", "production_ps", "n_atoms", "n_Li",
    "cell_volume_A3", "temperature_mean_K", "temperature_bias_fraction", "D_cm2_s", "D_lo_cm2_s",
    "D_hi_cm2_s", "D_direct_cm2_s", "Li_msd_100ps_A2", "D_bound_cm2_s", "framework_msd_100ps_A2",
    "framework_fraction_gt2A_200ps", "diffusion_resolved", "framework_intact", "eligible", "review_flags",
    "ne_factor_mS_cm_per_cm2_s", "trajectory_sha256",
)
_INT_COLUMNS = ("p1_rank", "md_seed", "n_atoms", "n_Li")
_FLOAT_COLUMNS = ("temperature_K", "production_ps", "cell_volume_A3", "temperature_mean_K",
                  "temperature_bias_fraction", "D_cm2_s", "D_lo_cm2_s", "D_hi_cm2_s", "D_direct_cm2_s",
                  "Li_msd_100ps_A2", "D_bound_cm2_s", "framework_msd_100ps_A2", "framework_fraction_gt2A_200ps",
                  "ne_factor_mS_cm_per_cm2_s")
_BOOL_COLUMNS = ("diffusion_resolved", "framework_intact", "eligible")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cell(value: Any) -> str:
    """Text of one CSV cell: ``repr`` for floats (reads back bit-identical), lower-case booleans."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    return str(value)


def csv_text(columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([cell(row.get(column)) for column in columns])
    return buffer.getvalue()


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text(columns, rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Eligibility checks (census criteria, unchanged)
# ---------------------------------------------------------------------------
def framework_is_intact(framework_msd_100ps_A2: float, fraction_gt2A_200ps: float) -> bool:
    return framework_msd_100ps_A2 <= FRAMEWORK_MSD_MAX_A2 and fraction_gt2A_200ps <= FRAMEWORK_FRACTION_MAX


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text not in ("true", "false"):
        raise ValueError(f"Expected a boolean value, got {value!r}")
    return text == "true"


def _float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    text = str(value).strip()
    if text in ("", "nan", "NaN", "None"):
        return None
    return float(text)


def _close(a: float, b: float, rel: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)


def normalize_run(row: Mapping[str, Any]) -> dict[str, Any]:
    """Type one run record in the ``RUN_COLUMNS`` contract and check its derived columns.

    ``framework_intact``, ``eligible``, ``D_bound_cm2_s`` and the NE factor are
    recomputed from the primary columns; a supplied value that disagrees is an error.
    """
    missing = [column for column in RUN_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"Run record is missing contract columns: {missing}")
    run: dict[str, Any] = {column: row[column] for column in RUN_COLUMNS}
    for column in _INT_COLUMNS:
        run[column] = int(float(row[column]))
    for column in _FLOAT_COLUMNS:
        run[column] = _float(row[column])
    for column in _BOOL_COLUMNS:
        run[column] = _truth(row[column])
    for column in ("formula", "candidate_key", "review_flags", "trajectory_sha256"):
        value = row[column]
        run[column] = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    key = (run["candidate_key"], run["temperature_K"], run["md_seed"])
    for column in ("temperature_K", "n_Li", "cell_volume_A3", "temperature_bias_fraction",
                   "framework_msd_100ps_A2", "framework_fraction_gt2A_200ps", "Li_msd_100ps_A2"):
        if run[column] is None:
            raise ValueError(f"Run {key} lacks {column}")
    intact = framework_is_intact(run["framework_msd_100ps_A2"], run["framework_fraction_gt2A_200ps"])
    if run["framework_intact"] != intact:
        raise ValueError(f"framework_intact disagrees with the framework thresholds for run {key}")
    eligible = run["diffusion_resolved"] and intact and abs(run["temperature_bias_fraction"]) <= T_BIAS_MAX
    if run["eligible"] != eligible:
        raise ValueError(f"eligible disagrees with resolved/intact/T-bias for run {key}")
    if run["diffusion_resolved"] and not ((run["D_lo_cm2_s"] or 0) > 0 and (run["D_cm2_s"] or 0) > 0):
        raise ValueError(f"Resolved run {key} needs a positive D and lower bound")
    bound = d_bound_from_msd(run["Li_msd_100ps_A2"])
    if run["D_bound_cm2_s"] is None or not _close(run["D_bound_cm2_s"], bound):
        raise ValueError(f"D_bound_cm2_s is not MSD_Li(100 ps)/(6 x 100 ps) for run {key}")
    factor = ne_factor_mS_cm_per_cm2_s(run["n_Li"], run["cell_volume_A3"], run["temperature_K"])
    if run["ne_factor_mS_cm_per_cm2_s"] is None or not _close(run["ne_factor_mS_cm_per_cm2_s"], factor):
        raise ValueError(f"ne_factor_mS_cm_per_cm2_s disagrees with n_Li, V and T for run {key}")
    return run


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_runs_table(path: Path = RUNS) -> list[dict[str, str]]:
    """All analysis records (census, replicate re-analysis, definitive campaigns) as text rows."""
    return read_csv(path)


def classifier_runs(rows: Sequence[Mapping[str, str]], legacy_sources: bool = False,
                    source_label: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Records the classifier uses (``used_by_classifier``), grouped per composition and ordered by
    (T, seed). ``legacy_sources`` keeps the legacy file names in ``data_sources`` so the output can be
    compared byte for byte with the frozen tables."""
    label = source_label or relpath(RUNS)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not _truth(row["used_by_classifier"]):
            continue
        record = {column: row[column] for column in RUN_COLUMNS if column != "eligible"}
        record["eligible"] = row["eligible_census_criteria"]
        run = normalize_run(record)
        run["_sources"] = (tuple(row["legacy_sources"].split(";")) if legacy_sources else (label,))
        run["_record_id"] = row["record_id"]
        run["_estimator"] = row["estimator"]
        grouped[run["candidate_key"]].append(run)
    for key, runs in grouped.items():
        slots = [(run["temperature_K"], run["md_seed"]) for run in runs]
        if len(slots) != len(set(slots)):
            raise ValueError(f"Two classifier records share a (T, seed) slot for {key}")
        runs.sort(key=lambda run: (run["temperature_K"], run["md_seed"]))
    return dict(grouped)


def load_catalog(path: Path = COMPOSITIONS) -> list[dict[str, str]]:
    return read_csv(path)


def load_replicate_decisions(path: Path = LEAD_DECISIONS, legacy_sources: bool = False) -> dict[str, dict[str, Any]]:
    """Seed 0/1/2 decisions of the six census leads (validated leads keep their decision)."""
    decisions = {}
    source = LEGACY_LEAD_DECISIONS if legacy_sources else relpath(path)
    for row in read_csv(path):
        passed = row["decision"] in REPLICATE_PASS_DECISIONS
        decisions[row["candidate_key"]] = {
            "decision": row["decision"], "passed": passed,
            "seeds": [int(seed) for seed in row["seeds_used"].split(";") if seed],
            "Ea_eV": _float(row["Ea_eV_mean"]), "sigma300_mS_cm": _float(row["sigma300_mS_cm_mean"]),
            "source": source,
        }
    return decisions
