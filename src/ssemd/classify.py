"""Classify the 62 compositions from the committed run records.

``build()`` writes ``data/classification.csv`` (the classifier columns plus four
presentation columns) and ``data/classification_summary.json``. ``legacy_outputs()``
reruns the same classification with the legacy file names as data sources; its output
equals the frozen tables in ``data/frozen/`` byte for byte (tests/test_classification.py).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .paths import (
    CLASSIFICATION,
    CLASSIFICATION_SUMMARY,
    COMPOSITIONS,
    LEAD_DECISIONS,
    LEGACY_FILES,
    PROTOCOL,
    RUNS,
    relpath,
)
from .protocol import FAST_VALIDATED, NEEDS_COMPUTE, THERMALLY_UNSTABLE
from .records import classifier_runs, load_catalog, load_replicate_decisions, load_runs_table, read_csv
from .rules import OUTPUT_COLUMNS, assert_terminal, classify, summarize, to_csv_text

STAGE = "post-compute"

# Inputs recorded by the frozen summary, in the order the legacy classifier listed them.
LEGACY_INPUTS = (
    "docs/CLASSIFICATION_PROTOCOL.md",
    "source_data/campaign/trajectory_summary_186.csv",
    "campaigns/p1_all62_analysis/detailed_trajectory_186.csv",
    "source_data/campaign/non_lead_classification_56.csv",
    "source_data/campaign/sample_catalog_62.csv",
    "source_data/campaign/validation_v2/lead_decisions_6.csv",
    "source_data/campaign/validation_v2/figure_sources/validation_seed012_20261010_v4/seed012_transport_source.csv",
    "source_data/campaign/validation_v2/figure_sources/validation_seed012_20261010_v4/seed012_primary_MSD_source.csv",
    "source_data/campaign/validation_v2/figure_sources/validation_remaining3_seed012_20261016_v1/seed012_transport_source.csv",
    "source_data/campaign/validation_v2/figure_sources/validation_remaining3_seed012_20261016_v1/seed012_primary_MSD_source.csv",
    "campaigns/p2_definitive_lowT_20261024/analysis/trajectory_results.csv",
    "campaigns/p2_definitive_batch2_20261024/analysis/trajectory_results.csv",
    "campaigns/p2_definitive_batch3_20261025/analysis/trajectory_results.csv",
)

# Presentation columns appended to the classifier output in data/classification.csv.
PRESENTATION_COLUMNS = ("decided_by", "sigma300_estimator", "qualifiers")
QUALIFIERS = {
    "near_threshold": "pooled 95% interval of sigma300 contains the 0.1 mS/cm threshold",
    "a32_dependent": "class differs without amendment A3.2 (failed kinisi fits excluded)",
    "d0_dependent": "class differs when the prefactor bound uses D0,min = 1e-5 cm2/s",
    "upper_value_estimate": "two-point estimate with an unresolved lower temperature (an upper value)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_classifier(legacy_sources: bool = False) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    grouped = classifier_runs(load_runs_table(RUNS), legacy_sources=legacy_sources)
    decisions = load_replicate_decisions(LEAD_DECISIONS, legacy_sources=legacy_sources)
    rows = classify(grouped, load_catalog(COMPOSITIONS), decisions)
    return rows, grouped


def decided_by(row: Mapping[str, Any]) -> str:
    """Which rule placed the composition (derived from the classifier's recorded basis)."""
    basis = row["basis"]
    if row["class"] == NEEDS_COMPUTE:
        return "pending"
    if row["class"] == THERMALLY_UNSTABLE:
        return "framework_not_intact"
    if basis.startswith("Replicate rule passed in validation_v2"):
        return "six_lead_validation"
    if "after the terminal seed/length limit" in basis:
        return "terminal_rule"
    if "the replicate rule passes" in basis:
        return "replicate_rule"
    if row["sigma_kind"] == "upper_bound":
        return f"{row['bound_kind']}_bound"
    if row["Ea_kind"] == "two_point":
        return "two_point_fit"
    if row["Ea_kind"] == "fit":
        return "single_seed_fit"
    raise ValueError(f"No decision label for {row['formula']}")


def sigma300_estimator(row: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> str:
    if row["sigma_kind"] == "upper_bound":
        return "msd_bound"
    if decided_by(row) == "six_lead_validation":
        return "fixed_cov_seed_mean"
    failed = set(row["kinisi_failed_runs"])
    used = {run["_estimator"] for run in runs if run["eligible"]
            and f"{run['temperature_K']:g}K_seed{run['md_seed']}" not in failed}
    names = {"kinisi_mcmc": "kinisi", "fixed_cov_analytic": "fixed_cov"}
    return "+".join(names[name] for name in sorted(used)) + "_pooled_fit"


def qualifiers(row: Mapping[str, Any]) -> list[str]:
    found = []
    if "near_threshold" in row["flags"]:
        found.append("near_threshold")
    if row["kinisi_failure_rule_changes_class"]:
        found.append("a32_dependent")
    if row["sensitivity_verdict_changes"]:
        found.append("d0_dependent")
    if "lower_T_unresolved_estimate_is_upper_value" in row["flags"]:
        found.append("upper_value_estimate")
    return found


def annotate(rows: list[dict[str, Any]], grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    for row in rows:
        runs = grouped[row["candidate_key"]]
        row["decided_by"] = decided_by(row)
        row["sigma300_estimator"] = sigma300_estimator(row, runs)
        row["qualifiers"] = ";".join(qualifiers(row))
    return rows


def summary_text(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, indent=2, ensure_ascii=False) + "\n"


def legacy_outputs() -> tuple[str, str]:
    """Classification CSV and summary JSON with the legacy data-source names (for the frozen check)."""
    rows, _ = run_classifier(legacy_sources=True)
    digests = {row["legacy_path"]: row["legacy_sha256"] for row in read_csv(LEGACY_FILES)}
    inputs = [(path, digests[path]) for path in LEGACY_INPUTS]
    summary = summarize(rows, inputs, STAGE, (relpath(PROTOCOL), sha256(PROTOCOL)))
    return to_csv_text(rows), summary_text(summary)


def build(require_terminal: bool = True) -> list[dict[str, Any]]:
    rows, grouped = run_classifier()
    if require_terminal:
        assert_terminal(rows)
    annotate(rows, grouped)
    CLASSIFICATION.write_text(to_csv_text(rows, OUTPUT_COLUMNS + PRESENTATION_COLUMNS), encoding="utf-8")
    inputs = [(relpath(path), sha256(path)) for path in (RUNS, COMPOSITIONS, LEAD_DECISIONS)]
    summary = summarize(rows, inputs, STAGE, (relpath(PROTOCOL), sha256(PROTOCOL)))
    CLASSIFICATION_SUMMARY.write_text(summary_text(summary), encoding="utf-8")
    return rows


def fast_conductors(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted((row for row in rows if row["class"] == FAST_VALIDATED),
                  key=lambda row: -float(row["sigma300_mS_cm"]))
