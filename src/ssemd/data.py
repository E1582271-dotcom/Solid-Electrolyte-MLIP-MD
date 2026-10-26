"""Typed loaders for the committed tables (pandas). Figures and tables read data only through here."""
from __future__ import annotations

from functools import cache

import pandas as pd

from . import paths

BOOL_TEXT = {"true": True, "false": False}


def _booleans(frame: pd.DataFrame, columns) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = frame[column].map(lambda value: BOOL_TEXT.get(str(value).strip().lower(), value))
    return frame


def _read(path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)


@cache
def runs() -> pd.DataFrame:
    """All 338 analysis records (census, replicate re-analysis, definitive campaigns)."""
    frame = _read(paths.RUNS, dtype={"review_flags": str, "superseded_by": str, "legacy_sources": str})
    frame["review_flags"] = frame["review_flags"].fillna("")
    frame["superseded_by"] = frame["superseded_by"].fillna("")
    return _booleans(frame, ("diffusion_resolved", "framework_intact", "eligible_census_criteria",
                             "kinisi_fit_failed", "eligible", "used_by_classifier"))


@cache
def compositions() -> pd.DataFrame:
    frame = _read(paths.COMPOSITIONS, dtype={"census_review_reasons": str, "replicate_outcome": str,
                                             "icsd_ids": str})
    for column in ("census_review_reasons", "replicate_outcome", "icsd_ids"):
        frame[column] = frame[column].fillna("")
    return frame


@cache
def classification() -> pd.DataFrame:
    frame = _read(paths.CLASSIFICATION, dtype={"flags": str, "qualifiers": str, "kinisi_failed_runs": str,
                                               "needs_compute_reason": str, "bound_kind": str})
    for column in ("flags", "qualifiers", "kinisi_failed_runs", "needs_compute_reason", "bound_kind"):
        frame[column] = frame[column].fillna("")
    return _booleans(frame, ("sensitivity_verdict_changes", "kinisi_failure_rule_changes_class"))


@cache
def msd_curves() -> pd.DataFrame:
    return _read(paths.MSD_CURVES, compression="gzip")


@cache
def windows() -> pd.DataFrame:
    return _read(paths.WINDOWS, dtype={"error": str})


@cache
def lead_decisions() -> pd.DataFrame:
    return _read(paths.LEAD_DECISIONS, dtype={"seeds_used": str, "seed_excluded": str})


@cache
def seed_fits() -> pd.DataFrame:
    return _booleans(_read(paths.SEED_FITS), ("fit_success", "seed_usable_all_T", "usable_extrapolation",
                                               "included_in_decision"))


@cache
def temperature_summary() -> pd.DataFrame:
    return _read(paths.TEMPERATURE_SUMMARY)


@cache
def li6ps5cl_reference() -> pd.DataFrame:
    return _read(paths.REFERENCES / "li6ps5cl_calibration.csv")


@cache
def workflow() -> pd.DataFrame:
    return _read(paths.WORKFLOW)


def classifier_runs() -> pd.DataFrame:
    """Records the classifier used, with the composition's class attached."""
    frame = runs()
    frame = frame[frame["used_by_classifier"]].copy()
    classes = classification()[["candidate_key", "class"]]
    return frame.merge(classes, on="candidate_key", how="left", validate="many_to_one")
