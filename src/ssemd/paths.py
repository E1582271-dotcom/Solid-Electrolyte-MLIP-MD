"""Repository layout. Every path the package reads or writes is defined here."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
FIGURES = REPO / "figures"
DOCS = REPO / "docs"

RUNS = DATA / "runs.csv"
COMPOSITIONS = DATA / "compositions.csv"
MSD_CURVES = DATA / "msd_curves.csv.gz"
WINDOWS = DATA / "window_sensitivity.csv"
REPLICATES = DATA / "replicates"
LEAD_DECISIONS = REPLICATES / "lead_decisions.csv"
SEED_FITS = REPLICATES / "seed_fits.csv"
TEMPERATURE_SUMMARY = REPLICATES / "temperature_summary.csv"
CLASSIFICATION = DATA / "classification.csv"
CLASSIFICATION_SUMMARY = DATA / "classification_summary.json"
WORKFLOW = DATA / "workflow.csv"
REFERENCES = DATA / "references"
FROZEN = DATA / "frozen"
PROVENANCE = DATA / "provenance"
LEGACY_FILES = PROVENANCE / "legacy_files.csv"
PROTOCOL = DOCS / "CLASSIFICATION_PROTOCOL.md"

# File names of the legacy repository that the frozen classification records as its inputs.
LEGACY_LEAD_DECISIONS = "source_data/campaign/validation_v2/lead_decisions_6.csv"


def relpath(path: Path | str) -> str:
    """Repository-relative POSIX path."""
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name
