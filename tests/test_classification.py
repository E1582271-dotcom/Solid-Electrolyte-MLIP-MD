"""The published classification is reproduced exactly from the committed run records."""
from __future__ import annotations

import json
import math
from collections import Counter

from ssemd import classify, data, paths
from ssemd.records import read_csv
from ssemd.rules import OUTPUT_COLUMNS, to_csv_text


def _same_json(ours, frozen) -> bool:
    """Equal JSON values, keys in the same order; floats may differ in the last bits."""
    if isinstance(frozen, float):
        return isinstance(ours, float) and math.isclose(ours, frozen, rel_tol=1e-12)
    if isinstance(frozen, dict):
        return list(ours) == list(frozen) and all(_same_json(ours[key], frozen[key]) for key in frozen)
    if isinstance(frozen, list):
        return len(ours) == len(frozen) and all(_same_json(a, b) for a, b in zip(ours, frozen))
    return type(ours) is type(frozen) and ours == frozen


def test_legacy_outputs_reproduce_the_frozen_tables():
    """The table is identical byte for byte; the summary's fitted D0 values depend on the LAPACK build."""
    csv_text, summary_text = classify.legacy_outputs()
    assert csv_text == (paths.FROZEN / "definitive_classification_62.csv").read_text(encoding="utf-8")
    frozen = (paths.FROZEN / "definitive_classification_62_summary.json").read_text(encoding="utf-8")
    assert _same_json(json.loads(summary_text), json.loads(frozen))


def test_frozen_files_match_their_checksums():
    import hashlib

    for line in (paths.FROZEN / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((paths.FROZEN / name).read_bytes()).hexdigest() == digest


def test_written_classification_is_current():
    rows, grouped = classify.run_classifier()
    classify.annotate(rows, grouped)
    columns = OUTPUT_COLUMNS + classify.PRESENTATION_COLUMNS
    assert paths.CLASSIFICATION.read_text(encoding="utf-8") == to_csv_text(rows, columns)


def test_every_composition_has_one_terminal_class():
    table = data.classification()
    assert len(table) == 62 and table["candidate_key"].is_unique
    assert set(table["class"]) <= {"FAST_VALIDATED", "SLOW_MEASURED", "SLOW_ESTIMATED", "SLOW_BOUNDED",
                                   "THERMALLY_UNSTABLE"}
    assert Counter(table["class"]) == {"FAST_VALIDATED": 11, "SLOW_MEASURED": 2, "SLOW_ESTIMATED": 6,
                                       "SLOW_BOUNDED": 43}
    slow = table[table["class"].str.startswith("SLOW")]
    assert (slow["sigma300_mS_cm"] < 0.1).all()
    assert (table.loc[table["class"] == "FAST_VALIDATED", "sigma300_mS_cm"] >= 0.1).all()


def test_qualifiers():
    table = data.classification().set_index("formula")
    flagged = {formula: set(text.split(";")) for formula, text in table["qualifiers"].items() if text}
    assert {f for f, q in flagged.items() if "a32_dependent" in q} == {"Li5P(S2Cl)2", "Li10Zn7(PS4)8", "Li2GePbS4"}
    assert {f for f, q in flagged.items() if "near_threshold" in q} == {"Li6AsS5I", "Li3BiS3"}
    assert {f for f, q in flagged.items() if "d0_dependent" in q} == {"Li2SiSnS4"}


def test_used_records_equal_the_frozen_run_records():
    frozen = read_csv(paths.FROZEN / "definitive_run_records.csv")
    runs = data.runs()
    used = runs[runs["used_by_classifier"]]
    assert len(frozen) == len(used) == 318
    key = lambda c, t, s: (c, int(float(t)), int(s))  # noqa: E731
    mine = {key(r.candidate_key, r.temperature_K, r.md_seed): r for r in used.itertuples()}
    for row in frozen:
        ours = mine[key(row["candidate_key"], row["temperature_K"], row["md_seed"])]
        for theirs, column in (("D_kinisi_cm2_s", "D_cm2_s"), ("D_direct_MSD_cm2_s", "D_direct_cm2_s"),
                               ("D_bound_msd_cm2_s", "D_bound_cm2_s"), ("Li_msd_100ps_A2", "Li_msd_100ps_A2"),
                               ("framework_msd_100ps_A2", "framework_msd_100ps_A2")):
            if row[theirs]:
                assert abs(float(row[theirs]) - getattr(ours, column)) <= 1e-9 * abs(float(row[theirs]))
        assert (row["eligible"] == "true") == bool(ours.eligible)
        assert (row["kinisi_fit_failed"] == "true") == bool(ours.kinisi_fit_failed)


def test_summary_records_current_inputs():
    summary = json.loads(paths.CLASSIFICATION_SUMMARY.read_text(encoding="utf-8"))
    assert summary["class_counts"]["FAST_VALIDATED"] == 11
    assert summary["protocol"] == "docs/CLASSIFICATION_PROTOCOL.md"
    assert [item["path"] for item in summary["inputs"]] == ["data/runs.csv", "data/compositions.csv",
                                                           "data/replicates/lead_decisions.csv"]


def test_readme_states_the_current_counts():
    from collections import Counter

    readme = (paths.REPO / "README.md").read_text(encoding="utf-8")
    counts = Counter(data.classification()["class"])
    assert f"**{counts['FAST_VALIDATED']} fast conductors**" in readme
    assert f"{counts['SLOW_MEASURED']} slow (measured)" in readme
    assert f"{counts['SLOW_ESTIMATED']} slow (two-point estimate)" in readme
    assert f"{counts['SLOW_BOUNDED']} slow (upper bound)" in readme
