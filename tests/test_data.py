"""Integrity of the committed data tables."""
from __future__ import annotations

import numpy as np

from ssemd import data, paths
from ssemd.arrhenius import d_bound_from_msd, ne_factor_mS_cm_per_cm2_s
from ssemd.protocol import ESTIMATOR_RATIO_FLAG
from ssemd.records import framework_is_intact, read_csv


def test_record_counts():
    runs = data.runs()
    assert len(runs) == 338 and runs["record_id"].is_unique
    assert runs["stage"].value_counts().to_dict() == {"census": 186, "definitive": 98, "replicate": 54}
    assert runs["trajectory_id"].nunique() == 320
    assert int(runs["used_by_classifier"].sum()) == 318
    unused = runs[~runs["used_by_classifier"]]
    assert (unused["superseded_by"] != "").all()
    assert set(unused["superseded_by"]) <= set(runs["record_id"])


def test_derived_columns_recompute():
    runs = data.runs()
    for row in runs.itertuples():
        assert np.isclose(row.D_bound_cm2_s, d_bound_from_msd(row.Li_msd_100ps_A2), rtol=1e-12)
        assert np.isclose(row.ne_factor_mS_cm_per_cm2_s,
                          ne_factor_mS_cm_per_cm2_s(row.n_Li, row.cell_volume_A3, row.temperature_K), rtol=1e-9)
        assert row.framework_intact == framework_is_intact(row.framework_msd_100ps_A2,
                                                           row.framework_fraction_gt2A_200ps)
        census_ok = row.diffusion_resolved and row.framework_intact and abs(row.temperature_bias_fraction) <= 0.05
        assert row.eligible_census_criteria == census_ok
        failed = bool(census_ok and row.D_direct_cm2_s > 0
                      and not 1 / ESTIMATOR_RATIO_FLAG <= row.D_cm2_s / row.D_direct_cm2_s <= ESTIMATOR_RATIO_FLAG)
        assert row.kinisi_fit_failed == failed
        assert row.eligible == (census_ok and not failed)


def test_trajectory_provenance():
    trajectories = read_csv(paths.PROVENANCE / "trajectories.csv")
    runs = data.runs()
    assert len(trajectories) == 320
    by_id = {row["trajectory_id"]: row for row in trajectories}
    for row in runs.itertuples():
        assert by_id[row.trajectory_id]["trajectory_sha256"] == row.trajectory_sha256
    assert all(len(row["trajectory_sha256"]) == 64 for row in trajectories)


def test_msd_curves_match_the_100ps_values():
    runs = data.runs().set_index("record_id")
    curves = data.msd_curves()
    at100 = curves[np.isclose(curves["lag_ps"], 100.0)].set_index("record_id")["Li_total_A2"]
    assert set(curves["record_id"]) == set(runs.index)
    for stage, tolerance in (("replicate", 1e-12), ("definitive", 1e-12), ("census", 0.05)):
        ids = runs.index[runs["stage"] == stage]
        ratio = runs.loc[ids, "Li_msd_100ps_A2"] / at100.loc[ids]
        # Census QC used MSD origins every 20 frames, the stored census curves every frame.
        assert (ratio - 1).abs().max() <= tolerance, stage


def test_compositions_and_structures():
    comps = data.compositions()
    assert list(comps["p1_rank"]) == list(range(1, 63))
    import hashlib

    for row in comps.itertuples():
        path = paths.DATA / row.structure_file
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.structure_sha256
    assert comps["census_category"].value_counts().to_dict() == {
        "unresolved_diffusion": 36, "framework_melting": 10, "undetermined_600K": 7, "lead": 6, "both": 3}


def test_workflow_counts():
    workflow = data.workflow().set_index("stage")["count"].to_dict()
    assert workflow["census"] == 186 and workflow["replicates"] == 36 and workflow["definitive"] == 98
    assert workflow["unique_trajectories"] == 320 and workflow["fast_conductors"] == 11
