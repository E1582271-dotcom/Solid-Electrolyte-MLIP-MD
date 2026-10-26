"""Reference estimators: settings identical to the analyses that produced the data."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ssemd import paths


def test_kinisi_settings_hash_matches_every_definitive_analysis():
    from ssemd.transport import kinisi_census

    settings = json.loads((paths.PROVENANCE / "analysis_settings.json").read_text(encoding="utf-8"))
    assert {batch: item["config_sha256"] for batch, item in settings.items()} == {
        batch: kinisi_census.config_sha256() for batch in ("def_b1", "def_b2", "def_b3")}


def test_fixed_covariance_estimator_recovers_a_known_slope():
    from ssemd.transport.fixed_cov import fixed_covariance_reference

    lags = np.arange(60.0, 101.0)
    slope = 0.6  # A^2/ps -> D = 1e-5 cm^2/s
    msd = 5.0 + slope * lags
    covariance = np.diag(np.full(lags.size, 0.01)) + 0.001
    result = fixed_covariance_reference(lags, msd, covariance)
    assert result["D_cm2_s"] == pytest.approx(slope / 6 * 1e-4, rel=1e-6)
    assert result["D_lo_cm2_s"] < result["D_cm2_s"] < result["D_hi_cm2_s"]


def test_fixed_covariance_estimator_rejects_bad_covariance():
    from ssemd.transport.fixed_cov import fixed_covariance_reference

    lags = np.arange(60.0, 70.0)
    with pytest.raises(ValueError):
        fixed_covariance_reference(lags, lags, -np.eye(lags.size))
