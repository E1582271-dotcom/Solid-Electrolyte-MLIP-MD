"""Classification rules on synthetic run records (ported from the tests of the original classifier)."""

from __future__ import annotations

import math
import types
import unittest

from ssemd import arrhenius, protocol, records, rules

# One namespace with every name the rule tests use.
dc = types.SimpleNamespace(**{name: getattr(module, name) for module in (protocol, arrhenius, records, rules)
                              for name in dir(module) if not name.startswith("__")})

KB = dc.K_B_EV_K
N_LI, VOLUME = 100, 5000.0  # 2e22 Li cm^-3: sigma300 = 0.1 mS/cm at D300 ~ 8.1e-10 cm^2/s


def arrhenius(temperature, ea, d0=1e-3):
    return d0 * math.exp(-ea / (KB * temperature))


def make_run(temperature, d, seed=0, key="k1", resolved=True, fw=0.2, frac=0.0, bias=0.0, msd100=None,
             production=200.0, d_lo=None, d_direct=None, **overrides):
    """A contract row (RUN_COLUMNS) with consistent derived columns, validated by normalize_run."""
    msd100 = msd100 if msd100 is not None else 6 * d * 1e4 * dc.BOUND_LAG_PS + 0.3
    intact = dc.framework_is_intact(fw, frac)
    row = {
        "p1_rank": 1, "formula": "LiX", "candidate_key": key, "temperature_K": temperature, "md_seed": seed,
        "production_ps": production, "n_atoms": 300, "n_Li": N_LI, "cell_volume_A3": VOLUME,
        "temperature_mean_K": temperature * (1 + bias), "temperature_bias_fraction": bias,
        "D_cm2_s": d, "D_lo_cm2_s": d_lo if d_lo is not None else 0.8 * d,
        "D_hi_cm2_s": 1.2 * d, "D_direct_cm2_s": d_direct if d_direct is not None else d,
        "Li_msd_100ps_A2": msd100, "D_bound_cm2_s": dc.d_bound_from_msd(msd100),
        "framework_msd_100ps_A2": fw, "framework_fraction_gt2A_200ps": frac,
        "diffusion_resolved": resolved, "framework_intact": intact,
        "eligible": resolved and intact and abs(bias) <= dc.T_BIAS_MAX, "review_flags": "",
        "ne_factor_mS_cm_per_cm2_s": dc.ne_factor_mS_cm_per_cm2_s(N_LI, VOLUME, temperature),
        "trajectory_sha256": "",
    }
    row.update(overrides)
    return dc.normalize_run({column: str(value) for column, value in row.items()})


def unresolved(temperature, msd100=0.3, **kwargs):
    return make_run(temperature, 1e-12, resolved=False, msd100=msd100, **kwargs)


def melted(temperature, d=5e-5, **kwargs):
    return make_run(temperature, d, fw=5.0, frac=0.5, **kwargs)


class PhysicsTests(unittest.TestCase):
    def test_ne_factor_value(self):
        # 2e22 cm^-3 x e^2 / (kB x 300 K) = 1.2395e5 S/cm per cm^2/s -> 1.2395e8 mS/cm per cm^2/s
        self.assertAlmostEqual(dc.ne_factor_mS_cm_per_cm2_s(N_LI, VOLUME, 300) / 1.23949e8, 1.0, places=4)

    def test_d_bound_is_msd_over_six_t(self):
        self.assertAlmostEqual(dc.d_bound_from_msd(60.0), 1e-5)  # 60 A^2 / 600 ps = 0.1 A^2/ps
        self.assertAlmostEqual(dc.d_bound_from_msd(0.6), 1e-7)

    def test_prefactor_bound_closed_form(self):
        ea, d300 = dc.prefactor_bound(600, 1e-6, 1e-4)
        self.assertAlmostEqual(ea, KB * 600 * math.log(100), places=12)
        self.assertAlmostEqual(d300 / 1e-8, 1.0, places=9)  # D300 = Db (Db/D0)^(T/300 - 1)
        ea, d300 = dc.prefactor_bound(1000, 1e-6, 1e-5)
        self.assertAlmostEqual(d300 / (1e-6 * (1e-6 / 1e-5) ** (1000 / 300 - 1)), 1.0, places=9)

    def test_prefactor_bound_is_clamped_at_zero_ea(self):
        self.assertEqual(dc.prefactor_bound(600, 5e-4, 1e-4), (0.0, 5e-4))

    def test_prefactor_bound_holds_for_any_d0_above_the_minimum(self):
        for d0 in (1e-4, 1e-3, 1e-1):
            for ea in (0.2, 0.4, 0.8):
                d600 = arrhenius(600, ea, d0)
                _, bound = dc.prefactor_bound(600, d600, 1e-4)
                self.assertLessEqual(arrhenius(300, ea, d0), bound * (1 + 1e-12))

    def test_pair_bound_is_exact_for_arrhenius_data(self):
        ea_true = 0.45
        ea, d300 = dc.pair_bound(1000, arrhenius(1000, ea_true), 600, arrhenius(600, ea_true))
        self.assertAlmostEqual(ea, ea_true, places=12)
        self.assertAlmostEqual(d300 / arrhenius(300, ea_true), 1.0, places=9)

    def test_arrhenius_fit_recovers_parameters(self):
        temps = (600, 800, 1000)
        fit = dc.arrhenius_fit(temps, [arrhenius(t, 0.3, 2e-3) for t in temps])
        self.assertAlmostEqual(fit["Ea_eV"], 0.3, places=9)
        self.assertAlmostEqual(fit["D0_cm2_s"] / 2e-3, 1.0, places=9)
        self.assertAlmostEqual(fit["R2"], 1.0, places=12)


class RunContractTests(unittest.TestCase):
    def test_contract_rejects_inconsistent_derived_columns(self):
        with self.assertRaises(ValueError):
            make_run(600, 1e-6, framework_intact=False)
        with self.assertRaises(ValueError):
            make_run(600, 1e-6, eligible=False)
        with self.assertRaises(ValueError):
            make_run(600, 1e-6, D_bound_cm2_s=1.0)
        with self.assertRaises(ValueError):
            make_run(600, 1e-6, ne_factor_mS_cm_per_cm2_s=1.0)

    def test_eligibility_gates(self):
        self.assertTrue(make_run(600, 1e-6)["eligible"])
        self.assertFalse(make_run(600, 1e-6, bias=0.06)["eligible"])
        self.assertFalse(make_run(600, 1e-6, fw=1.01)["eligible"])
        self.assertFalse(make_run(600, 1e-6, frac=0.11)["eligible"])
        self.assertTrue(make_run(600, 1e-6, fw=1.0, frac=0.10)["framework_intact"])
        self.assertFalse(unresolved(600)["eligible"])

class SyntheticClassTests(unittest.TestCase):
    def classify(self, runs, decision=None, d0_min=dc.D0_MIN_cm2_s):
        return dc.classify_composition(runs, decision, d0_min)

    def test_slow_measured(self):
        out = self.classify([make_run(t, arrhenius(t, 0.5)) for t in (600, 800, 1000)])
        self.assertEqual(out["class"], dc.SLOW_MEASURED)
        self.assertEqual((out["Ea_kind"], out["sigma_kind"], out["n_eligible"]), ("fit", "estimate", 3))
        self.assertAlmostEqual(out["Ea_eV"], 0.5, places=9)
        expected = arrhenius(300, 0.5) * dc.ne_factor_mS_cm_per_cm2_s(N_LI, VOLUME, 300)
        self.assertAlmostEqual(out["sigma300_mS_cm"] / expected, 1.0, places=9)

    def test_melted_temperature_is_excluded_not_a_veto(self):
        runs = [make_run(t, arrhenius(t, 0.5)) for t in (450, 525, 600)] + [melted(800), melted(1000)]
        out = self.classify(runs)
        self.assertEqual((out["class"], out["eligible_temperatures"]), (dc.SLOW_MEASURED, "450;525;600"))

    def test_single_seed_fast_fit_needs_replicates(self):
        out = self.classify([make_run(t, arrhenius(t, 0.2)) for t in (600, 800, 1000)])
        self.assertEqual(out["class"], dc.NEEDS_COMPUTE)
        self.assertTrue(out["needs_compute_reason"].startswith("replicate_validation_required"))

    def test_fast_validated_by_replicate_rule(self):
        jitter = {0: 1.0, 1: 1.05, 2: 0.95}
        runs = [make_run(t, arrhenius(t, 0.2) * jitter[s], seed=s) for t in (600, 800, 1000) for s in jitter]
        out = self.classify(runs)
        self.assertEqual(out["class"], dc.FAST_VALIDATED)
        self.assertGreaterEqual(out["sigma300_mS_cm"], dc.FAST_THRESHOLD_mS_cm)

    def test_failed_replicate_rule_then_terminal_pooled_interval(self):
        factors = {0: 1.0, 1: 2.5, 2: 1.0}
        runs = [make_run(t, arrhenius(t, 0.2) * factors[s], seed=s) for t in (600, 800, 1000) for s in factors]
        out = self.classify(runs)
        self.assertEqual(out["class"], dc.NEEDS_COMPUTE)
        self.assertIn("max/min", out["needs_compute_reason"])
        factors.update({3: 1.0, 4: 1.0})
        runs = [make_run(t, arrhenius(t, 0.2) * factors[s], seed=s) for t in (600, 800, 1000) for s in factors]
        out = self.classify(runs)
        self.assertEqual(out["class"], dc.FAST_VALIDATED)  # pooled 95 % lower bound far above 0.1 mS/cm

    def test_multi_seed_slow_fit_failing_replicates_goes_to_the_terminal_rule(self):
        # Amendment A3.1: with replicate seeds the replicate rule decides before SLOW_MEASURED.
        factors = {0: 1.0, 1: 2.5, 2: 1.0}
        runs = [make_run(t, arrhenius(t, 0.6) * factors[s], seed=s) for t in (600, 800, 1000) for s in factors]
        out = self.classify(runs)
        self.assertEqual(out["class"], dc.NEEDS_COMPUTE)
        self.assertTrue(out["needs_compute_reason"].startswith("terminal_rule_required"))
        factors.update({3: 1.0, 4: 1.0})
        runs = [make_run(t, arrhenius(t, 0.6) * factors[s], seed=s) for t in (600, 800, 1000) for s in factors]
        self.assertEqual(self.classify(runs)["class"], dc.SLOW_MEASURED)  # pooled 95 % upper bound below 0.1

    def test_multi_seed_slow_fit_passing_replicates_is_slow_measured(self):
        jitter = {0: 1.0, 1: 1.05, 2: 0.95}
        runs = [make_run(t, arrhenius(t, 0.6) * jitter[s], seed=s) for t in (600, 800, 1000) for s in jitter]
        out = self.classify(runs)
        self.assertEqual(out["class"], dc.SLOW_MEASURED)
        self.assertIn("replicate rule passes", out["basis"])

    def test_failed_kinisi_fit_is_treated_as_unresolved(self):
        # Amendment A3.2: kinisi D 4000x below the direct-MSD D of the same run.
        jitter = {0: 1.0, 1: 1.05, 2: 0.95}
        runs = [make_run(t, arrhenius(t, 0.2) * jitter[s], seed=s) for t in (600, 800, 1000) for s in jitter]
        bad = make_run(600, arrhenius(600, 0.2) / 4000, seed=1, d_direct=arrhenius(600, 0.2))
        self.assertTrue(dc.kinisi_fit_failed(bad))
        self.assertFalse(dc.kinisi_fit_failed(make_run(600, 1e-6, d_direct=2.9e-6)))
        runs = [bad if (r["temperature_K"], r["md_seed"]) == (600, 1) else r for r in runs]
        out = self.classify(runs)
        self.assertEqual(out["kinisi_failed_runs"], ["600K_seed1"])
        self.assertIn("kinisi_fit_failed_runs_excluded", out["flags"])
        self.assertEqual(out["class"], dc.NEEDS_COMPUTE)  # 2 usable seeds left at 600 K
        self.assertIn("2 usable seeds at 600 K", out["needs_compute_reason"])
        without = dc.classify_composition(runs, kinisi_failure_rule=False)
        self.assertEqual(without["kinisi_failed_runs"], [])
        self.assertIn("max/min", without["needs_compute_reason"])

    def test_slow_estimated_with_unresolved_lower_temperature(self):
        out = self.classify([unresolved(600)] + [make_run(t, arrhenius(t, 0.5)) for t in (800, 1000)])
        self.assertEqual((out["class"], out["Ea_kind"]), (dc.SLOW_ESTIMATED, "two_point"))
        self.assertIn("lower_T_unresolved_estimate_is_upper_value", out["flags"])

    def test_two_point_fast_needs_compute(self):
        out = self.classify([make_run(t, arrhenius(t, 0.2)) for t in (600, 800)] + [melted(1000)])
        self.assertEqual(out["class"], dc.NEEDS_COMPUTE)
        self.assertTrue(out["needs_compute_reason"].startswith("two_point_sigma300_ge_threshold"))

    def test_slow_bounded_by_prefactor_rule(self):
        out = self.classify([unresolved(t) for t in (600, 800, 1000)])
        self.assertEqual((out["class"], out["Ea_kind"], out["sigma_kind"]), (dc.SLOW_BOUNDED, "lower_bound", "upper_bound"))
        # The same MSD bound at a higher temperature implies a higher Ea, so 1000 K binds.
        d300 = min(dc.prefactor_bound(t, dc.d_bound_from_msd(0.3), dc.D0_MIN_cm2_s)[1] for t in (600, 800, 1000))
        self.assertEqual(d300, dc.prefactor_bound(1000, dc.d_bound_from_msd(0.3), dc.D0_MIN_cm2_s)[1])
        self.assertAlmostEqual(out["sigma300_mS_cm"] / (d300 * dc.ne_factor_mS_cm_per_cm2_s(N_LI, VOLUME, 300)), 1.0)
        self.assertEqual(out["bound_kind"], "prefactor")

    def test_single_eligible_temperature_uses_the_slope_bound(self):
        runs = [unresolved(600, msd100=0.4), make_run(1000, 2e-6)]
        out = self.classify(runs)
        ea, d300 = dc.pair_bound(1000, 0.8 * 2e-6, 600, dc.d_bound_from_msd(0.4))
        self.assertEqual((out["class"], out["bound_kind"]), (dc.SLOW_BOUNDED, "pair"))
        self.assertAlmostEqual(out["Ea_eV"], ea)
        self.assertAlmostEqual(out["sigma300_mS_cm"] / (d300 * dc.ne_factor_mS_cm_per_cm2_s(N_LI, VOLUME, 300)), 1.0)

    def test_slope_bound_uses_the_smaller_lower_d(self):
        run = make_run(1000, 2e-6, d_lo=1.6e-6, d_direct=5e-7)
        self.assertEqual(dc._lower_d(run), 5e-7)

    def test_fast_single_temperature_needs_compute(self):
        out = self.classify([make_run(600, 1e-5), melted(800), melted(1000)])
        self.assertEqual(out["class"], dc.NEEDS_COMPUTE)
        self.assertTrue(out["needs_compute_reason"].startswith("bound_sigma300_ge_threshold"))

    def test_thermally_unstable_only_when_tested_down_to_400K(self):
        out = self.classify([melted(t) for t in (400, 450, 500, 600)])
        self.assertEqual(out["class"], dc.THERMALLY_UNSTABLE)
        out = self.classify([melted(t) for t in (600, 800, 1000)])
        self.assertEqual(out["class"], dc.NEEDS_COMPUTE)
        self.assertTrue(out["needs_compute_reason"].startswith("framework_not_intact_at_any_tested_T"))

    def test_validation_decision(self):
        runs = [make_run(t, arrhenius(t, 0.2) * f, seed=s) for t in (600, 800, 1000) for s, f in ((0, 1), (1, 1.02), (2, 9))]
        decision = {"decision": "pass_seeds01", "passed": True, "seeds": [0, 1], "Ea_eV": 0.2,
                    "sigma300_mS_cm": 12.5, "source": "x.csv"}
        out = self.classify(runs, decision)
        self.assertEqual((out["class"], out["sigma300_mS_cm"], out["seeds"]), (dc.FAST_VALIDATED, 12.5, "0;1"))
        self.assertIn("x.csv", out["data_sources"])

    def test_sensitivity_flags_a_bounded_verdict_that_flips(self):
        # D_bound = 1.5e-7 at 600 K: sigma300 bound ~0.028 mS/cm at D0,min = 1e-4 but ~0.28 at 1e-5.
        runs = {"k1": [unresolved(600, msd100=0.9)]}
        catalog = [{"p1_rank": "1", "formula": "LiX", "candidate_key": "k1"}]
        (row,) = dc.classify(runs, catalog)
        self.assertEqual((row["class"], row["sensitivity_D0min_1e-5_class"]), (dc.SLOW_BOUNDED, dc.NEEDS_COMPUTE))
        self.assertTrue(row["sensitivity_verdict_changes"])
        self.assertIn("verdict_changes_at_D0min_1e-5", row["flags"])

    def test_assert_terminal(self):
        dc.assert_terminal([{"formula": "A", "class": dc.SLOW_BOUNDED}])
        with self.assertRaises(AssertionError):
            dc.assert_terminal([{"formula": "A", "class": dc.NEEDS_COMPUTE}])
