"""Classification rules of docs/CLASSIFICATION_PROTOCOL.md (amendments A1-A3).

Every composition receives exactly one class. A run is *eligible* when Li diffusion is
resolved, the framework is intact (non-Li MSD at 100 ps <= 1 A^2 and fraction of non-Li
atoms displaced > 2 A at 200 ps <= 0.10) and |T bias| <= 5 %; amendment A3.2 also treats
as unresolved an eligible run whose kinisi D differs from its direct-MSD D by more than 3x.
A temperature is eligible when at least one run at it is eligible.

* replicate rule passed (six-lead validation decision, or the replicate rule on the runs)
  and sigma300 >= 0.1 mS/cm -> FAST_VALIDATED;
* >= 3 eligible temperatures: Arrhenius fit of ln D on 1/T over every eligible run ->
  sigma300 (Nernst-Einstein at 300 K). One seed: < 0.1 -> SLOW_MEASURED, otherwise
  replicates are required. Several seeds (amendment A3.1): the replicate rule decides
  first; a failure goes to the terminal rule (seeds up to 5 or runs up to 800 ps, then the
  pooled 95 % interval);
* exactly 2: two-point estimate; < 0.1 -> SLOW_ESTIMATED (flagged as an upper value when
  a lower tested temperature was framework-intact but unresolved);
* < 2: prefactor rule. Every framework-intact run gives D <= MSD_Li(100 ps)/(6 x 100 ps);
  with D0 >= D0,min this bounds Ea from below and sigma300 from above. One eligible
  temperature with a lower framework-intact temperature also bounds Ea through the
  two-point slope. The tightest bound < 0.1 -> SLOW_BOUNDED;
* no framework-intact run at any tested temperature -> THERMALLY_UNSTABLE if tested down
  to 400 K;
* anything else -> NEEDS_COMPUTE (allowed only before the last batch is in).

Every SLOW_BOUNDED verdict is re-evaluated with D0,min = 1e-5 cm^2/s. The functions below
are the ones that produced the published classes; only their data loading was replaced.
"""
from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .arrhenius import arrhenius_fit, ne_factor_mS_cm_per_cm2_s, pair_bound, prefactor_bound
from .protocol import (
    BOUND_LAG_PS,
    CLASSES,
    ESTIMATOR_RATIO_FLAG,
    FAST_VALIDATED,
    FRAMEWORK_FRACTION_MAX,
    FRAMEWORK_MSD_MAX_A2,
    NEEDS_COMPUTE,
    PREREGISTERED_COMPUTE_LIST,
    REPLICATE_MAX_RATIO,
    REPLICATE_MIN_R2,
    REPLICATE_MIN_SEEDS,
    SLOW_BOUNDED,
    SLOW_ESTIMATED,
    SLOW_MEASURED,
    T_BIAS_MAX,
    T_REF_K,
    TERMINAL_CLASSES,
    TERMINAL_MAX_PRODUCTION_PS,
    TERMINAL_MAX_SEEDS,
    THERMALLY_UNSTABLE,
    UNSTABLE_LOWEST_T_K,
    D0_MIN_cm2_s,
    D0_MIN_SENSITIVITY_cm2_s,
    FAST_THRESHOLD_mS_cm,
)

OUTPUT_COLUMNS = (
    "p1_rank", "formula", "candidate_key", "class", "eligible_temperatures", "n_eligible", "Ea_eV", "Ea_kind",
    "D0_cm2_s", "sigma300_mS_cm", "sigma_kind", "sensitivity_D0min_1e-5_class", "needs_compute_reason",
    "basis", "data_sources",
    # Additional audit columns.
    "sensitivity_verdict_changes", "sigma300_upper_bound_D0min_1e-5_mS_cm", "bound_kind", "tested_temperatures",
    "framework_intact_temperatures", "seeds", "flags",
    # Amendment A3.2 audit: runs excluded as failed kinisi fits, and the class without that rule.
    "kinisi_failed_runs", "class_without_kinisi_failure_rule", "kinisi_failure_rule_changes_class",
    # Pooled 95 % interval of sigma300 for multi-seed fits over >= 3 eligible temperatures.
    "sigma300_ci95_lo_mS_cm", "sigma300_ci95_hi_mS_cm",
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _temps(values: Iterable[float]) -> str:
    return ";".join(f"{t:g}" for t in sorted(set(values)))


def _ne300(runs: Sequence[Mapping[str, Any]]) -> float:
    structures = {(run["n_Li"], round(run["cell_volume_A3"], 6)) for run in runs}
    if len(structures) != 1:
        raise ValueError(f"Inconsistent n_Li/volume across runs of {runs[0]['candidate_key']}")
    n_li, volume = runs[0]["n_Li"], runs[0]["cell_volume_A3"]
    return ne_factor_mS_cm_per_cm2_s(n_li, volume, T_REF_K)


def bound_candidates(runs: Sequence[Mapping[str, Any]], d0_min: float) -> list[dict[str, Any]]:
    """All sigma300 upper bounds available to a composition with < 2 eligible temperatures."""
    factor = _ne300(runs)
    intact = [run for run in runs if run["framework_intact"]]
    candidates = []
    for run in intact:
        ea, d300 = prefactor_bound(run["temperature_K"], run["D_bound_cm2_s"], d0_min)
        candidates.append({
            "kind": "prefactor", "Ea_eV": ea, "sigma300_mS_cm": d300 * factor,
            "text": (f"the prefactor rule at {run['temperature_K']:g} K (seed {run['md_seed']}; "
                     f"D <= MSD/6t = {run['D_bound_cm2_s']:.3g} cm2/s, D0 >= {d0_min:g} cm2/s)")})
    for high in (run for run in intact if run["eligible"]):
        for low in intact:
            if low["eligible"] or low["temperature_K"] >= high["temperature_K"]:
                continue
            d_high = _lower_d(high)
            ea, d300 = pair_bound(high["temperature_K"], d_high, low["temperature_K"], low["D_bound_cm2_s"])
            candidates.append({
                "kind": "pair", "Ea_eV": ea, "sigma300_mS_cm": d300 * factor,
                "text": (f"the two-point slope between the {high['temperature_K']:g} K lower D estimate "
                         f"({d_high:.3g} cm2/s) and the {low['temperature_K']:g} K bound "
                         f"(D <= {low['D_bound_cm2_s']:.3g} cm2/s)")})
    return candidates


def _lower_d(run: Mapping[str, Any]) -> float:
    """Conservative lower D of a resolved run: the kinisi lower credible bound, or the
    direct-MSD estimate when that is smaller (kinisi can overshoot it)."""
    values = [run["D_lo_cm2_s"]]
    if run["D_direct_cm2_s"] is not None and run["D_direct_cm2_s"] > 0:
        values.append(run["D_direct_cm2_s"])
    return min(values)


def kinisi_fit_failed(run: Mapping[str, Any]) -> bool:
    """Amendment A3.2: an eligible run whose kinisi D differs from its direct-MSD D by more
    than ESTIMATOR_RATIO_FLAG (either way) is a failed kinisi fit."""
    d_direct = run["D_direct_cm2_s"]
    return bool(run["eligible"] and d_direct is not None and d_direct > 0
                and not 1 / ESTIMATOR_RATIO_FLAG <= run["D_cm2_s"] / d_direct <= ESTIMATOR_RATIO_FLAG)


def _estimator_audit(eligible: Sequence[Mapping[str, Any]], factor: float, sigma: float) -> list[str]:
    """Audit flags when kinisi and direct-MSD D disagree on eligible runs used in a fit."""
    flags = []
    disagree = sorted({run["temperature_K"] for run in eligible
                       if run["D_direct_cm2_s"] is not None and run["D_direct_cm2_s"] > 0
                       and not (1 / ESTIMATOR_RATIO_FLAG <= run["D_cm2_s"] / run["D_direct_cm2_s"]
                                <= ESTIMATOR_RATIO_FLAG)})
    if disagree:
        flags.append(f"kinisi_vs_direct_D_ratio_gt{ESTIMATOR_RATIO_FLAG:g}_at_{_temps(disagree)}K")
        direct = [run["D_direct_cm2_s"] if (run["D_direct_cm2_s"] or 0) > 0 else run["D_cm2_s"]
                  for run in eligible]
        alt = arrhenius_fit([run["temperature_K"] for run in eligible], direct)["D_ref_cm2_s"] * factor
        if (alt >= FAST_THRESHOLD_mS_cm) != (sigma >= FAST_THRESHOLD_mS_cm):
            flags.append("verdict_depends_on_D_estimator")
    return flags


def replicate_rule(eligible: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    """Pre-registered replicate rule on eligible runs (>= 3 seeds per fitted T, D max/min <= 2,
    per-seed and pooled Arrhenius R^2 >= 0.95)."""
    by_t: dict[float, dict[int, float]] = defaultdict(dict)
    for run in eligible:
        by_t[run["temperature_K"]][run["md_seed"]] = run["D_cm2_s"]
    problems = []
    for temperature, seeds in sorted(by_t.items()):
        if len(seeds) < REPLICATE_MIN_SEEDS:
            problems.append(f"{len(seeds)} usable seeds at {temperature:g} K")
        elif max(seeds.values()) / min(seeds.values()) > REPLICATE_MAX_RATIO:
            problems.append(f"between-seed D max/min > {REPLICATE_MAX_RATIO:g} at {temperature:g} K")
    common = set.intersection(*(set(seeds) for seeds in by_t.values())) if by_t else set()
    for seed in sorted(common):
        fit = arrhenius_fit(list(by_t), [by_t[t][seed] for t in by_t])
        if fit["R2"] < REPLICATE_MIN_R2:
            problems.append(f"seed {seed} Arrhenius R2 {fit['R2']:.3f} < {REPLICATE_MIN_R2}")
    if len(by_t) >= 2:
        pooled = arrhenius_fit([run["temperature_K"] for run in eligible], [run["D_cm2_s"] for run in eligible])
        if pooled["R2"] < REPLICATE_MIN_R2:
            problems.append(f"pooled Arrhenius R2 {pooled['R2']:.3f} < {REPLICATE_MIN_R2}")
    return (not problems, "; ".join(problems))


def _terminal_reached(runs: Sequence[Mapping[str, Any]], temperatures: Iterable[float]) -> bool:
    """Terminal rule: seeds raised to 5, or runs extended to 800 ps, at every fitted temperature."""
    temperatures = list(temperatures)
    seeds = {t: {run["md_seed"] for run in runs if run["temperature_K"] == t} for t in temperatures}
    longest = {t: max(run["production_ps"] or 0 for run in runs if run["temperature_K"] == t) for t in temperatures}
    return bool(temperatures) and (all(len(seeds[t]) >= TERMINAL_MAX_SEEDS for t in temperatures)
                                   or all(longest[t] >= TERMINAL_MAX_PRODUCTION_PS for t in temperatures))


def classify_composition(runs: Sequence[Mapping[str, Any]], decision: Mapping[str, Any] | None = None,
                         d0_min: float = D0_MIN_cm2_s, kinisi_failure_rule: bool = True) -> dict[str, Any]:
    """Assign one class to one composition from its runs (and validation decision, if any)."""
    if not runs:
        raise ValueError("A composition needs at least one run")
    if len({run["candidate_key"] for run in runs}) != 1:
        raise ValueError("Runs of more than one composition passed to classify_composition")
    first = runs[0]
    if decision is not None and decision.get("passed"):
        seeds = set(decision["seeds"])
        runs = [run for run in runs if run["md_seed"] in seeds]
        if not runs:
            raise ValueError(f"No runs for the decision seeds of {first['candidate_key']}")
    failed = [run for run in runs if kinisi_failure_rule and kinisi_fit_failed(run)]
    if failed:
        runs = [dict(run, diffusion_resolved=False, eligible=False) if any(run is f for f in failed) else run
                for run in runs]
    factor = _ne300(runs)
    tested = sorted({run["temperature_K"] for run in runs})
    intact_t = sorted({run["temperature_K"] for run in runs if run["framework_intact"]})
    eligible = [run for run in runs if run["eligible"]]
    eligible_t = sorted({run["temperature_K"] for run in eligible})
    seeds = sorted({run["md_seed"] for run in runs})
    out: dict[str, Any] = {
        "p1_rank": first["p1_rank"], "formula": first["formula"], "candidate_key": first["candidate_key"],
        "class": None, "eligible_temperatures": _temps(eligible_t), "n_eligible": len(eligible_t),
        "Ea_eV": None, "Ea_kind": "", "D0_cm2_s": None, "sigma300_mS_cm": None, "sigma_kind": "",
        "needs_compute_reason": "", "basis": "",
        "data_sources": sorted({source for run in runs for source in run.get("_sources", ())}),
        "tested_temperatures": _temps(tested), "framework_intact_temperatures": _temps(intact_t),
        "seeds": ";".join(str(s) for s in seeds), "flags": [], "bound_kind": "",
        "kinisi_failed_runs": [f"{run['temperature_K']:g}K_seed{run['md_seed']}" for run in failed],
    }
    if failed:
        out["flags"].append("kinisi_fit_failed_runs_excluded")
    fit = None
    if len(eligible_t) >= 2:
        fit = arrhenius_fit([run["temperature_K"] for run in eligible], [run["D_cm2_s"] for run in eligible])
        out["D0_cm2_s"] = fit["D0_cm2_s"]
        if fit["Ea_eV"] <= 0:
            out["flags"].append("non_positive_Ea")
    if decision is not None:
        out["data_sources"] = sorted(set(out["data_sources"]) | {decision["source"]})
    run_text = f"{len(eligible)} eligible runs, seeds {';'.join(str(s) for s in sorted({r['md_seed'] for r in eligible}))}"

    # 1. Replicate-validated by the validation_v2 decision.
    if decision is not None and decision.get("passed"):
        sigma, ea = decision["sigma300_mS_cm"], decision["Ea_eV"]
        out.update(Ea_eV=ea, Ea_kind="fit", sigma300_mS_cm=sigma, sigma_kind="estimate")
        if fit is not None and (fit["D_ref_cm2_s"] * factor >= FAST_THRESHOLD_mS_cm) != (sigma >= FAST_THRESHOLD_mS_cm):
            out["flags"].append("pooled_fit_disagrees_with_decision")
        if sigma >= FAST_THRESHOLD_mS_cm:
            out["class"] = FAST_VALIDATED
            out["basis"] = (f"Replicate rule passed in validation_v2 (decision {decision['decision']}, seeds "
                            f"{';'.join(str(s) for s in decision['seeds'])}) and the mean seed sigma300 = "
                            f"{sigma:.3g} mS/cm >= {FAST_THRESHOLD_mS_cm:g} mS/cm (mean Ea = {ea:.3f} eV).")
        else:
            out["class"] = SLOW_MEASURED
            out["basis"] = (f"Replicate rule passed in validation_v2 (decision {decision['decision']}) with mean "
                            f"seed sigma300 = {sigma:.3g} mS/cm < {FAST_THRESHOLD_mS_cm:g} mS/cm.")
        return out

    # 2. No framework-intact run at any tested temperature.
    if not intact_t:
        if tested[0] <= UNSTABLE_LOWEST_T_K:
            out["class"] = THERMALLY_UNSTABLE
            out["basis"] = (f"The framework is not intact at any tested temperature ({_temps(tested)} K), "
                            f"tested down to {tested[0]:g} K.")
        else:
            out["class"] = NEEDS_COMPUTE
            out["needs_compute_reason"] = (f"framework_not_intact_at_any_tested_T: lowest tested {tested[0]:g} K "
                                           f"> {UNSTABLE_LOWEST_T_K:g} K; run lower temperatures down to "
                                           f"{UNSTABLE_LOWEST_T_K:g} K")
            out["basis"] = (f"The framework is not intact at {_temps(tested)} K, but THERMALLY_UNSTABLE needs "
                            f"testing down to {UNSTABLE_LOWEST_T_K:g} K.")
        return out

    if fit is not None:
        out["flags"] += _estimator_audit(eligible, factor, fit["D_ref_cm2_s"] * factor)

    # 3. >= 3 eligible temperatures: Arrhenius fit.
    if len(eligible_t) >= 3:
        sigma = fit["D_ref_cm2_s"] * factor
        out.update(Ea_eV=fit["Ea_eV"], Ea_kind="fit", sigma300_mS_cm=sigma, sigma_kind="estimate")
        multi_seed = len({run["md_seed"] for run in eligible}) > 1
        ci = [value * factor for value in fit["D_ref_ci95_cm2_s"]] if multi_seed and fit["D_ref_ci95_cm2_s"] else None
        ci_text = f" (pooled 95% CI {ci[0]:.3g}-{ci[1]:.3g} mS/cm)" if ci else ""
        if ci:
            out.update(sigma300_ci95_lo_mS_cm=ci[0], sigma300_ci95_hi_mS_cm=ci[1])
        if ci and ci[0] < FAST_THRESHOLD_mS_cm <= ci[1]:
            out["flags"].append("near_threshold")
        description = (f"Arrhenius fit of ln D over {len(eligible_t)} eligible temperatures ({_temps(eligible_t)} K; "
                       f"{run_text}) gives Ea = {fit['Ea_eV']:.3f} eV and sigma300 = {sigma:.3g} mS/cm{ci_text}")
        if not multi_seed:
            if sigma < FAST_THRESHOLD_mS_cm:
                out["class"] = SLOW_MEASURED
                out["basis"] = f"{description} < {FAST_THRESHOLD_mS_cm:g} mS/cm."
                return out
            _, problems = replicate_rule(eligible)
            out["class"] = NEEDS_COMPUTE
            out["needs_compute_reason"] = ("replicate_validation_required: sigma300 >= threshold without a passed "
                                           f"replicate rule ({problems or 'no replicate seeds'}); run seeds 1/2 at "
                                           f"{_temps(eligible_t)} K")
            out["basis"] = f"{description} >= {FAST_THRESHOLD_mS_cm:g} mS/cm without replicate validation."
            return out
        # Amendment A3.1: once replicate seeds exist, the replicate rule decides before the class.
        passed, problems = replicate_rule(eligible)
        if passed:
            fast = sigma >= FAST_THRESHOLD_mS_cm
            out["class"] = FAST_VALIDATED if fast else SLOW_MEASURED
            out["basis"] = (f"{description} {'>=' if fast else '<'} {FAST_THRESHOLD_mS_cm:g} mS/cm and the "
                            f"replicate rule passes.")
            return out
        if _terminal_reached(runs, eligible_t) and ci:
            if ci[0] >= FAST_THRESHOLD_mS_cm:
                out["class"] = FAST_VALIDATED
            elif ci[1] < FAST_THRESHOLD_mS_cm:
                out["class"] = SLOW_MEASURED
            else:
                out["class"] = FAST_VALIDATED if sigma >= FAST_THRESHOLD_mS_cm else SLOW_MEASURED
                if "near_threshold" not in out["flags"]:
                    out["flags"].append("near_threshold")
            out["basis"] = (f"{description}; the replicate rule failed ({problems}) after the terminal seed/length "
                            f"limit, so the pooled 95% interval decides.")
            return out
        out["class"] = NEEDS_COMPUTE
        out["needs_compute_reason"] = (f"terminal_rule_required: the replicate rule failed ({problems}); add seeds up "
                                       f"to {TERMINAL_MAX_SEEDS} (or extend runs to {TERMINAL_MAX_PRODUCTION_PS:g} ps) "
                                       f"at {_temps(eligible_t)} K, then the pooled 95% interval decides")
        out["basis"] = (f"{description}; the replicate rule failed ({problems}) before the terminal seed/length "
                        f"limit was reached.")
        return out

    # 4. Exactly 2 eligible temperatures: two-point estimate.
    if len(eligible_t) == 2:
        sigma = fit["D_ref_cm2_s"] * factor
        out.update(Ea_eV=fit["Ea_eV"], Ea_kind="two_point", sigma300_mS_cm=sigma, sigma_kind="estimate")
        lower_unresolved = sorted({run["temperature_K"] for run in runs
                                   if run["framework_intact"] and not run["eligible"]
                                   and run["temperature_K"] < eligible_t[0]})
        note = ""
        if lower_unresolved:
            out["flags"].append("lower_T_unresolved_estimate_is_upper_value")
            note = (f"; diffusion was not resolved at the lower tested {_temps(lower_unresolved)} K, so the "
                    f"true Ea is higher and this is an upper value")
        description = (f"Two-point Arrhenius estimate from {_temps(eligible_t)} K ({run_text}) gives Ea = "
                       f"{fit['Ea_eV']:.3f} eV and sigma300 = {sigma:.3g} mS/cm")
        if sigma < FAST_THRESHOLD_mS_cm:
            out["class"] = SLOW_ESTIMATED
            out["basis"] = f"{description} < {FAST_THRESHOLD_mS_cm:g} mS/cm{note}."
        else:
            out["class"] = NEEDS_COMPUTE
            out["needs_compute_reason"] = ("two_point_sigma300_ge_threshold: add a third eligible temperature, "
                                           "then replicate validation if still >= threshold")
            out["basis"] = f"{description} >= {FAST_THRESHOLD_mS_cm:g} mS/cm{note}."
        return out

    # 5. Fewer than 2 eligible temperatures: prefactor / slope bounds.
    candidates = bound_candidates(runs, d0_min)
    best = min(candidates, key=lambda c: c["sigma300_mS_cm"])
    out.update(Ea_eV=best["Ea_eV"], Ea_kind="lower_bound", sigma300_mS_cm=best["sigma300_mS_cm"],
               sigma_kind="upper_bound", bound_kind=best["kind"])
    lead = (f"{len(eligible_t)} eligible temperature{'s' if len(eligible_t) != 1 else ''}"
            f"{' (' + _temps(eligible_t) + ' K)' if eligible_t else ''}; the tightest bound, from {best['text']}, "
            f"gives Ea >= {best['Ea_eV']:.3f} eV and sigma300 <= {best['sigma300_mS_cm']:.3g} mS/cm")
    if best["sigma300_mS_cm"] < FAST_THRESHOLD_mS_cm:
        out["class"] = SLOW_BOUNDED
        out["basis"] = f"Only {lead}, below the {FAST_THRESHOLD_mS_cm:g} mS/cm threshold."
    else:
        out["class"] = NEEDS_COMPUTE
        out["needs_compute_reason"] = ("bound_sigma300_ge_threshold: fewer than 2 eligible temperatures and no "
                                       "bound below threshold; add lower framework-intact temperatures")
        out["basis"] = (f"Only {lead}, which is not below the {FAST_THRESHOLD_mS_cm:g} mS/cm threshold, so no slow "
                        f"class is supported.")
    return out


def classify(grouped: Mapping[str, Sequence[Mapping[str, Any]]], catalog: Sequence[Mapping[str, str]],
             decisions: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Classify every catalog composition; adds the D0,min = 1e-5 sensitivity columns."""
    decisions = decisions or {}
    keys = [row["candidate_key"] for row in catalog]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate candidate_key in the catalog")
    extra = set(grouped) - set(keys)
    if extra:
        raise ValueError(f"Runs for compositions outside the catalog: {sorted(extra)}")
    rows = []
    for entry in sorted(catalog, key=lambda row: int(row["p1_rank"])):
        key = entry["candidate_key"]
        runs = grouped.get(key)
        if not runs:
            raise ValueError(f"No runs for catalog composition {key}")
        if runs[0]["formula"] != entry["formula"] or runs[0]["p1_rank"] != int(entry["p1_rank"]):
            raise ValueError(f"Run identity disagrees with the catalog for {key}")
        row = classify_composition(runs, decisions.get(key))
        sensitivity = classify_composition(runs, decisions.get(key), D0_MIN_SENSITIVITY_cm2_s)
        row["sensitivity_D0min_1e-5_class"] = sensitivity["class"]
        row["sensitivity_verdict_changes"] = sensitivity["class"] != row["class"]
        row["sigma300_upper_bound_D0min_1e-5_mS_cm"] = (sensitivity["sigma300_mS_cm"]
                                                         if sensitivity["sigma_kind"] == "upper_bound" else None)
        if row["sensitivity_verdict_changes"]:
            row["flags"].append("verdict_changes_at_D0min_1e-5")
        without = classify_composition(runs, decisions.get(key), kinisi_failure_rule=False)
        row["class_without_kinisi_failure_rule"] = without["class"]
        row["kinisi_failure_rule_changes_class"] = without["class"] != row["class"]
        rows.append(row)
    for row in rows:
        if row["class"] not in CLASSES:
            raise ValueError(f"Unknown class for {row['candidate_key']}: {row['class']}")
    return rows


def assert_terminal(rows: Sequence[Mapping[str, Any]]) -> None:
    """Post-compute invariant: every composition has a terminal class (no NEEDS_COMPUTE)."""
    pending = [row["formula"] for row in rows if row["class"] not in TERMINAL_CLASSES]
    if pending:
        raise AssertionError(f"Compositions without a terminal class: {pending}")


def output_records(rows: Sequence[Mapping[str, Any]],
                   columns: Sequence[str] = OUTPUT_COLUMNS) -> list[dict[str, str]]:
    records = []
    for row in rows:
        record = {}
        for column in columns:
            value = row.get(column)
            if column in ("data_sources", "flags", "kinisi_failed_runs"):
                value = ";".join(value)
            record[column] = _fmt(value)
        records.append(record)
    return records


def to_csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] = OUTPUT_COLUMNS) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_records(rows, columns))
    return buffer.getvalue()


def summarize(rows: Sequence[Mapping[str, Any]], inputs: Sequence[tuple[str, str]], stage: str,
              protocol: tuple[str, str]) -> dict[str, Any]:
    """Class counts, D0 report, sensitivity and A3.2 audit. ``inputs`` and ``protocol`` are
    (repository path, sha256) pairs."""
    counts = Counter(row["class"] for row in rows)
    needs = [{"formula": row["formula"], "candidate_key": row["candidate_key"], "reason": row["needs_compute_reason"]}
             for row in rows if row["class"] == NEEDS_COMPUTE]
    needs_set = {item["formula"] for item in needs}
    d0 = sorted((row["D0_cm2_s"], row["formula"]) for row in rows if row["D0_cm2_s"] is not None)
    d0_values = [value for value, _ in d0]
    bounded = [row for row in rows if row["class"] == SLOW_BOUNDED]
    return {
        "protocol": protocol[0],
        "protocol_sha256": protocol[1],
        "stage": stage,
        "n_compositions": len(rows),
        "class_counts": {name: counts.get(name, 0) for name in CLASSES},
        "needs_compute": needs,
        "preregistered_compute_list": list(PREREGISTERED_COMPUTE_LIST),
        "needs_compute_matches_preregistered": needs_set == set(PREREGISTERED_COMPUTE_LIST),
        "deviations_from_preregistered": {
            "unexpected_needs_compute": sorted(needs_set - set(PREREGISTERED_COMPUTE_LIST)),
            "preregistered_but_decided": sorted(set(PREREGISTERED_COMPUTE_LIST) - needs_set),
        },
        "fast_validated": [row["formula"] for row in rows if row["class"] == FAST_VALIDATED],
        "fitted_D0_cm2_s": {
            "scope": "compositions with >= 2 eligible temperatures (pooled ln D vs 1/T fit over eligible runs)",
            "n": len(d0_values),
            "min": d0_values[0] if d0_values else None,
            "min_formula": d0[0][1] if d0 else None,
            "median": float(np.median(d0_values)) if d0_values else None,
            "max": d0_values[-1] if d0_values else None,
            "below_D0_min": [formula for value, formula in d0 if value < D0_MIN_cm2_s],
            "below_D0_min_sensitivity": [formula for value, formula in d0 if value < D0_MIN_SENSITIVITY_cm2_s],
            "by_composition": {formula: value for value, formula in d0},
        },
        "sensitivity_D0min_1e-5": {
            "n_slow_bounded": len(bounded),
            "binding_bound_kind_counts": dict(Counter(row["bound_kind"] for row in bounded)),
            "max_slow_bounded_sigma300_upper_bound_mS_cm": max((row["sigma300_mS_cm"] for row in bounded), default=None),
            "max_slow_bounded_sigma300_upper_bound_at_1e-5_mS_cm": max(
                (row["sigma300_upper_bound_D0min_1e-5_mS_cm"] for row in bounded), default=None),
            "n_unchanged": sum(not row["sensitivity_verdict_changes"] for row in bounded),
            "verdict_changes": [{"formula": row["formula"], "class_at_1e-5": row["sensitivity_D0min_1e-5_class"],
                                 "sigma300_upper_bound_at_1e-5_mS_cm": row["sigma300_upper_bound_D0min_1e-5_mS_cm"]}
                                for row in rows if row["sensitivity_verdict_changes"]],
        },
        "kinisi_failure_rule": {
            "rule": (f"amendment A3.2: an eligible run with kinisi D / direct-MSD D outside "
                     f"1/{ESTIMATOR_RATIO_FLAG:g}-{ESTIMATOR_RATIO_FLAG:g} is treated as not resolved"),
            "excluded_runs": {row["formula"]: list(row["kinisi_failed_runs"]) for row in rows if row["kinisi_failed_runs"]},
            "class_changes": [{"formula": row["formula"], "class": row["class"],
                               "class_without_rule": row["class_without_kinisi_failure_rule"]}
                              for row in rows if row["kinisi_failure_rule_changes_class"]],
        },
        "flagged": {row["formula"]: list(row["flags"]) for row in rows if row["flags"]},
        "parameters": {
            "fast_threshold_mS_cm": FAST_THRESHOLD_mS_cm, "T_ref_K": T_REF_K, "D0_min_cm2_s": D0_MIN_cm2_s,
            "D0_min_sensitivity_cm2_s": D0_MIN_SENSITIVITY_cm2_s, "T_bias_max": T_BIAS_MAX,
            "framework_msd_100ps_max_A2": FRAMEWORK_MSD_MAX_A2, "framework_fraction_gt2A_max": FRAMEWORK_FRACTION_MAX,
            "bound_lag_ps": BOUND_LAG_PS, "unstable_lowest_T_K": UNSTABLE_LOWEST_T_K,
        },
        "inputs": [{"path": path, "sha256": digest} for path, digest in inputs],
    }
