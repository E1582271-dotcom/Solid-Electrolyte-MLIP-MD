"""Figure 3: Arrhenius small multiples of the eleven FAST_VALIDATED compositions.

One panel per composition (a-k, ordered by sigma(300 K) descending): D of every classifier
record against 1000/T, marker shape = MD seed, filled = used in the fit, open = not used,
cross = amendment-A3.2 failed kinisi fit. The line is the pooled OLS Arrhenius fit over the
records used by the classifier (solid over the data, dashed to 300 K) with the 95 % band of
the mean line; the star is the classified sigma(300 K) converted to D at 300 K.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from .. import data
from ..arrhenius import arrhenius_fit, ne_factor_mS_cm_per_cm2_s
from ..protocol import FAST_VALIDATED, T_REF_K
from . import style

NAME = "fig3_fast_arrhenius"
FIG = "fig3"
LETTERS = "abcdefghijk"

X_MIN, X_MAX = 0.88, 3.48          # 1000/T axis (1136 K to 287 K)
Y_MIN, Y_MAX = 1e-10, 1e-3          # D axis, cm^2/s
SEED_DODGE = 0.024                  # deterministic horizontal offset per seed (1000/T units)
TOP_TICKS_K = (1000, 600, 400, 300)

FIT_COLOR = style.CLASS_COLORS[FAST_VALIDATED]
LINE_COLOR = "#000000"
FAILED_COLOR = "#000000"
DECIDED_BY = {"six_lead_validation": "Seed mean", "replicate_rule": "Replicate rule",
              "terminal_rule": "Terminal rule"}


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def formula_tex(formula: str) -> str:
    """Chemical formula with subscripted counts as matplotlib mathtext, e.g. Li$_{7}$PS$_{6}$."""
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def _decimals(value: float) -> int:
    return 0 if value >= 10 else 1 if value >= 1 else 2


def _fmt_sigma(value: float, decimals: int | None = None) -> str:
    """Two significant figures for the central value; an SD is printed with the same decimals."""
    return f"{value:.{_decimals(value) if decimals is None else decimals}f}"


def _inverse_1000(values):
    """1000/x for the secondary temperature axis (x > 0 on this axis; 0 maps to inf)."""
    values = np.asarray(values, float)
    with np.errstate(divide="ignore"):
        return 1000.0 / values


def _split(text) -> list[str]:
    return [item for item in str(text).split(";") if item and item != "nan"]


def _ols_band(inv_t: np.ndarray, ln_d: np.ndarray, grid: np.ndarray):
    """Mean line of ln D on 1/T and its 95 % band (Student t, n-2 dof), as in arrhenius_fit."""
    slope, intercept = np.polyfit(inv_t, ln_d, 1)
    mean = intercept + slope * grid
    n = inv_t.size
    if n < 3:
        return mean, None, None
    resid = ln_d - (intercept + slope * inv_t)
    s2 = float(np.sum(resid ** 2)) / (n - 2)
    sxx = float(np.sum((inv_t - inv_t.mean()) ** 2))
    se = np.sqrt(s2 * (1.0 / n + (grid - inv_t.mean()) ** 2 / sxx))
    q = float(stats.t.ppf(0.975, n - 2))
    return mean, mean - q * se, mean + q * se


def _status(row, seeds_used: set[int], temps_used: set[int]) -> str:
    if bool(row["kinisi_fit_failed"]):
        return "failed_kinisi_fit"
    if bool(row["eligible"]):
        if int(row["md_seed"]) in seeds_used and int(row["temperature_K"]) in temps_used:
            return "fit"
        return "excluded_seed"
    return "not_eligible"


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def _panel_data(comp: pd.Series, records: pd.DataFrame, leads: pd.DataFrame) -> dict:
    seeds_used = {int(s) for s in _split(comp["seeds"])}
    temps_used = {int(t) for t in _split(comp["eligible_temperatures"])}
    rec = records.sort_values(["temperature_K", "md_seed"]).copy()
    rec["status"] = [_status(row, seeds_used, temps_used) for _, row in rec.iterrows()]
    rec["inv_T_1000_per_K"] = 1000.0 / rec["temperature_K"]
    seeds_present = sorted(rec["md_seed"].unique())
    centre = float(np.mean(seeds_present))
    rec["x_plotted"] = rec["inv_T_1000_per_K"] + (rec["md_seed"] - centre) * SEED_DODGE

    used = rec[rec["status"] == "fit"]
    fit = arrhenius_fit(used["temperature_K"], used["D_cm2_s"])
    n_li, volume = float(rec["n_Li"].iloc[0]), float(rec["cell_volume_A3"].iloc[0])
    ne300 = ne_factor_mS_cm_per_cm2_s(n_li, volume, T_REF_K)
    sigma300 = float(comp["sigma300_mS_cm"])
    seed_mean = comp["decided_by"] == "six_lead_validation"
    if seed_mean:
        lead = leads[leads["candidate_key"] == comp["candidate_key"]].iloc[0]
        sd = float(lead["sigma300_mS_cm_sd"])
        sigma_lo, sigma_hi, err_kind = sigma300 - sd, sigma300 + sd, "seed_sd"
        if not math.isclose(float(lead["sigma300_mS_cm_mean"]), sigma300, rel_tol=1e-5):
            raise ValueError(f"lead_decisions disagrees with classification for {comp['formula']}")
    else:
        sigma_lo = float(comp["sigma300_ci95_lo_mS_cm"])
        sigma_hi = float(comp["sigma300_ci95_hi_mS_cm"])
        err_kind = "pooled_ci95"
        pooled = fit["D_ref_cm2_s"] * ne300
        if not math.isclose(pooled, sigma300, rel_tol=2e-3):
            raise ValueError(f"pooled fit {pooled:.4g} does not reproduce sigma300 {sigma300:.4g} "
                             f"for {comp['formula']}")

    inv_used = 1.0 / used["temperature_K"].to_numpy(float)
    ln_d = np.log(used["D_cm2_s"].to_numpy(float))
    x_lo, x_hi = float(inv_used.min()), float(inv_used.max())
    grid = np.unique(np.concatenate([np.linspace(x_lo, 1.0 / T_REF_K, 60), [x_hi]]))
    mean, lo, hi = _ols_band(inv_used, ln_d, grid)
    line = pd.DataFrame({"inv_T_1000_per_K": grid * 1000.0, "D_fit_cm2_s": np.exp(mean),
                         "segment": np.where(grid <= x_hi + 1e-12, "data_range", "extrapolation")})
    if lo is not None and not seed_mean:
        line["D_band_lo_cm2_s"] = np.exp(lo)
        line["D_band_hi_cm2_s"] = np.exp(hi)
    else:
        line["D_band_lo_cm2_s"] = np.nan
        line["D_band_hi_cm2_s"] = np.nan
    return {"comp": comp, "records": rec, "fit": fit, "line": line, "ne300": ne300,
            "sigma300": sigma300, "sigma_lo": sigma_lo, "sigma_hi": sigma_hi, "err_kind": err_kind,
            "seed_mean": seed_mean, "seeds_used": sorted(seeds_used)}


def _source_frame(panel: dict) -> pd.DataFrame:
    comp, rec, line = panel["comp"], panel["records"], panel["line"]
    records = pd.DataFrame({
        "element": "record", "formula": comp["formula"], "record_id": rec["record_id"],
        "stage": rec["stage"], "estimator": rec["estimator"], "temperature_K": rec["temperature_K"],
        "md_seed": rec["md_seed"], "inv_T_1000_per_K": rec["inv_T_1000_per_K"],
        "x_plotted": rec["x_plotted"], "D_cm2_s": rec["D_cm2_s"],
        "D_lo_cm2_s": np.where(rec["status"] == "failed_kinisi_fit", np.nan, rec["D_lo_cm2_s"]),
        "D_hi_cm2_s": np.where(rec["status"] == "failed_kinisi_fit", np.nan, rec["D_hi_cm2_s"]),
        "status": rec["status"]})
    fit_rows = pd.DataFrame({"element": "fit_line", "formula": comp["formula"],
                             "inv_T_1000_per_K": line["inv_T_1000_per_K"], "x_plotted": line["inv_T_1000_per_K"],
                             "D_cm2_s": line["D_fit_cm2_s"], "D_lo_cm2_s": line["D_band_lo_cm2_s"],
                             "D_hi_cm2_s": line["D_band_hi_cm2_s"], "status": line["segment"]})
    ne = panel["ne300"]
    star = pd.DataFrame([{
        "element": "sigma300_marker", "formula": comp["formula"], "temperature_K": T_REF_K,
        "inv_T_1000_per_K": 1000.0 / T_REF_K, "x_plotted": 1000.0 / T_REF_K,
        "D_cm2_s": panel["sigma300"] / ne, "D_lo_cm2_s": panel["sigma_lo"] / ne, "D_hi_cm2_s": panel["sigma_hi"] / ne,
        "status": panel["err_kind"], "sigma300_mS_cm": panel["sigma300"],
        "sigma300_lo_mS_cm": panel["sigma_lo"], "sigma300_hi_mS_cm": panel["sigma_hi"],
        "Ea_eV": float(comp["Ea_eV"]), "pooled_fit_Ea_eV": panel["fit"]["Ea_eV"],
        "pooled_fit_D0_cm2_s": panel["fit"]["D0_cm2_s"], "pooled_fit_R2": panel["fit"]["R2"],
        "ne_factor_300K_mS_cm_per_cm2_s": ne, "decided_by": comp["decided_by"],
        "qualifiers": comp["qualifiers"]}])
    return pd.concat([records, fit_rows, star], ignore_index=True)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_panel(ax, panel: dict) -> None:
    rec, line, comp = panel["records"], panel["line"], panel["comp"]

    # band and line
    band = line.dropna(subset=["D_band_lo_cm2_s"])
    if len(band):
        ax.fill_between(band["inv_T_1000_per_K"], band["D_band_lo_cm2_s"], band["D_band_hi_cm2_s"],
                        color=FIT_COLOR, alpha=0.16, linewidth=0, zorder=1)
    inside = line[line["segment"] == "data_range"]
    x_join = inside["inv_T_1000_per_K"].max()
    outside = line[line["inv_T_1000_per_K"] >= x_join - 1e-12]
    ax.plot(inside["inv_T_1000_per_K"], inside["D_fit_cm2_s"], color=LINE_COLOR, lw=0.8, zorder=3)
    ax.plot(outside["inv_T_1000_per_K"], outside["D_fit_cm2_s"], color=LINE_COLOR, lw=0.8, ls=(0, (2.2, 1.6)),
            zorder=3)

    # records
    for _, row in rec.iterrows():
        marker = style.SEED_MARKERS[int(row["md_seed"])]
        x, d = row["x_plotted"], row["D_cm2_s"]
        status = row["status"]
        if status == "failed_kinisi_fit":
            ax.plot(x, d, marker="x", ms=3.4, mew=0.8, color=FAILED_COLOR, linestyle="none", zorder=5)
            continue
        if status == "fit":
            face, edge, ecolor, z = FIT_COLOR, "#1F4F3F", FIT_COLOR, 4
        elif status == "excluded_seed":
            face, edge, ecolor, z = "white", FIT_COLOR, FIT_COLOR, 4
        else:
            face, edge, ecolor, z = "white", style.GREY, style.GREY, 2
        ax.errorbar(x, d, yerr=[[d - row["D_lo_cm2_s"]], [row["D_hi_cm2_s"] - d]], fmt="none", ecolor=ecolor,
                    elinewidth=0.45, capsize=0, zorder=z - 0.5)
        ax.plot(x, d, marker=marker, ms=2.9, mfc=face, mec=edge, mew=0.45, linestyle="none", zorder=z)

    # classified sigma(300 K) as D(300 K)
    ne = panel["ne300"]
    x300 = 1000.0 / T_REF_K
    d300 = panel["sigma300"] / ne
    ax.errorbar(x300, d300, yerr=[[d300 - panel["sigma_lo"] / ne], [panel["sigma_hi"] / ne - d300]], fmt="none",
                ecolor=LINE_COLOR, elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=6)
    ax.plot(x300, d300, marker="*", ms=6.0, mfc=LINE_COLOR, mec="white", mew=0.3, linestyle="none", zorder=7)

    # annotation
    qualifiers = set(_split(comp["qualifiers"]))
    title = formula_tex(comp["formula"]) + ("$^{\\dagger}$" if "a32_dependent" in qualifiers else "")
    ax.text(0.04, 0.965, title, transform=ax.transAxes, ha="left", va="top", fontsize=style.FONT_SIZE)
    if panel["seed_mean"]:
        seeds = panel["seeds_used"]
        tag = f"Seed mean ({len(seeds)} seeds)" if len(seeds) >= 3 else "Seed mean (seeds " + ", ".join(
            str(s) for s in seeds) + ")"
        decimals = _decimals(panel["sigma300"])
        sigma_line = (f"$\\sigma$(300 K) = {_fmt_sigma(panel['sigma300'])} $\\pm$ "
                      f"{_fmt_sigma(panel['sigma_hi'] - panel['sigma300'], decimals)} mS cm$^{{-1}}$")
    else:
        tag = DECIDED_BY.get(comp["decided_by"], comp["decided_by"])
        sigma_line = f"$\\sigma$(300 K) = {_fmt_sigma(panel['sigma300'])} mS cm$^{{-1}}$"
    lines = [sigma_line, f"$E_a$ = {float(comp['Ea_eV']):.2f} eV", tag]
    if "near_threshold" in qualifiers:
        lines.append("95% interval spans 0.1 mS cm$^{-1}$")
    ax.text(0.965, 0.965, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=style.SMALL, linespacing=1.25)


def _format_axes(ax, *, left: bool, bottom: bool, top: bool) -> None:
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_yscale("log")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.yaxis.set_major_locator(FixedLocator([1e-10, 1e-8, 1e-6, 1e-4]))
    ax.yaxis.set_minor_locator(FixedLocator([1e-9, 1e-7, 1e-5, 1e-3]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(FixedLocator([1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.xaxis.set_minor_locator(NullLocator())
    if left:
        ax.set_ylabel("D (cm$^2$ s$^{-1}$)", labelpad=1.5)
    else:
        ax.tick_params(axis="y", which="both", labelleft=False)
    if bottom:
        ax.set_xlabel("1000/T (K$^{-1}$)", labelpad=1.5)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if top:
        sec = ax.secondary_xaxis("top", functions=(_inverse_1000, _inverse_1000))
        sec.xaxis.set_major_locator(FixedLocator(list(TOP_TICKS_K)))
        sec.xaxis.set_minor_locator(NullLocator())
        sec.set_xlabel("T (K)", labelpad=1.5)
        sec.tick_params(axis="x", pad=1.5)
        sec.spines["top"].set_linewidth(style.LINE)


def _legend(ax) -> None:
    ax.axis("off")
    seed_handles = [Line2D([0], [0], marker=style.SEED_MARKERS[s], ls="none", ms=3.2, mfc=FIT_COLOR, mec="#1F4F3F",
                           mew=0.45, label=f"Seed {s}") for s in sorted(style.SEED_MARKERS)]
    status_handles = [
        Line2D([0], [0], marker="o", ls="none", ms=3.2, mfc=FIT_COLOR, mec="#1F4F3F", mew=0.45,
               label="Eligible, used in fit"),
        Line2D([0], [0], marker="o", ls="none", ms=3.2, mfc="white", mec=style.GREY, mew=0.45,
               label="Not eligible (framework\nnot intact or unresolved)"),
        Line2D([0], [0], marker="o", ls="none", ms=3.2, mfc="white", mec=FIT_COLOR, mew=0.45,
               label="Eligible, seed excluded"),
        Line2D([0], [0], marker="x", ls="none", ms=3.4, mew=0.8, color=FAILED_COLOR,
               label="Failed kinisi fit (A3.2)"),
    ]
    fit_handles = [
        Line2D([0], [0], color=LINE_COLOR, lw=0.8, label="Pooled Arrhenius fit"),
        Line2D([0], [0], color=LINE_COLOR, lw=0.8, ls=(0, (2.2, 1.6)), label="Extrapolation to 300 K"),
        Patch(facecolor=FIT_COLOR, alpha=0.16, edgecolor="none", label="95% band of the fit"),
        Line2D([0], [0], marker="*", ls="none", ms=6.0, mfc=LINE_COLOR, mec="white", mew=0.3,
               label="Classified $\\sigma$(300 K) as D"),
    ]
    first = ax.legend(handles=seed_handles, loc="upper left", bbox_to_anchor=(0.0, 0.80), ncol=1,
                      handletextpad=0.4, labelspacing=0.45, borderaxespad=0.0)
    ax.add_artist(first)
    second = ax.legend(handles=status_handles + fit_handles, loc="upper left", bbox_to_anchor=(0.30, 0.80),
                       handletextpad=0.5, labelspacing=0.45, borderaxespad=0.0)
    ax.add_artist(second)
    ax.text(0.0, 0.0, "$^{\\dagger}$Class depends on amendment A3.2\n"
            "Error bars: D interval of the estimator;\nat 300 K, pooled 95% interval or seed SD",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=style.SMALL, linespacing=1.3)


def make() -> list[Path]:
    classes = data.classification()
    fast = classes[classes["class"] == FAST_VALIDATED].sort_values("sigma300_mS_cm", ascending=False)
    records = data.classifier_runs()
    leads = data.lead_decisions()
    panels = [_panel_data(comp, records[records["candidate_key"] == comp["candidate_key"]], leads)
              for _, comp in fast.iterrows()]
    if len(panels) > len(LETTERS):
        raise ValueError("more FAST_VALIDATED compositions than panel letters")

    fig = style.figure(style.DOUBLE, 150 * style.MM)
    width_mm, height_mm = 183.0, 150.0
    left, right, top, bottom = 13.0, 2.5, 11.0, 9.5
    col_gap, row_gap = 3.0, 7.0
    ncol, nrow = 4, 3
    w = (width_mm - left - right - (ncol - 1) * col_gap) / ncol
    h = (height_mm - top - bottom - (nrow - 1) * row_gap) / nrow

    def rect(row: int, col: int):
        x0 = left + col * (w + col_gap)
        y0 = height_mm - top - (row + 1) * h - row * row_gap
        return [x0 / width_mm, y0 / height_mm, w / width_mm, h / height_mm]

    n_panels = len(panels)
    last_row_of_col = {}
    for index in range(n_panels):
        last_row_of_col[index % ncol] = index // ncol
    for index, panel in enumerate(panels):
        row, col = divmod(index, ncol)
        ax = fig.add_axes(rect(row, col))
        _draw_panel(ax, panel)
        _format_axes(ax, left=col == 0, bottom=row == last_row_of_col[col], top=row == 0)
        # top row: letters level with the "T (K)" label, clear of the top-axis tick labels
        label_y = 1.0 if row else (height_mm - top + 5.2) / height_mm
        style.panel_label(ax, LETTERS[index], x=-0.004 if col else -0.035, y=label_y)
        style.save_source(_source_frame(panel), FIG, LETTERS[index])
    legend_ax = fig.add_axes(rect(nrow - 1, ncol - 1))
    _legend(legend_ax)
    return style.save(fig, NAME, "main")
