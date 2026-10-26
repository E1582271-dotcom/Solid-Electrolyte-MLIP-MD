"""Figure 5: robustness of the classification.

a, between-seed spread of D (max/min over eligible seeds) at every composition x temperature
with >= 2 eligible seeds, against the 2x replicate limit. b, prefactor sensitivity of the
SLOW_BOUNDED upper bounds (D0 >= 1e-4 against D0 >= 1e-5 cm^2/s). c, kinisi D against the
direct-MSD D for every kinisi record that meets the census eligibility criteria, with the 3x
amendment-A3.2 band. d, sigma(300 K) with its pooled 95 % interval. e, the census kinisi
estimator against the fixed-covariance estimator on the same 18 seed-0 trajectories.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter

from .. import data
from ..protocol import (
    ESTIMATOR_RATIO_FLAG,
    REPLICATE_MAX_RATIO,
    D0_MIN_cm2_s,
    D0_MIN_SENSITIVITY_cm2_s,
    FAST_THRESHOLD_mS_cm,
)
from . import style

NAME = "fig5_robustness"
FIG = "fig5"
RULE_COLOR = "#000000"
FAILED_COLOR = "#000000"
HIGHLIGHT = style.FAST_THRESHOLD_COLOR
HIGHLIGHT_FILL = "#FBE3D6"     # pale vermillion for rows/regions tied to the threshold
X_LO_D = 3e-4                  # panel d sigma axis lower limit (mS/cm)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def formula_tex(formula: str) -> str:
    """Chemical formula with subscripted counts as matplotlib mathtext, e.g. Li$_{7}$PS$_{6}$."""
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def _split(text) -> list[str]:
    return [item for item in str(text).split(";") if item and item != "nan"]


class _LogMM:
    """Map data coordinates of a log-log axes to millimetres within the axes box."""

    def __init__(self, xlim, ylim, width_mm: float, height_mm: float):
        self.lx, self.ly = np.log10(xlim), np.log10(ylim)
        self.w, self.h = width_mm, height_mm

    def __call__(self, x, y):
        x_mm = (np.log10(x) - self.lx[0]) / (self.lx[1] - self.lx[0]) * self.w
        y_mm = (np.log10(y) - self.ly[0]) / (self.ly[1] - self.ly[0]) * self.h
        return np.column_stack([np.atleast_1d(x_mm), np.atleast_1d(y_mm)])

    def y_data(self, y_mm: float) -> float:
        return 10 ** (self.ly[0] + y_mm / self.h * (self.ly[1] - self.ly[0]))


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _place_right_labels(to_mm: _LogMM, xs, ys, text_x: float, obstacles_mm: np.ndarray, label_h_mm: float,
                        clearance_mm: float = 1.0, step_mm: float = 0.2) -> list[float]:
    """Deterministic label heights for labels left-aligned at ``text_x``, with straight leaders.

    Each label (placed in order of its point's height) takes the candidate height closest to its
    point such that its leader keeps ``clearance_mm`` from every other plotted marker and its text
    does not overlap a label already placed. Returns heights in data units.
    """
    anchors = to_mm(xs, ys)
    tx = float(to_mm(np.array([text_x]), np.array([1.0]))[0, 0])
    candidates = np.arange(label_h_mm / 2 + 0.3, to_mm.h - label_h_mm / 2 - 0.3, step_mm)
    placed: list[float] = []
    heights = [0.0] * len(anchors)
    for index in np.argsort(anchors[:, 1], kind="stable"):
        anchor = anchors[index]
        others = obstacles_mm[np.linalg.norm(obstacles_mm - anchor, axis=1) > 1e-6]
        best, best_cost = None, np.inf
        for y in candidates:
            end = np.array([tx - 0.6, y])
            cost = abs(y - anchor[1])
            if np.any(_segment_distance(others, anchor + 0.01 * (end - anchor), end) < clearance_mm):
                cost += 100.0
            if any(abs(y - p) < label_h_mm + 0.3 for p in placed):
                cost += 100.0
            covered = (obstacles_mm[:, 0] > tx - 0.8) & (np.abs(obstacles_mm[:, 1] - y) < label_h_mm / 2 + 0.6)
            if np.any(covered):
                cost += 100.0
            if cost < best_cost:
                best, best_cost = y, cost
        placed.append(best)
        heights[index] = to_mm.y_data(best)
    return heights


def _log_axis(ax, axis: str, lo: float, hi: float, majors, minors=None) -> None:
    target = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    (ax.set_xlim if axis == "x" else ax.set_ylim)(lo, hi)
    target.set_major_locator(FixedLocator(list(majors)))
    if minors is None:
        minors = [10.0 ** e for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)]
    target.set_minor_locator(FixedLocator(list(minors)))
    target.set_minor_formatter(NullFormatter())


def _class_order(frame: pd.DataFrame) -> pd.DataFrame:
    rank = {c: i for i, c in enumerate(style.CLASS_ORDER)}
    frame = frame.copy()
    frame["_class_rank"] = frame["class"].map(rank)
    return frame.sort_values(["_class_rank", "sigma300_mS_cm"], ascending=[True, False]).drop(columns="_class_rank")


# ---------------------------------------------------------------------------
# a: between-seed spread
# ---------------------------------------------------------------------------
def _panel_a(ax, classes: pd.DataFrame) -> pd.DataFrame:
    records = data.classifier_runs()
    eligible = records[records["eligible"]]
    grouped = (eligible.groupby(["candidate_key", "formula", "class", "temperature_K"])
               .agg(n_seeds=("md_seed", "nunique"), D_max_cm2_s=("D_cm2_s", "max"), D_min_cm2_s=("D_cm2_s", "min"),
                    seeds=("md_seed", lambda s: ";".join(str(v) for v in sorted(s))))
               .reset_index())
    grouped = grouped[grouped["n_seeds"] >= 2].copy()
    grouped["D_max_min_ratio"] = grouped["D_max_cm2_s"] / grouped["D_min_cm2_s"]
    order = _class_order(classes[classes["candidate_key"].isin(grouped["candidate_key"])])
    row_of = {key: i for i, key in enumerate(order["candidate_key"])}
    grouped["row"] = grouped["candidate_key"].map(row_of)
    grouped = grouped.sort_values(["row", "temperature_K"]).reset_index(drop=True)

    ax.axvline(REPLICATE_MAX_RATIO, color=RULE_COLOR, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for i in range(len(order)):
        ax.axhline(i, color=style.LIGHT_GREY, lw=0.35, zorder=0)
    for _, g in grouped.iterrows():
        color = style.CLASS_COLORS[g["class"]]
        ax.plot(g["D_max_min_ratio"], g["row"], marker="o", ms=3.0, mfc=color, mec="white", mew=0.35, ls="none",
                zorder=3)
        if g["D_max_min_ratio"] > REPLICATE_MAX_RATIO:
            ax.text(g["D_max_min_ratio"] + 0.07, g["row"], f"{int(g['temperature_K'])} K", ha="left", va="center",
                    fontsize=style.SMALL)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([formula_tex(f) for f in order["formula"]], fontsize=style.SMALL)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(order) - 0.5, -0.6)
    ax.set_xlim(0.9, 4.4)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4]))
    ax.set_xlabel("Between-seed D max/min", labelpad=1.5)
    ax.text(REPLICATE_MAX_RATIO + 0.05, -0.55, f"{REPLICATE_MAX_RATIO:g}$\\times$ limit", ha="left", va="top",
            fontsize=style.SMALL)
    return grouped[["formula", "class", "temperature_K", "n_seeds", "seeds", "D_max_cm2_s", "D_min_cm2_s",
                    "D_max_min_ratio"]]


# ---------------------------------------------------------------------------
# b: prefactor sensitivity of the bounds
# ---------------------------------------------------------------------------
def _panel_b(ax, classes: pd.DataFrame) -> pd.DataFrame:
    bounded = classes[classes["class"] == "SLOW_BOUNDED"].copy()
    lo, hi = 1e-11, 10.0
    thr = FAST_THRESHOLD_mS_cm
    ax.fill_between([lo, thr], [thr, thr], [hi, hi], color=HIGHLIGHT_FILL, linewidth=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=RULE_COLOR, lw=0.5, zorder=1)
    ax.axhline(thr, color=style.FAST_THRESHOLD_COLOR, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axvline(thr, color=style.FAST_THRESHOLD_COLOR, lw=0.6, ls=(0, (3, 2)), zorder=1)
    markers = {"prefactor": "o", "pair": "^"}
    for kind, marker in markers.items():
        sub = bounded[bounded["bound_kind"] == kind]
        ax.plot(sub["sigma300_mS_cm"], sub["sigma300_upper_bound_D0min_1e-5_mS_cm"], marker=marker,
                ms=3.0 if marker == "o" else 3.2, mfc=style.CLASS_COLORS["SLOW_BOUNDED"], mec="white", mew=0.3,
                ls="none", zorder=3)
    flagged = bounded[bounded["qualifiers"].map(lambda q: "d0_dependent" in _split(q))]
    for _, row in flagged.iterrows():
        px, py = row["sigma300_mS_cm"], row["sigma300_upper_bound_D0min_1e-5_mS_cm"]
        ax.plot(px, py, marker="o", ms=4.6, mfc="none", mec=HIGHLIGHT, mew=0.7, ls="none", zorder=4)
        ax.annotate(formula_tex(row["formula"]), xy=(px, py), xytext=(px * 10 ** -3.2, py * 10 ** 0.15),
                    ha="right", va="center", fontsize=style.SMALL,
                    arrowprops={"arrowstyle": "-", "lw": 0.4, "color": style.GREY, "shrinkA": 0.5, "shrinkB": 2.0})
    ax.text(lo * 10 ** 0.25, hi * 10 ** -0.25, "Class would change", ha="left", va="top", fontsize=style.SMALL,
            color=HIGHLIGHT)
    _log_axis(ax, "x", lo, hi, [1e-10, 1e-6, 1e-2])
    _log_axis(ax, "y", lo, hi, [1e-10, 1e-6, 1e-2])
    for setter, d0 in ((ax.set_xlabel, D0_MIN_cm2_s), (ax.set_ylabel, D0_MIN_SENSITIVITY_cm2_s)):
        setter(f"$\\sigma$(300 K) bound, D$_0$ ≥ 10$^{{{int(np.log10(d0))}}}$ (mS cm$^{{-1}}$)", labelpad=1.5)
    handles = [Line2D([0], [0], marker="o", ls="none", ms=3.0, mfc=style.CLASS_COLORS["SLOW_BOUNDED"], mec="white",
                      mew=0.3, label="Prefactor bound"),
               Line2D([0], [0], marker="^", ls="none", ms=3.2, mfc=style.CLASS_COLORS["SLOW_BOUNDED"], mec="white",
                      mew=0.3, label="Slope (pair) bound")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.80), borderaxespad=0.2, handletextpad=0.3,
              labelspacing=0.3)
    out = bounded[["formula", "bound_kind", "sigma300_mS_cm", "sigma300_upper_bound_D0min_1e-5_mS_cm",
                   "sensitivity_D0min_1e-5_class", "qualifiers"]].rename(
        columns={"sigma300_mS_cm": "sigma300_bound_D0min_1e-4_mS_cm"})
    return out.sort_values("sigma300_bound_D0min_1e-4_mS_cm", ascending=False)


# ---------------------------------------------------------------------------
# c: kinisi against direct MSD
# ---------------------------------------------------------------------------
def _panel_c(ax, classes: pd.DataFrame) -> pd.DataFrame:
    runs = data.runs()
    rec = runs[runs["eligible_census_criteria"] & (runs["estimator"] == "kinisi_mcmc")].copy()
    rec = rec.merge(classes[["candidate_key", "class", "qualifiers"]], on="candidate_key", how="left",
                    validate="many_to_one")
    rec["ratio_kinisi_direct"] = rec["D_cm2_s"] / rec["D_direct_cm2_s"]
    x_lo, x_hi, y_lo, y_hi = 1e-7, 1e-4, 1e-12, 1e-4
    grid = np.array([x_lo, x_hi])
    ax.fill_between(grid, grid / ESTIMATOR_RATIO_FLAG, grid * ESTIMATOR_RATIO_FLAG, color=style.LIGHT_GREY,
                    alpha=0.6, linewidth=0, zorder=0)
    ax.plot(grid, grid, color=RULE_COLOR, lw=0.5, zorder=1)
    ok = rec[~rec["kinisi_fit_failed"]]
    for cls in style.CLASS_ORDER:
        sub = ok[ok["class"] == cls]
        if len(sub):
            ax.plot(sub["D_direct_cm2_s"], sub["D_cm2_s"], marker="o", ms=2.3, mfc=style.CLASS_COLORS[cls],
                    mec="white", mew=0.25, ls="none", zorder=2)
    failed = rec[rec["kinisi_fit_failed"]]
    ax.plot(failed["D_direct_cm2_s"], failed["D_cm2_s"], marker="x", ms=3.2, mew=0.75, color=FAILED_COLOR,
            ls="none", zorder=4)

    # label the failed fits of the compositions whose class depends on A3.2
    dependent = failed[failed["qualifiers"].map(lambda q: "a32_dependent" in _split(q))].copy()
    dependent = dependent.sort_values("D_cm2_s")
    text_x = 1.15e-5
    box = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches() / style.MM
    to_mm = _LogMM((x_lo, x_hi), (y_lo, y_hi), box.width * fig_w, box.height * fig_h)
    obstacles = to_mm(rec["D_direct_cm2_s"].to_numpy(), rec["D_cm2_s"].to_numpy())
    text_y = _place_right_labels(to_mm, dependent["D_direct_cm2_s"].to_numpy(), dependent["D_cm2_s"].to_numpy(),
                                 text_x, obstacles, label_h_mm=4.3)
    for (_, row), ty in zip(dependent.iterrows(), text_y):
        label = f"{formula_tex(row['formula'])}\n{int(row['temperature_K'])} K, seed {int(row['md_seed'])}"
        ax.annotate(label, xy=(row["D_direct_cm2_s"], row["D_cm2_s"]), xytext=(text_x, ty), ha="left", va="center",
                    fontsize=style.SMALL, linespacing=1.15,
                    arrowprops={"arrowstyle": "-", "lw": 0.4, "color": style.GREY, "shrinkA": 0.5, "shrinkB": 2.2})
    _log_axis(ax, "x", x_lo, x_hi, [1e-7, 1e-6, 1e-5, 1e-4])
    _log_axis(ax, "y", y_lo, y_hi, [1e-11, 1e-9, 1e-7, 1e-5])
    ax.set_xlabel("Direct-MSD D (cm$^2$ s$^{-1}$)", labelpad=1.5)
    ax.set_ylabel("kinisi D (cm$^2$ s$^{-1}$)", labelpad=1.5)
    handles = [Line2D([0], [0], color=RULE_COLOR, lw=0.5, label="1:1"),
               Patch(facecolor=style.LIGHT_GREY, alpha=0.6, edgecolor="none",
                     label=f"Within {ESTIMATOR_RATIO_FLAG:g}$\\times$"),
               Line2D([0], [0], marker="x", ls="none", ms=3.2, mew=0.75, color=FAILED_COLOR,
                      label="Failed fit (A3.2)")]
    ax.legend(handles=handles, loc="upper left", borderaxespad=0.2, handletextpad=0.4, labelspacing=0.3)
    out = rec[["record_id", "formula", "class", "temperature_K", "md_seed", "stage", "used_by_classifier",
               "D_direct_cm2_s", "D_cm2_s", "ratio_kinisi_direct", "kinisi_fit_failed"]].rename(
        columns={"D_cm2_s": "D_kinisi_cm2_s"})
    out["labelled"] = out["record_id"].isin(set(dependent["record_id"]))
    return out


# ---------------------------------------------------------------------------
# d: pooled 95 % intervals
# ---------------------------------------------------------------------------
def _panel_d(ax, classes: pd.DataFrame) -> pd.DataFrame:
    with_ci = classes.dropna(subset=["sigma300_ci95_lo_mS_cm", "sigma300_ci95_hi_mS_cm"])
    with_ci = with_ci.sort_values("sigma300_mS_cm", ascending=False).reset_index(drop=True)
    thr = FAST_THRESHOLD_mS_cm
    with_ci["contains_threshold"] = (with_ci["sigma300_ci95_lo_mS_cm"] <= thr) & (
        with_ci["sigma300_ci95_hi_mS_cm"] >= thr)
    for i, row in with_ci.iterrows():
        if row["contains_threshold"]:
            ax.axhspan(i - 0.42, i + 0.42, color=HIGHLIGHT_FILL, linewidth=0, zorder=0)
    ax.axvline(thr, color=style.FAST_THRESHOLD_COLOR, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for i, row in with_ci.iterrows():
        color = style.CLASS_COLORS[row["class"]]
        ax.plot([row["sigma300_ci95_lo_mS_cm"], row["sigma300_ci95_hi_mS_cm"]], [i, i], color=color, lw=0.9,
                solid_capstyle="butt", zorder=2)
        ax.plot(row["sigma300_mS_cm"], i, marker="o", ms=3.2, mfc=color, mec="white", mew=0.35, ls="none", zorder=3)
        tag = {"replicate_rule": "replicate rule", "terminal_rule": "terminal rule"}.get(row["decided_by"],
                                                                                       row["decided_by"])
        seeds = len(_split(row["seeds"]))
        background = HIGHLIGHT_FILL if row["contains_threshold"] else "white"
        ax.text(row["sigma300_ci95_hi_mS_cm"] * 10 ** 0.08, i, f"{tag}, {seeds} seeds", ha="left", va="center",
                fontsize=style.SMALL, color="#4D4D4D", zorder=4,
                bbox={"boxstyle": "square,pad=0.1", "facecolor": background, "edgecolor": "none"})
        if row["contains_threshold"]:
            ax.text(X_LO_D * 10 ** 0.06, i, "near threshold", ha="left", va="center", fontsize=style.SMALL,
                    color=HIGHLIGHT)
    ax.set_yticks(range(len(with_ci)))
    ax.set_yticklabels([formula_tex(f) for f in with_ci["formula"]], fontsize=style.SMALL)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(with_ci) - 0.5, -0.6)
    _log_axis(ax, "x", X_LO_D, 1e3, [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
              minors=[m * 10.0 ** e for e in range(-4, 3) for m in range(2, 10) if m * 10.0 ** e >= X_LO_D])
    ax.set_xlabel("$\\sigma$(300 K) (mS cm$^{-1}$)", labelpad=1.5)
    return with_ci[["formula", "class", "decided_by", "seeds", "sigma300_mS_cm", "sigma300_ci95_lo_mS_cm",
                    "sigma300_ci95_hi_mS_cm", "contains_threshold", "qualifiers"]]


# ---------------------------------------------------------------------------
# e: estimator comparison on identical trajectories
# ---------------------------------------------------------------------------
def _panel_e(ax) -> pd.DataFrame:
    runs = data.runs()
    census = runs[(runs["stage"] == "census") & (runs["estimator"] == "kinisi_mcmc")]
    replicate = runs[(runs["stage"] == "replicate") & (runs["estimator"] == "fixed_cov_analytic")]
    pairs = census.merge(replicate, on="trajectory_id", suffixes=("_kinisi", "_fixed"), validate="one_to_one")
    pairs = pairs.sort_values(["formula_kinisi", "temperature_K_kinisi"]).reset_index(drop=True)
    pairs["ratio_fixed_kinisi"] = pairs["D_cm2_s_fixed"] / pairs["D_cm2_s_kinisi"]
    temps = sorted(pairs["temperature_K_kinisi"].unique())
    cmap = colormaps[style.TEMPERATURE_CMAP]
    colors = {t: cmap(v) for t, v in zip(temps, np.linspace(0.1, 0.8, len(temps)))}
    lo, hi = 3e-7, 2e-4
    ax.plot([lo, hi], [lo, hi], color=RULE_COLOR, lw=0.5, zorder=1)
    for t in temps:
        sub = pairs[pairs["temperature_K_kinisi"] == t]
        xk, yf = sub["D_cm2_s_kinisi"], sub["D_cm2_s_fixed"]
        ax.errorbar(xk, yf, xerr=[xk - sub["D_lo_cm2_s_kinisi"], sub["D_hi_cm2_s_kinisi"] - xk],
                    yerr=[yf - sub["D_lo_cm2_s_fixed"], sub["D_hi_cm2_s_fixed"] - yf], fmt="none",
                    ecolor=colors[t], elinewidth=0.4, capsize=0, alpha=0.8, zorder=2)
        ax.plot(xk, yf, marker="o", ms=3.0, mfc=colors[t], mec="white", mew=0.3, ls="none", zorder=3,
                label=f"{int(t)} K")
    _log_axis(ax, "x", lo, hi, [1e-6, 1e-5, 1e-4])
    _log_axis(ax, "y", lo, hi, [1e-6, 1e-5, 1e-4])
    ax.set_xlabel("kinisi D, census (cm$^2$ s$^{-1}$)", labelpad=1.5)
    ax.set_ylabel("Fixed-covariance D (cm$^2$ s$^{-1}$)", labelpad=1.5)
    ratio = pairs["ratio_fixed_kinisi"]
    ax.text(0.97, 0.03, f"n = {len(pairs)} trajectories\nratio {ratio.min():.2f}–{ratio.max():.2f}\n"
            f"(median {ratio.median():.2f})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=style.SMALL, linespacing=1.25)
    ax.legend(loc="upper left", borderaxespad=0.2, handletextpad=0.2, labelspacing=0.3)
    return pd.DataFrame({
        "trajectory_id": pairs["trajectory_id"], "formula": pairs["formula_kinisi"],
        "temperature_K": pairs["temperature_K_kinisi"], "md_seed": pairs["md_seed_kinisi"],
        "census_record_id": pairs["record_id_kinisi"], "replicate_record_id": pairs["record_id_fixed"],
        "D_kinisi_cm2_s": pairs["D_cm2_s_kinisi"], "D_kinisi_lo_cm2_s": pairs["D_lo_cm2_s_kinisi"],
        "D_kinisi_hi_cm2_s": pairs["D_hi_cm2_s_kinisi"], "D_fixed_cov_cm2_s": pairs["D_cm2_s_fixed"],
        "D_fixed_cov_lo_cm2_s": pairs["D_lo_cm2_s_fixed"], "D_fixed_cov_hi_cm2_s": pairs["D_hi_cm2_s_fixed"],
        "ratio_fixed_kinisi": ratio})


def _class_legend(fig, rect, classes: pd.DataFrame) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    present = [c for c in style.CLASS_ORDER if (classes["class"] == c).any()]
    handles = [Line2D([0], [0], marker="o", ls="none", ms=3.2, mfc=style.CLASS_COLORS[c], mec="white", mew=0.35,
                      label=style.CLASS_LABELS[c]) for c in present]
    ax.legend(handles=handles, loc="upper left", borderaxespad=0.0, handletextpad=0.3, labelspacing=0.45,
              title="Class", title_fontsize=style.SMALL, alignment="left")


def make() -> list[Path]:
    classes = data.classification()
    width_mm, height_mm = 183.0, 110.0
    fig = style.figure(style.DOUBLE, height_mm * style.MM)

    def rect(x0, y0, w, h):
        return [x0 / width_mm, y0 / height_mm, w / width_mm, h / height_mm]

    row1_y, row2_y, row_h = 62.0, 10.0, 42.0
    ax_a = fig.add_axes(rect(20.0, row1_y, 38.0, row_h))
    ax_b = fig.add_axes(rect(72.0, row1_y, 42.0, row_h))
    ax_c = fig.add_axes(rect(130.0, row1_y, 50.0, row_h))
    ax_d = fig.add_axes(rect(20.0, row2_y, 66.0, row_h))
    ax_e = fig.add_axes(rect(100.0, row2_y, 42.0, row_h))

    sources = {"a": _panel_a(ax_a, classes), "b": _panel_b(ax_b, classes), "c": _panel_c(ax_c, classes),
               "d": _panel_d(ax_d, classes), "e": _panel_e(ax_e)}
    _class_legend(fig, rect(150.0, row2_y + 22.0, 30.0, 20.0), classes)
    for letter, ax in zip("abcde", (ax_a, ax_b, ax_c, ax_d, ax_e)):
        style.panel_label(ax, letter, x=-0.075 if letter in "ad" else -0.045)
        style.save_source(sources[letter], FIG, letter)
    return style.save(fig, NAME, "main")
