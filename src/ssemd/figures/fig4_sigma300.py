"""Figure 4: room-temperature conductivity estimates and bounds for all 62 compositions.

a, every composition ranked by sigma(300 K): estimates (filled, class colour) with the pooled
95 % interval where one exists, and upper bounds (bar, stem and downward caret) at the
pre-registered prefactor D0 >= 1e-4 cm^2/s, joined to the looser bound at D0 >= 1e-5 cm^2/s.
b, activation energy against sigma(300 K): fitted or two-point estimates as points, and for
bounded compositions the corner glyph (Ea lower bound, sigma(300 K) upper bound).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FixedLocator, NullFormatter

from .. import data
from ..protocol import D0_MIN_cm2_s, D0_MIN_SENSITIVITY_cm2_s, FAST_THRESHOLD_mS_cm
from . import style

NAME = "fig4_sigma300"
FIG = "fig4"

Y_MIN_LOG = -11.0               # sigma axis lower limit (log10 mS/cm)
SPINE_TOP_LOG = 2.0             # left spine and ticks stop here; the labels sit above
LABEL_STEP_MM = 2.05            # vertical pitch of the staircase labels (5 pt text)
LABEL_CLEAR_DECADES = 0.30      # clearance between a label and the points below it
BOUND_FAINT = "#BDBDBD"
FOOTNOTE_MARKS = (("a32_dependent", "\\dagger"), ("near_threshold", "\\ddagger"), ("d0_dependent", "\\S"))
FOOTNOTES = ("$^{\\dagger}$Class depends on amendment A3.2 (failed kinisi fits)",
             "$^{\\ddagger}$Pooled 95% interval contains 0.1 mS cm$^{-1}$",
             "$^{\\S}$Bound exceeds 0.1 mS cm$^{-1}$ at D$_0$ ≥ 10$^{-5}$ cm$^2$ s$^{-1}$")


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def formula_tex(formula: str) -> str:
    """Chemical formula with subscripted counts as matplotlib mathtext, e.g. Li$_{7}$PS$_{6}$."""
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def _split(text) -> list[str]:
    return [item for item in str(text).split(";") if item and item != "nan"]


def _sci(value: float) -> str:
    exponent = int(np.round(np.log10(value)))
    return f"10$^{{{exponent}}}$"


def _corner_marker() -> MplPath:
    """'Value lies up and to the left': arms from the bound point to the left and upwards, with heads."""
    codes, verts = [], []

    def seg(points):
        verts.extend(points)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))

    seg([(0.0, 0.0), (-1.0, 0.0)])
    seg([(-0.62, 0.32), (-1.0, 0.0), (-0.62, -0.32)])
    seg([(0.0, 0.0), (0.0, 1.0)])
    seg([(-0.32, 0.62), (0.0, 1.0), (0.32, 0.62)])
    return MplPath(verts, codes)


CORNER = _corner_marker()


def _upper_bound_marker() -> MplPath:
    """Upper-limit glyph anchored at the bound: short bar, downward stem and caret."""
    verts = [(-0.26, 0.0), (0.26, 0.0), (0.0, 0.0), (0.0, -1.0), (-0.24, -0.70), (0.0, -1.0), (0.24, -0.70)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.MOVETO, MplPath.LINETO, MplPath.MOVETO, MplPath.LINETO,
             MplPath.LINETO]
    return MplPath(verts, codes)


UPPER_BOUND = _upper_bound_marker()
BOUND_MS = 10.0                 # glyph height = BOUND_MS / 2 points (about 1.8 mm)


def _point_top(row) -> float:
    """Highest plotted value of one composition in panel a (point, interval top or loose bound)."""
    values = [row["sigma300_mS_cm"], row["sigma300_ci95_hi_mS_cm"], row["sigma300_upper_bound_D0min_1e-5_mS_cm"]]
    return float(np.nanmax(np.asarray(values, float)))


def _ranked() -> pd.DataFrame:
    frame = data.classification().sort_values("sigma300_mS_cm", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["qualifier_set"] = frame["qualifiers"].map(lambda q: set(_split(q)))
    return frame


# ---------------------------------------------------------------------------
# Panel a
# ---------------------------------------------------------------------------
def _staircase(frame: pd.DataFrame, labelled: pd.DataFrame, axis_height_mm: float):
    """Label heights (log10) for a descending staircase above the labelled points.

    Label i (rank order) sits LABEL_STEP above label i+1, so the vertical leader of every later
    point ends below every earlier label: no leader crosses a label. The lowest label clears
    every point under its horizontal extent. Returns label log-heights and the axis top (log10).
    """
    n = len(labelled)
    # The highest point a label of the last labelled rank must clear: points within ~8 ranks to its right.
    last_rank = int(labelled["rank"].iloc[-1])
    right = frame[(frame["rank"] > last_rank) & (frame["rank"] <= last_rank + 8)]
    tops = np.log10([_point_top(row) for _, row in right.iterrows()])
    lowest_needed = max(float(np.max(tops)), float(np.log10(_point_top(labelled.iloc[-1])))) + LABEL_CLEAR_DECADES
    # Solve top/step self-consistently: step (decades) = LABEL_STEP_MM * (top - Y_MIN) / height.
    top = SPINE_TOP_LOG + 1.0
    for _ in range(50):
        step = LABEL_STEP_MM * (top - Y_MIN_LOG) / axis_height_mm
        first = lowest_needed + (n - 1) * step
        # the first label must also clear its own point
        first = max(first, float(np.log10(_point_top(labelled.iloc[0]))) + LABEL_CLEAR_DECADES)
        new_top = first + 0.6 * step
        if abs(new_top - top) < 1e-6:
            break
        top = new_top
    heights = first - step * np.arange(n)
    return heights, top


def _panel_a(ax, frame: pd.DataFrame, axis_height_mm: float) -> pd.DataFrame:
    threshold = FAST_THRESHOLD_mS_cm
    ax.axhline(threshold, color=style.FAST_THRESHOLD_COLOR, lw=0.6, ls=(0, (3, 2)), zorder=1)

    est = frame[frame["sigma_kind"] == "estimate"]
    bnd = frame[frame["sigma_kind"] == "upper_bound"]

    # estimates with pooled 95 % interval
    for _, row in est.iterrows():
        color = style.CLASS_COLORS[row["class"]]
        lo, hi = row["sigma300_ci95_lo_mS_cm"], row["sigma300_ci95_hi_mS_cm"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([row["rank"]] * 2, [lo, hi], color=color, lw=0.7, solid_capstyle="butt", zorder=2)
        ax.plot(row["rank"], row["sigma300_mS_cm"], marker="o", ms=3.0, mfc=color, mec="white", mew=0.3,
                ls="none", zorder=4)

    # upper bounds: faint D0 >= 1e-5 bound joined to the D0 >= 1e-4 bound, then bar + stem + caret
    for _, row in bnd.iterrows():
        color = style.CLASS_COLORS[row["class"]]
        x, y = row["rank"], row["sigma300_mS_cm"]
        loose = row["sigma300_upper_bound_D0min_1e-5_mS_cm"]
        if np.isfinite(loose) and loose > y * 1.0001:
            ax.plot([x, x], [y, loose], color=BOUND_FAINT, lw=0.45, zorder=1.5)
            ax.plot(x, loose, marker="_", ms=3.2, mew=0.7, color=BOUND_FAINT, ls="none", zorder=1.6)
        ax.plot(x, y, marker=UPPER_BOUND, ms=BOUND_MS, mfc="none", mec=color, mew=0.65, ls="none", zorder=3)

    # labels: compositions above the threshold, the near-threshold and the prefactor-dependent ones,
    # as a staircase with vertical leaders
    flags = frame["qualifier_set"].map(lambda q: bool(q & {"near_threshold", "d0_dependent"}))
    labelled = frame[(frame["sigma300_mS_cm"] >= threshold) | flags]
    heights, top = _staircase(frame, labelled, axis_height_mm)
    for (_, row), h in zip(labelled.iterrows(), heights):
        x = row["rank"]
        start = _point_top(row) * 10 ** 0.12
        ax.plot([x, x], [start, 10 ** h], color=style.GREY, lw=0.35, zorder=0.5)
        marks = "".join(symbol for key, symbol in FOOTNOTE_MARKS if key in row["qualifier_set"])
        ax.text(x + 0.3, 10 ** h, formula_tex(row["formula"]) + (f"$^{{{marks}}}$" if marks else ""),
                ha="left", va="center", fontsize=style.SMALL)

    ax.text(len(frame) + 0.4, threshold * 10 ** 0.12, f"{threshold:g} mS cm$^{{-1}}$", ha="right", va="bottom",
            fontsize=style.SMALL, color=style.FAST_THRESHOLD_COLOR)

    ax.set_xlim(0, len(frame) + 1)
    ax.set_yscale("log")
    ax.set_ylim(10 ** Y_MIN_LOG, 10 ** top)
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(-10, int(SPINE_TOP_LOG) + 1, 2)]))
    ax.yaxis.set_minor_locator(FixedLocator([10.0 ** e for e in range(int(Y_MIN_LOG), int(SPINE_TOP_LOG) + 1)]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.spines["left"].set_bounds(10 ** Y_MIN_LOG, 10 ** SPINE_TOP_LOG)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 20, 30, 40, 50, 60]))
    ax.set_xlabel("Rank by $\\sigma$(300 K)", labelpad=1.5)
    ax.set_ylabel("$\\sigma$(300 K) (mS cm$^{-1}$)", labelpad=1.5)

    # legend in the empty upper right
    counts = frame["class"].value_counts()
    class_handles = [Line2D([0], [0], marker="s", ls="none", ms=3.6, mfc=style.CLASS_COLORS[c], mec="none",
                            label=f"{style.CLASS_LABELS[c]} ({counts[c]})")
                     for c in style.CLASS_ORDER if counts.get(c, 0)]
    kind_handles = [
        Line2D([0], [0], marker="o", ls="-", lw=0.7, ms=3.0, color=style.GREY, mfc=style.GREY, mec="white",
               mew=0.3, label="Estimate (pooled 95% interval)"),
        Line2D([0], [0], marker=UPPER_BOUND, ls="none", ms=BOUND_MS, mfc="none", mec=style.GREY, mew=0.65,
               label=f"Upper bound, D$_0$ ≥ {_sci(D0_MIN_cm2_s)} cm$^2$ s$^{{-1}}$"),
        Line2D([0], [0], marker="_", ls="none", ms=3.2, mew=0.7, color=BOUND_FAINT,
               label=f"Upper bound, D$_0$ ≥ {_sci(D0_MIN_SENSITIVITY_cm2_s)} cm$^2$ s$^{{-1}}$"),
    ]
    first = ax.legend(handles=class_handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0,
                      handletextpad=0.3, labelspacing=0.4)
    ax.add_artist(first)
    ax.legend(handles=kind_handles, loc="upper right", bbox_to_anchor=(0.62, 1.0), borderaxespad=0.0,
              handletextpad=0.4, labelspacing=0.4)
    ax.text(0.012, 0.02, "\n".join(FOOTNOTES), transform=ax.transAxes, ha="left", va="bottom",
            fontsize=style.SMALL, linespacing=1.3)

    source = frame[["rank", "formula", "class", "sigma_kind", "sigma300_mS_cm", "sigma300_ci95_lo_mS_cm",
                    "sigma300_ci95_hi_mS_cm", "bound_kind", "sigma300_upper_bound_D0min_1e-5_mS_cm",
                    "decided_by", "qualifiers"]].copy()
    source["labelled"] = source["formula"].isin(set(labelled["formula"]))
    return source


# ---------------------------------------------------------------------------
# Panel b
# ---------------------------------------------------------------------------
def _panel_b(ax, frame: pd.DataFrame) -> pd.DataFrame:
    threshold = FAST_THRESHOLD_mS_cm
    ax.axvline(threshold, color=style.FAST_THRESHOLD_COLOR, lw=0.6, ls=(0, (3, 2)), zorder=1)
    est = frame[frame["Ea_kind"] != "lower_bound"]
    bnd = frame[frame["Ea_kind"] == "lower_bound"]
    for _, row in bnd.iterrows():
        ax.plot(row["sigma300_mS_cm"], row["Ea_eV"], marker=CORNER, ms=6.0, mew=0.45, mfc="none",
                mec=style.CLASS_COLORS[row["class"]], ls="none", zorder=2)
    for _, row in est.iterrows():
        color = style.CLASS_COLORS[row["class"]]
        lo, hi = row["sigma300_ci95_lo_mS_cm"], row["sigma300_ci95_hi_mS_cm"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [row["Ea_eV"]] * 2, color=color, lw=0.7, solid_capstyle="butt", zorder=3)
        marker = "o" if row["Ea_kind"] == "fit" else "D"
        ax.plot(row["sigma300_mS_cm"], row["Ea_eV"], marker=marker, ms=3.0 if marker == "o" else 2.6, mfc=color,
                mec="white", mew=0.3, ls="none", zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(10 ** Y_MIN_LOG, 10 ** 3)
    ax.xaxis.set_major_locator(FixedLocator([1e-10, 1e-6, 1e-2, 1e2]))
    ax.xaxis.set_minor_locator(FixedLocator([10.0 ** e for e in range(-11, 4)]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.0, 1.15)
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.set_xlabel("$\\sigma$(300 K) (mS cm$^{-1}$)", labelpad=1.5)
    ax.set_ylabel("E$_a$ (eV)", labelpad=1.5)
    handles = [
        Line2D([0], [0], marker="o", ls="-", lw=0.7, ms=3.0, color=style.GREY, mfc=style.GREY, mec="white", mew=0.3,
               label="Arrhenius fit (pooled 95% interval)"),
        Line2D([0], [0], marker="D", ls="none", ms=2.6, mfc=style.GREY, mec="white", mew=0.3,
               label="Two-point estimate"),
        Line2D([0], [0], marker=CORNER, ls="none", ms=4.2, mew=0.5, mfc="none", mec=style.GREY,
               label="Bound: E$_a$ ≥ and $\\sigma$(300 K) ≤ value"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), borderaxespad=0.2,
              handletextpad=0.4, labelspacing=0.4)
    return frame[["formula", "class", "sigma300_mS_cm", "sigma_kind", "sigma300_ci95_lo_mS_cm",
                  "sigma300_ci95_hi_mS_cm", "Ea_eV", "Ea_kind", "decided_by"]].copy()


def make() -> list[Path]:
    frame = _ranked()
    width_mm, height_mm = 183.0, 85.0
    fig = style.figure(style.DOUBLE, height_mm * style.MM)
    bottom, top_margin = 9.5, 4.0
    axis_h = height_mm - bottom - top_margin
    ax_a = fig.add_axes([13.0 / width_mm, bottom / height_mm, 110.0 / width_mm, axis_h / height_mm])
    ax_b = fig.add_axes([137.0 / width_mm, bottom / height_mm, 44.0 / width_mm, axis_h / height_mm])
    source_a = _panel_a(ax_a, frame, axis_h)
    source_b = _panel_b(ax_b, frame)
    style.panel_label(ax_a, "a", x=-0.045)
    style.panel_label(ax_b, "b", x=-0.035)
    style.save_source(source_a, FIG, "a")
    style.save_source(source_b, FIG, "b")
    return style.save(fig, NAME, "main")
