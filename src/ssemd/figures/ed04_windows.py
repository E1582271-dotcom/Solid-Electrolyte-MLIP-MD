"""Extended Data Fig. 4 - time-window and covariance sensitivity of the kinisi D estimates.

For every record the ratio max(D)/min(D) of the kinisi diffusion coefficient across
alternative analysis settings (``data.windows()``):

a) census records, four trajectory windows (0-100, 0-150, 0-200 and 100-200 ps);
b) census records, three covariance condition-number limits (1e6, 1e8, 1e10);
c) definitive records, trajectory windows (first half, first three quarters, full,
   second half and first 200 ps).

Points are grouped by the composition's final class and spread with a deterministic
beeswarm (no random jitter). Filled = diffusion resolved, open = not resolved, cross =
failed kinisi fit (amendment A3.2). The dashed line is the 2x replicate-agreement limit.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import data
from ..protocol import REPLICATE_MAX_RATIO
from . import style

NAME = "ed04_windows"
FIGURE = "ed04"
GROUPS = style.CLASS_ORDER[:4]
Y_LIM = (0.7, 1e6)
PANELS = (
    ("a", "census", "trajectory_window", "Max/min D (census, 4 time windows)"),
    ("b", "census", "covariance_condition_limit", "Max/min D (census, 3 covariance limits)"),
    ("c", "definitive", "trajectory_window", "Max/min D (definitive, 5 time windows)"),
)
SHORT_LABELS = {"FAST_VALIDATED": "Fast\nvalidated", "SLOW_MEASURED": "Slow\nmeasured",
                "SLOW_ESTIMATED": "Slow\nestimated", "SLOW_BOUNDED": "Slow\nbounded"}
MARKER_SIZE = 2.6  # points
DIAMETER_MM = 1.05  # spacing between swarm points (slightly above the marker size)
GROUP_WIDTH_MM = 13.0
MAX_HALF_WIDTH_MM = 5.9


def beeswarm(y_mm: np.ndarray, diameter: float, max_half: float) -> np.ndarray:
    """Deterministic beeswarm: x offsets (same units as ``y_mm``) placing points in input order.

    Each point takes the admissible offset closest to the centre line that does not
    overlap an already placed point; if none exists within ``max_half`` the offset
    with the fewest overlaps is used.
    """
    xs = np.zeros(len(y_mm))
    placed: list[tuple[float, float]] = []
    for i, y in enumerate(y_mm):
        near = [(px, math.sqrt(max(diameter ** 2 - (py - y) ** 2, 0.0))) for px, py in placed
                if abs(py - y) < diameter]
        candidates = {0.0}
        for px, half in near:
            candidates.update((px - half - 1e-6, px + half + 1e-6))
        ordered = sorted(candidates, key=lambda c: (abs(c), c))
        choice = None
        for c in ordered:
            if abs(c) <= max_half and all(abs(c - px) >= half for px, half in near):
                choice = c
                break
        if choice is None:
            bounded = [c for c in ordered if abs(c) <= max_half] or [0.0]
            choice = min(bounded, key=lambda c: (sum(abs(c - px) < half for px, half in near), abs(c), c))
        xs[i] = choice
        placed.append((choice, y))
    return xs


def ratios(stage: str, axis: str) -> pd.DataFrame:
    windows = data.windows()
    windows = windows[windows["axis"] == axis]
    runs = data.runs()
    runs = runs[runs["stage"] == stage][["record_id", "formula", "p1_rank", "candidate_key", "temperature_K",
                                         "md_seed", "diffusion_resolved", "kinisi_fit_failed"]]
    grouped = windows.groupby("record_id")["D_cm2_s"].agg(D_max_cm2_s="max", D_min_cm2_s="min",
                                                          n_settings="count").reset_index()
    grouped["settings"] = windows.groupby("record_id")["setting"].agg(lambda s: ";".join(s)).values
    frame = runs.merge(grouped, on="record_id", how="inner", validate="one_to_one")
    frame["ratio_max_min"] = frame["D_max_cm2_s"] / frame["D_min_cm2_s"]
    classes = data.classification()[["candidate_key", "class"]]
    frame = frame.merge(classes, on="candidate_key", how="left", validate="many_to_one")
    frame["state"] = np.select([frame["kinisi_fit_failed"], frame["diffusion_resolved"]],
                               ["kinisi_fit_failed", "resolved"], "not_resolved")
    return frame.drop(columns="candidate_key")


def make() -> list[Path]:
    fig = style.figure(style.DOUBLE, 70 * style.MM)
    left, right, bottom, top, gap = 0.07, 0.985, 0.17, 0.93, 0.075
    width = (right - left - 2 * gap) / 3
    height_mm = (top - bottom) * 70.0
    decades = math.log10(Y_LIM[1]) - math.log10(Y_LIM[0])
    for index, (letter, stage, axis, ylabel) in enumerate(PANELS):
        ax = fig.add_axes([left + index * (width + gap), bottom, width, top - bottom])
        frame = ratios(stage, axis)
        frame["group_index"] = [GROUPS.index(c) for c in frame["class"]]
        frame = frame.sort_values(["group_index", "ratio_max_min", "record_id"]).reset_index(drop=True)
        frame["x_plot"] = frame["group_index"].astype(float)
        for group, members in frame.groupby("group_index"):
            y_mm = (np.log10(members["ratio_max_min"]) - math.log10(Y_LIM[0])) / decades * height_mm
            offsets = beeswarm(y_mm.to_numpy(), DIAMETER_MM, MAX_HALF_WIDTH_MM) / GROUP_WIDTH_MM
            frame.loc[members.index, "x_plot"] = group + offsets
        for _, point in frame.iterrows():
            colour = style.CLASS_COLORS[point["class"]]
            if point["state"] == "kinisi_fit_failed":
                ax.plot(point["x_plot"], point["ratio_max_min"], marker="x", markersize=MARKER_SIZE + 0.4,
                        color=colour, markeredgewidth=0.7, linestyle="none", zorder=4)
            else:
                filled = point["state"] == "resolved"
                ax.plot(point["x_plot"], point["ratio_max_min"], marker="o", markersize=MARKER_SIZE,
                        markerfacecolor=colour if filled else "white", markeredgecolor=colour,
                        markeredgewidth=0.5, linestyle="none", zorder=3)
        ax.axhline(REPLICATE_MAX_RATIO, color="0.2", lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(len(GROUPS) - 0.5, REPLICATE_MAX_RATIO * 1.15, f"{REPLICATE_MAX_RATIO:g}$\\times$",
                fontsize=style.SMALL, ha="right", va="bottom", color="0.2")
        counts = frame["group_index"].value_counts()
        for group in range(len(GROUPS)):
            ax.text(group, Y_LIM[1] * 0.55, f"n = {counts.get(group, 0)}", fontsize=style.SMALL, ha="center",
                    va="top", color="0.35")
        ax.set_yscale("log")
        ax.set_ylim(*Y_LIM)
        style.log_ticks(ax, "y")
        ax.set_xlim(-0.5, len(GROUPS) - 0.5)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([SHORT_LABELS[g] for g in GROUPS], fontsize=style.SMALL)
        for tick, group in zip(ax.get_xticklabels(), GROUPS):
            tick.set_color(style.CLASS_COLORS[group])
        ax.tick_params(axis="x", length=0, pad=2.0)
        ax.set_ylabel(ylabel)
        style.panel_label(ax, letter, x=-0.05)
        columns = ["record_id", "formula", "p1_rank", "class", "temperature_K", "md_seed", "state",
                   "n_settings", "settings", "D_max_cm2_s", "D_min_cm2_s", "ratio_max_min", "x_plot"]
        style.save_source(frame[columns], FIGURE, letter)

    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=MARKER_SIZE + 0.4, markerfacecolor="0.3",
                      markeredgecolor="0.3", label="Diffusion resolved"),
               Line2D([0], [0], marker="o", linestyle="none", markersize=MARKER_SIZE + 0.4, markerfacecolor="white",
                      markeredgecolor="0.3", markeredgewidth=0.6, label="Not resolved"),
               Line2D([0], [0], marker="x", linestyle="none", markersize=MARKER_SIZE + 0.6, color="0.3",
                      markeredgewidth=0.7, label="Failed kinisi fit (A3.2)"),
               Line2D([0], [0], color="0.2", lw=0.6, ls=(0, (3, 2)), label="Replicate limit (2$\\times$)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=style.SMALL,
               handletextpad=0.3, columnspacing=1.6, borderaxespad=0.2)
    return style.save(fig, NAME, "extended")
