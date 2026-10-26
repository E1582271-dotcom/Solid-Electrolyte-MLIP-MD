"""Extended Data Fig. 6 - temperature control and energy drift of the NVT trajectories.

a) Relative deviation of the mean temperature from the target, (T_mean - T)/T in %, for
   every analysis record (338), against the target temperature; dashed lines mark the
   +/-5 % eligibility limit. Colour = stage.
b) Total-energy drift of the seed-0 census trajectories (linear slope of the energy,
   meV per atom per ps) against the target temperature; filled = framework intact, open
   = framework not intact. The energy slope is stored for census records only.

Temperatures are categorical; points are spread with a deterministic beeswarm.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import data
from ..protocol import T_BIAS_MAX
from . import style
from .ed04_windows import beeswarm

NAME = "ed06_thermostat"
FIGURE = "ed06"
STAGES = ("census", "replicate", "definitive")
STAGE_LABELS = {"census": "Census (seed 0)", "replicate": "Replicate re-analysis", "definitive": "Definitive batches"}
INTACT_COLOUR = "0.15"
MOTION_COLOUR = style.OKABE_ITO["vermillion"]
MARKER = 2.3  # points
DIAMETER_MM = 0.95
FIG_HEIGHT_MM = 68.0


def _swarm_x(frame: pd.DataFrame, value: str, categories, y_lim, axis_height_mm, slot_mm, half_mm):
    """Category index plus a deterministic beeswarm offset for every row of ``frame``."""
    x = np.zeros(len(frame))
    span = y_lim[1] - y_lim[0]
    for index, category in enumerate(categories):
        mask = (frame["temperature_K"] == category).to_numpy()
        if not mask.any():
            continue
        y_mm = (frame.loc[mask, value].to_numpy() - y_lim[0]) / span * axis_height_mm
        x[mask] = index + beeswarm(y_mm, DIAMETER_MM, half_mm) / slot_mm
    return x


def make() -> list[Path]:
    runs = data.runs()
    fig = style.figure(style.DOUBLE, FIG_HEIGHT_MM * style.MM)
    bottom, top = 0.15, 0.93
    axis_height_mm = (top - bottom) * FIG_HEIGHT_MM
    width_mm = 183.0

    # a) temperature bias of every record
    ax = fig.add_axes([0.065, bottom, 0.565, top - bottom])
    temperatures = sorted(runs["temperature_K"].unique())
    frame = runs[["record_id", "stage", "formula", "p1_rank", "temperature_K", "md_seed", "temperature_mean_K",
                  "temperature_bias_fraction"]].copy()
    frame["temperature_bias_percent"] = 100.0 * frame["temperature_bias_fraction"]
    frame = frame.sort_values(["temperature_K", "temperature_bias_percent", "record_id"]).reset_index(drop=True)
    y_lim = (-6.5, 6.5)
    slot_mm = 0.565 * width_mm / len(temperatures)
    frame["x_plot"] = _swarm_x(frame, "temperature_bias_percent", temperatures, y_lim, axis_height_mm, slot_mm,
                               0.45 * slot_mm)
    for stage in STAGES:
        points = frame[frame["stage"] == stage]
        ax.plot(points["x_plot"], points["temperature_bias_percent"], marker="o", linestyle="none",
                markersize=MARKER, markerfacecolor=style.STAGE_COLORS[stage], markeredgecolor="white",
                markeredgewidth=0.2, zorder=3)
    limit = 100.0 * T_BIAS_MAX
    for sign in (-1, 1):
        ax.axhline(sign * limit, color="0.2", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0.0, color=style.LIGHT_GREY, lw=0.5, zorder=0)
    ax.text(len(temperatures) - 0.55, limit + 0.25, f"+{limit:g} % limit", fontsize=style.SMALL, ha="right",
            va="bottom", color="0.2")
    ax.text(len(temperatures) - 0.55, -limit - 0.25, f"-{limit:g} % limit", fontsize=style.SMALL, ha="right",
            va="top", color="0.2")
    ax.set_xlim(-0.6, len(temperatures) - 0.4)
    ax.set_ylim(*y_lim)
    ax.set_xticks(range(len(temperatures)))
    ax.set_xticklabels([str(t) for t in temperatures])
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_xlabel("Target temperature (K)")
    ax.set_ylabel("(T$_{mean}$ $-$ T)/T (%)")
    style.panel_label(ax, "a", x=-0.035)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=MARKER + 0.6,
                      markerfacecolor=style.STAGE_COLORS[s], markeredgecolor="white", markeredgewidth=0.2,
                      label=f"{STAGE_LABELS[s]} ({int((frame['stage'] == s).sum())})") for s in STAGES]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=style.SMALL,
              handletextpad=0.2, columnspacing=1.2, borderaxespad=0.2)
    style.save_source(frame[["record_id", "stage", "formula", "p1_rank", "temperature_K", "md_seed",
                             "temperature_mean_K", "temperature_bias_percent", "x_plot"]], FIGURE, "a")

    # b) census energy drift
    bx = fig.add_axes([0.715, bottom, 0.275, top - bottom])
    census = runs[runs["stage"] == "census"][["record_id", "formula", "p1_rank", "temperature_K",
                                              "energy_slope_meV_atom_ps", "framework_intact"]].copy()
    census = census.sort_values(["temperature_K", "energy_slope_meV_atom_ps", "record_id"]).reset_index(drop=True)
    census_temperatures = sorted(census["temperature_K"].unique())
    top_value = census["energy_slope_meV_atom_ps"].max()
    y_lim_b = (-0.03, math.ceil(top_value * 1.08 / 0.05) * 0.05)
    slot_b = 0.275 * width_mm / len(census_temperatures)
    census["x_plot"] = _swarm_x(census, "energy_slope_meV_atom_ps", census_temperatures, y_lim_b,
                                axis_height_mm, slot_b, 0.42 * slot_b)
    for intact in (True, False):
        points = census[census["framework_intact"] == intact]
        colour = INTACT_COLOUR if intact else MOTION_COLOUR
        bx.plot(points["x_plot"], points["energy_slope_meV_atom_ps"], marker="o", linestyle="none",
                markersize=MARKER + 0.2, markerfacecolor=colour if intact else "white", markeredgecolor=colour,
                markeredgewidth=0.45 if not intact else 0.2, zorder=3 if intact else 4)
    bx.axhline(0.0, color=style.LIGHT_GREY, lw=0.5, zorder=0)
    bx.set_xlim(-0.6, len(census_temperatures) - 0.4)
    bx.set_ylim(*y_lim_b)
    bx.set_xticks(range(len(census_temperatures)))
    bx.set_xticklabels([str(t) for t in census_temperatures])
    bx.set_xlabel("Target temperature (K)")
    bx.set_ylabel("Energy drift (meV atom$^{-1}$ ps$^{-1}$)")
    style.panel_label(bx, "b", x=-0.06)
    handles_b = [Line2D([0], [0], marker="o", linestyle="none", markersize=MARKER + 0.6,
                        markerfacecolor=INTACT_COLOUR, markeredgecolor=INTACT_COLOUR, label="Framework intact"),
                 Line2D([0], [0], marker="o", linestyle="none", markersize=MARKER + 0.6, markerfacecolor="white",
                        markeredgecolor=MOTION_COLOUR, markeredgewidth=0.5, label="Framework not intact")]
    bx.legend(handles=handles_b, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=style.SMALL,
              handletextpad=0.2, borderaxespad=0.2)
    style.save_source(census[["record_id", "formula", "p1_rank", "temperature_K", "framework_intact",
                              "energy_slope_meV_atom_ps", "x_plot"]], FIGURE, "b")
    return style.save(fig, NAME, "extended")
