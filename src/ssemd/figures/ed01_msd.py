"""Extended Data Fig. 1 - Li and framework mean-squared displacement against lag time.

One representative composition per final class and one framework-melting example from
the seed-0 census. The representative of a class is the lowest-P1-rank member whose
number of eligible temperatures equals the class mode (3 for the fitted classes, 2 for
SLOW_ESTIMATED, 0 for SLOW_BOUNDED), so it shows the typical evidence behind the class.
The melting example is the census composition labelled ``framework_melting`` with the
largest framework MSD at 100 ps among its 800 K runs (it melts at the lowest census
temperature). At each temperature the longest trajectory with a full MSD curve is drawn
(lowest seed on ties); replicate re-analysis records store only 60-100 ps and are not
used. Li MSD solid, framework (non-Li) MSD dashed, colour = target temperature.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .. import data
from ..protocol import FRAMEWORK_MSD_MAX_A2
from . import style

NAME = "ed01_msd"
FIGURE = "ed01"
T_MIN, T_MAX = 400.0, 1000.0
CLASS_PANELS = ("FAST_VALIDATED", "SLOW_MEASURED", "SLOW_ESTIMATED", "SLOW_BOUNDED")
X_LIM = (0.04, 600.0)
Y_LIM = (0.01, 3000.0)


def formula_tex(formula: str) -> str:
    """Chemical formula with mathtext subscripts, e.g. Li5P(S2Cl)2 -> Li$_{5}$P(S$_{2}$Cl)$_{2}$."""
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def temperature_cmap() -> mcolors.Colormap:
    """Viridis without its palest yellow end, so 1000 K stays visible on white."""
    base = plt.get_cmap(style.TEMPERATURE_CMAP)
    return mcolors.ListedColormap(base(np.linspace(0.0, 0.9, 256)), name="viridis_trunc")


def _representatives() -> list[tuple[str, str, str]]:
    """(candidate_key, formula, role) for the five panels, chosen deterministically."""
    classes = data.classification().sort_values("p1_rank")
    chosen = []
    for name in CLASS_PANELS:
        members = classes[classes["class"] == name]
        mode = members["n_eligible"].mode().min()
        row = members[members["n_eligible"] == mode].iloc[0]
        chosen.append((row["candidate_key"], row["formula"], name))
    comps = data.compositions()
    melting_keys = set(comps.loc[comps["census_category"] == "framework_melting", "candidate_key"])
    runs = data.runs()
    census_800 = runs[(runs["stage"] == "census") & (runs["temperature_K"] == 800)
                      & runs["candidate_key"].isin(melting_keys)]
    census_800 = census_800.sort_values(["framework_msd_100ps_A2", "p1_rank"], ascending=[False, True])
    melt = census_800.iloc[0]
    chosen.append((melt["candidate_key"], melt["formula"], "MELTING"))
    return chosen


def _curve_records(candidate_key: str, census_only: bool) -> pd.DataFrame:
    """Longest full-curve trajectory per temperature (lowest seed on ties)."""
    runs = data.runs()
    stages = ("census",) if census_only else ("census", "definitive")
    pool = runs[(runs["candidate_key"] == candidate_key) & runs["stage"].isin(stages)].copy()
    full_curve = set(pool["record_id"])
    # drop records superseded by another full-curve record (e.g. a longer rerun)
    pool = pool[~pool["superseded_by"].isin(full_curve)]
    pool = pool.sort_values(["temperature_K", "production_ps", "md_seed"], ascending=[True, False, True])
    return pool.groupby("temperature_K", as_index=False).head(1).reset_index(drop=True)


def make() -> list[Path]:
    classes = data.classification().set_index("candidate_key")
    curves = data.msd_curves()
    cmap = temperature_cmap()
    norm = mcolors.Normalize(T_MIN, T_MAX)

    fig = style.figure(style.DOUBLE, 104 * style.MM)
    grid = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.085, top=0.955, wspace=0.28, hspace=0.32)
    letters = "abcde"
    for index, (key, formula, role) in enumerate(_representatives()):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        records = _curve_records(key, census_only=(role == "MELTING"))
        rows = []
        for _, record in records.iterrows():
            curve = curves[curves["record_id"] == record["record_id"]].sort_values("lag_ps")
            colour = cmap(norm(record["temperature_K"]))
            ax.plot(curve["lag_ps"], curve["Li_total_A2"], color=colour, lw=0.9, solid_capstyle="round")
            ax.plot(curve["lag_ps"], curve["framework_A2"], color=colour, lw=0.8, ls=(0, (3, 1.5)))
            frame = curve[["lag_ps", "Li_total_A2", "framework_A2"]].copy()
            for position, column in enumerate(("record_id", "temperature_K", "md_seed", "production_ps",
                                               "eligible")):
                frame.insert(position, column, record[column])
            rows.append(frame)
        source = pd.concat(rows, ignore_index=True)
        source.insert(0, "formula", formula)
        source.insert(1, "class", classes.loc[key, "class"])
        source.insert(2, "panel_role", "census_framework_melting" if role == "MELTING" else "class_representative")
        style.save_source(source, FIGURE, letters[index])

        ax.axhline(FRAMEWORK_MSD_MAX_A2, color=style.GREY, lw=0.5, ls=":", zorder=0)
        # slope-1 guide (diffusive regime) in the lower right corner
        guide_x = np.array([60.0, 500.0])
        ax.plot(guide_x, 0.012 * guide_x / guide_x[0], color=style.GREY, lw=0.6, zorder=0)
        ax.text(55.0, 0.0125, "slope 1", fontsize=style.SMALL, color=style.GREY, ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*X_LIM)
        ax.set_ylim(*Y_LIM)
        style.log_ticks(ax, "x")
        style.log_ticks(ax, "y")
        ax.set_xlabel("Lag time (ps)")
        if index % 3 == 0:
            ax.set_ylabel("MSD (Å$^2$)")
        if role == "MELTING":
            tag, colour = "Census, framework melting", style.OKABE_ITO["vermillion"]
        else:
            tag, colour = style.CLASS_LABELS[role], style.CLASS_COLORS[role]
        ax.text(0.03, 0.97, formula_tex(formula), transform=ax.transAxes, fontsize=style.FONT_SIZE,
                ha="left", va="top")
        ax.text(0.03, 0.885, tag, transform=ax.transAxes, fontsize=style.SMALL, color=colour,
                ha="left", va="top")
        style.panel_label(ax, letters[index], x=-0.045)

    # sixth slot: temperature colour bar and line key
    slot = fig.add_subplot(grid[1, 2])
    slot.axis("off")
    pos = slot.get_position()
    cax = fig.add_axes([pos.x0 + 0.04 * pos.width, pos.y0 + 0.80 * pos.height, 0.80 * pos.width,
                        0.045 * pos.height])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    bar.set_ticks([400, 500, 600, 700, 800, 900, 1000])
    bar.outline.set_linewidth(style.LINE * 0.8)
    bar.ax.tick_params(labelsize=style.FONT_SIZE, length=2.0, width=style.LINE)
    bar.set_label("Target temperature (K)", fontsize=style.FONT_SIZE, labelpad=2.0)
    handles = [Line2D([0], [0], color="0.25", lw=0.9, label="Li"),
               Line2D([0], [0], color="0.25", lw=0.8, ls=(0, (3, 1.5)), label="Framework (non-Li)"),
               Line2D([0], [0], color=style.GREY, lw=0.5, ls=":", label="Framework limit (1 Å$^2$ at 100 ps)"),
               Line2D([0], [0], color=style.GREY, lw=0.6, label="Slope 1 (diffusive)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(pos.x0 + 0.02 * pos.width, pos.y0 + 0.50 * pos.height),
               fontsize=style.FONT_SIZE, handlelength=2.4, borderaxespad=0.0, labelspacing=0.6)
    return style.save(fig, NAME, "extended")
