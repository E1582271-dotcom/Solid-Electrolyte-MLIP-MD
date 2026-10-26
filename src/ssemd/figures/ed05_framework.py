"""Extended Data Fig. 5 - the two framework-integrity criteria for every analysis record.

x: framework (non-Li) MSD at 100 ps lag (log); y: fraction of non-Li atoms displaced by
more than 2 A at 200 ps (symmetric-log axis so that runs with no displaced atom sit on
the 0 line). Dashed lines mark the protocol limits (1 A^2 and 0.10); the shaded box is the
framework-intact region (both criteria met). Colour = target temperature, marker = stage.
All 338 records are drawn; the 18 seed-0 replicate re-analyses re-use census trajectories
and nearly coincide with their census records.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter

from .. import data
from ..protocol import FRAMEWORK_FRACTION_MAX, FRAMEWORK_MSD_MAX_A2
from . import style
from .ed01_msd import temperature_cmap

NAME = "ed05_framework"
FIGURE = "ed05"
T_MIN, T_MAX = 400.0, 1000.0
STAGE_MARKERS = {"census": "o", "replicate": "s", "definitive": "^"}
STAGE_LABELS = {"census": "Census (seed 0)", "replicate": "Replicate re-analysis", "definitive": "Definitive batches"}
LINTHRESH = 1e-3
X_LIM = (0.06, 400.0)
Y_LIM = (-2e-4, 1.6)



def make() -> list[Path]:
    runs = data.runs()
    cmap = temperature_cmap()
    norm = mcolors.Normalize(T_MIN, T_MAX)

    fig = style.figure(style.ONE_HALF, 82 * style.MM)
    ax = fig.add_axes([0.11, 0.13, 0.66, 0.82])
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.35)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.add_patch(Rectangle((X_LIM[0], Y_LIM[0]), FRAMEWORK_MSD_MAX_A2 - X_LIM[0],
                           FRAMEWORK_FRACTION_MAX - Y_LIM[0], facecolor="#E8F4EF", edgecolor="none", zorder=0))
    ax.axvline(FRAMEWORK_MSD_MAX_A2, color="0.2", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(FRAMEWORK_FRACTION_MAX, color="0.2", lw=0.6, ls=(0, (3, 2)), zorder=1)

    order = runs.sort_values(["temperature_K", "stage", "record_id"])
    for stage, marker in STAGE_MARKERS.items():
        points = order[order["stage"] == stage]
        ax.scatter(points["framework_msd_100ps_A2"], points["framework_fraction_gt2A_200ps"], marker=marker,
                   s=9 if marker != "^" else 10, c=cmap(norm(points["temperature_K"].to_numpy(dtype=float))),
                   edgecolors="white", linewidths=0.25, zorder=3, clip_on=False)

    intact = runs["framework_intact"]
    n_intact, n_total = int(intact.sum()), len(runs)
    ax.text(0.075, 0.035, f"Framework intact\n{n_intact} of {n_total} records", fontsize=style.SMALL,
            ha="left", va="bottom", color=style.OKABE_ITO["green"])
    ax.text(FRAMEWORK_MSD_MAX_A2 * 1.08, 1.25, "1 Å$^2$", fontsize=style.SMALL, ha="left", va="center", color="0.2")
    ax.text(X_LIM[1] * 0.9, FRAMEWORK_FRACTION_MAX * 1.12, "0.10", fontsize=style.SMALL, ha="right",
            va="bottom", color="0.2")

    ax.yaxis.set_major_locator(FixedLocator([0.0, 1e-3, 1e-2, 1e-1, 1.0]))
    ax.set_yticklabels(["0", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "1"])
    minor = [m * 10.0 ** e for e in (-3, -2, -1) for m in range(2, 10)]
    ax.yaxis.set_minor_locator(FixedLocator(minor))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style.log_ticks(ax, "x")
    ax.set_xlabel("Framework MSD at 100 ps (Å$^2$)")
    ax.set_ylabel("Fraction of non-Li atoms displaced > 2 Å at 200 ps")

    cax = fig.add_axes([0.80, 0.50, 0.025, 0.42])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.set_ticks([400, 600, 800, 1000])
    bar.outline.set_linewidth(style.LINE * 0.8)
    bar.ax.tick_params(labelsize=style.FONT_SIZE, length=2.0, width=style.LINE)
    bar.set_label("Target temperature (K)", fontsize=style.FONT_SIZE)
    handles = [Line2D([0], [0], marker=marker, linestyle="none", markersize=3.4, markerfacecolor="0.45",
                      markeredgecolor="white", markeredgewidth=0.25,
                      label=f"{STAGE_LABELS[stage]} ({int((runs['stage'] == stage).sum())})")
               for stage, marker in STAGE_MARKERS.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.79, 0.40), fontsize=style.SMALL,
               handletextpad=0.3, borderaxespad=0.0, labelspacing=0.7)

    source = order[["record_id", "stage", "formula", "p1_rank", "temperature_K", "md_seed",
                    "framework_msd_100ps_A2", "framework_fraction_gt2A_200ps", "framework_intact"]]
    style.save_source(source.reset_index(drop=True), FIGURE, "a")
    return style.save(fig, NAME, "extended")
