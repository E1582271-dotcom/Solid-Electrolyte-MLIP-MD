"""Extended Data Fig. 2 - atlas of the 62 compositions (Arrhenius evidence per composition).

Small multiples in Project 1 rank order (7 columns x 9 rows). Each panel plots log D
against 1000/T for every record the classifier used (census, replicate re-analysis and
definitive runs, all seeds):

* filled circle - eligible run, kinisi D;
* open circle - diffusion resolved but framework not intact, kinisi D (not used);
* filled downward caret - diffusion not resolved, framework intact: the protocol upper
  bound D <= MSD_Li(100 ps)/(6 x 100 ps), drawn with its base at the bound;
* open downward caret - the same bound for a run whose framework is not intact (unused);
* cross - failed kinisi fit (amendment A3.2), drawn at the direct-MSD D of the trajectory.

Marker colour and the corner square give the final class; the text gives the formula and
sigma(300 K) in mS/cm ("<=" for prefactor/pair upper bounds). Axes are shared.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from .. import data
from . import style
from .ed01_msd import formula_tex

NAME = "ed02_atlas"
FIGURE = "ed02"
N_COLS, N_ROWS = 7, 9
X_LIM = (0.88, 2.62)
Y_LIM = (1e-8, 1e-3)
CARET_DOWN_BASE = 11  # matplotlib CARETDOWNBASE: base at the value, tip below

CATEGORIES = {
    # name: (marker, filled, size, y column)
    "eligible": ("o", True, 2.6, "D_cm2_s"),
    "framework_not_intact": ("o", False, 2.6, "D_cm2_s"),
    "bound_intact": (CARET_DOWN_BASE, True, 3.2, "D_bound_cm2_s"),
    "bound_not_intact": (CARET_DOWN_BASE, False, 3.2, "D_bound_cm2_s"),
    "kinisi_fit_failed": ("x", True, 2.8, "D_direct_cm2_s"),
}
CATEGORY_LABELS = {
    "eligible": "Eligible run (kinisi D)",
    "framework_not_intact": "Resolved, framework not intact",
    "bound_intact": "Not resolved: MSD bound (intact)",
    "bound_not_intact": "MSD bound, framework not intact",
    "kinisi_fit_failed": "Failed kinisi fit (A3.2), direct-MSD D",
}



def sigma_text(value: float, kind: str) -> str:
    """sigma(300 K) in mS/cm with two significant figures; '<=' prefix for bounds."""
    prefix = r"$\leq$" if kind == "upper_bound" else "="
    if 0.01 <= value < 100:
        decimals = max(0, 1 - int(math.floor(math.log10(value))))
        number = f"{value:.{decimals}f}"
    elif value >= 100:
        number = f"{value:.0f}"
    else:
        exponent = int(math.floor(math.log10(value)))
        mantissa = value / 10 ** exponent
        if round(mantissa, 1) >= 10:
            mantissa, exponent = mantissa / 10, exponent + 1
        number = f"{mantissa:.1f}" + r"$\times$10$^{" + str(exponent) + "}$"
    return r"$\sigma_{300}$ " + prefix + " " + number


def categorise(frame: pd.DataFrame) -> pd.Series:
    resolved = frame["diffusion_resolved"] & ~frame["kinisi_fit_failed"]
    return pd.Series(np.select(
        [frame["eligible"], frame["kinisi_fit_failed"], resolved & ~frame["framework_intact"],
         ~frame["diffusion_resolved"] & frame["framework_intact"],
         ~frame["diffusion_resolved"] & ~frame["framework_intact"]],
        ["eligible", "kinisi_fit_failed", "framework_not_intact", "bound_intact", "bound_not_intact"],
        "other"), index=frame.index)


def make() -> list[Path]:
    records = data.classifier_runs().copy()
    records["category"] = categorise(records)
    if (records["category"] == "other").any():
        raise ValueError("uncategorised classifier record")
    records["inv_T_1000_per_K"] = 1000.0 / records["temperature_K"]
    records["D_plotted_cm2_s"] = [row[CATEGORIES[row["category"]][3]] for _, row in records.iterrows()]
    records["D_plotted_column"] = [CATEGORIES[c][3] for c in records["category"]]
    classes = data.classification().sort_values("p1_rank").reset_index(drop=True)

    width, height = style.DOUBLE, 238 * style.MM
    fig = style.figure(width, height)
    left, right, bottom, top = 0.07, 0.99, 0.045, 0.948
    grid = fig.add_gridspec(N_ROWS, N_COLS, left=left, right=right, bottom=bottom, top=top,
                            wspace=0.10, hspace=0.12)
    labels = []
    for index, composition in classes.iterrows():
        row, col = divmod(index, N_COLS)
        ax = fig.add_subplot(grid[row, col])
        colour = style.CLASS_COLORS[composition["class"]]
        mine = records[records["candidate_key"] == composition["candidate_key"]]
        for category, (marker, filled, size, _) in CATEGORIES.items():
            points = mine[mine["category"] == category]
            if points.empty:
                continue
            kwargs = dict(marker=marker, linestyle="none", markersize=size, zorder=3, clip_on=False)
            if marker == "x":
                kwargs.update(color=colour, markeredgewidth=0.7)
            elif filled:
                kwargs.update(markerfacecolor=colour, markeredgecolor=colour, markeredgewidth=0.4)
            else:
                kwargs.update(markerfacecolor="white", markeredgecolor=colour, markeredgewidth=0.6)
            ax.plot(points["inv_T_1000_per_K"], points["D_plotted_cm2_s"], **kwargs)
        ax.set_yscale("log")
        ax.set_xlim(*X_LIM)
        ax.set_ylim(*Y_LIM)
        ax.set_xticks([1.0, 1.5, 2.0, 2.5])
        ax.set_yticks([1e-8, 1e-6, 1e-4])
        ax.yaxis.set_minor_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=style.SMALL, pad=1.5, length=2.0)
        ax.tick_params(which="minor", length=1.0)
        if col != 0:
            ax.tick_params(labelleft=False)
        if row != N_ROWS - 1:
            ax.tick_params(labelbottom=False)
        ax.plot([0.955], [0.90], marker="s", markersize=3.0, color=colour, transform=ax.transAxes,
                clip_on=False, markeredgewidth=0)
        ax.text(0.90, 0.90, formula_tex(composition["formula"]), transform=ax.transAxes,
                fontsize=style.SMALL, ha="right", va="center")
        text = sigma_text(composition["sigma300_mS_cm"], composition["sigma_kind"])
        ax.text(0.98, 0.76, text, transform=ax.transAxes, fontsize=style.SMALL, ha="right", va="center",
                color="0.25")
        ax.text(0.98, 0.04, f"{composition['p1_rank']}", transform=ax.transAxes, fontsize=style.SMALL,
                ha="right", va="bottom", color=style.GREY)
        labels.append({"p1_rank": composition["p1_rank"], "formula": composition["formula"],
                       "class": composition["class"], "sigma300_mS_cm": composition["sigma300_mS_cm"],
                       "sigma_kind": composition["sigma_kind"], "panel_row": row + 1, "panel_col": col + 1})

    fig.text(0.5 * (left + right), 0.008, "1000/T (K$^{-1}$)", ha="center", va="bottom", fontsize=style.FONT_SIZE)
    fig.text(0.012, 0.5 * (bottom + top), "D (cm$^2$ s$^{-1}$)", ha="left", va="center", rotation=90,
             fontsize=style.FONT_SIZE)

    marker_handles = []
    for category, (marker, filled, size, _) in CATEGORIES.items():
        legend_marker = "v" if marker == CARET_DOWN_BASE else marker  # centred triangle in the key
        kwargs = dict(marker=legend_marker, linestyle="none", markersize=size + 0.4,
                      label=CATEGORY_LABELS[category])
        if marker == "x":
            kwargs.update(color="0.2", markeredgewidth=0.7)
        elif filled:
            kwargs.update(markerfacecolor="0.2", markeredgecolor="0.2")
        else:
            kwargs.update(markerfacecolor="white", markeredgecolor="0.2", markeredgewidth=0.6)
        marker_handles.append(Line2D([0], [0], **kwargs))
    class_handles = style.legend_handles([(style.CLASS_LABELS[name], style.CLASS_COLORS[name])
                                          for name in style.CLASS_ORDER[:4]], size=3.5)
    fig.legend(handles=marker_handles, loc="upper left", bbox_to_anchor=(left, 0.995), ncol=3,
               fontsize=style.SMALL, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.0)
    fig.legend(handles=class_handles, loc="upper right", bbox_to_anchor=(right, 0.995), ncol=2,
               fontsize=style.SMALL, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0,
               title=r"Final class; $\sigma_{300}$ in mS cm$^{-1}$, grey number = P1 rank",
               title_fontsize=style.SMALL)

    source = records[["p1_rank", "formula", "class", "record_id", "stage", "md_seed", "temperature_K",
                      "inv_T_1000_per_K", "category", "D_plotted_column", "D_plotted_cm2_s"]]
    source = source.sort_values(["p1_rank", "temperature_K", "md_seed", "record_id"]).reset_index(drop=True)
    style.save_source(source, FIGURE, "a")
    style.save_source(pd.DataFrame(labels), FIGURE, "a_labels")
    return style.save(fig, NAME, "extended")
