"""Extended Data Fig. 3 - the seed-0 census as evidence: per-temperature flags against the final class.

One row per composition (Project 1 rank order, two blocks of 31), one glyph per census
temperature (600, 800, 1000 K; seed 0, 200 ps, kinisi): filled = Li diffusion resolved,
open = not resolved; black = framework intact, vermillion = framework motion flag
(framework MSD at 100 ps > 1 A^2 or > 10 % of non-Li atoms displaced > 2 A at 200 ps).
The two text columns give the census screen label (where one failing temperature vetoed
the whole composition) and the final class of the pre-registered protocol, which uses
per-temperature eligibility instead. Only census records are used for the glyphs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import data
from . import style
from .ed01_msd import formula_tex

NAME = "ed03_census_flags"
FIGURE = "ed03"
TEMPERATURES = (600, 800, 1000)
ROWS_PER_BLOCK = 31
INTACT_COLOUR = "0.15"
MOTION_COLOUR = style.OKABE_ITO["vermillion"]
# census screen labels use colours outside the class palette (open square = not reliable)
SCREEN_COLOURS = {"conditional": style.OKABE_ITO["sky"], "review": style.OKABE_ITO["orange"],
                  "not_reliable": "white"}
SCREEN_EDGE = "0.25"
SCREEN_LABELS = {"conditional": "Conditional", "review": "Review", "not_reliable": "Not reliable"}
# column x positions (mm from the block's left edge)
X_RANK, X_FORMULA, X_GLYPH0, GLYPH_STEP, X_SCREEN, X_CLASS, BLOCK_WIDTH = 5.0, 6.4, 25.5, 6.8, 44.0, 62.0, 89.0



def _table() -> pd.DataFrame:
    census = data.runs()
    census = census[census["stage"] == "census"]
    comps = data.compositions().sort_values("p1_rank")
    classes = data.classification().set_index("candidate_key")
    rows = []
    for _, comp in comps.iterrows():
        row = {"p1_rank": comp["p1_rank"], "formula": comp["formula"]}
        mine = census[census["candidate_key"] == comp["candidate_key"]].set_index("temperature_K")
        for temperature in TEMPERATURES:
            record = mine.loc[temperature]
            row[f"diffusion_resolved_{temperature}K"] = bool(record["diffusion_resolved"])
            row[f"framework_intact_{temperature}K"] = bool(record["framework_intact"])
        row["census_screen_label"] = comp["census_screen_label"]
        row["census_category"] = comp["census_category"]
        row["final_class"] = classes.loc[comp["candidate_key"], "class"]
        rows.append(row)
    return pd.DataFrame(rows)


def _glyph(ax, x, y, resolved: bool, intact: bool) -> None:
    colour = INTACT_COLOUR if intact else MOTION_COLOUR
    ax.plot([x], [y], marker="o", markersize=3.6, linestyle="none", markeredgewidth=0.7,
            markeredgecolor=colour, markerfacecolor=colour if resolved else "white", zorder=3)


def make() -> list[Path]:
    table = _table()
    style.save_source(table, FIGURE, "a")

    fig = style.figure(style.DOUBLE, 124 * style.MM)
    block_axes = []
    for block in range(2):
        ax = fig.add_axes([0.005 + block * 0.5, 0.105, 0.49, 0.88])
        ax.set_xlim(0, BLOCK_WIDTH)
        ax.set_ylim(ROWS_PER_BLOCK + 0.6, -1.4)
        ax.axis("off")
        block_axes.append(ax)
        header_y = -0.9
        for label, x, ha in (("Rank", X_RANK, "right"), ("Composition", X_FORMULA, "left"),
                             ("Census screen", X_SCREEN, "left"), ("Final class", X_CLASS, "left")):
            ax.text(x, header_y, label, fontsize=style.SMALL, ha=ha, va="center", fontweight="bold")
        for index, temperature in enumerate(TEMPERATURES):
            ax.text(X_GLYPH0 + index * GLYPH_STEP, header_y, f"{temperature} K", fontsize=style.SMALL,
                    ha="center", va="center", fontweight="bold")
        ax.plot([0.5, BLOCK_WIDTH - 0.5], [-0.35, -0.35], color="0.2", lw=0.5)
        rows = table.iloc[block * ROWS_PER_BLOCK:(block + 1) * ROWS_PER_BLOCK].reset_index(drop=True)
        for i, row in rows.iterrows():
            y = i + 0.5
            if i % 2 == 1:
                ax.add_patch(Rectangle((0.5, i), BLOCK_WIDTH - 1.0, 1.0, facecolor="#F2F2F2", edgecolor="none",
                                       zorder=0))
            ax.text(X_RANK, y, str(row["p1_rank"]), fontsize=style.SMALL, ha="right", va="center", color="0.35")
            ax.text(X_FORMULA, y, formula_tex(row["formula"]), fontsize=style.SMALL, ha="left", va="center")
            for index, temperature in enumerate(TEMPERATURES):
                _glyph(ax, X_GLYPH0 + index * GLYPH_STEP, y, row[f"diffusion_resolved_{temperature}K"],
                       row[f"framework_intact_{temperature}K"])
            screen = row["census_screen_label"]
            ax.plot([X_SCREEN + 0.8], [y], marker="s", markersize=2.8, markerfacecolor=SCREEN_COLOURS[screen],
                    markeredgecolor=SCREEN_EDGE if screen == "not_reliable" else SCREEN_COLOURS[screen],
                    markeredgewidth=0.5)
            ax.text(X_SCREEN + 2.2, y, SCREEN_LABELS[screen], fontsize=style.SMALL, ha="left", va="center")
            final = row["final_class"]
            ax.plot([X_CLASS + 0.8], [y], marker="s", markersize=3.0, color=style.CLASS_COLORS[final],
                    markeredgewidth=0)
            ax.text(X_CLASS + 2.2, y, style.CLASS_LABELS[final], fontsize=style.SMALL, ha="left", va="center")
        ax.plot([0.5, BLOCK_WIDTH - 0.5], [len(rows), len(rows)], color="0.2", lw=0.5)

    glyph_handles = []
    for label, resolved, intact in (("Diffusion resolved, framework intact", True, True),
                                    ("Not resolved, framework intact", False, True),
                                    ("Diffusion resolved, framework motion flag", True, False),
                                    ("Not resolved, framework motion flag", False, False)):
        colour = INTACT_COLOUR if intact else MOTION_COLOUR
        glyph_handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=3.6, markeredgewidth=0.7,
                                    markeredgecolor=colour, markerfacecolor=colour if resolved else "white",
                                    label=label))
    fig.legend(handles=glyph_handles, loc="upper left", bbox_to_anchor=(0.03, 0.093), ncol=2,
               fontsize=style.SMALL, handletextpad=0.3, columnspacing=1.5, borderaxespad=0.0,
               title="Census run (seed 0, 200 ps)", title_fontsize=style.SMALL, alignment="left")
    counts = table["census_screen_label"].value_counts()
    screen_handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=2.8, markeredgewidth=0.5,
                             markerfacecolor=SCREEN_COLOURS[k],
                             markeredgecolor=SCREEN_EDGE if k == "not_reliable" else SCREEN_COLOURS[k],
                             label=f"{SCREEN_LABELS[k]} ({counts.get(k, 0)})") for k in SCREEN_COLOURS]
    fig.legend(handles=screen_handles, loc="upper left", bbox_to_anchor=(0.52, 0.093), ncol=1,
               fontsize=style.SMALL, handletextpad=0.3, borderaxespad=0.0,
               title="Census screen label", title_fontsize=style.SMALL, alignment="left")
    class_counts = table["final_class"].value_counts()
    class_handles = style.legend_handles([(f"{style.CLASS_LABELS[k]} ({class_counts.get(k, 0)})",
                                           style.CLASS_COLORS[k]) for k in style.CLASS_ORDER[:4]], size=3.0)
    fig.legend(handles=class_handles, loc="upper left", bbox_to_anchor=(0.70, 0.093), ncol=2,
               fontsize=style.SMALL, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0,
               title="Final class", title_fontsize=style.SMALL, alignment="left")
    return style.save(fig, NAME, "extended")
