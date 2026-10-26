"""Extended Data Fig. 7 - per-seed Arrhenius data of the six census leads.

Each panel shows one lead (Project 1 rank order): D against 1000/T from its replicate
records (stage ``replicate``: MD seeds 0, 1 and 2 re-analysed with the fixed-covariance
estimator over 60-100 ps lags) and, for Li5SbS3I2 and Li3BiS3, the definitive seeds 3
and 4 (kinisi, 800 ps at 600 K). Marker and colour = MD seed; filled = eligible run, open
= not eligible; bars = the stored D interval. Lines are per-seed ordinary least-squares
fits of ln D on 1/T through all plotted points of that seed (their slopes equal the
``slope_K`` of ``data.seed_fits()`` for seeds 0-2). Text: the six-lead replicate decision
(``data.lead_decisions()``) and the final class. For Li4Re6S11 seed 2 was excluded from the
decision (seeds used 0 and 1) and is drawn in grey.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import data
from . import style
from .ed01_msd import formula_tex

NAME = "ed07_six_leads"
FIGURE = "ed07"
SEED_COLOURS = {0: style.OKABE_ITO["black"], 1: style.OKABE_ITO["orange"], 2: style.OKABE_ITO["sky"],
                3: style.OKABE_ITO["vermillion"], 4: style.OKABE_ITO["blue"]}
EXCLUDED_COLOUR = "#B3B3B3"
DECISION_TEXT = {"pass": "pass", "pass_seeds01": "pass on seeds 0 and 1",
                 "pass_low_sigma": "pass, low $\\sigma_{300}$", "method_pending": "method pending"}
X_LIM = (0.95, 1.72)
Y_LIM = (2e-7, 2e-4)



def lead_records() -> pd.DataFrame:
    """Replicate records of the six leads plus definitive seeds 3/4 of the same compositions."""
    runs = data.runs()
    leads = data.lead_decisions()
    keys = set(leads["candidate_key"])
    replicate = runs[(runs["stage"] == "replicate") & runs["candidate_key"].isin(keys)]
    definitive = runs[(runs["stage"] == "definitive") & runs["candidate_key"].isin(keys)
                      & runs["md_seed"].isin([3, 4])]
    frame = pd.concat([replicate, definitive], ignore_index=True)
    return frame.sort_values(["p1_rank", "md_seed", "temperature_K"]).reset_index(drop=True)


def seed_lines(frame: pd.DataFrame) -> pd.DataFrame:
    """OLS of ln D on 1/T for every (composition, seed); endpoints at the plotted temperature range."""
    rows = []
    for (key, seed), group in frame.groupby(["candidate_key", "md_seed"]):
        x = 1.0 / group["temperature_K"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, np.log(group["D_cm2_s"].to_numpy()), 1)
        for temperature in (group["temperature_K"].min(), group["temperature_K"].max()):
            rows.append({"candidate_key": key, "formula": group["formula"].iloc[0], "md_seed": seed,
                         "slope_lnD_per_inv_K": slope, "intercept_lnD_cm2_s": intercept,
                         "temperature_K": temperature, "inv_T_1000_per_K": 1000.0 / temperature,
                         "D_fit_cm2_s": float(np.exp(intercept + slope / temperature))})
    return pd.DataFrame(rows)


def make() -> list[Path]:
    records = lead_records()
    records["inv_T_1000_per_K"] = 1000.0 / records["temperature_K"]
    lines = seed_lines(records)
    leads = data.lead_decisions().sort_values("p1_rank").reset_index(drop=True)
    classes = data.classification().set_index("candidate_key")

    fig = style.figure(style.DOUBLE, 112 * style.MM)
    grid = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.155, top=0.955, wspace=0.12, hspace=0.16)
    letters = "abcdef"
    for index, lead in leads.iterrows():
        ax = fig.add_subplot(grid[index // 3, index % 3])
        key = lead["candidate_key"]
        used = {int(s) for s in str(lead["seeds_used"]).split(";")}
        mine = records[records["candidate_key"] == key]
        for seed, group in mine.groupby("md_seed"):
            excluded = seed <= 2 and seed not in used
            colour = EXCLUDED_COLOUR if excluded else SEED_COLOURS[seed]
            fit = lines[(lines["candidate_key"] == key) & (lines["md_seed"] == seed)]
            ax.plot(fit["inv_T_1000_per_K"], fit["D_fit_cm2_s"], color=colour, lw=0.7,
                    ls=(0, (2.5, 1.5)) if excluded else "-", zorder=2)
            offset = (seed - 2) * 0.006  # deterministic horizontal offset per seed
            x = group["inv_T_1000_per_K"] + offset
            ax.errorbar(x, group["D_cm2_s"],
                        yerr=[group["D_cm2_s"] - group["D_lo_cm2_s"], group["D_hi_cm2_s"] - group["D_cm2_s"]],
                        fmt="none", ecolor=colour, elinewidth=0.5, capsize=0, zorder=3)
            for eligible, points in group.assign(x=x).groupby("eligible"):
                ax.plot(points["x"], points["D_cm2_s"], marker=style.SEED_MARKERS[seed], linestyle="none",
                        markersize=3.0, markerfacecolor=colour if eligible else "white", markeredgecolor=colour,
                        markeredgewidth=0.6, zorder=4)
            records.loc[group.index, "x_plot"] = x
            records.loc[group.index, "excluded_from_decision"] = excluded
        ax.set_yscale("log")
        ax.set_xlim(*X_LIM)
        ax.set_ylim(*Y_LIM)
        style.log_ticks(ax, "y")
        ax.set_xticks([1.0, 1.2, 1.4, 1.6])
        if index // 3 == 1:
            ax.set_xlabel("1000/T (K$^{-1}$)")
        else:
            ax.tick_params(labelbottom=False)
        if index % 3 == 0:
            ax.set_ylabel("D (cm$^2$ s$^{-1}$)")
        else:
            ax.tick_params(labelleft=False)
        final = classes.loc[key, "class"]
        ax.text(0.97, 0.96, formula_tex(lead["formula"]), transform=ax.transAxes, fontsize=style.FONT_SIZE,
                ha="right", va="top")
        ax.text(0.97, 0.865, f"Six-lead decision: {DECISION_TEXT[lead['decision']]}", transform=ax.transAxes,
                fontsize=style.SMALL, ha="right", va="top", color="0.25")
        ax.text(0.97, 0.79, f"Final: {style.CLASS_LABELS[final]}", transform=ax.transAxes, fontsize=style.SMALL,
                ha="right", va="top", color=style.CLASS_COLORS[final])
        excluded_seeds = sorted({0, 1, 2} - used)
        if excluded_seeds:
            ax.text(0.97, 0.715, "Seed " + ", ".join(str(s) for s in excluded_seeds)
                    + " excluded (seeds used " + ", ".join(str(s) for s in sorted(used)) + ")",
                    transform=ax.transAxes, fontsize=style.SMALL, ha="right", va="top", color="0.45")
        style.panel_label(ax, letters[index], x=-0.03)

    handles = [Line2D([0], [0], marker=style.SEED_MARKERS[s], color=SEED_COLOURS[s], lw=0.7, markersize=3.0,
                      markerfacecolor=SEED_COLOURS[s], markeredgewidth=0.6,
                      label=f"Seed {s}" + (" (fixed-cov)" if s <= 2 else " (kinisi)")) for s in SEED_COLOURS]
    handles += [Line2D([0], [0], marker="o", color=EXCLUDED_COLOUR, lw=0.7, ls=(0, (2.5, 1.5)), markersize=3.0,
                       markerfacecolor=EXCLUDED_COLOUR, markeredgewidth=0.6, label="Seed excluded from decision"),
                Line2D([0], [0], marker="o", linestyle="none", markersize=3.0, markerfacecolor="white",
                       markeredgecolor="0.3", markeredgewidth=0.6, label="Not eligible run")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=4, fontsize=style.SMALL,
               handletextpad=0.3, columnspacing=1.4, borderaxespad=0.3, handlelength=2.0)

    point_columns = ["formula", "p1_rank", "record_id", "stage", "estimator", "md_seed", "temperature_K",
                     "inv_T_1000_per_K", "x_plot", "D_cm2_s", "D_lo_cm2_s", "D_hi_cm2_s", "eligible",
                     "excluded_from_decision"]
    style.save_source(records[point_columns], FIGURE, "points")
    style.save_source(lines, FIGURE, "fits")
    decisions = leads[["p1_rank", "formula", "decision", "seeds_used", "seed_excluded"]].copy()
    decisions["final_class"] = [classes.loc[k, "class"] for k in leads["candidate_key"]]
    style.save_source(decisions, FIGURE, "annotations")
    return style.save(fig, NAME, "extended")
