"""Extended Data Fig. 8 - Project 1 screening score against the final MD conductivity.

x: the Project 1 machine-learning screening score (predicted log10 sigma, S/cm);
y: the final sigma(300 K) of the classification (mS/cm, log). Colour = final class;
marker = Project 1 evidence state. Filled markers are estimates (fits or two-point
estimates); open markers with a downward arrow are upper bounds (prefactor or pair rule).
The dashed line is the fast threshold (0.1 mS/cm). Descriptive only: no regression is
drawn, and both quantities are model predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .. import data
from ..protocol import FAST_THRESHOLD_mS_cm
from . import style

NAME = "ed08_screening"
FIGURE = "ed08"
EVIDENCE_MARKERS = {"high_score_lower_risk": "o", "unflagged": "s", "extrapolative": "D"}
EVIDENCE_LABELS = {"high_score_lower_risk": "High score, lower risk", "unflagged": "Unflagged",
                   "extrapolative": "Extrapolative"}
MARKER_SIZE = {"o": 3.4, "s": 3.0, "D": 2.8}
ARROW_DECADES = 0.55
Y_LIM = (1e-11, 1e3)
X_LIM = (-8.2, -2.0)


def make() -> list[Path]:
    classes = data.classification()[["candidate_key", "class", "sigma300_mS_cm", "sigma_kind"]]
    comps = data.compositions()[["candidate_key", "p1_rank", "formula", "p1_score_log10_S_cm", "p1_evidence_state"]]
    frame = comps.merge(classes, on="candidate_key", how="inner", validate="one_to_one")
    frame = frame.sort_values("p1_rank").reset_index(drop=True)
    frame["is_upper_bound"] = frame["sigma_kind"] == "upper_bound"
    frame["arrow_tip_mS_cm"] = np.where(frame["is_upper_bound"],
                                        frame["sigma300_mS_cm"] / 10 ** ARROW_DECADES, np.nan)

    fig = style.figure(style.ONE_HALF, 80 * style.MM)
    ax = fig.add_axes([0.105, 0.125, 0.60, 0.835])
    ax.axhline(FAST_THRESHOLD_mS_cm, color=style.FAST_THRESHOLD_COLOR, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(X_LIM[1] - 0.08, FAST_THRESHOLD_mS_cm / 1.5, f"Fast threshold, {FAST_THRESHOLD_mS_cm:g} mS cm$^{{-1}}$",
            fontsize=style.SMALL, ha="right", va="top", color=style.FAST_THRESHOLD_COLOR)
    # bounds first (drawn underneath), then estimates; class order fixes the drawing order
    for bound in (True, False):
        for name in style.CLASS_ORDER:
            colour = style.CLASS_COLORS[name]
            for state, marker in EVIDENCE_MARKERS.items():
                points = frame[(frame["class"] == name) & (frame["p1_evidence_state"] == state)
                               & (frame["is_upper_bound"] == bound)]
                if points.empty:
                    continue
                if bound:
                    for _, point in points.iterrows():
                        ax.plot([point["p1_score_log10_S_cm"]] * 2, [point["sigma300_mS_cm"], point["arrow_tip_mS_cm"]],
                                color=colour, lw=0.6, zorder=2)
                    ax.plot(points["p1_score_log10_S_cm"], points["arrow_tip_mS_cm"], marker=7, linestyle="none",
                            markersize=2.6, color=colour, zorder=2)
                ax.plot(points["p1_score_log10_S_cm"], points["sigma300_mS_cm"], marker=marker, linestyle="none",
                        markersize=MARKER_SIZE[marker], markerfacecolor="white" if bound else colour,
                        markeredgecolor=colour, markeredgewidth=0.6, zorder=3 if bound else 4)
    ax.set_yscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    style.log_ticks(ax, "y")
    ax.set_yticks([10.0 ** e for e in range(-10, 4, 2)])
    ax.set_xlabel("Project 1 screening score (log$_{10}$ $\\sigma$ / S cm$^{-1}$)")
    ax.set_ylabel("MD $\\sigma_{300}$ (mS cm$^{-1}$)")

    # colour swatches as short thick lines, so they are not read as an evidence-state marker shape
    class_handles = [Line2D([0], [0], color=style.CLASS_COLORS[name], lw=2.2, solid_capstyle="butt",
                            label=f"{style.CLASS_LABELS[name]} ({int((frame['class'] == name).sum())})")
                     for name in style.CLASS_ORDER[:4]]
    fig.legend(handles=class_handles, loc="upper left", bbox_to_anchor=(0.73, 0.96), fontsize=style.SMALL,
               title="Final class", title_fontsize=style.SMALL, alignment="left", handletextpad=0.5, handlelength=1.0,
               borderaxespad=0.0)
    evidence_handles = [Line2D([0], [0], marker=marker, linestyle="none", markersize=MARKER_SIZE[marker],
                               markerfacecolor="0.35", markeredgecolor="0.35",
                               label=f"{EVIDENCE_LABELS[state]} ({int((frame['p1_evidence_state'] == state).sum())})")
                        for state, marker in EVIDENCE_MARKERS.items()]
    fig.legend(handles=evidence_handles, loc="upper left", bbox_to_anchor=(0.73, 0.70), fontsize=style.SMALL,
               title="Project 1 evidence state", title_fontsize=style.SMALL, alignment="left", handletextpad=0.3,
               borderaxespad=0.0)
    kind_handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.4, markerfacecolor="0.35",
                           markeredgecolor="0.35", label="Estimate"),
                    Line2D([0], [0], marker="o", linestyle="none", markersize=3.4, markerfacecolor="white",
                           markeredgecolor="0.35", markeredgewidth=0.6, label="Upper bound (with arrow)")]
    fig.legend(handles=kind_handles, loc="upper left", bbox_to_anchor=(0.73, 0.47), fontsize=style.SMALL,
               title="$\\sigma_{300}$ kind", title_fontsize=style.SMALL, alignment="left", handletextpad=0.3,
               borderaxespad=0.0)

    source = frame[["p1_rank", "formula", "p1_score_log10_S_cm", "p1_evidence_state", "class", "sigma_kind",
                    "sigma300_mS_cm", "arrow_tip_mS_cm"]]
    style.save_source(source, FIGURE, "a")
    return style.save(fig, NAME, "extended")
