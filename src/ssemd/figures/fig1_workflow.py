"""Fig. 1: from 21,712 Materials Project entries to one class per composition.

a, the screening and simulation pipeline as a left-to-right step diagram (counts from
``data.workflow()``; temperature, seed and run-length ranges from ``data.runs()``).
b, seed-0 census outcome of each composition against its final class (``data.compositions()``
joined to ``data.classification()``): the census veto is replaced by per-temperature
classification, so several compositions whose framework melted at high temperature end fast.
c, sigma(300 K) of the FAST_VALIDATED compositions with pooled 95 % intervals, marked by the
rule that decided them, against the fast threshold and the Li6PS5Cl benchmark
(``data.li6ps5cl_reference()``).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter

from .. import data
from ..protocol import FAST_THRESHOLD_mS_cm
from . import style

NAME = "fig1_workflow"
WIDTH_MM = 183.0
HEIGHT_MM = 96.0
LETTER_Y_BC = 63.0  # baseline of the panel letters b and c (mm from the bottom)

_DIGITS = re.compile(r"(\d+(?:\.\d+)?)")


def formula_tex(formula: str) -> str:
    """Chemical formula with mathtext subscripts: ``Li5P(S2Cl)2`` -> ``Li$_{5}$P(S$_{2}$Cl)$_{2}$``."""
    return _DIGITS.sub(lambda match: f"$_{{{match.group(1)}}}$", formula)


def _tint(color: str, amount: float) -> tuple[float, float, float]:
    """``color`` mixed with white; ``amount`` is the share of the colour (0 = white)."""
    rgb = np.array(to_rgb(color))
    return tuple(1.0 - amount * (1.0 - rgb))


def _span(values) -> str:
    low, high = min(values), max(values)
    return f"{low:g}" if low == high else f"{low:g}–{high:g}"


# ----------------------------------------------------------------------------------------------
# a: pipeline
# ----------------------------------------------------------------------------------------------

UPSTREAM_TEXT = "#7A7A7A"
UPSTREAM_EDGE = "#BDBDBD"
INK = "#262626"


def _pipeline_steps() -> list[dict]:
    flow = data.workflow().set_index("stage")
    runs = data.runs()
    classes = data.classification()["class"].value_counts()

    census = runs[runs["stage"] == "census"]
    census_ids = set(census["trajectory_id"])
    replicate = runs[(runs["stage"] == "replicate") & ~runs["trajectory_id"].isin(census_ids)]
    definitive = runs[runs["stage"] == "definitive"]

    steps = [
        dict(stage="mp_li_entries", group="p1", note="Li-containing"),
        dict(stage="mp_funnel", group="p1", note="sulfide, stable,\ninsulating"),
        dict(stage="candidate_structures", group="p1", note="P1 candidates", unit_override="structures"),
        dict(stage="queue", group="p1", note="one structure\neach"),
        dict(stage="census", group="census",
             note=f"{_span(census['temperature_K'])} K,\nseed {_span(census['md_seed'])}"),
        dict(stage="census_leads", group="census", note="conditional\nor review"),
        dict(stage="replicates", group="replicate",
             note=f"seeds {_span(replicate['md_seed'])},\nleads only"),
        dict(stage="definitive", group="definitive",
             note=f"{_span(definitive['temperature_K'])} K,\n{_span(definitive['production_ps'])} ps"),
        dict(stage="classified", group="class", note="one class each"),
        dict(stage="fast_conductors", group="class",
             note=f"σ(300 K) ≥ {FAST_THRESHOLD_mS_cm:g}\nmS cm$^{{-1}}$"),
    ]
    for step in steps:
        step["count"] = int(flow.loc[step["stage"], "count"])
        # The 79 Project 1 candidates are MP entries; "structures" names them as in the text.
        step["unit"] = step.pop("unit_override", None) or str(flow.loc[step["stage"], "unit"])
    steps[8]["classes"] = [(name, int(classes.get(name, 0))) for name in style.CLASS_ORDER
                           if classes.get(name, 0) > 0]
    return steps


def _panel_a(fig, top_mm: float) -> None:
    steps = _pipeline_steps()
    flow = data.workflow().set_index("stage")

    height = 27.0
    ax = fig.add_axes([0, (top_mm - height) / HEIGHT_MM, 1, height / HEIGHT_MM])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(0, height)
    ax.axis("off")

    left, right, gap = 6.0, 181.5, 3.4
    n = len(steps)
    width = (right - left - gap * (n - 1)) / n
    box_top, box_bottom = height - 7.2, height - 20.6
    xs = [left + i * (width + gap) for i in range(n)]

    for i, (x, step) in enumerate(zip(xs, steps)):
        group = step["group"]
        if group == "p1":
            face, edge, text, lw = "white", UPSTREAM_EDGE, UPSTREAM_TEXT, 0.5
        elif step["unit"] == "trajectories":
            color = style.STAGE_COLORS[group]
            face, edge, text, lw = _tint(color, 0.16), color, INK, 0.6
        elif step["stage"] == "fast_conductors":
            color = style.CLASS_COLORS["FAST_VALIDATED"]
            face, edge, text, lw = "white", color, INK, 0.9
        else:
            face, edge, text, lw = "white", INK, INK, 0.5
        ax.add_patch(FancyBboxPatch((x, box_bottom), width, box_top - box_bottom,
                                    boxstyle="round,pad=0,rounding_size=0.9", facecolor=face, edgecolor=edge,
                                    linewidth=lw, zorder=2))
        cx = x + width / 2
        count_color = style.CLASS_COLORS["FAST_VALIDATED"] if step["stage"] == "fast_conductors" else text
        # Explicit baselines keep every box typographically aligned.
        ax.text(cx, box_top - 3.7, f"{step['count']:,}", ha="center", va="baseline", fontsize=style.LABEL,
                fontweight="bold", color=count_color, zorder=3)
        ax.text(cx, box_top - 6.5, step["unit"], ha="center", va="baseline", fontsize=style.SMALL, color=text,
                zorder=3)
        for k, line in enumerate((step["note"] or "").split("\n")):
            ax.text(cx, box_top - 9.0 - 2.05 * k, line, ha="center", va="baseline", fontsize=style.SMALL,
                    color=UPSTREAM_TEXT if group == "p1" else "#555555", zorder=3)
        if step.get("classes"):
            total = sum(c for _, c in step["classes"])
            bar_x, bar_w, bar_y, bar_h = x + 1.4, width - 2.8, box_bottom + 1.5, 2.0
            offset = bar_x
            for name, c in step["classes"]:
                w = bar_w * c / total
                ax.add_patch(Rectangle((offset, bar_y), w, bar_h, facecolor=style.CLASS_COLORS[name],
                                       edgecolor="white", linewidth=0.25, zorder=3))
                offset += w
        if i < n - 1:
            arrow_color = UPSTREAM_EDGE if steps[i + 1]["group"] == "p1" else "#4D4D4D"
            y = (box_top + box_bottom) / 2
            ax.annotate("", xy=(xs[i + 1] - 0.25, y), xytext=(x + width + 0.25, y),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=arrow_color,
                                        linewidth=0.5, shrinkA=0, shrinkB=0), zorder=1)

    # Stage brackets above the boxes.
    groups = [("Project 1 screen (upstream)", 0, 3, UPSTREAM_EDGE, UPSTREAM_TEXT),
              ("Seed-0 census", 4, 5, style.STAGE_COLORS["census"], INK),
              ("Replicates", 6, 6, style.STAGE_COLORS["replicate"], INK),
              ("Definitive", 7, 7, style.STAGE_COLORS["definitive"], INK),
              ("Classification", 8, 9, INK, INK)]
    y_line = box_top + 2.2
    for label, first, last, color, text_color in groups:
        x0, x1 = xs[first] + 0.3, xs[last] + width - 0.3
        ax.plot([x0, x1], [y_line, y_line], color=color, linewidth=0.9, solid_capstyle="butt")
        ax.text((x0 + x1) / 2, y_line + 0.9, label, ha="center", va="bottom", fontsize=style.FONT_SIZE,
                color=text_color)

    # MD bracket below the simulation boxes.
    x0, x1 = xs[4] + 0.3, xs[7] + width - 0.3
    y = box_bottom - 1.6
    ax.plot([x0, x0, x1, x1], [y + 0.8, y, y, y + 0.8], color="#4D4D4D", linewidth=0.5)
    unique = int(flow.loc["unique_trajectories", "count"])
    ax.text((x0 + x1) / 2, y - 0.7, f"MACE-MP-0 NVT MD: {unique} unique trajectories", ha="center", va="top",
            fontsize=style.SMALL, color="#4D4D4D")

    style.panel_label(ax, "a", x=3.6 / WIDTH_MM, y=(top_mm - 3.4) / HEIGHT_MM, transform=fig.transFigure)

    source = pd.DataFrame({"step": range(1, n + 1), "stage": [s["stage"] for s in steps],
                           "count": [s["count"] for s in steps], "unit": [s["unit"] for s in steps],
                           "note": [(s["note"] or "").replace("\n", " ").replace("$^{-1}$", "^-1")
                                    for s in steps],
                           "upstream_project1": [s["group"] == "p1" for s in steps]})
    style.save_source(source, "fig1", "a")


# ----------------------------------------------------------------------------------------------
# b: census outcome -> final class
# ----------------------------------------------------------------------------------------------

CENSUS_GROUPS = [
    (("lead", "conditional"), "Lead (conditional)"),
    (("lead", "review"), "Lead (review)"),
    (("unresolved_diffusion", None), "Unresolved diffusion"),
    (("framework_melting", None), "Framework melting"),
    (("both", None), "Unresolved + melting"),
    (("undetermined_600K", None), "Undetermined at 600 K"),
]


def _panel_b(fig, rect_mm) -> None:
    comps = data.compositions()[["candidate_key", "census_category", "census_screen_label"]]
    classes = data.classification()[["candidate_key", "class"]]
    merged = comps.merge(classes, on="candidate_key", how="left", validate="one_to_one")
    present = [name for name in style.CLASS_ORDER if (merged["class"] == name).any()]

    rows = []
    for (category, label), text in CENSUS_GROUPS:
        subset = merged[merged["census_category"] == category]
        if label is not None:
            subset = subset[subset["census_screen_label"] == label]
        for name in present:
            rows.append(dict(census_group=text, census_category=category, census_screen_label=label or "",
                             final_class=name, n_compositions=int((subset["class"] == name).sum()),
                             group_total=len(subset)))
    table = pd.DataFrame(rows)
    table["share_percent"] = 100.0 * table["n_compositions"] / table["group_total"]
    assert table.groupby("census_group")["n_compositions"].sum().sum() == len(merged)

    x0, y0, x1, y1 = rect_mm
    ax = fig.add_axes([x0 / WIDTH_MM, y0 / HEIGHT_MM, (x1 - x0) / WIDTH_MM, (y1 - y0) / HEIGHT_MM])
    labels = [text for _, text in CENSUS_GROUPS]
    ys = np.arange(len(labels))[::-1].astype(float)
    bar_h = 0.62
    for y, group in zip(ys, labels):
        part = table[table["census_group"] == group]
        left = 0.0
        for _, row in part.iterrows():
            if row["n_compositions"] == 0:
                continue
            w = row["share_percent"]
            color = style.CLASS_COLORS[row["final_class"]]
            ax.barh(y, w, left=left, height=bar_h, color=color, edgecolor="white", linewidth=0.4)
            ax.text(left + w / 2, y, f"{row['n_compositions']}", ha="center", va="center",
                    fontsize=style.SMALL, color="white", fontweight="bold")
            left += w
        ax.text(101.5, y, f"n = {int(part['group_total'].iloc[0])}", ha="left", va="center",
                fontsize=style.SMALL, color="#4D4D4D", clip_on=False)
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(ys.min() - 0.55, ys.max() + 0.55)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Compositions in census group (%)")

    # Bracket for the census-vetoed groups ('not reliable').
    trans = ax.get_yaxis_transform()
    vetoed = [y for y, (key, _) in zip(ys, CENSUS_GROUPS) if key[0] != "lead"]
    leads = [y for y, (key, _) in zip(ys, CENSUS_GROUPS) if key[0] == "lead"]
    bx = -0.53
    for members, text in ((leads, "Leads"), (vetoed, "Not reliable (veto)")):
        lo, hi = min(members) - 0.3, max(members) + 0.3
        ax.plot([bx + 0.025, bx, bx, bx + 0.025], [lo, lo, hi, hi], transform=trans, color="#4D4D4D",
                linewidth=0.5, clip_on=False)
        ax.text(bx - 0.03, (lo + hi) / 2, text, transform=trans, rotation=90, ha="right", va="center",
                fontsize=style.SMALL, color="#4D4D4D", clip_on=False)

    handles = style.legend_handles([(style.CLASS_LABELS[name], style.CLASS_COLORS[name]) for name in present],
                                   size=4.0)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.0,
              handletextpad=0.3, borderaxespad=0.2, title="Final class", title_fontsize=style.SMALL,
              alignment="left")
    style.panel_label(ax, "b", x=3.6 / WIDTH_MM, y=LETTER_Y_BC / HEIGHT_MM, transform=fig.transFigure)
    style.save_source(table, "fig1", "b")


# ----------------------------------------------------------------------------------------------
# c: sigma(300 K) of the fast conductors
# ----------------------------------------------------------------------------------------------

DECIDED_BY = {"six_lead_validation": ("o", "Six-lead validation"),
              "replicate_rule": ("s", "Replicate rule"),
              "terminal_rule": ("D", "Terminal rule")}


def _qualifiers(text: str) -> set[str]:
    return {item for item in str(text).split(";") if item}


def _panel_c(fig, rect_mm) -> None:
    table = data.classification()
    fast = table[table["class"] == "FAST_VALIDATED"].sort_values("sigma300_mS_cm", ascending=False).copy()
    fast["a32_dependent"] = fast["qualifiers"].map(lambda q: "a32_dependent" in _qualifiers(q))
    fast["near_threshold"] = fast["qualifiers"].map(lambda q: "near_threshold" in _qualifiers(q))
    unknown = set(fast["decided_by"]) - set(DECIDED_BY)
    if unknown:
        raise ValueError(f"unexpected decided_by values: {sorted(unknown)}")

    ref = data.li6ps5cl_reference()
    mace = ref[ref["model"].str.contains("MACE-MP-0") & ref["model"].str.contains("un-fine-tuned")].iloc[0]
    experiment = float(mace["experiment_sintered_mS_cm"])
    model = float(mace["sigma300_mS_cm"])

    x0, y0, x1, y1 = rect_mm
    ax = fig.add_axes([x0 / WIDTH_MM, y0 / HEIGHT_MM, (x1 - x0) / WIDTH_MM, (y1 - y0) / HEIGHT_MM])
    green = style.CLASS_COLORS["FAST_VALIDATED"]
    n = len(fast)
    ys = np.arange(n)[::-1].astype(float)

    ref_lines = [(FAST_THRESHOLD_mS_cm, style.FAST_THRESHOLD_COLOR, (2.2, 1.4)),
                 (experiment, "#4D4D4D", (None)),
                 (model, "#4D4D4D", (2.2, 1.4))]
    for value, color, dashes in ref_lines:
        line = ax.axvline(value, color=color, linewidth=0.6, zorder=1)
        if dashes:
            line.set_dashes(dashes)

    for y, (_, row) in zip(ys, fast.iterrows()):
        lo, hi = row["sigma300_ci95_lo_mS_cm"], row["sigma300_ci95_hi_mS_cm"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=green, linewidth=0.8, solid_capstyle="butt", zorder=2)
            for end in (lo, hi):
                ax.plot([end, end], [y - 0.16, y + 0.16], color=green, linewidth=0.6, zorder=2)
        marker = DECIDED_BY[row["decided_by"]][0]
        ax.plot(row["sigma300_mS_cm"], y, marker=marker, linestyle="none",
                markersize=3.6 if marker != "D" else 3.1,
                markerfacecolor="white" if row["near_threshold"] else green,
                markeredgecolor=green, markeredgewidth=0.7, zorder=3)

    labels = [formula_tex(f) + ("$^{\\dagger}$" if dep else "") for f, dep in zip(fast["formula"],
                                                                                    fast["a32_dependent"])]
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xscale("log")
    style.log_ticks(ax, "x")
    ax.set_xlim(0.02, 500)
    ax.set_ylim(-0.7, n - 1 + 2.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("σ(300 K) (mS cm$^{-1}$)")

    upper, lower = n - 1 + 1.55, n - 1 + 0.85
    ax.text(FAST_THRESHOLD_mS_cm * 1.12, upper, "Fast threshold", ha="left", va="center", fontsize=style.SMALL,
            color=style.FAST_THRESHOLD_COLOR)
    ax.text(experiment / 1.12, lower, "Li$_6$PS$_5$Cl expt.", ha="right", va="center", fontsize=style.SMALL,
            color="#4D4D4D")
    ax.text(model * 1.12, upper, "Li$_6$PS$_5$Cl MACE-MP-0", ha="left", va="center", fontsize=style.SMALL,
            color="#4D4D4D")

    handles = [Line2D([0], [0], marker=marker, linestyle="none", markersize=3.4 if marker != "D" else 3.0,
                      markerfacecolor=green, markeredgecolor=green, markeredgewidth=0.7, label=label)
               for key, (marker, label) in DECIDED_BY.items() if (fast["decided_by"] == key).any()]
    if fast["near_threshold"].any():
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=3.4, markerfacecolor="white",
                              markeredgecolor=green, markeredgewidth=0.7, label="Near threshold"))
    if fast["a32_dependent"].any():
        handles.append(Line2D([0], [0], marker="$\\dagger$", linestyle="none", markersize=4.0,
                              markerfacecolor=INK, markeredgecolor=INK, markeredgewidth=0.0,
                              label="A3.2-dependent"))
    legend = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=1.0,
                       handletextpad=0.4, labelspacing=0.35, borderaxespad=0.1)
    legend.set_zorder(5)
    style.panel_label(ax, "c", x=(x0 - 24.0) / WIDTH_MM, y=LETTER_Y_BC / HEIGHT_MM, transform=fig.transFigure)

    source = fast[["p1_rank", "formula", "sigma300_mS_cm", "sigma300_ci95_lo_mS_cm", "sigma300_ci95_hi_mS_cm",
                   "decided_by", "a32_dependent", "near_threshold"]].copy()
    source["reference_fast_threshold_mS_cm"] = FAST_THRESHOLD_mS_cm
    source["reference_li6ps5cl_experiment_sintered_mS_cm"] = experiment
    source["reference_li6ps5cl_mace_mp0_mS_cm"] = model
    style.save_source(source, "fig1", "c")


def make() -> list[Path]:
    fig = style.figure(style.DOUBLE, HEIGHT_MM * style.MM)
    _panel_a(fig, top_mm=HEIGHT_MM - 1.0)
    _panel_b(fig, rect_mm=(33.0, 11.0, 80.0, 52.0))
    _panel_c(fig, rect_mm=(125.0, 11.0, 181.0, 61.0))
    return style.save(fig, NAME, "main")
