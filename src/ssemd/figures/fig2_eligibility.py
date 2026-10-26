"""Fig. 2: per-temperature evidence behind every class.

One row per composition (62), one column per tested temperature. Each cell summarises every
record the classifier used at that (composition, temperature) (``data.classifier_runs()``):
eligible (filled with the class colour), diffusion not resolved, framework not intact, failed
kinisi fit (amendment A3.2) or temperature bias above the protocol limit; untested cells are
blank. Rows are grouped by final class and sorted by sigma(300 K) within a class; the rule that
decided the class and sigma(300 K) (estimate, or upper bound with a "<=" sign) are printed on
the right (``data.classification()``).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from .. import data
from ..protocol import T_BIAS_MAX
from . import style

NAME = "fig2_eligibility"
WIDTH_MM = 183.0

INK = "#262626"
MUTED = "#6E6E6E"
OPEN_EDGE = "#7A7A7A"

# Cell states, in the priority used when a cell has no eligible record.
ELIGIBLE = "eligible"
FRAMEWORK = "framework_not_intact"
T_BIAS = "temperature_bias"
KINISI = "kinisi_fit_failed"
UNRESOLVED = "diffusion_not_resolved"
NON_ELIGIBLE_PRIORITY = (FRAMEWORK, T_BIAS, KINISI, UNRESOLVED)
STATE_LABELS = {ELIGIBLE: "Eligible (fill: class colour)",
                UNRESOLVED: "Diffusion not resolved",
                FRAMEWORK: "Framework not intact",
                KINISI: "Failed kinisi fit (A3.2)",
                T_BIAS: f"T bias > {T_BIAS_MAX * 100:g} %"}

DECIDED_BY = {"six_lead_validation": "six-lead decision", "replicate_rule": "replicate rule",
              "terminal_rule": "terminal rule", "single_seed_fit": "single-seed fit",
              "two_point_fit": "two-point fit", "prefactor_bound": "prefactor bound",
              "pair_bound": "slope bound"}

# Geometry (mm).
ROW = 2.62          # row pitch
GROUP_GAP = 4.2     # gap above each class group (holds the group label)
TOP = 13.0          # space above the first group (column header)
BOTTOM = 14.0       # space below the last row (legend)
PILL_W, PILL_H = 6.4, 1.8

_DIGITS = re.compile(r"(\d+(?:\.\d+)?)")


def formula_tex(formula: str) -> str:
    """Chemical formula with mathtext subscripts: ``Li5P(S2Cl)2`` -> ``Li$_{5}$P(S$_{2}$Cl)$_{2}$``."""
    return _DIGITS.sub(lambda match: f"$_{{{match.group(1)}}}$", formula)


def sigma_text(value: float, upper_bound: bool) -> str:
    """Two significant figures; values below 0.1 in scientific notation (mathtext)."""
    rounded = float(f"{value:.2g}")
    if rounded >= 10:
        text = f"{rounded:.0f}"
    elif rounded >= 1:
        text = f"{rounded:.1f}"
    elif rounded >= 0.1:
        text = f"{rounded:.2f}"
    else:
        exponent = int(np.floor(np.log10(rounded)))
        mantissa = rounded / 10 ** exponent
        if round(mantissa, 1) >= 10:
            mantissa, exponent = mantissa / 10, exponent + 1
        text = f"{mantissa:.1f} × 10$^{{{exponent}}}$"
    return f"≤ {text}" if upper_bound else text


def _record_state(row) -> str:
    if row["eligible"]:
        return ELIGIBLE
    if row["kinisi_fit_failed"]:
        return KINISI
    if not row["framework_intact"]:
        return FRAMEWORK
    if abs(row["temperature_bias_fraction"]) > T_BIAS_MAX:
        return T_BIAS
    if not row["diffusion_resolved"]:
        return UNRESOLVED
    raise ValueError(f"record {row['record_id']} is not eligible for an unknown reason")


def _seed_list(text) -> set[int]:
    return {int(float(item)) for item in str(text).split(";") if item not in ("", "nan")}


def _cells(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    runs["state"] = runs.apply(_record_state, axis=1)
    runs["temperature_biased"] = runs["temperature_bias_fraction"].abs() > T_BIAS_MAX
    rows = []
    for (key, temperature), group in runs.groupby(["candidate_key", "temperature_K"], sort=True):
        states = set(group["state"])
        n_eligible = int((group["state"] == ELIGIBLE).sum())
        if n_eligible:
            state = ELIGIBLE
        else:
            state = next(s for s in NON_ELIGIBLE_PRIORITY if s in states)
        rows.append(dict(candidate_key=key, temperature_K=int(temperature), state=state,
                         n_records=len(group), n_seeds=int(group["md_seed"].nunique()), n_eligible=n_eligible,
                         seeds=";".join(str(s) for s in sorted(group["md_seed"].unique())),
                         n_diffusion_not_resolved=int((group["state"] == UNRESOLVED).sum()),
                         n_framework_not_intact=int((group["state"] == FRAMEWORK).sum()),
                         n_kinisi_fit_failed=int((group["state"] == KINISI).sum()),
                         n_temperature_bias=int(group["temperature_biased"].sum()),
                         max_production_ps=float(group["production_ps"].max())))
    return pd.DataFrame(rows)


def _row_order(classes: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = []
    for name in style.CLASS_ORDER:
        members = classes[classes["class"] == name]
        if len(members):
            members = members.sort_values(["sigma300_mS_cm", "p1_rank"], ascending=[False, True])
            groups.append((name, members))
    unknown = set(classes["class"]) - set(style.CLASS_ORDER)
    if unknown:
        raise ValueError(f"classes outside style.CLASS_ORDER: {sorted(unknown)}")
    return groups


# ----------------------------------------------------------------------------------------------
# glyphs (drawn in millimetre data coordinates)
# ----------------------------------------------------------------------------------------------

def _pill(ax, cx, cy, face, edge, lw=0.5, linestyle="-"):
    ax.add_patch(FancyBboxPatch((cx - PILL_W / 2, cy - PILL_H / 2), PILL_W, PILL_H,
                                boxstyle="round,pad=0,rounding_size=0.45", facecolor=face, edgecolor=edge,
                                linewidth=lw, linestyle=linestyle, zorder=3))


def _glyph(ax, state, cx, cy, color):
    """Draw one cell glyph; returns the colour for a seed-count label inside it."""
    if state == ELIGIBLE:
        _pill(ax, cx, cy, color, color, lw=0.3)
        return "white"
    if state == UNRESOLVED:
        _pill(ax, cx, cy, "white", OPEN_EDGE, lw=0.5)
        return MUTED
    if state == KINISI:
        _pill(ax, cx, cy, "white", OPEN_EDGE, lw=0.5)
        half_w, half_h = PILL_W / 2 - 0.9, PILL_H / 2 - 0.15
        ax.plot([cx - half_w, cx + half_w], [cy - half_h, cy + half_h], color=INK, linewidth=0.6,
                solid_capstyle="round", zorder=4)
        return MUTED
    if state == FRAMEWORK:
        half = PILL_H / 2 - 0.05
        for sign in (1, -1):
            ax.plot([cx - half, cx + half], [cy - sign * half, cy + sign * half], color=INK, linewidth=0.75,
                    solid_capstyle="round", zorder=4)
        return INK
    if state == T_BIAS:
        half = PILL_H / 2
        ax.add_patch(Polygon([(cx - half, cy - half), (cx + half, cy - half), (cx, cy + half)], closed=True,
                             facecolor="white", edgecolor=INK, linewidth=0.6, zorder=4))
        return INK
    raise ValueError(state)


def make() -> list[Path]:
    classes = data.classification()
    runs = data.classifier_runs()
    cells = _cells(runs)
    temperatures = sorted(cells["temperature_K"].unique())
    groups = _row_order(classes)
    n_rows = sum(len(members) for _, members in groups)

    height = TOP + len(groups) * GROUP_GAP + n_rows * ROW + BOTTOM
    fig = style.figure(style.DOUBLE, height * style.MM)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(0, height)
    ax.axis("off")

    # Columns (mm).
    x_rank = 8.5          # right edge of P1 rank
    x_formula = 10.5      # left edge of formula
    x_strip = 28.0        # class strip
    x_first, x_last = 36.0, 134.0
    pitch = (x_last - x_first) / (len(temperatures) - 1)
    x_of = {t: x_first + i * pitch for i, t in enumerate(temperatures)}
    x_rule = 141.0        # left edge of decided-by text
    x_sigma = 181.0       # right edge of sigma text

    # Header.
    y_head = height - TOP + 2.4
    for t, x in x_of.items():
        ax.text(x, y_head, f"{t}", ha="center", va="baseline", fontsize=style.FONT_SIZE, color=INK)
    ax.text((x_first + x_last) / 2, y_head + 4.4, "Simulation temperature (K)", ha="center", va="baseline",
            fontsize=style.FONT_SIZE, color=INK)
    ax.plot([x_first - PILL_W / 2, x_last + PILL_W / 2], [y_head + 3.0, y_head + 3.0], color=INK, linewidth=0.5)
    ax.text(x_rank, y_head, "P1 rank", ha="right", va="baseline", fontsize=style.SMALL, color=MUTED)
    ax.text(x_formula, y_head, "Composition", ha="left", va="baseline", fontsize=style.SMALL, color=MUTED)
    ax.text(x_rule, y_head, "Decided by", ha="left", va="baseline", fontsize=style.SMALL, color=MUTED)
    ax.text(x_sigma, y_head + 2.6, "σ(300 K)", ha="right", va="baseline", fontsize=style.SMALL, color=MUTED)
    ax.text(x_sigma, y_head, "(mS cm$^{-1}$)", ha="right", va="baseline", fontsize=style.SMALL, color=MUTED)

    record_seeds = runs.groupby("candidate_key")["md_seed"].apply(lambda s: set(int(v) for v in s))
    cell_index = cells.set_index(["candidate_key", "temperature_K"])
    source_rows = []
    excluded_seed_rows = []
    y = height - TOP
    order = 0
    for name, members in groups:
        color = style.CLASS_COLORS[name]
        y_group_top = y - GROUP_GAP
        ax.text(x_formula, y - GROUP_GAP + 1.05, f"{style.CLASS_LABELS[name]}, n = {len(members)}", ha="left",
                va="baseline", fontsize=style.FONT_SIZE, color=INK, fontweight="bold")
        y = y_group_top
        strip_top = y
        for _, comp in members.iterrows():
            order += 1
            cy = y - ROW / 2
            key = comp["candidate_key"]
            used = _seed_list(comp["seeds"])
            excluded = sorted(record_seeds.get(key, set()) - used) if used else []
            mark = "$^{\\ddagger}$" if excluded else ""
            if excluded:
                excluded_seed_rows.append((comp["formula"], excluded))
            ax.text(x_rank, cy, f"{comp['p1_rank']}", ha="right", va="center_baseline", fontsize=style.SMALL,
                    color=MUTED)
            ax.text(x_formula, cy, formula_tex(comp["formula"]) + mark, ha="left", va="center_baseline",
                    fontsize=style.SMALL, color=INK)
            ax.text(x_rule, cy, DECIDED_BY.get(comp["decided_by"], comp["decided_by"]), ha="left",
                    va="center_baseline", fontsize=style.SMALL, color=MUTED)
            upper = comp["sigma_kind"] == "upper_bound"
            ax.text(x_sigma, cy, sigma_text(comp["sigma300_mS_cm"], upper), ha="right", va="center_baseline",
                    fontsize=style.SMALL, color=INK)
            for t in temperatures:
                if (key, t) not in cell_index.index:
                    continue
                cell = cell_index.loc[(key, t)]
                label_color = _glyph(ax, cell["state"], x_of[t], cy, color)
                if cell["n_seeds"] > 1:
                    text = (f"{cell['n_eligible']}/{cell['n_seeds']}"
                            if cell["state"] == ELIGIBLE and cell["n_eligible"] < cell["n_seeds"]
                            else f"{cell['n_seeds']}")
                    ax.text(x_of[t], cy, text, ha="center", va="center_baseline", fontsize=style.SMALL,
                            color=label_color, zorder=5)
                source_rows.append(dict(row_order=order, p1_rank=comp["p1_rank"], formula=comp["formula"],
                                        candidate_key=key, **{"class": name}, temperature_K=t,
                                        state=cell["state"], n_seeds=cell["n_seeds"],
                                        n_eligible=cell["n_eligible"], seeds=cell["seeds"],
                                        n_diffusion_not_resolved=cell["n_diffusion_not_resolved"],
                                        n_framework_not_intact=cell["n_framework_not_intact"],
                                        n_kinisi_fit_failed=cell["n_kinisi_fit_failed"],
                                        n_temperature_bias=cell["n_temperature_bias"],
                                        max_production_ps=cell["max_production_ps"],
                                        decided_by=comp["decided_by"], sigma300_mS_cm=comp["sigma300_mS_cm"],
                                        sigma_kind=comp["sigma_kind"],
                                        seeds_used_by_class=comp["seeds"],
                                        excluded_seeds=";".join(str(s) for s in excluded)))
            y -= ROW
        ax.add_patch(Rectangle((x_strip, y + 0.25), 1.1, strip_top - y - 0.5, facecolor=color, edgecolor="none",
                               zorder=2))

    # Legend: glyph, label, then a fixed gap; label widths are measured so the spacing is even.
    present = set(cells["state"])
    y_leg = BOTTOM - 4.6
    entries = [s for s in (ELIGIBLE, UNRESOLVED, FRAMEWORK, KINISI, T_BIAS) if s in present]
    renderer = fig.canvas.get_renderer()
    mm_per_px = WIDTH_MM / fig.bbox.width
    legend_color = style.CLASS_COLORS["FAST_VALIDATED"]
    x = x_formula
    for state in entries:
        glyph_w = PILL_W if state in (ELIGIBLE, UNRESOLVED, KINISI) else PILL_H
        _glyph(ax, state, x + glyph_w / 2, y_leg, legend_color)
        label = ax.text(x + glyph_w + 1.2, y_leg, STATE_LABELS[state], ha="left", va="center_baseline",
                        fontsize=style.SMALL, color=INK)
        x += glyph_w + 1.2 + label.get_window_extent(renderer).width * mm_per_px + 5.0
    first = "Numbers: MD seeds at that temperature (eligible/total where not all are eligible); blank: not simulated."
    if T_BIAS not in present:
        first += f" No record has a temperature bias above {T_BIAS_MAX * 100:g} %."
    ax.text(x_formula, y_leg - 3.3, first, ha="left", va="center_baseline", fontsize=style.SMALL, color=MUTED)
    if excluded_seed_rows:
        seed_sets = {tuple(seeds) for _, seeds in excluded_seed_rows}
        which = (f"Seed {', '.join(str(s) for s in next(iter(seed_sets)))}" if len(seed_sets) == 1
                 else "Some eligible seeds")
        ax.text(x_formula, y_leg - 6.1, f"$^{{\\ddagger}}${which} excluded from the class decision "
                "(documented exception of the six-lead validation).", ha="left", va="center_baseline",
                fontsize=style.SMALL, color=MUTED)

    source = pd.DataFrame(source_rows)
    source["sigma300_text"] = [sigma_text(v, k == "upper_bound").replace("$^{", "^").replace("}$", "")
                               for v, k in zip(source["sigma300_mS_cm"], source["sigma_kind"])]
    style.save_source(source, "fig2", "a")
    return style.save(fig, NAME, "main")
