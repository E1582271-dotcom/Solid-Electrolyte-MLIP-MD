"""
Project 2 -- the AI4SSB portfolio-level pipeline diagram (core deliverable of the
July 2026 figure-presentation pass): screen (P1) + generate (P3) -> validate (P2, this
repo) -> future electrochemical closure. A schematic, not a data plot -- drawn with
matplotlib patches/annotate so it shares the Nature-style fonts/palette/sizing of every
other figure in this portfolio (no Graphviz/draw.io).

All four boxes' numbers are HARDCODED Python literals, each commented with its exact
source file/field -- a deliberate choice: this script must render a complete, correct
figure even when this repo (Solid-Electrolyte-MLIP-MD) is cloned on its own (the common
case for a GitHub visitor), without reading sibling repos at runtime. Cross-check the
comments against the cited files before changing a number.

Colors: P1=cool (PALETTE.blue_main), P3=warm (PALETTE.violet) -- the same provenance
split 06_compare_leads.py uses for LEAD_COLORS; the two ranking-reversal callouts inside
the validate box reuse LEAD_COLORS['lips3_gen021'] (red_strong) and LEAD_COLORS['li8tis6']
(teal) verbatim, so this figure doesn't re-derive colors that already mean something in
06_compare_leads.png.

Run:
    python 08_pipeline_overview.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, HERE)
from src import plotstyle as ps  # noqa: E402

# Same provenance colors as 06_compare_leads.py's LEAD_COLORS -- do not re-derive.
C_P1 = ps.PALETTE["blue_main"]     # cool: screen (Project 1)
C_P3 = ps.PALETTE["violet"]        # warm: generate (Project 3)
C_LIPS3 = ps.PALETTE["red_strong"]   # LEAD_COLORS["lips3_gen021"]
C_LI8TIS6 = ps.PALETTE["teal"]       # LEAD_COLORS["li8tis6"]
C_FUTURE = ps.PALETTE["neutral_mid"]


def _box(ax, xy, w, h, color, dashed=False, lw=1.1, fill_alpha=0.10):
    x, y = xy
    style = "round,pad=0.02,rounding_size=0.08"
    patch = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                           edgecolor=color, facecolor=color, alpha=fill_alpha,
                           linestyle="--" if dashed else "-", zorder=2)
    ax.add_patch(patch)
    edge = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                          edgecolor=color, facecolor="none",
                          linestyle="--" if dashed else "-", zorder=3)
    ax.add_patch(edge)
    return x, y, w, h


def _arrow(ax, p0, p1, color, dashed=False):
    ax.annotate("", xy=p1, xytext=p0, zorder=1,
               arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                               linestyle="--" if dashed else "-",
                               shrinkA=2, shrinkB=2, mutation_scale=14))


def main():
    ps.apply_publication_style()
    fig, ax = plt.subplots(figsize=(ps.COL_DOUBLE_IN, 3.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # ── Box 1: Screen (Project 1) ──────────────────────────────────────────
    b1 = _box(ax, (0.3, 2.35), 2.5, 1.3, C_P1)
    ax.text(0.3 + 1.25, 2.35 + 1.3 - 0.22, "Screen (Project 1)", ha="center", va="top",
           fontsize=ps.FS_LABEL, fontweight="bold", color=ps.PALETTE["neutral_black"])
    ax.text(0.3 + 1.25, 2.35 + 0.72, "CatBoost + SHAP +\nadversarial audit",
           ha="center", va="center", fontsize=ps.FS_ANNOT, color=ps.PALETTE["neutral_dark"])
    # 328 (loose MP query) -> 184 (post audited-filter): project1_screening/
    # source_data/fig06a_screen_audit_funnel.csv
    ax.text(0.3 + 1.25, 2.35 + 0.22, "328 → 184 candidates", ha="center", va="center",
           fontsize=ps.FS_ANNOT, fontweight="bold", color=C_P1)

    # ── Box 2: Generate (Project 3) ────────────────────────────────────────
    b2 = _box(ax, (0.3, 0.35), 2.5, 1.3, C_P3)
    ax.text(0.3 + 1.25, 0.35 + 1.3 - 0.22, "Generate (Project 3)", ha="center", va="top",
           fontsize=ps.FS_LABEL, fontweight="bold", color=ps.PALETTE["neutral_black"])
    ax.text(0.3 + 1.25, 0.35 + 0.72, "MatterGen + self-consistent\nMLIP hull",
           ha="center", va="center", fontsize=ps.FS_ANNOT, color=ps.PALETTE["neutral_dark"])
    # 64 generated -> 43 S.U.N.: project3_generative/data/p3_runs.json counts
    # (n_generated field starts post Li-filter at 61; the raw 64 and the 3 dropped
    # Li-free P-S binaries are recorded in project3_generative/README.md only)
    ax.text(0.3 + 1.25, 0.35 + 0.22, "64 → 43 S.U.N.", ha="center", va="center",
           fontsize=ps.FS_ANNOT, fontweight="bold", color=C_P3)

    # ── Box 3: Validate (Project 2, this repo) ─────────────────────────────
    b3 = _box(ax, (3.95, 1.15), 3.15, 1.85, ps.PALETTE["neutral_dark"], fill_alpha=0.05)
    ax.text(3.95 + 1.575, 1.15 + 1.85 - 0.22, "Validate (Project 2, this repo)",
           ha="center", va="top", fontsize=ps.FS_LABEL, fontweight="bold",
           color=ps.PALETTE["neutral_black"])
    ax.text(3.95 + 1.575, 1.15 + 1.85 - 0.48, "Same-protocol MLIP-MD,\n4 funnel leads",
           ha="center", va="top", fontsize=ps.FS_ANNOT, color=ps.PALETTE["neutral_dark"])
    # Ranking-reversal callouts: source_data/fig06_compare_leads.csv (this repo)
    # LiPS3 (gen021): prior_rank 4, sigma300 10.19 mS/cm -> 2nd by sigma among the 4 leads
    ax.text(3.95 + 1.575, 1.15 + 0.62,
           "LiPS$_3$: prior rank 4th (worst) → MD rank 2nd (confirmed)",
           ha="center", va="center", fontsize=ps.FS_ANNOT, color=C_LIPS3)
    # Li8TiS6: prior_rank 2, sigma300 1.4e-7 mS/cm (near-insulator) -> falsified
    ax.text(3.95 + 1.575, 1.15 + 0.30,
           "Li$_8$TiS$_6$: prior rank 2nd → MD falsified (insulator)",
           ha="center", va="center", fontsize=ps.FS_ANNOT, color=C_LI8TIS6)

    # ── Box 4: Future -- electrochemical closure (dashed, aspirational) ───
    b4 = _box(ax, (7.6, 1.45), 2.1, 1.25, C_FUTURE, dashed=True)
    ax.text(7.6 + 1.05, 1.45 + 1.25 - 0.22, "Future", ha="center", va="top",
           fontsize=ps.FS_LABEL, fontweight="bold", color=ps.PALETTE["neutral_black"])
    ax.text(7.6 + 1.05, 1.45 + 0.62, "Electrochemical\nvalidation\n(EIS / ARC)\n— next step",
           ha="center", va="center", fontsize=ps.FS_ANNOT, color=C_FUTURE, style="italic")

    # ── Arrows: 1->3, 2->3 (colored by source), 3->4 (dashed, aspirational) ─
    _arrow(ax, (2.8, 3.0), (3.95, 2.55), C_P1)
    _arrow(ax, (2.8, 1.0), (3.95, 1.65), C_P3)
    _arrow(ax, (7.1, 2.075), (7.6, 2.075), C_FUTURE, dashed=True)

    fig.text(0.5, 0.015, "Every stage ships a CPU-only plumbing fixture; "
            "honest limitations disclosed per stage.",
            ha="center", va="bottom", fontsize=ps.FS_ANNOT - 1, color=ps.PALETTE["neutral_mid"],
            style="italic")

    rows = [
        ["screen", "Project 1", "328 -> 184 candidates"],
        ["generate", "Project 3", "64 -> 43 S.U.N."],
        ["validate", "Project 2", "LiPS3 prior rank 4 -> MD rank 2 (confirmed); "
                                  "Li8TiS6 prior rank 2 -> MD falsified"],
        ["future", "n/a", "EIS / ARC electrochemical validation"],
    ]
    ps.save_source_data(os.path.join(FIG, "08_pipeline_overview.png"),
                        ["stage", "project", "headline"], rows)
    ps.finalize_figure(fig, os.path.join(FIG, "08_pipeline_overview.png"),
                       formats=("png", "svg"), pad=0.3)
    print("Saved figures/08_pipeline_overview.png, source_data/08_pipeline_overview.csv")


if __name__ == "__main__":
    main()
