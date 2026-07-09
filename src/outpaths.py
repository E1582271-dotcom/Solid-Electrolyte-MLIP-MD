"""Figure/source-data output tiering (Project 2 local — NOT part of the shared CORE).

Routes each figure and its provenance CSV into a ``main/`` (headline Fig. 1–4) or
``supplementary/`` subfolder so that *regeneration* lands in the right tier, keeping
``figures/`` and ``source_data/`` organised without editing the byte-identical CORE
block of ``plotstyle.py`` (shared verbatim with the P1/P3 repos).

Usage in a plotting script::

    from src import outpaths as op
    out = op.fig(args.fig_dir, "03_arrhenius_prod.png")   # -> figures/main/03_arrhenius_prod.png
    op.save_source_data(out, columns, rows)               # -> source_data/main/03_arrhenius_prod.csv
    ps.finalize_figure(fig, out)
"""
from __future__ import annotations

import os

from src import plotstyle as ps

# The seven headline stems = docs/FIGURES.md "Main figures" (Fig. 1–4). Everything else
# (02 stability, baseline/_long transport, per-composition doping, per-lead, the
# leads comparison and the pipeline schematic) is supplementary.
MAIN = {
    "01_structures",
    "03_arrhenius_prod", "03_sigma300_vs_expt_prod",
    "04_finetune_convergence", "03_arrhenius_ft", "03_sigma300_vs_expt_ft",
    "07_doping_trend",
}


def tier(name: str) -> str:
    """"main" | "supplementary" for a figure name (path/basename, with or without ext)."""
    stem = os.path.basename(str(name)).rsplit(".", 1)[0]
    return "main" if stem in MAIN else "supplementary"


def fig(fig_dir: str, basename: str) -> str:
    """<fig_dir>/<tier>/<basename> — the tiered path to hand to finalize_figure."""
    return os.path.join(fig_dir, tier(basename), basename)


def save_source_data(fig_path: str, columns, rows):
    """CORE save_source_data, routed to source_data/<tier>/. The '../../' resolves the
    CORE path logic (base.parent / subdir, since the figure's parent is now a tier dir,
    not literally 'figures') back to <repo>/source_data/<tier>/."""
    return ps.save_source_data(fig_path, columns, rows,
                               subdir=f"../../source_data/{tier(fig_path)}")
