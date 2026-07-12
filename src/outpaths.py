"""Figure/source-data output tiering (Project 2 local — NOT part of the shared CORE).

Routes each figure and its provenance CSV into a ``main/`` (headline Fig. 1–4) or
``supplementary/`` subfolder so that *regeneration* lands in the right tier, keeping
``figures/`` and ``source_data/`` organised without editing the byte-identical CORE
block of ``plotstyle.py`` (shared verbatim with the P1/P3 repos).

Usage in a plotting script::

    from src import outpaths as op
    out = op.fig(args.fig_dir, "03_transport_prod.png")   # -> figures/main/03_transport_prod.png
    op.save_source_data(out, columns, rows)               # -> source_data/main/03_transport_prod.csv
    ps.finalize_figure(fig, out)
"""
from __future__ import annotations

import os

from src import plotstyle as ps

# The six headline stems = docs/FIGURES.md "Main figures" (Fig. 1–5). Everything else
# (02 stability, the merged baseline/doping/leads transport figures, and the pipeline
# schematic) is supplementary.
MAIN = {
    "01_structures",
    "03_transport_prod", "03_transport_ft",
    "04_finetune_convergence",
    "06_compare_leads",
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
