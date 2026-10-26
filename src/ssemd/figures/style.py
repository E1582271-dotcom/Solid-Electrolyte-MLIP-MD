"""Figure style (Nature family) and deterministic export shared by every figure.

Sizes follow Nature's column widths (89 mm single, 183 mm double, at most 247 mm tall);
text is 5-7 pt sans-serif, panel letters 8 pt bold lower case. Colours are the Okabe-Ito
colour-blind-safe set. Every figure is saved as PDF and SVG (editable text) and as a
600 dpi PNG, without timestamps, so a rebuild gives identical bytes. Each panel's plotted
values are written to ``figures/source_data/<figure>_<panel>.csv``.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..paths import FIGURES  # noqa: E402

MM = 1 / 25.4
SINGLE = 89 * MM
ONE_HALF = 120 * MM
DOUBLE = 183 * MM
MAX_HEIGHT = 247 * MM

FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
FONT_SIZE = 6.0
SMALL = 5.0
LABEL = 7.0
PANEL = 8.0
LINE = 0.6
HASHSALT = "solid-electrolyte-mlip-md"

# Okabe-Ito
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
GREY = "#8C8C8C"
LIGHT_GREY = "#D9D9D9"

CLASS_ORDER = ("FAST_VALIDATED", "SLOW_MEASURED", "SLOW_ESTIMATED", "SLOW_BOUNDED", "THERMALLY_UNSTABLE")
CLASS_COLORS = {"FAST_VALIDATED": OKABE_ITO["green"], "SLOW_MEASURED": OKABE_ITO["blue"],
                "SLOW_ESTIMATED": OKABE_ITO["purple"], "SLOW_BOUNDED": GREY,
                "THERMALLY_UNSTABLE": OKABE_ITO["vermillion"], "NEEDS_COMPUTE": OKABE_ITO["orange"]}
CLASS_LABELS = {"FAST_VALIDATED": "Fast (validated)", "SLOW_MEASURED": "Slow (measured)",
                "SLOW_ESTIMATED": "Slow (two-point estimate)", "SLOW_BOUNDED": "Slow (upper bound)",
                "THERMALLY_UNSTABLE": "Thermally unstable", "NEEDS_COMPUTE": "Pending"}
STAGE_COLORS = {"census": OKABE_ITO["sky"], "replicate": OKABE_ITO["orange"], "definitive": OKABE_ITO["blue"]}
SEED_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}
TEMPERATURE_CMAP = "viridis"
FAST_THRESHOLD_COLOR = OKABE_ITO["vermillion"]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": SMALL,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.borderaxespad": 0.3,
        "axes.linewidth": LINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": LINE,
        "ytick.major.width": LINE,
        "xtick.minor.width": LINE * 0.7,
        "ytick.minor.width": LINE * 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "errorbar.capsize": 1.5,
        "patch.linewidth": LINE,
        "svg.fonttype": "none",
        "svg.hashsalt": HASHSALT,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "standard",
        "path.simplify": False,
        "axes.unicode_minus": False,
        "mathtext.default": "regular",
    })


def figure(width: float, height: float):
    """A new figure of the given size (inches); width must be a Nature column width."""
    if round(width / MM) not in (89, 120, 183):
        raise ValueError("Figure width must be 89, 120 or 183 mm")
    if height > MAX_HEIGHT + 1e-9:
        raise ValueError("Figure taller than 247 mm")
    apply_style()
    return plt.figure(figsize=(width, height))


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.0, transform=None) -> None:
    """Bold lower-case panel letter at the top-left of ``ax`` (figure coordinates by default)."""
    fig = ax.figure
    bbox = ax.get_position()
    if transform is None:
        fig.text(bbox.x0 + x, bbox.y1 + 0.004 if y == 1.0 else y, letter, fontsize=PANEL, fontweight="bold",
                 ha="right", va="bottom")
    else:
        ax.text(x, y, letter, transform=transform, fontsize=PANEL, fontweight="bold", ha="right", va="bottom")


def save(fig, name: str, folder: str = "main") -> list[Path]:
    """Write ``figures/<folder>/<name>.{pdf,svg,png}`` deterministically."""
    target = FIGURES / folder
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for extension, metadata in (("pdf", {"CreationDate": None, "ModDate": None, "Producer": None,
                                         "Creator": None}),
                                ("svg", {"Date": None, "Creator": None}),
                                ("png", {"Software": None})):
        path = target / f"{name}.{extension}"
        fig.savefig(path, metadata=metadata, dpi=600 if extension == "png" else None)
        written.append(path)
    plt.close(fig)
    return written


def save_source(frame: pd.DataFrame, figure_name: str, panel: str) -> Path:
    """Plotted values of one panel: ``figures/source_data/<figure>_<panel>.csv``."""
    target = FIGURES / "source_data"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{figure_name}_{panel}.csv"
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.6g")
    return path


def log_ticks(ax, axis: str = "y") -> None:
    from matplotlib.ticker import LogLocator, NullFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10, numticks=12))
    target.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12))
    target.set_minor_formatter(NullFormatter())


def legend_handles(labels_colors: Iterable[tuple[str, str]], marker: str = "s", size: float = 4.0):
    from matplotlib.lines import Line2D

    return [Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor=color, markeredgecolor="none",
                   markersize=size, label=label) for label, color in labels_colors]
