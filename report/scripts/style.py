"""Shared figure/table style for the report assets.

Vector PDF output, Computer Modern math, Okabe-Ito colour-blind-safe palette,
grayscale-safe line styles and markers, widths 3.3 in (half) or 6.5 in (full),
axis labels with units, no titles, legends outside the data.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HALF = 3.3   # inches
FULL = 6.5   # inches

# Okabe-Ito palette (colour-blind safe)
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#999999",
}
PALETTE = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["green"],
    OKABE_ITO["orange"],
    OKABE_ITO["purple"],
    OKABE_ITO["sky"],
    OKABE_ITO["black"],
]
# markers also distinguishable in grayscale
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), "-"]


def apply():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.grid": False,
        }
    )


def new(width=HALF, height=None, ratio=0.62):
    if height is None:
        height = width * ratio
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def save(fig, path):
    fig.savefig(path, format="pdf")
    plt.close(fig)


def legend_outside(ax, loc="upper left", **kw):
    return ax.legend(loc=loc, bbox_to_anchor=(1.02, 1.0), **kw)
