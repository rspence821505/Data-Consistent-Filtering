"""
Plotting conventions of the Lorenz-96 figures (paper Figures 1-3), ported
verbatim from ``01_calibration.ipynb`` (cells 4 and 10).
"""

from __future__ import annotations

from matplotlib.ticker import FixedLocator, FuncFormatter

# The first three colours are shared across the paper so a filter keeps its
# identity in every figure.
COLORS = {
    "Seq-EnKF": "#2E86AB",      # professional blue
    "Seq-ETKF": "#C570FF",      # amethyst
    "4D-EnKF": "#A23B72",       # rich magenta
    "4D-ETKF": "#4A3AA7",       # violet
    "QPCA-EnDCF": "#F18F01",    # vibrant orange
}
# Text wears ink, never the series colour; a coloured mark beside it carries
# identity. That also keeps labels legible where a hue is low-contrast on white.
INK, INK_SOFT = "#1A1A1A", "#555555"

plain_ticks = FuncFormatter(lambda v, _: f"{v:g}")


def style_axes(ax, xlabel=None, ylabel=None, label_fontsize=19, tick_fontsize=15):
    """Bold axis labels, subtle grid behind the data, bold enlarged tick labels."""
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=label_fontsize, labelpad=11, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=label_fontsize, labelpad=11, color=INK)
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.9, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(tick_fontsize)
        tick.set_color(INK)


def log_ticks(axis, values):
    """Plain decimal labels on a log axis (3 rather than 3 x 10^0)."""
    axis.set_major_locator(FixedLocator(values))
    axis.set_minor_locator(FixedLocator([]))
    axis.set_major_formatter(plain_ticks)
