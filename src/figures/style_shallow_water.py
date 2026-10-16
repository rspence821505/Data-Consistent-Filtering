"""Shared plot style for the shallow-water (idealized inlet) figures, paper
Figures 7 and 8 (``fig02_rmse_calibration_trajectories.png`` and
``fig05_metric_sensitivity.png``).

Reproduces the visual conventions of paper Figures 1 and 2
(``fig04t_calibration_collapse_ribbons.png`` and
``fig04s_postburnin_coverage_reliability.png``), which style per-axes with a
``style_axes()`` helper plus a ``COLORS`` palette (see ``style_lorenz96.py``).
This module ports those conventions so the inlet figures share the same
palette, grid, legend, and typography.

Public API:
    COLORS     -- method-to-hex mapping consistent across the paper
    INK, INK_SOFT, GUIDE -- text + reference-line ink colors
    apply_rcparams()    -- set once before drawing
    style_axes(ax, ...) -- apply the panel grid/tick/label style
    top_legend(ax, handles, labels, **kw) -- standard legend
    save(fig, path)     -- standard savefig call
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# Palette shared with the Lorenz-96 figures (style_lorenz96.COLORS).
# Only QPCA-EnDCF and 4D-LETKF appear in the two inlet figures; the
# other four slots are retained so later scripts can reuse the key.
COLORS = {
    "Seq-EnKF":    "#2E86AB",
    "Seq-ETKF":    "#C570FF",
    "4D-EnKF":     "#A23B72",
    "4D-ETKF":     "#4A3AA7",
    "4D-LETKF":    "#4A3AA7",  # reuses the 4D-ETKF slot; 4D-ETKF never
                               # appears alongside 4D-LETKF in the paper
    "QPCA-EnDCF":  "#F18F01",
}

INK      = "#1A1A1A"
INK_SOFT = "#555555"
GUIDE    = "#9AA3AB"


def apply_rcparams() -> None:
    """Install rcParams matching the mainv3.tex figure family.

    The notebook does not call `plt.rcParams.update` itself (it styles
    per-axes), but it inherits matplotlib's default sans-serif font.
    Setting bold weight here matches the heavier text visible in the
    rendered PNGs without changing font family.
    """
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset":   "dejavusans",
        "mathtext.default":   "bf",
        "font.weight":        "bold",
        "font.size":          11,
        "axes.titlesize":     13, "axes.titleweight": "bold",
        "axes.labelsize":     12, "axes.labelweight": "bold",
        "xtick.labelsize":    10, "ytick.labelsize": 10,
        "legend.fontsize":    10,
        "axes.edgecolor":     INK,
        "axes.linewidth":     0.9,
        "axes.labelcolor":    INK,
        "xtick.color":        INK,
        "ytick.color":        INK,
        "axes.axisbelow":     True,
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
        "savefig.edgecolor":  "none",
        "savefig.dpi":        400,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.constrained_layout.use": False,
        "figure.autolayout":  False,
    })


def style_axes(ax, grid_axis: str = "both") -> None:
    """Apply the mainv3.tex panel style: light grid, bold ticks, no
    top/right spines, INK-colored labels.
    """
    ax.grid(True, which="major", axis=grid_axis,
            color="#DDDDDD", alpha=0.9, lw=0.9, zorder=0)
    ax.grid(True, which="minor", axis=grid_axis,
            color="#EEEEEE", alpha=0.9, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(which="both", colors=INK, labelcolor=INK)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontweight("bold")


def panel_title(ax, letter: str, title: str, pad: int = 10) -> None:
    """Add a 'A  Reliability'-style panel title in bold, left-aligned."""
    ax.set_title(f"{letter}  {title}", loc="left",
                 fontsize=14, fontweight="bold", color=INK, pad=pad)


def top_legend(fig_or_ax, handles, labels=None,
               ncol: int | None = None, y: float = 1.0,
               **kwargs):
    """Place a horizontal, frameless legend above the plot area."""
    if labels is None:
        labels = [h.get_label() for h in handles]
    if ncol is None:
        ncol = len(handles)
    return fig_or_ax.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, y),
        ncol=ncol, frameon=False, fontsize=10.5,
        handlelength=1.9, columnspacing=1.6, handletextpad=0.6,
        **kwargs,
    )


def save(fig, path) -> None:
    """savefig with the paper-figure conventions."""
    fig.savefig(path, dpi=400, bbox_inches="tight",
                facecolor="white", edgecolor="none")
