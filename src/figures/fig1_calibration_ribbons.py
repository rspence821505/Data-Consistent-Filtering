"""
Figure 1 -- median cycle-wise spread--skill ratio with interquartile ranges.

Paper label: ``fig:calibration_ratio_ribbons``; file
``fig04t_calibration_collapse_ribbons.png``.  Plotting code verbatim from
``01_calibration.ipynb`` (sections 4-5, "Figure 1b").

Input: ``lorenz96/calibration_N10.npz``.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from src.figures._common import STYLE_SHEET, Report, load_lorenz96
from src.figures.style_lorenz96 import COLORS, INK, INK_SOFT, log_ticks, style_axes

OUTPUT = "fig04t_calibration_collapse_ribbons.png"
SAVE_KWARGS = dict(dpi=400, bbox_inches="tight", facecolor="white", edgecolor="none")
STYLE = STYLE_SHEET
TITLE = "Median spread-skill ratio with interquartile ranges (Lorenz-96, N = 10)"


def load(results_root):
    with load_lorenz96(results_root, "calibration_N10.npz") as data:
        filters = [str(f) for f in data["filters"]]
        rmse = {label: data["rmse"][i] for i, label in enumerate(filters)}
        spread = {label: data["spread"][i] for i, label in enumerate(filters)}
        climatology = float(data["climatology_rmse"])
        n_windows = int(data["rmse"].shape[2])
    return filters, rmse, spread, climatology, n_windows


def build(results_root, save):
    FILTERS, rmse_by_seed, spread_by_seed, CLIMATOLOGY, n_windows = load(results_root)
    ratio_by_seed = {label: spread_by_seed[label] / rmse_by_seed[label] for label in FILTERS}
    windows = np.arange(1, n_windows + 1)

    # Windows before this are treated as burn-in and excluded from any statistic
    # describing the settled regime.
    BURN_IN = 40

    # Figure 1b: the same calibration ratios as Figure 1, reduced to a median and an
    # interquartile band across the 100 seeds so all five filters fit on one axes.
    # The distributions are unimodal and near-symmetric in log (checked over five
    # filters x five windows), so five quantiles lose nothing a reader would use.
    q_ratio = {label: np.percentile(ratio_by_seed[label], [25, 50, 75], axis=0)
               for label in FILTERS}

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    fig.subplots_adjust(left=0.088, right=0.985, bottom=0.145, top=0.855)

    ax.axhline(1.0, color=INK_SOFT, ls="--", lw=1.4, zorder=3)
    ax.axvline(BURN_IN, color=INK_SOFT, ls=":", lw=1.6, zorder=3)

    for label in FILTERS:
        q25, med, q75 = q_ratio[label]
        ax.fill_between(windows, q25, q75, color=COLORS[label], alpha=0.15, lw=0, zorder=4)
    for label in FILTERS:
        ax.plot(windows, q_ratio[label][1], color=COLORS[label], lw=2.0, zorder=6,
                solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, n_windows)
    ax.set_ylim(0.05, 1.30)          # 1.0 has to be on the axis: it is the reference
    log_ticks(ax.xaxis, [1, 10, 100, 1000])
    log_ticks(ax.yaxis, [0.05, 0.1, 0.25, 0.5, 1.0])
    style_axes(ax, xlabel="Assimilation window", ylabel="Spread / RMSE",
               label_fontsize=13, tick_fontsize=11)
    ax.grid(True, alpha=0.35, lw=0.6, color="#E4E4E4")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.annotate("perfect calibration", xy=(1.25, 1.0), xytext=(0, 5),
                textcoords="offset points", ha="left", va="bottom", fontsize=10.5,
                color=INK_SOFT, zorder=8)
    ax.annotate(f"burn-in (window {BURN_IN})", xy=(BURN_IN, 0.052), xytext=(5, 0),
                textcoords="offset points",
                ha="left", va="bottom", fontsize=10.5, color=INK_SOFT, zorder=8)

    handles = [mpl.lines.Line2D([], [], color=COLORS[l], lw=2.4, label=l) for l in FILTERS]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                     ncol=len(FILTERS), frameon=False, fontsize=10.5,
                     handlelength=1.9, columnspacing=1.6, handletextpad=0.6)
    for t in leg.get_texts():
        t.set_color(INK)

    save(fig)

    # ---- numerical summary (01_calibration.ipynb, sections 4 and 5) ----------
    report = Report()
    settled = {label: ratio_by_seed[label][:, BURN_IN:] for label in FILTERS}
    settled_rmse = {label: rmse_by_seed[label][:, BURN_IN:] for label in FILTERS}
    report(f"Climatological RMSE (no-skill reference): {CLIMATOLOGY:.3f}\n")
    report(f"{'filter':<14}{'ratio, window 1':>17}{'ratio, settled':>16}"
           f"{'RMSE, settled':>15}{'vs climatology':>16}")
    for label in FILTERS:
        rmse = np.median(settled_rmse[label])
        verdict = "skill" if rmse < CLIMATOLOGY else "NO SKILL"
        report(f"{label:<14}{np.median(ratio_by_seed[label][:, 0]):>17.3f}"
               f"{np.median(settled[label]):>16.3f}{rmse:>15.3f}{verdict:>16}")
    report(f"\nsettled medians and interquartile band (windows {BURN_IN + 1}-{n_windows}):")
    for label in FILTERS:
        v = ratio_by_seed[label][:, BURN_IN:]
        a, b, c = np.percentile(v, [25, 50, 75])
        report(f"  {label:<12} {b:.3f}   IQR {a:.3f}-{c:.3f}")
    return report.text()
