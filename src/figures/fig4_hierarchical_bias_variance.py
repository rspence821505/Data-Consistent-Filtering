"""
Figure 4 -- full-run bias--variance composition in the hierarchical
Lorenz-96 experiment, with its temporal stability.

Paper label: ``fig:hierarchical_bias_variance_summary``; file
``fig07j2_hierarchical_bias_variance_summary.png``.  Plotting code verbatim
from ``10_bias_variance.ipynb`` (section 2).

Input: ``lorenz96/hierarchical_bias_variance.npz``.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from src.figures._common import STYLE_SHEET, Report, load_lorenz96

OUTPUT = "fig07j2_hierarchical_bias_variance_summary.png"
STYLE = STYLE_SHEET
TITLE = "Hierarchical bias-variance composition (Lorenz-96, N = 10)"

# --- shared figure chrome of 10_bias_variance.ipynb ------------------------ #
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e3df"
FIG_RC = {
    "font.family": "serif",
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "bold",
}
# Burn-in for the trace summaries, measured from the traces (see the notebook);
# the stability traces start at the first window whose 25-window average holds
# no window before it.
SPINUP = 25

SUMMARY_SMOOTH = 25
C_BIAS, C_VAR, C_TOTAL = "#a21caf", "#008300", "#222222"
QPCA_NAME = "QPCA-EnDCF"
PAGE = "#ffffff"
SAVE_KWARGS = dict(dpi=300, facecolor=PAGE)


def load(results_root):
    with load_lorenz96(results_root, "hierarchical_bias_variance.npz") as data:
        filters = [str(f) for f in data["filters"]]
        traces = {
            name: {
                "bias_squared_trace": data["bias_squared_trace"][i],
                "variance_trace": data["variance_trace"][i],
                "mse_trace": data["mse_trace"][i],
            }
            for i, name in enumerate(filters)
        }
        meta = json.loads(str(data["meta"]))
        seeds = [int(s) for s in data["truth_seeds"]]
    return filters, traces, meta, seeds


def build(results_root, save, smooth=SUMMARY_SMOOTH):
    FILTER_NAMES, traces, meta, complete_truth_seeds = load(results_root)

    def seed_matrix(name, metric):
        """(n_seeds, n_windows). Window 0 is the initial ensemble, dropped."""
        return traces[name][metric][:, 1:]

    # Means commute with the sum, so averaging bias^2, the stored variance trace
    # and MSE across truth seeds keeps MSE = Bias^2 + Variance exactly.
    mean_trace = {}
    for name in FILTER_NAMES:
        mean_trace[name] = {
            "bias2": seed_matrix(name, "bias_squared_trace").mean(axis=0),
            "variance": seed_matrix(name, "variance_trace").mean(axis=0),
            "mse": seed_matrix(name, "mse_trace").mean(axis=0),
        }
    n_windows = mean_trace[FILTER_NAMES[0]]["mse"].size
    windows_full = np.arange(1, n_windows + 1)

    def identity_residual(traces_):
        return max(
            float(np.abs(t["mse"] - t["bias2"] - t["variance"]).max())
            for t in traces_.values()
        )

    residual_pre = identity_residual(mean_trace)

    kernel = np.ones(smooth) / smooth
    smooth_trace = {
        name: {k: np.convolve(v, kernel, mode="valid") for k, v in t.items()}
        for name, t in mean_trace.items()
    }
    residual_post = identity_residual(smooth_trace)
    assert residual_pre < 1e-9 and residual_post < 1e-9, (
        f"additive identity broken: {residual_pre:.3e} / {residual_post:.3e}"
    )

    valid = windows_full[smooth // 2: smooth // 2 + smooth_trace[FILTER_NAMES[0]]["mse"].size]
    keep = valid >= SPINUP + 1 + smooth // 2
    windows_smooth = valid[keep]
    smooth_trace = {name: {k: v[keep] for k, v in t.items()}
                    for name, t in smooth_trace.items()}

    summary = {}
    for name in FILTER_NAMES:
        t, st = mean_trace[name], smooth_trace[name]
        bias2, variance, mse = t["bias2"].mean(), t["variance"].mean(), t["mse"].mean()
        summary[name] = dict(
            bias2=bias2, variance=variance, mse=mse,
            bias_share=100 * bias2 / mse, var_share=100 * variance / mse,
            rel_range=100 * (st["mse"].max() - st["mse"].min()) / st["mse"].mean(),
        )
    baselines = [n for n in FILTER_NAMES if n != QPCA_NAME]
    ratios = {b: summary[b]["variance"] / summary[QPCA_NAME]["variance"]
              for b in baselines}
    reductions = {b: 100 * (1 - summary[QPCA_NAME]["variance"] / summary[b]["variance"])
                  for b in baselines}

    def _deepen(color, t=0.30):
        """The same hue, darkened, for a hairline that bounds a fill."""
        return tuple(v * (1 - t) for v in to_rgb(color))

    def text_color(fill):
        """Ink or white, whichever actually has more contrast against `fill`."""
        def luminance(hex_color):
            rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                   for c in rgb]
            return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

        def ratio(a, b):
            high, low = max(luminance(a), luminance(b)), min(luminance(a), luminance(b))
            return (high + 0.05) / (low + 0.05)

        return INK if ratio(fill, INK) >= ratio(fill, "#ffffff") else "#ffffff"

    # ---- layout ---------------------------------------------------------- #
    # Rows are ordered by total MSE, best first.
    order = sorted(FILTER_NAMES, key=lambda name: summary[name]["mse"])
    row_of = {name: i for i, name in enumerate(order)}

    rc = dict(FIG_RC, **{"font.weight": "bold", "axes.labelweight": "bold",
                         "axes.titleweight": "bold"})

    def _style_axes(target, grid_axis="both"):
        target.grid(False)
        target.grid(True, axis=grid_axis, alpha=0.25, linestyle="-",
                    linewidth=0.9, color="#DDDDDD")
        target.set_axisbelow(True)
        target.minorticks_off()
        for side in ("left", "bottom"):
            if target.spines[side].get_visible():
                target.spines[side].set_color(INK)
                target.spines[side].set_linewidth(1.2)
        target.tick_params(colors=INK, width=1.2, length=6, direction="out")
        for text in target.get_xticklabels() + target.get_yticklabels():
            text.set_fontweight("bold")
            text.set_color(INK)
        for label in (target.xaxis.label, target.yaxis.label):
            label.set_fontweight("bold")
            label.set_color(INK)

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(14.6, 6.4), facecolor=PAGE)
        gs = GridSpec(len(order), 2, figure=fig, width_ratios=[1.0, 1.0],
                      left=0.108, right=0.988, top=0.845, bottom=0.125,
                      wspace=0.075, hspace=0.0)

        # ---- left: full-run composition ---------------------------------- #
        ax = fig.add_subplot(gs[:, 0])
        x_top = float(np.ceil(max(s["mse"] for s in summary.values()) * 1.13 / 2) * 2)
        bar_h = 0.58
        qpca_mse = summary[QPCA_NAME]["mse"]

        ax.axvline(qpca_mse, color=INK, lw=1.0, ls=(0, (3.0, 2.4)), zorder=2,
                   alpha=0.55)

        for name in order:
            s, y = summary[name], row_of[name]
            # Variance is stacked FIRST, against the common zero, because it is
            # the component that separates the methods.
            ax.barh(y, s["variance"], height=bar_h, color=C_VAR,
                    edgecolor=_deepen(C_VAR), linewidth=0.9, zorder=3)
            ax.barh(y, s["bias2"], left=s["variance"], height=bar_h, color=C_BIAS,
                    edgecolor=_deepen(C_BIAS), linewidth=0.9, zorder=3)
            ax.plot([s["variance"]] * 2, [y - bar_h / 2, y + bar_h / 2],
                    color=PAGE, lw=1.6, zorder=5)
            ax.text(s["mse"] + 0.42, y, f"{s['mse']:.1f}", va="center", ha="left",
                    fontsize=12.5, color=C_TOTAL, zorder=6, fontweight="bold")
            for value, left, fill in ((s["variance"], 0.0, C_VAR),
                                      (s["bias2"], s["variance"], C_BIAS)):
                if value > 0.075 * x_top:  # wide enough to hold its own label
                    ax.text(left + value / 2, y, f"{value:.1f}", va="center",
                            ha="center", fontsize=11.5, fontweight="bold",
                            color=text_color(fill), zorder=6)
                else:
                    ax.text(left + value / 2, y - 0.37, f"{value:.1f}", va="top",
                            ha="center", fontsize=11.5, fontweight="bold",
                            color=INK, zorder=6)

        ax.set_yticks(range(len(order)), order, fontsize=12.5)
        for label in ax.get_yticklabels():
            if label.get_text() == QPCA_NAME:
                label.set_fontweight("bold")
            label.set_color(INK)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlim(0, x_top)
        ax.set_xticks(np.arange(0, x_top + 1, 4))
        ax.set_xlabel("Mean squared error", fontsize=12, color=INK, labelpad=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        _style_axes(ax, grid_axis="x")
        ax.tick_params(labelsize=11)
        ax.tick_params(axis="y", length=0)

        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=C_VAR, ec="none", label="Variance"),
            plt.Rectangle((0, 0), 1, 1, fc=C_BIAS, ec="none", label="Bias$^2$"),
            plt.Line2D([], [], color=C_TOTAL, lw=2.0, label="Total MSE"),
        ]

        # ---- right: the same rows, over the run --------------------------- #
        relative = {
            name: {k: 100 * (v / v.mean() - 1) for k, v in t.items()}
            for name, t in smooth_trace.items()
        }
        pooled = np.concatenate([v for t in relative.values() for v in t.values()])
        y_lim = float(np.ceil(np.abs(pooled).max() * 1.06 * 10) / 10)

        spark = []
        for name in order:
            sax = fig.add_subplot(gs[row_of[name], 1],
                                  sharey=spark[0] if spark else None)
            spark.append(sax)
            sax.axhline(0, color=INK_MUTED, lw=0.7, zorder=2)
            for key, color in (("bias2", C_BIAS), ("variance", C_VAR)):
                sax.fill_between(windows_smooth, relative[name]["mse"],
                                 relative[name][key], color=color, alpha=0.42,
                                 lw=0, zorder=2.5)
            for key, color, width, style in (
                ("bias2", C_BIAS, 1.4, "-"),
                ("variance", C_VAR, 1.4, "-"),
                ("mse", C_TOTAL, 1.6, "-"),
            ):
                sax.plot(windows_smooth, relative[name][key], color=color,
                         lw=width, ls=style,
                         zorder=3 if key != "mse" else 4)
            sax.set_xlim(windows_smooth[0], windows_smooth[-1])
            sax.set_ylim(-y_lim, y_lim)
            sax.set_yticks([-2, 0, 2] if y_lim > 2.3 else [-1, 0, 1])
            for side in ("top", "right", "left"):
                sax.spines[side].set_visible(False)
            sax.spines["bottom"].set_visible(False)
            _style_axes(sax, grid_axis="y")
            sax.grid(True, axis="y", color="#D8D8D8", alpha=1.0, linewidth=0.9,
                     linestyle="-")
            sax.tick_params(labelsize=10.5, labelbottom=False, left=False,
                            labelleft=False)

        base, head = spark[-1], spark[0]
        base.spines["bottom"].set_visible(True)
        base.spines["bottom"].set_color(INK)
        base.spines["bottom"].set_linewidth(1.2)
        base.set_xticks([200, 500, 800])
        base.tick_params(labelbottom=True, colors=INK, width=1.2, length=6)
        base.set_xlabel("Assimilation window", fontsize=12, color=INK, labelpad=7)

        head.tick_params(labelleft=True, left=True, colors=INK, width=1.2,
                         length=6)
        head.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:+.0f}%")
        )
        for text in base.get_xticklabels() + head.get_yticklabels():
            text.set_fontweight("bold")
            text.set_color(INK)

        fig.canvas.draw()
        span_left = ax.get_position().x0
        span_right = max(s.get_position().x1 for s in spark)
        legend = fig.legend(handles=handles, loc="upper center",
                            bbox_to_anchor=((span_left + span_right) / 2, 0.995),
                            ncol=3, frameon=False, fontsize=13.5,
                            handlelength=1.9, handleheight=1.1,
                            columnspacing=2.6, borderpad=0.0, handletextpad=0.7)
        for text in legend.get_texts():
            text.set_color(INK)
            text.set_fontweight("bold")

        save(fig)

    # ---- numerical summary (10_bias_variance.ipynb, section 2) --------------
    report = Report()
    report(f"truth seeds used: {len(complete_truth_seeds)}; "
           f"excluded because a filter diverged: {meta['excluded_truth_seeds']}")
    report(f"bars: mean over windows {windows_full[0]}-{windows_full[-1]};  "
           f"panels: {smooth}-window mean over windows {windows_smooth[0]}-"
           f"{windows_smooth[-1]}")
    report(f"\n{'filter':<12}{'Bias²':>8}{'Variance':>10}{'MSE':>8}{'Bias² %':>9}"
           f"{'Var %':>7}{'MSE range, % of mean':>23}")
    for name in FILTER_NAMES:
        s = summary[name]
        report(f"{name:<12}{s['bias2']:>8.2f}{s['variance']:>10.2f}{s['mse']:>8.2f}"
               f"{s['bias_share']:>9.1f}{s['var_share']:>7.1f}{s['rel_range']:>23.1f}")
    report("\nvariance of each baseline against QPCA-EnDCF")
    for b in baselines:
        report(f"  {b:<12}{ratios[b]:5.2f}x   QPCA-EnDCF is {reductions[b]:.1f}% lower")
    report("\nMSE of each baseline against QPCA-EnDCF")
    for b in baselines:
        report(f"  {b:<12}QPCA-EnDCF is "
               f"{100 * (1 - summary[QPCA_NAME]['mse'] / summary[b]['mse']):.1f}% lower")
    report(f"\nadditive identity, max |MSE - Bias² - Variance|:"
           f"  before smoothing {residual_pre:.3e},  after {residual_post:.3e}")
    gram_start = int(meta["bootstrap_spinup_windows"])
    report(f"\nfull-run trace means against the mean from window {gram_start} on:")
    for name in FILTER_NAMES:
        full = seed_matrix(name, "mse_trace").mean()
        post = np.array([traces[name]["mse_trace"][i][gram_start:].mean()
                         for i in range(len(complete_truth_seeds))]).mean()
        report(f"  {name:<12} full-run {full:6.3f}   from window {gram_start} "
               f"{post:6.3f}   difference {post - full:+.3f}")
    return report.text()
