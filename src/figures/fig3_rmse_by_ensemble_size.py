"""
Figure 3 -- settled analysis RMSE as a function of ensemble size.

Paper label: ``fig:calibration_ensemble_size``; file
``fig04p_rmse_by_ensemble_size.png``.  Plotting code verbatim from
``01_calibration.ipynb`` (section 6, "Figure 2").

Input: ``lorenz96/ensemble_size_sweep.npz``.
"""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.figures._common import STYLE_SHEET, Report, load_lorenz96
from src.figures.style_lorenz96 import (
    COLORS,
    INK,
    INK_SOFT,
    log_ticks,
    plain_ticks,
    style_axes,
)

OUTPUT = "fig04p_rmse_by_ensemble_size.png"
SAVE_KWARGS = dict(dpi=400, bbox_inches="tight", facecolor="white", edgecolor="none")
STYLE = STYLE_SHEET
TITLE = "Settled analysis RMSE versus ensemble size (Lorenz-96)"


def load(results_root):
    with load_lorenz96(results_root, "ensemble_size_sweep.npz") as data:
        filters = [str(f) for f in data["filters"]]
        sizes = [int(n) for n in data["ensemble_sizes"]]
        settled = data["settled_median_rmse"]
        per_seed = data["per_seed_settled_median_rmse"]
        n_seeds = int(data["seeds"].size)
        meta = json.loads(str(data["meta"]))
        climatology = float(data["climatology_rmse"])
    ens_settled = {(label, n): float(settled[i, j])
                   for i, label in enumerate(filters) for j, n in enumerate(sizes)}
    ens_per_seed = {(label, n): per_seed[i, j]
                    for i, label in enumerate(filters) for j, n in enumerate(sizes)}
    return filters, sizes, ens_settled, ens_per_seed, n_seeds, climatology, meta


def build(results_root, save):
    (FILTERS, ENSEMBLE_SIZES, ens_settled, ens_per_seed, n_seeds, CLIMATOLOGY,
     meta) = load(results_root)
    BURN_IN = int(meta["burn_in"])
    n_windows = int(meta["n_windows"])

    # --- the reductions the figure draws -------------------------------------
    # The point is the settled median the table below prints; the interval is
    # over the 100 per-seed settled medians -- one number per seed, so it is
    # seed-to-seed disagreement rather than 96,000 pseudo-independent samples.
    ens_q = {k: np.percentile(v, [25, 75]) for k, v in ens_per_seed.items()}
    # The point and the interval are different statistics, so check the point
    # lies inside its interval rather than assume it.
    for k, (q25, q75) in ens_q.items():
        assert q25 <= ens_settled[k] <= q75, f"{k}: settled median outside the seed IQR"

    # Everything the figure states is derived here, never typed in.
    N0, NL = ENSEMBLE_SIZES[0], ENSEMBLE_SIZES[-1]
    ens_curve = {label: np.array([ens_settled[(label, n)] for n in ENSEMBLE_SIZES])
                 for label in FILTERS}
    PLATEAU = ens_settled[("QPCA-EnDCF", NL)]

    # Panel C compares QPCA-EnDCF against whichever conventional filter is best
    # at each N. That is Seq-ETKF at every size here, so the panel can name it;
    # the assert is what makes naming it safe rather than assumed.
    OTHERS = [l for l in FILTERS if l != "QPCA-EnDCF"]
    best_other = {n: min(OTHERS, key=lambda l: ens_settled[(l, n)]) for n in ENSEMBLE_SIZES}
    assert len(set(best_other.values())) == 1, f"best conventional filter varies: {best_other}"
    RIVAL = best_other[N0]
    margin = ens_curve[RIVAL] - ens_curve["QPCA-EnDCF"]   # > 0 means QPCA-EnDCF ahead
    peak_i = int(np.argmax(margin))
    # where the margin changes sign, by linear interpolation between sizes
    cross_i = next((i for i in range(len(margin) - 1) if margin[i] >= 0 > margin[i + 1]), None)
    cross_n = (ENSEMBLE_SIZES[cross_i] + (ENSEMBLE_SIZES[cross_i + 1] - ENSEMBLE_SIZES[cross_i])
               * margin[cross_i] / (margin[cross_i] - margin[cross_i + 1])) if cross_i is not None else None

    lead_at_N0 = margin[0]
    lead_pct = 100 * lead_at_N0 / ens_settled[(RIVAL, N0)]
    banked = ((ens_settled[("QPCA-EnDCF", N0)] - ens_settled[("QPCA-EnDCF", ENSEMBLE_SIZES[1])])
              / (ens_settled[("QPCA-EnDCF", N0)] - PLATEAU))
    overtake_n = next((n for n in ENSEMBLE_SIZES
                       if any(ens_settled[(l, n)] < ens_settled[("QPCA-EnDCF", n)]
                              for l in OTHERS)), None)
    overtaker = min(OTHERS, key=lambda l: ens_settled[(l, overtake_n)])
    # panel B: the RMSE bought per added member, on each step of the sweep
    ens_steps = list(zip(ENSEMBLE_SIZES, ENSEMBLE_SIZES[1:]))
    ens_gain = {label: np.array([(ens_settled[(label, a)] - ens_settled[(label, b)]) / (b - a)
                                 for a, b in ens_steps]) for label in FILTERS}
    ratio_last = ens_gain[overtaker][-1] / ens_gain["QPCA-EnDCF"][-1]

    # --- figure-local conventions --------------------------------------------
    FIGURE_RC = {
        "font.family": "serif",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }
    PANEL_LABEL_FS, PANEL_TICK_FS = 11.0, 10.0
    LEAD = ("QPCA-EnDCF", RIVAL)
    LINE_KW = dict(lw=2.2, zorder=9)

    def panel(ax, ylabel):
        """One visual grammar for all three panels."""
        style_axes(ax, ylabel=ylabel, label_fontsize=PANEL_LABEL_FS,
                   tick_fontsize=PANEL_TICK_FS)

    with plt.rc_context(FIGURE_RC):
        fig = plt.figure(figsize=(9.6, 3.95))
        gs = fig.add_gridspec(1, 3, wspace=0.46, left=0.098, right=0.988,
                              bottom=0.215, top=0.885)
        axA = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1], sharex=axA)
        axC = fig.add_subplot(gs[2], sharex=axA)

        # ---- A: where each filter lands -------------------------------------
        panel(axA, "Settled median RMSE")
        y_hi = max(ens_q[k][1] for k in ens_q) + 0.10
        y_lo = min(ens_q[k][0] for k in ens_q) - 0.10

        for label in LEAD:
            q = np.array([ens_q[(label, n)] for n in ENSEMBLE_SIZES]).T
            axA.fill_between(ENSEMBLE_SIZES, q[0], q[1], color=COLORS[label],
                             alpha=0.16, lw=0, zorder=2)
        for label in FILTERS:
            axA.plot(ENSEMBLE_SIZES, ens_curve[label], color=COLORS[label],
                     marker="o", ms=5.0, mec="white", mew=1.3,
                     solid_capstyle="round", **LINE_KW)
        if cross_n is not None:
            axA.axvline(cross_n, color=INK_SOFT, lw=0.9, ls=(0, (3, 3)), zorder=3)
        axA.set_ylim(y_lo, y_hi)
        axA.set_yticks(np.arange(3.5, 5.3, 0.5))

        # ---- B: what the next member buys -----------------------------------
        panel(axB, "Δ RMSE per member")
        # Each rate belongs to an interval of N, not to a point, so it spans that
        # interval; a marker at the midpoint would attach it to a size it was
        # never measured at.
        edges = np.array(ENSEMBLE_SIZES, dtype=float)
        step_x = np.repeat(edges, 2)[1:-1]
        b_lo = min(g.min() for g in ens_gain.values()) * 0.70
        b_hi = max(g.max() for g in ens_gain.values()) * 1.45
        for label in FILTERS:
            y = np.repeat(ens_gain[label], 2)
            for glow_lw, glow_alpha in ((7.5, 0.10), (4.6, 0.16)):
                axB.plot(step_x, y, color=COLORS[label], lw=glow_lw,
                         alpha=glow_alpha, solid_capstyle="butt",
                         solid_joinstyle="miter", zorder=6)
            axB.plot(step_x, y, color=COLORS[label],
                     solid_capstyle="butt", solid_joinstyle="miter",
                     path_effects=[pe.Stroke(linewidth=3.6, foreground="white"),
                                   pe.Normal()], **LINE_KW)
        axB.set_yscale("log")
        axB.set_ylim(b_lo, b_hi)
        log_ticks(axB.yaxis, [0.003, 0.01, 0.03, 0.1])

        # ---- C: the net trade -----------------------------------------------
        panel(axC, f"RMSE lead over {RIVAL}")
        # Sign is carried by the fill: whichever filter is ahead tints its side.
        axC.fill_between(ENSEMBLE_SIZES, 0, margin, where=margin >= 0,
                         color=COLORS["QPCA-EnDCF"], alpha=0.20, lw=0,
                         interpolate=True, zorder=2)
        axC.fill_between(ENSEMBLE_SIZES, 0, margin, where=margin <= 0,
                         color=COLORS[RIVAL], alpha=0.20, lw=0,
                         interpolate=True, zorder=2)
        axC.axhline(0, color=INK_SOFT, lw=1.0, zorder=4)
        axC.plot(ENSEMBLE_SIZES, margin, color=INK, lw=1.8, marker="o", ms=4.6,
                 mec="white", mew=1.3, zorder=9, solid_capstyle="round")
        if cross_n is not None:
            axC.axvline(cross_n, color=INK_SOFT, lw=0.9, ls=(0, (3, 3)), zorder=3)
        c_hi = margin.max() * 1.30
        c_lo = min(margin.min() * 1.55, -0.10)
        axC.set_ylim(c_lo, c_hi)
        axC.set_yticks([-0.5, 0.0, 0.5, 1.0])
        axC.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: (f"{v:+g}".replace("-", "−")) if v else "0"))
        # The two numbers the other panels cannot show: the peak of the arc and
        # where it runs out.
        axC.annotate(f"{margin[peak_i]:+.2f}",
                     xy=(ENSEMBLE_SIZES[peak_i], margin[peak_i]), xytext=(0, 9),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=10.5, fontweight="bold", color=INK, zorder=12)
        if cross_n is not None:
            axC.annotate(f"N ≈ {cross_n:.0f}", xy=(cross_n, c_lo), xytext=(4, 5),
                         textcoords="offset points", ha="left", va="bottom",
                         fontsize=9.5, color=INK_SOFT, zorder=12)

        # ---- shared x, one legend, no figure title --------------------------
        for ax in (axA, axB, axC):
            ax.set_xlim(N0 - 1.4, NL + 1.4)
            ax.set_xticks(ENSEMBLE_SIZES)
            ax.xaxis.set_major_formatter(plain_ticks)
            ax.minorticks_off()
            # Re-applied once every locator and formatter is final: `style_axes`
            # styles the tick labels that exist when it is called, and each panel
            # replaces its own after `panel` has run.
            style_axes(ax, tick_fontsize=PANEL_TICK_FS)
        fig.supxlabel("Ensemble size   N", fontsize=PANEL_LABEL_FS, fontweight="bold",
                      color=INK, y=0.045)

        handles = [mpl.lines.Line2D([], [], color=COLORS[l], lw=2.2, marker="o",
                                    ms=5.0, mec="white", mew=1.3, label=l)
                   for l in FILTERS]
        leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
                         ncol=len(FILTERS), frameon=False, fontsize=10.5,
                         handlelength=1.9, columnspacing=1.5, handletextpad=0.6)
        for text in leg.get_texts():
            text.set_color(INK)
            text.set_fontweight("bold")

        save(fig)

    # ---- numerical summary (01_calibration.ipynb, section 6) -----------------
    report = Report()
    report(f"settled median analysis RMSE  (windows {BURN_IN + 1}-{n_windows}, "
           f"{n_seeds} seeds; climatology {CLIMATOLOGY:.2f})")
    report(f"{'filter':<14}" + "".join(f"{('N=' + str(n)):>8}" for n in ENSEMBLE_SIZES))
    for label in FILTERS:
        report(f"{label:<14}"
               + "".join(f"{ens_settled[(label, n)]:>8.2f}" for n in ENSEMBLE_SIZES))
    report("\nRMSE bought per added member, by step:")
    report(f"{'filter':<14}" + "".join(f"{f'{a}->{b}':>10}" for a, b in ens_steps))
    for label in FILTERS:
        report(f"{label:<14}" + "".join(f"{v:>10.4f}" for v in ens_gain[label]))
    report("\nfilters below climatology, by ensemble size:")
    for n in ENSEMBLE_SIZES:
        below = [l for l in FILTERS if ens_settled[(l, n)] < CLIMATOLOGY]
        report(f"  N = {n:<3} {', '.join(below) if below else 'none'}")
    report("\nwhat the figure is claiming, in numbers:")
    report(f"  QPCA-EnDCF lead at N = {N0:<12} {lead_at_N0:.2f} RMSE "
           f"({lead_pct:.1f}% lower) over {RIVAL}")
    report(f"  banked by N = {ENSEMBLE_SIZES[1]:<19} {banked:.0%} of its whole "
           f"N = {N0} -> {NL} gain")
    report(f"  QPCA-EnDCF plateau{'':<15} {PLATEAU:.3f}  "
           f"({100 * (CLIMATOLOGY - PLATEAU) / CLIMATOLOGY:.1f}% below climatology)")
    report(f"  peak advantage{'':<19} {margin[peak_i]:+.2f} RMSE at "
           f"N = {ENSEMBLE_SIZES[peak_i]}")
    report(f"  advantage runs out{'':<15} N = {cross_n:.1f} "
           f"(first overtaken at N = {overtake_n} by {overtaker})")
    report(f"  final-step return ratio{'':<10} {ratio_last:.0f}x "
           f"({overtaker} {ens_gain[overtaker][-1]:.4f} vs "
           f"QPCA-EnDCF {ens_gain['QPCA-EnDCF'][-1]:.4f})")
    report(f"  the other four, per-member{'':<7} "
           f"{min(g.min() for l, g in ens_gain.items() if l in OTHERS):.3f}-"
           f"{max(g.max() for l, g in ens_gain.items() if l in OTHERS):.3f} RMSE, "
           f"flat across the sweep")
    return report.text()
