"""
Figure 2 -- post-burn-in coverage, tail balance and persistence.

Paper label: ``fig:postburnin_coverage``; file
``fig04s_postburnin_coverage_reliability.png``.  Plotting code verbatim from
``01_calibration.ipynb`` (section 7, "Figure 3").

Inputs: ``lorenz96/coverage_ranks_N10.npz`` and, for the printed summary table,
``lorenz96/calibration_N10.npz``.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter

from src.experiments.coverage import (
    central_intervals,
    coverage_from_counts,
    outside_rate_by_window,
    tail_rates,
)
from src.figures._common import STYLE_SHEET, Report, load_lorenz96
from src.figures.style_lorenz96 import COLORS, INK, log_ticks, style_axes

OUTPUT = "fig04s_postburnin_coverage_reliability.png"
SAVE_KWARGS = dict(dpi=400, bbox_inches="tight", facecolor="white", edgecolor="none")
STYLE = STYLE_SHEET
TITLE = "Post-burn-in coverage, tail balance and persistence (Lorenz-96, N = 10)"


def load(results_root):
    with load_lorenz96(results_root, "coverage_ranks_N10.npz") as data:
        filters = [str(f) for f in data["filters"]]
        ranks = {label: data["rank_counts"][i] for i, label in enumerate(filters)}
        seeds = [int(s) for s in data["seeds"]]
        meta = json.loads(str(data["meta"]))
    with load_lorenz96(results_root, "calibration_N10.npz") as data:
        assert [str(f) for f in data["filters"]] == filters
        rmse = {label: data["rmse"][i] for i, label in enumerate(filters)}
        spread = {label: data["spread"][i] for i, label in enumerate(filters)}
        climatology = float(data["climatology_rmse"])
    return filters, ranks, seeds, meta, rmse, spread, climatology


def build(results_root, save):
    FILTERS, ranks, seeds, meta, rmse_by_seed, spread_by_seed, CLIMATOLOGY = load(results_root)
    ratio_by_seed = {label: spread_by_seed[label] / rmse_by_seed[label] for label in FILTERS}
    N_ENSEMBLE = int(meta["ensemble_size"])
    STATE_DIM = int(meta["state_dim"])
    n_windows = int(meta["n_windows"])
    window_length = int(meta["window_length"])
    windows = np.arange(1, n_windows + 1)
    BURN_IN = 40

    # ---- validate before believing anything ------------------------------
    for label in FILTERS:
        block = ranks[label]
        assert block.shape == (len(seeds), n_windows, N_ENSEMBLE + 1), (
            f"{label}: ranks are {block.shape}, expected "
            f"{(len(seeds), n_windows, N_ENSEMBLE + 1)}"
        )
        assert (block.sum(axis=2) == STATE_DIM).all(), f"{label}: a row is not {STATE_DIM}"
        assert (block >= 0).all(), f"{label}: negative counts"
    assert list(ranks) == list(FILTERS), "filter order differs from the main experiment"

    report = Report()

    # BURN_IN = 40 and the slice is [BURN_IN:], so window 41 is the first kept
    # one when windows are counted from 1.
    settled_ranks = {label: ranks[label][:, BURN_IN:, :] for label in FILTERS}
    n_settled = n_windows - BURN_IN
    ranks_per_seed = n_settled * STATE_DIM
    report(f"settled slice: windows {BURN_IN + 1}-{n_windows} "
           f"({n_settled} windows), {len(seeds)} paired seeds, "
           f"{ranks_per_seed:,} component-window ranks per seed and filter")

    # ---- the three reductions, one value per seed ------------------------
    INTERVALS = central_intervals(N_ENSEMBLE)          # (q, nominal, lo_bin, hi_bin)
    NOMINAL = np.array([nom for _, nom, _, _ in INTERVALS])

    seed_cov = {label: np.column_stack(
        [coverage_from_counts(settled_ranks[label], lo, hi) for _, _, lo, hi in INTERVALS])
        for label in FILTERS}                          # (n_seeds, n_intervals)
    seed_tail = {label: np.column_stack(tail_rates(settled_ranks[label]))
                 for label in FILTERS}                 # (n_seeds, 2) lower, upper
    seed_outside = {label: seed_tail[label].sum(axis=1) for label in FILTERS}
    outside_by_window = {label: outside_rate_by_window(ranks[label], STATE_DIM)
                         for label in FILTERS}         # (n_seeds, n_windows)

    # Every seed contributes the same number of ranks, so pooling and averaging
    # the seed-level values are the same number. Check rather than assume.
    for label in FILTERS:
        pooled = np.array([coverage_from_counts(settled_ranks[label].sum(axis=0), lo, hi)
                           for _, _, lo, hi in INTERVALS])
        assert np.allclose(pooled, seed_cov[label].mean(axis=0), atol=1e-12), label

    # ---- paired seed bootstrap -------------------------------------------
    # Ranks are dependent in space and time, so the resampling unit is a whole
    # seed. The same resampled seeds are used for every filter.
    N_BOOT = 2000
    boot_rng = np.random.default_rng(20260810)
    boot_idx = boot_rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))

    def paired_ci(per_seed):
        """95% percentile interval of the seed-mean, over the shared draws."""
        draws = per_seed[boot_idx].mean(axis=1)        # (N_BOOT, ...)
        return np.percentile(draws, [2.5, 97.5], axis=0)

    cov_est = {label: seed_cov[label].mean(axis=0) for label in FILTERS}
    cov_ci = {label: paired_ci(seed_cov[label]) for label in FILTERS}
    tail_est = {label: seed_tail[label].mean(axis=0) for label in FILTERS}
    tail_ci = {label: paired_ci(seed_tail[label]) for label in FILTERS}
    out_est = {label: float(seed_outside[label].mean()) for label in FILTERS}

    EXPECTED_TAIL = 1.0 / (N_ENSEMBLE + 1)
    EXPECTED_OUTSIDE = 2.0 / (N_ENSEMBLE + 1)

    # ---- visual hierarchy ------------------------------------------------
    LEAD = {"QPCA-EnDCF": dict(lw=3.4, ms=8.5, alpha=1.0, z=9),
            "4D-ETKF": dict(lw=2.2, ms=6.0, alpha=0.95, z=7)}
    REST = dict(lw=1.5, ms=4.6, alpha=0.6, z=5)
    style = {label: LEAD.get(label, REST) for label in FILTERS}

    GUIDE = "#9AA3AB"          # every reference mark, in one colour
    pct = FuncFormatter(lambda v, _: f"{v:g}%")

    def ideal(ax, x, y, rotation=0, ha="left", va="bottom"):
        """The one word each panel needs: where a calibrated filter would sit."""
        ax.annotate("ideal", xy=(x, y), rotation=rotation, rotation_mode="anchor",
                    ha=ha, va=va, fontsize=10.5, color=GUIDE, zorder=10)

    def rolling(y, k):
        """Centred rolling mean that shortens at the edges rather than dropping
        them. Display only -- every scalar below is computed unsmoothed."""
        c = np.concatenate([[0.0], np.cumsum(y)])
        idx = np.arange(len(y))
        lo = np.maximum(0, idx - k // 2)
        hi = np.minimum(len(y), idx + k // 2 + 1)
        return (c[hi] - c[lo]) / (hi - lo)

    SMOOTH = 25

    fig = plt.figure(figsize=(13.6, 9.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.95],
                          height_ratios=[1.0, 0.62], wspace=0.22, hspace=0.34,
                          left=0.058, right=0.985, bottom=0.075, top=0.945)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # ---- A: reliability --------------------------------------------------
    X_HI, Y_HI = 97.0, 88.0
    axA.plot([0, Y_HI], [0, Y_HI], ls=(0, (4, 3)), lw=1.2, color=GUIDE, zorder=2)
    for label in FILTERS:
        s = style[label]
        lo, hi = cov_ci[label] * 100
        est = cov_est[label] * 100
        axA.errorbar(NOMINAL * 100, est, yerr=[est - lo, hi - est],
                     color=COLORS[label], lw=s["lw"], marker="o", ms=s["ms"],
                     mec="white", mew=1.2, alpha=s["alpha"], zorder=s["z"],
                     elinewidth=1.2, capsize=2.4, capthick=1.0)
    axA.set_xlim(0, X_HI)
    axA.set_ylim(0, Y_HI)
    axA.set_aspect("equal", adjustable="box")
    axA.set_xticks(np.arange(0, 81, 20))
    axA.set_yticks(np.arange(0, 81, 20))
    axA.xaxis.set_major_formatter(pct)
    axA.yaxis.set_major_formatter(pct)
    ideal(axA, Y_HI - 5, Y_HI - 5, rotation=45, ha="right", va="bottom")
    axA.annotate("QPCA-EnDCF", xy=(NOMINAL[0] * 100 + 2.5, cov_est["QPCA-EnDCF"][0] * 100),
                 ha="left", va="center", fontsize=11.5, fontweight="bold",
                 color=COLORS["QPCA-EnDCF"], zorder=10)

    # ---- B: tail balance -------------------------------------------------
    def pale(color, t=0.74):
        """Step a hue toward the page."""
        return tuple(1 - (1 - v) * (1 - t) for v in to_rgb(color))

    def deepen(color, t=0.28):
        """The same hue, darkened, for a hairline that bounds the fill."""
        return tuple(v * (1 - t) for v in to_rgb(color))

    BAR_H = 0.62
    ypos = np.arange(len(FILTERS))[::-1]
    axB.axvspan(-EXPECTED_TAIL * 100, EXPECTED_TAIL * 100, color=GUIDE,
                alpha=0.10, lw=0, zorder=0.5)
    for y, label in zip(ypos, FILTERS):
        s = style[label]
        low, up = tail_est[label] * 100
        fill_alpha = 0.78 + 0.22 * s["alpha"]
        for sign, value in ((-1, low), (1, up)):
            allowed = min(value, EXPECTED_TAIL * 100)
            axB.barh(y, sign * allowed, height=BAR_H, color=pale(COLORS[label]),
                     lw=0, alpha=fill_alpha, zorder=s["z"])
            if value > EXPECTED_TAIL * 100:
                axB.barh(y, sign * (value - EXPECTED_TAIL * 100),
                         left=sign * EXPECTED_TAIL * 100, height=BAR_H,
                         color=COLORS[label], lw=0, alpha=fill_alpha,
                         zorder=s["z"])
                axB.plot([sign * EXPECTED_TAIL * 100] * 2,
                         [y - BAR_H / 2, y + BAR_H / 2], color="white", lw=1.5,
                         solid_capstyle="butt", zorder=s["z"] + 0.4)
            axB.barh(y, sign * value, height=BAR_H, fill=False,
                     edgecolor=deepen(COLORS[label]), linewidth=0.9,
                     alpha=fill_alpha, zorder=s["z"] + 0.6)
            axB.annotate(f"{value:.1f}%", xy=(sign * value, y), xytext=(7 * sign, 0),
                         textcoords="offset points", ha="left" if sign > 0 else "right",
                         va="center", fontsize=10.5, color=INK, zorder=10)
    for sign in (-1, 1):
        axB.axvline(sign * EXPECTED_TAIL * 100, color=GUIDE, ls=(0, (4, 3)),
                    lw=1.2, zorder=3)
    axB.axvline(0, color=INK, lw=0.9, zorder=4)
    _span = max(np.concatenate([tail_est[l] for l in FILTERS])) * 100 * 1.34
    axB.set_xlim(-_span, _span)
    axB.set_ylim(-0.62, len(FILTERS) - 0.4)
    axB.set_yticks(ypos)
    axB.set_yticklabels(FILTERS)
    axB.set_xticks([-40, -20, 0, 20, 40])
    axB.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}%"))
    ideal(axB, EXPECTED_TAIL * 100 + 3.5, len(FILTERS) - 0.5, ha="left",
          va="top")

    # ---- C: persistence --------------------------------------------------
    for label in FILTERS:
        q25, _, q75 = np.percentile(outside_by_window[label] * 100, [25, 50, 75], axis=0)
        axC.fill_between(windows, rolling(q25, SMOOTH), rolling(q75, SMOOTH),
                         color=COLORS[label], alpha=0.10, lw=0, zorder=2)
    for label in FILTERS:
        s = style[label]
        med = np.median(outside_by_window[label] * 100, axis=0)
        axC.plot(windows, rolling(med, SMOOTH), color=COLORS[label], lw=s["lw"],
                 alpha=s["alpha"], zorder=s["z"], solid_capstyle="round")
    axC.axhline(EXPECTED_OUTSIDE * 100, color=GUIDE, ls=(0, (4, 3)), lw=1.2, zorder=6)
    axC.axvline(BURN_IN, color=GUIDE, ls=":", lw=1.6, zorder=6)
    axC.set_xscale("log")
    axC.set_xlim(1, n_windows)
    axC.set_ylim(0, 100)
    log_ticks(axC.xaxis, [1, 10, 100, 1000])
    axC.set_yticks(np.arange(0, 101, 25))
    axC.yaxis.set_major_formatter(pct)
    ideal(axC, 1.3, EXPECTED_OUTSIDE * 100 + 2)
    axC.annotate("burn-in", xy=(BURN_IN, 97), xytext=(5, 0), textcoords="offset points",
                 ha="left", va="top", fontsize=10.5, color=GUIDE, zorder=10)

    # ---- shared typography ------------------------------------------------
    for ax, letter, name, xlab, ylab in (
        (axA, "A", "Reliability", "Nominal coverage", "Empirical coverage"),
        (axB, "B", "Tail balance", "◀  truth below        truth above  ▶", None),
        (axC, "C", "Persistence", "Assimilation window", "Truth outside ensemble"),
    ):
        style_axes(ax, xlabel=xlab, ylabel=ylab, label_fontsize=12.5, tick_fontsize=11)
        ax.set_title(f"{letter}   {name}", loc="left", fontsize=13.5,
                     fontweight="bold", color=INK, pad=8)
        ax.grid(True, alpha=0.35, lw=0.6, color="#E4E4E4")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    save(fig)

    # ---- printed summary --------------------------------------------------
    settled_rmse_median = {l: float(np.median(rmse_by_seed[l][:, BURN_IN:])) for l in FILTERS}
    settled_ratio_median = {l: float(np.median(ratio_by_seed[l][:, BURN_IN:])) for l in FILTERS}

    report(f"\nrank convention: zero-based, rank = members strictly below the truth, "
           f"0..{N_ENSEMBLE} over {N_ENSEMBLE + 1} bins")
    report(f"common-endpoint evaluation: every filter is scored at the same "
           f"assimilation-window endpoints (4D filters at each window end, "
           f"sequential filters at observation indices "
           f"{window_length - 1}, "
           f"{2 * window_length - 1}, ..., "
           f"{n_windows * window_length - 1})")
    report(f"settled slice: windows {BURN_IN + 1}-{n_windows} ({n_settled} windows); "
           f"{len(seeds)} paired seeds; {ranks_per_seed:,} ranks per seed and filter")
    report(f"\nouter central interval = ensemble min-max, nominal "
           f"{INTERVALS[0][1]:.1%}; calibrated outside rate {EXPECTED_OUTSIDE:.1%}\n")
    head = (f"{'filter':<12}{'cover':>7}{'95% CI':>16}{'lower':>8}{'upper':>8}"
            f"{'outside':>9}{'RMSE':>7}{'sp/RMSE':>9}  verdict")
    report(head)
    report("-" * len(head))
    for label in FILTERS:
        lo, hi = cov_ci[label][:, 0] * 100
        low, up = tail_est[label] * 100
        verdict = ("below climatology" if settled_rmse_median[label] < CLIMATOLOGY
                   else "above climatology (no skill)")
        report(f"{label:<12}{cov_est[label][0]:>6.1%} "
               f"{f'[{lo:.2f}, {hi:.2f}]%':>16}{low:>7.1f}%{up:>7.1f}%"
               f"{out_est[label]:>8.1%}{settled_rmse_median[label]:>7.2f}"
               f"{settled_ratio_median[label]:>9.3f}  {verdict}")
    _worst = max(FILTERS, key=lambda l: abs(out_est[l] - EXPECTED_OUTSIDE))
    report(f"\nfurthest from the calibrated outside rate: {_worst} "
           f"({out_est[_worst]:.1%} against {EXPECTED_OUTSIDE:.1%})")
    _asym_by = {l: abs(float(np.diff(tail_est[l])[0])) * 100 for l in FILTERS}
    _asym_worst = max(_asym_by, key=_asym_by.get)
    _ci_half = max(float(np.abs(np.diff(tail_ci[l], axis=0)).max())
                   for l in FILTERS) * 100 / 2
    report(f"tail asymmetry |upper - lower|: at most {_asym_by[_asym_worst]:.2f} "
           f"points ({_asym_worst}), so every filter's misses are two-sided")
    report(f"widest 95% tail interval: ±{_ci_half:.2f} points")
    _skilful = [l for l in FILTERS if settled_rmse_median[l] < CLIMATOLOGY]
    report(f"below the climatological RMSE ({CLIMATOLOGY:.2f}) at N = {N_ENSEMBLE}: "
           f"{', '.join(_skilful) if _skilful else 'none'} — coverage calibration "
           f"does not imply forecast skill.")
    return report.text()
