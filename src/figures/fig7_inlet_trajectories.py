"""
Figure 7 -- per-window energy-equivalent error RMSE_{E,w} and spread--skill
ratio gamma_{E,w} = sigma_{E,w} / RMSE_{E,w} for QPCA-EnDCF and 4D-LETKF.

Paper label: ``fig:inlet_scale_matching_trajectories``; file
``fig02_rmse_calibration_trajectories.png``.  Plotting code verbatim from the
script ``fig02_rmse_calibration_trajectories.py`` that
``02_shallow_water_figures.ipynb`` ran; only the input and output paths changed.

Layout: 2 rows x 2 cols
  Row 1: RMSE_{E,w} (m), log-y;  Row 2: gamma_{E,w}, log-y
  Col 1: slow storm (12 h traverse);  Col 2: fast storm (7 h traverse)
Encoding: QPCA-EnDCF orange, 4D-LETKF violet; r_loc 2 km solid, 10 km dashed;
  three-seed min-max band and three-seed arithmetic mean.
  Censoring: RMSE_E = inf/nan at a seed -> top-of-axis triangle;
             gamma = 0 (floor collapse) -> bottom-of-axis triangle.
Reference lines: physical shelf crossing (slow t ~ 411 min, fast t = 240 min),
  comparison endpoint (slow W_6 end = 420 min, fast W_4 end = 300 min), and
  sqrt(N/(N+1)) = 0.953 for N = 10.

Inputs (from the result set): the per-run JSON files
``shallow_water/runs/idealized_inlet_5filter_energy_metric_v1_sigmaE_<cell>_gh_<method>_seed<s>/result_qpca_endcf.json``
(v3 campaign, seeds {2027, 2028, 2029}) and, for the printed cross-check,
``shallow_water/scale_matching_regime_map_v3.csv``.
"""

from __future__ import annotations

import csv
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from src.figures._common import Report, shallow_water_root
from src.figures.style_shallow_water import (
    COLORS, INK, INK_SOFT, GUIDE,
    apply_rcparams, style_axes,
)

OUTPUTS = ("fig02_rmse_calibration_trajectories.pdf", "fig02_rmse_calibration_trajectories.png")
OUTPUT = OUTPUTS[-1]
SAVE_KWARGS = dict()  # the original called fig.savefig(p): dpi and colours come from rcParams
STYLE = None
# The shallow-water experiment has no Level 2: this figure is always drawn
# from the saved run outputs in results/published/shallow_water/.
PUBLISHED_ONLY = True
TITLE = "Idealized inlet: per-window RMSE_E and gamma_E trajectories"

SEEDS = (2027, 2028, 2029)
DIR_TAG = "sigmaE"

# (cell_tag, r_loc_m, n_windows, comparison-window index 1-based)
SLOW = [
    ("capstone60",     2000,  11, 6),
    ("cap60_L10km",   10000,  11, 6),
]
FAST = [
    ("cap60cmp",       2000,   6, 4),
    ("cap60cmp_L10km",10000,   6, 4),
]
SHELF_T_MIN   = {"slow": 6.857 * 60.0, "fast": 4.0 * 60.0}  # 411.4 / 240.0
COMPARE_T_MIN = {"slow": 420.0,        "fast": 300.0}        # end of W_6 / W_4
XMAX_T_MIN    = {"slow": 480.0,        "fast": 360.0}        # one window past W_cmp

# Palette: shared with the main-paper figure family.
QPCA_C  = COLORS["QPCA-EnDCF"]  # #F18F01 vibrant orange
LETKF_C = COLORS["4D-LETKF"]    # #4A3AA7 violet (shared 4D- family hue)

# Censoring sentinels for display
GAMMA_FLOOR = 1e-4
RMSE_TOP    = 1e35


# --- Data loading -----------------------------------------------------------
def _to_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if v != v or not math.isfinite(v):
        return float("inf") if (v == v and v > 0) else float("nan")
    return v


def _load_trajectories(results_root, cell: str, method: str) -> dict:
    """Return per-seed series for a (cell, method) pair."""
    rmse, gE, gC, nfail = [], [], [], []
    times = None
    for s in SEEDS:
        p = (shallow_water_root(results_root) / "runs" /
             f"idealized_inlet_5filter_energy_metric_v1_{DIR_TAG}_{cell}"
             f"_gh_{method}_seed{s}" / "result_qpca_endcf.json")
        d = json.load(open(p))
        rows = d["windows"]
        r, ge, gc, nf, t = [], [], [], [], []
        for w in rows:
            r.append(_to_float(w.get("analysis_rmse_energy_equiv_m")))
            ge.append(_to_float(w.get("analysis_spread_skill_energy_equiv")))
            gc.append(_to_float(w.get("analysis_spread_skill_truth")))
            nf.append(int(w.get("n_failed_members", 0)))
            t.append(_to_float(w.get("t_window_end_s")) / 60.0)
        rmse.append(r); gE.append(ge); gC.append(gc); nfail.append(nf)
        if times is None:
            times = np.asarray(t, dtype=float)
    return dict(
        times_min=times,
        rmse_E=np.asarray(rmse, dtype=float),
        gamma_E=np.asarray(gE, dtype=float),
        gamma_coef=np.asarray(gC, dtype=float),
        n_failed=np.asarray(nfail, dtype=int),
        seeds=SEEDS,
    )


def _mean_strict(x_row: np.ndarray) -> float:
    """Strict arithmetic mean: NaN if any seed is non-finite in that window."""
    finite = x_row[np.isfinite(x_row)]
    if finite.size != x_row.size or x_row.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _minmax_strict(x_row: np.ndarray):
    """Strict min/max band: NaN pair if any seed is non-finite."""
    if x_row.size == 0 or np.any(~np.isfinite(x_row)):
        return float("nan"), float("nan")
    return float(np.min(x_row)), float(np.max(x_row))


def _gamma_mean(x_row: np.ndarray) -> float:
    """Mean of gamma; 0.0 treated as floor-collapse display value."""
    if np.any(np.isnan(x_row)):
        return float("nan")
    clean = np.where(x_row <= 0.0, GAMMA_FLOOR, x_row)
    return float(np.mean(clean))


def _gamma_minmax(x_row: np.ndarray):
    """Strict min/max band for gamma; NaN pair if any seed is NaN;
    zero (floor collapse) replaced by GAMMA_FLOOR for display."""
    if np.any(np.isnan(x_row)):
        return float("nan"), float("nan")
    clean = np.where(x_row <= 0.0, GAMMA_FLOOR, x_row)
    return float(np.min(clean)), float(np.max(clean))


# --- Plotting helpers ------------------------------------------------------
def _plot_cell_rmse(ax, series_q, series_l, r_loc_m: int, times: np.ndarray):
    ls = "-" if r_loc_m == 2000 else "--"
    r_loc_km = r_loc_m // 1000

    # 3-seed min/max envelope (shaded band)
    q_lo = np.array([_minmax_strict(series_q["rmse_E"][:, j])[0]
                     for j in range(times.size)])
    q_hi = np.array([_minmax_strict(series_q["rmse_E"][:, j])[1]
                     for j in range(times.size)])
    l_lo = np.array([_minmax_strict(series_l["rmse_E"][:, j])[0]
                     for j in range(times.size)])
    l_hi = np.array([_minmax_strict(series_l["rmse_E"][:, j])[1]
                     for j in range(times.size)])
    band_mask_q = np.isfinite(q_lo) & np.isfinite(q_hi)
    band_mask_l = np.isfinite(l_lo) & np.isfinite(l_hi)
    ax.fill_between(times, q_lo, q_hi, where=band_mask_q,
                    color=QPCA_C, alpha=0.18, linewidth=0, zorder=2)
    ax.fill_between(times, l_lo, l_hi, where=band_mask_l,
                    color=LETKF_C, alpha=0.18, linewidth=0, zorder=2)

    # Thick three-seed means (strict)
    q_mean = np.array([_mean_strict(series_q["rmse_E"][:, j])
                       for j in range(times.size)])
    l_mean = np.array([_mean_strict(series_l["rmse_E"][:, j])
                       for j in range(times.size)])
    ax.plot(times, np.where(np.isfinite(q_mean), q_mean, np.nan),
            color=QPCA_C, linestyle=ls, linewidth=1.8, marker="o",
            markersize=3.5, markeredgecolor="white", markeredgewidth=0.3,
            zorder=4,
            label=f"QPCA-EnDCF, $r_\\mathrm{{loc}}={r_loc_km}$ km")
    ax.plot(times, np.where(np.isfinite(l_mean), l_mean, np.nan),
            color=LETKF_C, linestyle=ls, linewidth=1.8, marker="s",
            markersize=3.5, markeredgecolor="white", markeredgewidth=0.3,
            zorder=4,
            label=f"4D-LETKF, $r_\\mathrm{{loc}}={r_loc_km}$ km")

    # Overflow markers
    for si in range(len(SEEDS)):
        for j in range(times.size):
            if not np.isfinite(series_q["rmse_E"][si, j]):
                ax.plot(times[j], RMSE_TOP, marker="^", color=QPCA_C,
                        markersize=6, markeredgecolor="black",
                        markeredgewidth=0.3, linestyle="None",
                        alpha=0.85, zorder=5)
            if not np.isfinite(series_l["rmse_E"][si, j]):
                ax.plot(times[j], RMSE_TOP, marker="^", color=LETKF_C,
                        markersize=6, markeredgecolor="black",
                        markeredgewidth=0.3, linestyle="None",
                        alpha=0.85, zorder=5)


def _plot_cell_gamma(ax, series_q, series_l, r_loc_m: int, times: np.ndarray):
    ls = "-" if r_loc_m == 2000 else "--"

    q_lo = np.array([_gamma_minmax(series_q["gamma_E"][:, j])[0]
                     for j in range(times.size)])
    q_hi = np.array([_gamma_minmax(series_q["gamma_E"][:, j])[1]
                     for j in range(times.size)])
    l_lo = np.array([_gamma_minmax(series_l["gamma_E"][:, j])[0]
                     for j in range(times.size)])
    l_hi = np.array([_gamma_minmax(series_l["gamma_E"][:, j])[1]
                     for j in range(times.size)])
    band_mask_q = np.isfinite(q_lo) & np.isfinite(q_hi)
    band_mask_l = np.isfinite(l_lo) & np.isfinite(l_hi)
    ax.fill_between(times, q_lo, q_hi, where=band_mask_q,
                    color=QPCA_C, alpha=0.18, linewidth=0, zorder=2)
    ax.fill_between(times, l_lo, l_hi, where=band_mask_l,
                    color=LETKF_C, alpha=0.18, linewidth=0, zorder=2)

    q_mean = np.array([_gamma_mean(series_q["gamma_E"][:, j])
                       for j in range(times.size)])
    l_mean = np.array([_gamma_mean(series_l["gamma_E"][:, j])
                       for j in range(times.size)])
    ax.plot(times, np.where(np.isfinite(q_mean), q_mean, np.nan),
            color=QPCA_C, linestyle=ls, linewidth=1.8, marker="o",
            markersize=3.5, markeredgecolor="white", markeredgewidth=0.3,
            zorder=4)
    ax.plot(times, np.where(np.isfinite(l_mean), l_mean, np.nan),
            color=LETKF_C, linestyle=ls, linewidth=1.8, marker="s",
            markersize=3.5, markeredgecolor="white", markeredgewidth=0.3,
            zorder=4)

    for si in range(len(SEEDS)):
        for j in range(times.size):
            if series_q["gamma_E"][si, j] == 0.0:
                ax.plot(times[j], GAMMA_FLOOR, marker="v", color=QPCA_C,
                        markersize=5, markeredgecolor="black",
                        markeredgewidth=0.3, linestyle="None",
                        alpha=0.75, zorder=5)
            if series_l["gamma_E"][si, j] == 0.0:
                ax.plot(times[j], GAMMA_FLOOR, marker="v", color=LETKF_C,
                        markersize=5, markeredgecolor="black",
                        markeredgewidth=0.3, linestyle="None",
                        alpha=0.75, zorder=5)


_PANEL_TITLES = {
    "a": "Slow storm, 12 h traverse, RMSE$_{E,w}$ (m)",
    "b": "Fast storm, 7 h traverse, RMSE$_{E,w}$ (m)",
    "c": "Slow storm, $\\gamma_{E,w}$",
    "d": "Fast storm, $\\gamma_{E,w}$",
}


def _panel_letter(ax, letter):
    """Letter + descriptor title above the panel."""
    ax.set_title(f"{letter.upper()}  {_PANEL_TITLES[letter]}",
                 loc="left", fontsize=12, fontweight="bold",
                 color=INK, pad=8)


# --- Build the figure -------------------------------------------------------
def build_figure(results_root):
    fig = plt.figure(figsize=(11.0, 7.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.10,
                          left=0.085, right=0.985,
                          top=0.90, bottom=0.17)
    axSR = fig.add_subplot(gs[0, 0]); axFR = fig.add_subplot(gs[0, 1])
    axSG = fig.add_subplot(gs[1, 0]); axFG = fig.add_subplot(gs[1, 1])

    # Upper (RMSE_E) axes -- y-range sized for pre-shelf + one post-W_cmp
    # window (W_7 slow / W_5 fast).
    for ax in (axSR, axFR):
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e4)
        style_axes(ax)
        ax.tick_params(axis="x", labelbottom=False)

    # Lower (gamma_E) axes
    gamma_ref = math.sqrt(10 / 11)   # 0.9535
    for ax in (axSG, axFG):
        ax.set_yscale("log")
        ax.set_ylim(GAMMA_FLOOR / 3, 10)
        ax.axhline(gamma_ref, color=GUIDE, linestyle="--",
                   linewidth=1.2, alpha=1.0, zorder=3,
                   label=r"leading-order calibration $\sqrt{N/(N+1)}=0.953$")
        style_axes(ax)
        ax.set_xlabel("Simulation time (min, incl. 60-min ramp)")

    axFR.tick_params(axis="y", labelleft=False)
    axFG.tick_params(axis="y", labelleft=False)
    axSR.set_ylabel(r"$\mathrm{RMSE}_{E,w}$  (m)")
    axSG.set_ylabel(r"$\gamma_{E,w}=\sigma_{E,w}/\mathrm{RMSE}_{E,w}$")

    # Load + plot the slow + fast columns
    for cfg_list, axR, axG, side in (
        (SLOW, axSR, axSG, "slow"),
        (FAST, axFR, axFG, "fast"),
    ):
        for (cell, r_loc_m, _n_w, _w_cmp) in cfg_list:
            sq = _load_trajectories(results_root, cell, "qpca")
            sl = _load_trajectories(results_root, cell, "letkf")
            assert np.allclose(sq["times_min"], sl["times_min"]), \
                f"time grids disagree for {cell}"
            _plot_cell_rmse(axR, sq, sl, r_loc_m, sq["times_min"])
            _plot_cell_gamma(axG, sq, sl, r_loc_m, sq["times_min"])

        t_shelf   = SHELF_T_MIN[side]
        t_compare = COMPARE_T_MIN[side]
        for ax in (axR, axG):
            ax.axvline(t_shelf, color=GUIDE, linestyle="-.",
                       linewidth=1.3, alpha=1.0, zorder=3,
                       label="physical shelf crossing")
            ax.axvline(t_compare, color=INK_SOFT, linestyle=":",
                       linewidth=1.3, alpha=1.0, zorder=3,
                       label=r"comparison endpoint ($W_6$ slow / $W_4$ fast)")
        # Inline event labels in ink grey at the top edge.
        trans = axR.get_xaxis_transform()
        axR.annotate("shelf", xy=(t_shelf, 0.97), xycoords=trans,
                     xytext=(-4, -2), textcoords="offset points",
                     ha="right", va="top", fontsize=10,
                     fontweight="bold", color=INK_SOFT,
                     zorder=11)
        axR.annotate(f"$W_{{{int(cfg_list[0][3])}}}$",
                     xy=(t_compare, 0.97), xycoords=trans,
                     xytext=(4, -2), textcoords="offset points",
                     ha="left", va="top", fontsize=10,
                     fontweight="bold", color=INK_SOFT,
                     zorder=11)

    # Shared x-axis range per column: start at the first window end and
    # cap one window past W_cmp (W_7 slow / W_5 fast).
    for col_cfg, axR, axG, side in ((SLOW, axSR, axSG, "slow"),
                                      (FAST, axFR, axFG, "fast")):
        tmin = min(_load_trajectories(results_root, c[0], "qpca")["times_min"][0]
                   for c in col_cfg)
        tmax = XMAX_T_MIN[side]
        span = tmax - tmin
        for ax in (axR, axG):
            ax.set_xlim(tmin - 0.03 * span, tmax + 0.03 * span)

    # Panel letters
    for ax, letter in ((axSR, "a"), (axFR, "b"), (axSG, "c"), (axFG, "d")):
        _panel_letter(ax, letter)

    # Top-centered horizontal legend: series on top, reference lines beneath.
    series_handles = []
    for h, l in zip(*axSR.get_legend_handles_labels()):
        if l.startswith(("QPCA", "4D-LETKF")):
            series_handles.append(h)
    event_handles = []
    event_labels = []
    for h, l in zip(*axSR.get_legend_handles_labels()):
        if l.startswith(("physical shelf", "comparison endpoint")):
            event_handles.append(h); event_labels.append(l)
    for h, l in zip(*axSG.get_legend_handles_labels()):
        if "leading-order" in l:
            event_handles.append(h); event_labels.append(l)

    leg1 = fig.legend(
        series_handles, [h.get_label() for h in series_handles],
        loc="upper center", bbox_to_anchor=(0.5, 1.00),
        ncol=4, frameon=False, fontsize=10.5,
        handlelength=2.0, columnspacing=1.6, handletextpad=0.6,
    )
    leg2 = fig.legend(
        event_handles, event_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.045),
        ncol=3, frameon=False, fontsize=9.5,
        handlelength=2.4, columnspacing=1.8, handletextpad=0.6,
    )
    fig.add_artist(leg1); fig.add_artist(leg2)

    return fig


def build(results_root, save):
    # Install the shared rcParams (bold sans-serif, white facecolor, ink colours).
    apply_rcparams()
    fig = build_figure(results_root)
    save(fig)

    report = Report()
    rmap_csv = shallow_water_root(results_root) / "scale_matching_regime_map_v3.csv"
    if rmap_csv.is_file():
        rows = {(r["cell"], r["method"]): r
                for r in csv.DictReader(rmap_csv.open())}
        report("[validation] shelf-window three-seed means (regime-map CSV, "
               f"seeds={SEEDS}):")
        for (cell, _r, _n, w_cmp) in SLOW + FAST:
            q = float(rows[(cell, "qpca")]["mean_rmse_E_m"])
            l = float(rows[(cell, "letkf")]["mean_rmse_E_m"])
            sE = float(rows[(cell, "qpca")]["mean_sigma_E_m"])
            gE = float(rows[(cell, "qpca")]["mean_gamma_E"])
            gL = float(rows[(cell, "letkf")]["mean_gamma_E"])
            report(f"  {cell:<16} w={w_cmp}  QPCA={q:>9.4f}  LETKF={l:>9.4f}  "
                   f"R_E={q/l:>8.3f}  QPCA sigma_E={sE:>8.3f}  gamma_E QPCA={gE:>6.3f}"
                   f"  LETKF={gL:>6.3f}")
    return report.text()
