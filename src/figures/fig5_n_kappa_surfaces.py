"""
Figure 5 -- bias--variance response to ensemble size N and retained rank kappa
under independent restarts.

Paper label: ``fig:n_kappa_bias_variance_surfaces``; file
``fig07p_n_kappa_bias_variance_surfaces.png``.  Plotting code verbatim from
``10_bias_variance.ipynb`` (section 4.2); the surface is the bootstrap median,
as ``n_kappa_bias_variance.surface`` computes it.

Input: ``lorenz96/n_kappa_bootstrap.npz``.
"""

from __future__ import annotations

import json

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from src.figures._common import STYLE_SHEET, Report, load_lorenz96

OUTPUT = "fig07p_n_kappa_bias_variance_surfaces.png"
SAVE_KWARGS = dict(dpi=300, bbox_inches="tight", facecolor="white")
STYLE = STYLE_SHEET
TITLE = "Bias-variance response to N and kappa (Lorenz-96, independent restarts)"

INK, INK_MUTED = "#0b0b0b", "#52514e"
FIG_RC = {
    "font.family": "serif",
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "bold",
}

# metric, panel title, colormap, colour-scale bounds, tick step. The bounds are
# fixed rather than taken from each surface's min and max, so every colourbar is
# labelled in whole numbers. Where a bound sits inside the data the affected
# cells are clamped to the end colour, and the summary prints exactly which ones.
NK_SURFACE_COLUMNS = (
    ("bias_squared_debiased", "Bias$^2$ (debiased)", "BuPu", (12.0, 16.0), 1.0),
    ("variance", "Variance across realizations", "YlOrRd", (1.0, 7.0), 1.0),
    ("total", "Total  $T=$ Bias$^2_{\\mathrm{db}}+$Variance", "YlGnBu",
     (15.0, 20.0), 1.0),
)
NK_LABEL_FS, NK_TICK_FS = 12.0, 11.0
NK_TILE, NK_ROUND = 0.86, 0.11
NK_CBAR_Y, NK_CBAR_H, NK_CBAR_TARGET_TICKS = 0.075, 0.042, 6


def load(results_root):
    with load_lorenz96(results_root, "n_kappa_bootstrap.npz") as data:
        sizes = [int(N) for N in data["ensemble_sizes"]]
        kappas = [int(k) for k in data["kappa_values"]]
        design_points = [(int(N), int(k)) for N, k in data["design_points"]]
        metric_names = [str(m) for m in data["metric_names"]]
        draws = {point: data["restart_draws"][i] for i, point in enumerate(design_points)}
        n_seeds = int(data["truth_seeds"].size)
        meta = json.loads(str(data["meta"]))
    return sizes, kappas, design_points, metric_names, draws, n_seeds, meta


def surface(draws, metric_names, sizes, kappas, metric):
    """Bootstrap-median surface; cells excluded by kappa <= N - 1 are NaN."""
    column = metric_names.index(metric)
    grid = np.full((len(sizes), len(kappas)), np.nan)
    for i, N in enumerate(sizes):
        for j, kappa in enumerate(kappas):
            if (N, kappa) not in draws:
                continue
            grid[i, j] = float(np.median(draws[(N, kappa)][:, column]))
    return grid


def _tile_cmap(name, start=0.14):
    """Sequential map with its palest end trimmed, and white for a missing cell."""
    base = plt.get_cmap(name)
    cmap = LinearSegmentedColormap.from_list(
        f"{name}-tiles", base(np.linspace(start, 1.0, 256))
    )
    cmap.set_bad("white")
    return cmap


def _nk_heatmap(ax, grid, cmap, vmin, vmax, NK_SIZES, NK_KAPPAS, fmt="{:.2f}"):
    """One triangular (N, kappa) surface, drawn as rounded, separated tiles."""
    colormap = _tile_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isfinite(grid[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", color=INK_MUTED,
                        fontsize=NK_TICK_FS, fontweight="bold", zorder=3)
                continue
            ax.add_patch(FancyBboxPatch(
                (j - NK_TILE / 2, i - NK_TILE / 2), NK_TILE, NK_TILE,
                boxstyle=f"round,pad=0,rounding_size={NK_ROUND}",
                facecolor=colormap(norm(grid[i, j])), edgecolor="none", zorder=2))
            shade = (grid[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            ink = "white" if shade > 0.55 else INK
            ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                    color=ink, fontsize=NK_TICK_FS, fontweight="bold", zorder=3,
                    path_effects=[pe.withStroke(
                        linewidth=2.2, alpha=0.30,
                        foreground=INK if ink == "white" else "white")])

    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(grid.shape[1]), [str(k) for k in NK_KAPPAS])
    ax.set_yticks(range(grid.shape[0]), [str(N) for N in NK_SIZES])
    ax.tick_params(which="major", colors=INK, labelsize=NK_TICK_FS,
                   length=6, width=1.2)
    ax.minorticks_off()
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")
        text.set_color(INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.2)
    return ScalarMappable(norm=norm, cmap=colormap)


def build(results_root, save):
    NK_SIZES, NK_KAPPAS, design_points, metric_names, draws, n_seeds, meta = load(results_root)
    report = Report()

    # The decomposition-consistent total T = Bias^2_db + Variance is what every
    # comparison ranks on. It is NOT the plug-in MSE, which exceeds it by exactly
    # Var/(M - 1); the assertion pins that (10_bias_variance.ipynb, section 4.1).
    n_mc = int(meta["n_mc"])
    _residual = max(
        float(np.abs(
            d[:, metric_names.index("total")]
            - d[:, metric_names.index("mse")]
            + d[:, metric_names.index("variance")] / (n_mc - 1)
        ).max())
        for d in draws.values()
    )
    assert _residual < 1e-9, f"T != MSE - Var/(M-1): {_residual:.3e}"

    # No figure title: the panel titles name the three quantities and the
    # caption carries the design. Everything is bold, on white.
    rc = dict(FIG_RC, **{"axes.grid": False, "font.weight": "bold",
                         "axes.labelweight": "bold", "axes.titleweight": "bold"})
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), facecolor="white")
        mappables, clamped = [], []
        for ax, (metric, title, cmap, (vmin, vmax), _step) in zip(
                axes, NK_SURFACE_COLUMNS):
            ax.set_facecolor("white")
            grid = surface(draws, metric_names, NK_SIZES, NK_KAPPAS, metric)
            finite = grid[np.isfinite(grid)]
            outside = finite[(finite < vmin) | (finite > vmax)]
            if outside.size:
                clamped.append((metric, vmin, vmax, np.sort(outside)))
            mappables.append(_nk_heatmap(ax, grid, cmap, vmin, vmax, NK_SIZES, NK_KAPPAS))
            ax.set_title(title, fontsize=NK_LABEL_FS + 0.5, color=INK,
                         fontweight="bold", pad=10)
            ax.set_xlabel("Retained rank  $\\kappa$", fontsize=NK_LABEL_FS,
                          color=INK, fontweight="bold", labelpad=9)
        axes[0].set_ylabel("Ensemble size  $N$", fontsize=NK_LABEL_FS, color=INK,
                           fontweight="bold", labelpad=9)

        # The equal aspect shrinks each axes inside its gridspec slot, so the
        # colourbars are placed by hand once the layout is settled and drawn.
        fig.tight_layout(w_pad=2.2,
                         rect=(0, NK_CBAR_Y + NK_CBAR_H + 0.055, 1, 1))
        fig.canvas.draw()
        for ax, mappable, column in zip(axes, mappables, NK_SURFACE_COLUMNS):
            (vmin, vmax), step = column[3], column[4]
            position = ax.get_position()
            cax = fig.add_axes((position.x0, NK_CBAR_Y, position.width, NK_CBAR_H))
            bar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
            bar.set_ticks(np.round(np.arange(vmin, vmax + 1e-9, step), 10))
            bar.formatter = FuncFormatter(lambda v, _: f"{v:.1f}")
            bar.update_ticks()
            bar.outline.set_visible(False)
            bar.ax.tick_params(colors=INK, width=1.2, length=4, labelsize=9.5)
            for text in bar.ax.get_xticklabels():
                text.set_fontweight("bold")
        save(fig)

    # ---- numerical summary (10_bias_variance.ipynb, sections 4.1 and 4.4) ----
    report(f"run {meta['run_directory']} (config hash {meta['config_hash'][:12]}); "
           f"restart law {meta['restart_law'].get('name')} "
           f"({meta['restart_law'].get('identifier')})")
    report(f"design: {len(design_points)} valid (N, kappa) points x {n_seeds} complete "
           f"truth seeds x M={n_mc} realizations x {meta['n_windows']} evaluated windows "
           f"(stride {meta['window_stride']}, spin-up {meta['spinup_windows']})")
    report(f"excluded truth seeds: {meta['excluded_truth_seeds']}")
    report(f"T = MSE - Var/(M-1) holds to {_residual:.2e} over all design points.")
    for metric, vmin, vmax, values in clamped:
        report(f"  {metric}: {values.size} cell(s) outside the "
               f"[{vmin:g}, {vmax:g}] colour scale and clamped to its end -- "
               + ", ".join(f"{v:.2f}" for v in values))

    total_column = metric_names.index("total")
    variance_column = metric_names.index("variance")
    bias_column = metric_names.index("bias_squared_debiased")

    def _median(N, kappa, column):
        return float(np.median(draws[(N, kappa)][:, column]))

    report("\n1. Variance response to N at fixed kappa")
    for kappa in NK_KAPPAS:
        sizes = [N for N in NK_SIZES if (N, kappa) in set(design_points)]
        values = [_median(N, kappa, variance_column) for N in sizes]
        slope = np.polyfit(np.log(sizes), np.log(values), 1)[0]
        report(f"  kappa={kappa}: {values[0]:.3f} -> {values[-1]:.3f}; "
               f"descriptive log-log slope {slope:+.2f}")
    report("\n2-3. Bias and variance response to kappa at fixed N")
    for N in NK_SIZES:
        kappas = [k for Np, k in design_points if Np == N]
        bias = [_median(N, k, bias_column) for k in kappas]
        variance = [_median(N, k, variance_column) for k in kappas]
        report(f"  N={N}: Bias^2 {bias[0]:.3f} -> {bias[-1]:.3f}; "
               f"Variance {variance[0]:.3f} -> {variance[-1]:.3f}")
    report("\n5. Rank minimizing total error at each tested N")
    for N in NK_SIZES:
        kappas = [k for Np, k in design_points if Np == N]
        totals = {k: _median(N, k, total_column) for k in kappas}
        best = min(totals, key=totals.get)
        runner_up = sorted(totals.values())[1]
        report(f"  N={N}: kappa={best}, T={totals[best]:.3f} "
               f"(next best T={runner_up:.3f}, gap {runner_up - totals[best]:.3f})")
    return report.text()
