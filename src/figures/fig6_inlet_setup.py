"""
Figure 6 -- configuration of the idealized-inlet shallow-water experiment.

Paper label: ``fig:inlet_scale_matching_setup``; file
``fig01_scale_matching_setup.png``.  Plotting code verbatim from the script
``fig01_scale_matching_setup.py`` that ``02_shallow_water_figures.ipynb`` ran;
only the input and output paths changed.

Panel A: inlet geometry -- mesh, analytic bathymetry, the observation points
  (reproduced deterministically from the driver's interior-point sampler with
  obs_seed=42, obs_fraction=0.043, area weighting), the ocean boundary, the
  y = 20 km shelf transition, the shared Rankine-vortex track and the
  window-end vortex positions of the slow and fast storms.
Panel B: the 2 x 2 storm--localization design, with the endpoint error ratio
  and the lower-error filter of each cell read from the regime-map table.

Inputs: ``data/shallow_water/Ideal_Inlet.h5`` (the mesh) and
``shallow_water/scale_matching_regime_map_v3.csv`` from the result set.
"""

from __future__ import annotations

import csv as _csv

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.tri import Triangulation

from src.figures._common import Report, shallow_water_root
from src.paths import SHALLOW_WATER_MESH_H5, require

OUTPUTS = ("fig01_scale_matching_setup.pdf", "fig01_scale_matching_setup.png")
OUTPUT = OUTPUTS[-1]
SAVE_KWARGS = dict(dpi=300, bbox_inches="tight")
STYLE = None
# The shallow-water experiment has no Level 2: this figure is always drawn
# from the saved run outputs in results/published/shallow_water/.
PUBLISHED_ONLY = True
TITLE = "Idealized-inlet configuration and storm--localization design"

# Storm track endpoints (from CartesianVortexConfig)
TRACK_START = np.array([40_000.0, -20_000.0])   # m
TRACK_END   = np.array([25_000.0, +50_000.0])   # m
# Window end times (minutes) for slow (12 h = 43200 s) and fast (7 h = 25200 s)
# Ramp = 60 min, 60-min cycling, slow 11 windows, fast 6
SLOW_WINDOW_END_MIN = np.array([60 + 60 * k for k in range(1, 12)])   # W1..W11
FAST_WINDOW_END_MIN = np.array([60 + 60 * k for k in range(1, 7)])    # W1..W6
SLOW_T_END_S = 43_200.0
FAST_T_END_S = 25_200.0

QPCA_C = "#1f77b4"
LETKF_C = "#d62728"
SLOW_C = "#4C72B0"
FAST_C = "#C44E52"

# Cells whose three-seed mean R_E is not multi-seed-robust at n_trial=3.
_SEED_SENSITIVE_CELLS = {"cap60cmp", "cap60cmp_L10km"}


def bathymetry(y_m):
    """Analytical bathymetry H(y) in meters, as defined by the idealized-inlet
    problem of the shallow-water solver that produced the runs."""
    return np.where(y_m >= 20_000.0,
                    5.0,
                    14.0 - (9.0 / 20_000.0) * y_m)


def reproduce_obs_points(h5_path, obs_seed=42, obs_fraction=0.043,
                         boundary_tol=1e-10):
    """Pure numpy+h5py reimplementation of the driver's
    _generate_interior_observation_points with obs_area_weighted=True."""
    with h5py.File(h5_path, "r") as f:
        geom = f["Mesh/mesh/geometry"][:]       # (n_v, 2 or 3)
        topo = f["Mesh/mesh/topology"][:]       # (n_tri, 3)
    coords = np.zeros((geom.shape[0], 3))
    coords[:, : min(geom.shape[1], 3)] = geom[:, : min(geom.shape[1], 3)]
    # NB: driver deduplicates after gather(); the h5 geometry is already unique
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    interior_mask = ((coords[:, 0] > x_min + boundary_tol) &
                     (coords[:, 0] < x_max - boundary_tol) &
                     (coords[:, 1] > y_min + boundary_tol) &
                     (coords[:, 1] < y_max - boundary_tol))
    interior = np.where(interior_mask)[0]
    n_obs = max(1, int(len(interior) * obs_fraction))

    v = coords[:, :2]
    tri = v[topo]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    tri_area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    dual = np.zeros(len(coords))
    for k in range(3):
        np.add.at(dual, topo[:, k], tri_area / 3.0)
    weights = dual[interior]
    weights = weights / weights.sum()
    rng = np.random.default_rng(obs_seed)
    sel = rng.choice(len(interior), size=n_obs, replace=False, p=weights)
    selected = interior[sel]
    return coords[selected], coords, topo


class _HandlerCircleKey(HandlerPatch):
    """Draw a legend key as a circle so it mirrors the in-panel glyph."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        key = Circle((width / 2.0 - xdescent, height / 2.0 - ydescent),
                     0.44 * min(width, height))
        key.update_from(orig_handle)
        key.set_transform(trans)
        return [key]


def vortex_pos_at_min(t_min, t_total_s):
    """Linear-in-time vortex position; clamps when t > t_total_s."""
    frac = min(1.0, t_min * 60.0 / t_total_s)
    return TRACK_START + frac * (TRACK_END - TRACK_START)


def build(results_root, save):
    _RMAP_CSV = require(shallow_water_root(results_root) / "scale_matching_regime_map_v3.csv")
    mesh_h5 = require(SHALLOW_WATER_MESH_H5)
    report = Report()

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "figure.constrained_layout.use": True,
        # Bold throughout (text and math) for both panels.
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.titleweight": "bold", "figure.titleweight": "bold",
        "mathtext.default": "bfit",
    })

    # Load mesh + obs
    obs_coords, coords, topo = reproduce_obs_points(mesh_h5)
    report(f"Mesh: {len(coords)} vertices, {len(topo)} triangles")
    report(f"Obs: {len(obs_coords)} points generated (fraction=0.043, seed=42)")

    # Build figure
    fig = plt.figure(figsize=(13.0, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ===== Panel A: geometry =====
    # Convert coords to km for display
    x_km = coords[:, 0] / 1000.0
    y_km = coords[:, 1] / 1000.0
    tri = Triangulation(x_km, y_km, topo)
    # Vertex bathymetry
    H_vertex = bathymetry(coords[:, 1])

    # Bathymetry filled contour
    cf = axA.tricontourf(tri, H_vertex, levels=np.linspace(5, 14, 10),
                         cmap="Blues", alpha=0.85)
    # Inset cax so the bar tracks the aspect="equal" axes box.
    caxA = axA.inset_axes([1.025, 0.0, 0.028, 1.0])
    cbar = plt.colorbar(cf, cax=caxA)
    cbar.set_label("Bathymetric depth $H$ (m)")

    # Mesh edges (lightly)
    axA.triplot(tri, color="black", linewidth=0.15, alpha=0.5)

    # Observation points
    axA.scatter(obs_coords[:, 0] / 1000.0, obs_coords[:, 1] / 1000.0,
                s=12, color="darkorange", edgecolors="black", linewidths=0.3,
                zorder=5)

    # Ocean boundary line y=0
    axA.axhline(0.0, color="navy", linewidth=1.6, linestyle="-",
                alpha=0.9, zorder=4)
    axA.text(0.8, -0.6, "Ocean boundary", color="navy", fontsize=9,
             ha="left", va="top", fontweight="bold")

    # Shelf transition y=20 km
    axA.axhline(20.0, color="saddlebrown", linewidth=1.4, linestyle="--",
                alpha=0.9, zorder=4)
    axA.text(0.8, 20.5, "Shelf transition\n($y=20$ km, $H=5$ m)",
             color="saddlebrown", fontsize=9, ha="left", va="bottom",
             fontweight="bold", linespacing=1.25)

    # Storm track (shared geometry)
    tk_x = np.array([TRACK_START[0], TRACK_END[0]]) / 1000.0
    tk_y = np.array([TRACK_START[1], TRACK_END[1]]) / 1000.0
    axA.plot(tk_x, tk_y, color="black", linewidth=1.8, zorder=6)
    axA.annotate("", xy=tk_x[1:2].tolist() + tk_y[1:2].tolist(),
                 xytext=tk_x[0:1].tolist() + tk_y[0:1].tolist(),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.6),
                 zorder=6)

    # Window-end vortex positions: slow (circles) and fast (triangles)
    slow_pos = np.array([vortex_pos_at_min(t, SLOW_T_END_S)
                         for t in SLOW_WINDOW_END_MIN])
    fast_pos = np.array([vortex_pos_at_min(t, FAST_T_END_S)
                         for t in FAST_WINDOW_END_MIN])
    axA.scatter(slow_pos[:, 0] / 1000.0, slow_pos[:, 1] / 1000.0,
                marker="o", s=70, facecolor=SLOW_C, edgecolor="black",
                linewidths=0.8, zorder=7)
    axA.scatter(fast_pos[:, 0] / 1000.0, fast_pos[:, 1] / 1000.0,
                marker="^", s=80, facecolor=FAST_C, edgecolor="black",
                linewidths=0.8, zorder=7)

    # Label W_crit specifically: slow W6, fast W4
    axA.annotate("$\\mathbf{W_6^{slow}}$",
                 xy=slow_pos[5] / 1000.0,
                 xytext=(slow_pos[5][0] / 1000.0 + 7.0,
                         slow_pos[5][1] / 1000.0 + 3.0),
                 fontsize=11, color=SLOW_C, fontweight="bold",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=SLOW_C, lw=1.2))
    axA.annotate("$\\mathbf{W_4^{fast}}$",
                 xy=fast_pos[3] / 1000.0,
                 xytext=(fast_pos[3][0] / 1000.0 + 7.0,
                         fast_pos[3][1] / 1000.0 - 1.5),
                 fontsize=11, color=FAST_C, fontweight="bold",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=FAST_C, lw=1.2))

    # First *visible* window of each series.  Slow W1 lies at y = -8.3 km,
    # outside the plotted range, so without this anchor the slow markers are
    # easily miscounted (W6 is only the fifth blue marker in view).
    axA.annotate("$\\mathbf{W_2}$", xy=slow_pos[1] / 1000.0,
                 xytext=(slow_pos[1][0] / 1000.0 - 2.0,
                         slow_pos[1][1] / 1000.0 - 0.1),
                 fontsize=9, color=SLOW_C, fontweight="bold",
                 ha="right", va="center",
                 arrowprops=dict(arrowstyle="-", color=SLOW_C, lw=0.8))
    axA.annotate("$\\mathbf{W_1}$", xy=fast_pos[0] / 1000.0,
                 xytext=(fast_pos[0][0] / 1000.0 + 1.6,
                         fast_pos[0][1] / 1000.0 - 1.4),
                 fontsize=9, color=FAST_C, fontweight="bold",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=FAST_C, lw=0.8))

    # Track-start / end labels
    axA.annotate("Start\n($t{=}0$)", xy=(TRACK_START[0] / 1000.0,
                                          TRACK_START[1] / 1000.0),
                 xytext=(TRACK_START[0] / 1000.0 + 3,
                         TRACK_START[1] / 1000.0 - 1),
                 fontsize=8, color="black", ha="left")
    axA.annotate("Track end\n(slow: 12 h, fast: 7 h)",
                 xy=(TRACK_END[0] / 1000.0, TRACK_END[1] / 1000.0),
                 xytext=(TRACK_END[0] / 1000.0 - 20,
                         TRACK_END[1] / 1000.0 + 1),
                 fontsize=8, color="black", ha="right")

    axA.set_xlim(x_km.min(), x_km.max())   # no x padding: domain fills the axes
    axA.set_ylim(y_km.min() - 3, y_km.max() + 3)
    axA.set_aspect("equal")
    axA.set_xlabel("$x$ (km)")
    axA.set_ylabel("$y$ (km)")
    # Panel-letter marker only; the panel descriptor is carried in the
    # manuscript figure caption rather than as an in-figure title.
    axA.text(0.015, 0.98, "A", transform=axA.transAxes,
             fontsize=16, fontweight="bold", ha="left", va="top",
             bbox=dict(facecolor="white", edgecolor="none",
                       boxstyle="round,pad=0.2", alpha=0.9),
             zorder=12)
    # ----- Panel A legend -----
    legA = axA.legend(
        handles=[
            Line2D([], [], linestyle="none", marker="o", markersize=5.0,
                   markerfacecolor="darkorange", markeredgecolor="black",
                   markeredgewidth=0.4),
            Line2D([], [], color="black", linewidth=1.8),
            Line2D([], [], linestyle="none", marker="o", markersize=8.0,
                   markerfacecolor=SLOW_C, markeredgecolor="black",
                   markeredgewidth=0.8),
            Line2D([], [], linestyle="none", marker="^", markersize=8.5,
                   markerfacecolor=FAST_C, markeredgecolor="black",
                   markeredgewidth=0.8),
        ],
        labels=[
            f"Observations ($n={len(obs_coords)}$)",
            "Rankine-vortex track (shared)",
            "Slow storm, $W_2$–$W_8$ in view",
            "Fast storm, $W_1$–$W_4$ in view",
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=2, fontsize=9, frameon=True, fancybox=False, framealpha=1.0,
        edgecolor="0.75", facecolor="white",
        handlelength=1.8, handletextpad=0.6,
        columnspacing=2.6, labelspacing=0.6, borderpad=0.8, borderaxespad=0.0,
    )
    legA.get_frame().set_linewidth(0.6)
    axA.grid(True, alpha=0.25)

    # ===== Panel B: factorial design matrix =====
    # Every cell is drawn at ONE shared length scale (1 cell unit = 40 km) so the
    # four sketches are directly comparable; a scale bar states that scale.  The
    # filled disk is the surge correlation scale ell, and the dashed ring is the
    # Gaspari--Cohn radius r_loc.  The glyphs show the scale relationships only;
    # the outcomes are read from the regime-map CSV.
    #
    # Each cell shows the ratio
    #   R_E = mean(RMSE_E, QPCA-EnDCF) / mean(RMSE_E, 4D-LETKF)
    # at the fixed shelf-crossing window (slow W_6, fast W_4).
    # R_E < 1 => QPCA-EnDCF has the lower three-seed mean error by a factor of
    # 1/R_E; R_E > 1 => 4D-LETKF has the lower mean by R_E.
    def _r_e_by_cell():
        rows = list(_csv.DictReader(_RMAP_CSV.open()))

        def _m(cell, method):
            for r in rows:
                if r["cell"] == cell and r["method"] == method:
                    return float(r["mean_rmse_E_m"])
            raise KeyError((cell, method))
        out = {}
        for cell in ("capstone60", "cap60_L10km", "cap60cmp", "cap60cmp_L10km"):
            q, l = _m(cell, "qpca"), _m(cell, "letkf")
            out[cell] = dict(R_E=q / l, qpca=q, letkf=l)
        return out
    _RE = _r_e_by_cell()

    def _fmt_factor(cell):
        """Return (winner_label, factor_text, r_e_text) for a cell."""
        R = _RE[cell]["R_E"]
        # The two fast-storm cells remain seed-sensitive (_SEED_SENSITIVE_CELLS);
        # that caveat lives in the caption/text rather than on the figure.
        mark = ""
        if R < 1.0:
            inv = 1.0 / R
            factor = f"{inv:.2f}" if inv < 10 else f"{inv:.1f}"
            return (f"QPCA-EnDCF{mark}", factor, f"$R_E={R:.2f}$")
        factor = f"{R:.2f}" if R < 10 else f"{R:.0f}"
        return (f"4D-LETKF{mark}", factor, f"$R_E={R:.2f}$")
    UNIT_KM = 40.0            # 1 cell unit
    ELL_SLOW_KM = 15.0        # dispersed slow-storm surge
    ELL_FAST_KM = 2.0         # concentrated fast-storm surge
    CW, CH = 1.0, 1.30        # cell width / height (room for three text lines)
    DISK_DY = 0.84            # disk centre height within a cell

    axB.set_xlim(0, 2 * CW)
    axB.set_ylim(-0.28, 2 * CH)
    axB.set_aspect("equal")
    axB.axis("off")

    TINT_QPCA, TINT_LETKF = "#eaf2fa", "#fdedee"
    SURGE_FC, SURGE_EC = "#9ec4e8", "#2d5f8f"

    cells = [
        # (x0, y0, r_loc km, ell km, relation, cell_tag)
        (0.0, CH, 2.0,  ELL_SLOW_KM, "$r_{\\mathrm{loc}} \\ll \\ell$",     "capstone60"),
        (CW,  CH, 10.0, ELL_SLOW_KM, "$r_{\\mathrm{loc}} < \\ell$",          "cap60_L10km"),
        (0.0, 0.0, 2.0,  ELL_FAST_KM, "$r_{\\mathrm{loc}} \\approx \\ell$", "cap60cmp"),
        (CW,  0.0, 10.0, ELL_FAST_KM, "$r_{\\mathrm{loc}} \\gg \\ell$",     "cap60cmp_L10km"),
    ]

    for (x0, y0, L_km, ell_km, rel, cell_tag) in cells:
        winner, factor, _ = _fmt_factor(cell_tag)
        is_qpca = winner.startswith("QPCA")
        col = QPCA_C if is_qpca else LETKF_C
        # Cell wash encodes the winner, so the regime map reads at a glance.
        axB.add_patch(Rectangle((x0, y0), CW, CH, zorder=0, edgecolor="none",
                                facecolor=TINT_QPCA if is_qpca else TINT_LETKF))
        cx, cy = x0 + CW / 2.0, y0 + DISK_DY
        axB.add_patch(Circle((cx, cy), ell_km / UNIT_KM, zorder=3,
                             facecolor=SURGE_FC, edgecolor=SURGE_EC,
                             linewidth=1.1, alpha=0.70))
        axB.add_patch(Circle((cx, cy), L_km / UNIT_KM, zorder=4, facecolor="none",
                             edgecolor=LETKF_C, linewidth=1.9,
                             linestyle=(0, (4.0, 2.2))))
        # Three stacked text lines:
        #   scale relation  ·  winner name  ·  "{factor}x lower RMSE_E"
        axB.text(cx, y0 + 0.300, rel, ha="center", va="center",
                 fontsize=11, color="0.30", zorder=5)
        axB.text(cx, y0 + 0.175, winner, ha="center", va="center",
                 fontsize=9.5, color=col, zorder=5)
        axB.text(cx, y0 + 0.055,
                 f"{factor}× lower $\\mathrm{{RMSE}}_E$",
                 ha="center", va="center", fontsize=9.5, color=col, zorder=5)

    # Cell borders last, so no wash or glyph sits on top of a rule.
    for i in range(3):
        axB.plot([0, 2 * CW], [i * CH, i * CH], color="0.35", lw=0.9, zorder=6)
        axB.plot([i * CW, i * CW], [0, 2 * CH], color="0.35", lw=0.9, zorder=6)

    # Column headers
    axB.text(0.5 * CW, 2 * CH + 0.07, "$r_{\\mathrm{loc}}$ = 2 km", ha="center", va="bottom",
             fontsize=11)
    axB.text(1.5 * CW, 2 * CH + 0.07, "$r_{\\mathrm{loc}}$ = 10 km", ha="center", va="bottom",
             fontsize=11)

    # Row headers
    axB.text(-0.07, 1.5 * CH,
             "Slow storm\n$\\ell \\approx$ 15 km"
             "\nshelf: $W_6$",
             ha="right", va="center", fontsize=9.5, linespacing=1.45)
    axB.text(-0.07, 0.5 * CH,
             "Fast storm\n$\\ell \\approx$ 2 km"
             "\nshelf: $W_4$",
             ha="right", va="center", fontsize=9.5, linespacing=1.45)

    # Shared scale bar: states the one length scale used by all four cells.
    bar_y, bar_len = -0.17, 10.0 / UNIT_KM
    axB.plot([0.0, bar_len], [bar_y, bar_y], color="black", lw=1.6,
             solid_capstyle="butt", zorder=6)
    for bx in (0.0, bar_len):
        axB.plot([bx, bx], [bar_y - 0.035, bar_y + 0.035], color="black",
                 lw=1.6, zorder=6)
    axB.text(bar_len + 0.07, bar_y, "10 km", ha="left", va="center", fontsize=9)

    # ----- Panel B legend -----
    # Proxy artists drawn as true circles (not the default rectangles) so the
    # legend keys are literally scaled-down copies of the in-panel glyphs.
    _key_surge = Circle((0, 0), 1, facecolor=SURGE_FC, edgecolor=SURGE_EC,
                        linewidth=1.1, alpha=0.70)
    _key_loc = Circle((0, 0), 1, facecolor="none", edgecolor=LETKF_C,
                      linewidth=1.6, linestyle=(0, (2.0, 1.35)))
    legB = axB.legend(
        handles=[_key_surge, _key_loc],
        labels=["Surge scale $\\ell$", "Gaspari–Cohn radius $r_{\\mathrm{loc}}$"],
        handler_map={Circle: _HandlerCircleKey()},
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=2, fontsize=8.5, frameon=True, fancybox=False, framealpha=1.0,
        edgecolor="0.75", facecolor="white",
        handlelength=1.5, handleheight=1.5, handletextpad=0.5,
        columnspacing=1.4, borderpad=0.6, borderaxespad=0.0,
    )
    legB.get_frame().set_linewidth(0.6)

    axB.text(0.015, 0.98, "B", transform=axB.transAxes,
             fontsize=16, fontweight="bold", ha="left", va="top",
             bbox=dict(facecolor="white", edgecolor="none",
                       boxstyle="round,pad=0.2", alpha=0.9),
             zorder=12)

    save(fig)

    report("\nPanel B (regime map, shelf-crossing window, three-realization means):")
    for (_x0, _y0, L_km, _ell, _rel, cell_tag) in cells:
        winner, factor, r_e = _fmt_factor(cell_tag)
        report(f"  {cell_tag:<16} r_loc={L_km:>4.0f} km  "
               f"QPCA-EnDCF={_RE[cell_tag]['qpca']:.4g} m  "
               f"4D-LETKF={_RE[cell_tag]['letkf']:.4g} m  "
               f"R_E={_RE[cell_tag]['R_E']:.3f}  lower: {winner} ({factor}x)")
    return report.text()
