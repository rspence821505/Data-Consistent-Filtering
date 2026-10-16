"""
Figure 8 -- endpoint energy-equivalent error and its component composition for
QPCA-EnDCF and 4D-LETKF at the shelf-crossing comparison windows.

Paper label: ``fig:inlet_energy_metric_sensitivity``; file
``fig05_metric_sensitivity.png``.  Plotting code verbatim from the script
``fig05_metric_sensitivity.py`` that ``02_shallow_water_figures.ipynb`` ran;
only the input and output paths changed.

Panel A: per-realization and mean RMSE_{E,w} (m, log axis) in the four
  storm--localization cells, matched-seed connectors, and the ratio R_E of the
  arithmetic means.
Panel B: per-seed-then-averaged fractional composition (f_zeta, f_u, f_v) of
  RMSE_{E,w}^2 as 100% stacked bars (largest-remainder rounding).

Both panels use the shelf-crossing window: slow storm W6, fast storm W4.

Input (from the result set): the per-run JSON files
``shallow_water/runs/idealized_inlet_5filter_energy_metric_v1_sigmaE_<cell>_gh_<method>_seed<s>/result_qpca_endcf.json``.
"""

from __future__ import annotations

import json

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.figures._common import Report, shallow_water_root
from src.figures.style_shallow_water import COLORS, INK, apply_rcparams, style_axes

OUTPUTS = ("fig05_metric_sensitivity.pdf", "fig05_metric_sensitivity.png")
OUTPUT = OUTPUTS[-1]
SAVE_KWARGS = dict(dpi=300, bbox_inches="tight")
STYLE = None
# The shallow-water experiment has no Level 2: this figure is always drawn
# from the saved run outputs in results/published/shallow_water/.
PUBLISHED_ONLY = True
TITLE = "Idealized inlet: endpoint RMSE_E and its component composition"

# v3 campaign.
SEEDS = [2027, 2028, 2029]
# The driver tags v3 result dirs with ``_sigmaE_``.
DIR_TAG = "sigmaE"
CELLS = [
    # (tag, two-line label, w_shelf index)
    ("capstone60",     "Slow storm\n2 km loc",  5),
    ("cap60_L10km",    "Slow storm\n10 km loc", 5),
    ("cap60cmp",       "Fast storm\n2 km loc",  3),
    ("cap60cmp_L10km", "Fast storm\n10 km loc", 3),
]

# Component palette (Panel B).
ZETA_C = "#2E86AB"   # surface elevation (shared with Seq-EnKF slot)
U_C    = "#A23B72"   # u-velocity (shared with 4D-EnKF slot)
V_C    = "#C570FF"   # v-velocity (shared with Seq-ETKF slot)

# Method palette (Panel A); shared with the Lorenz-96 figures.
QPCA_C  = COLORS["QPCA-EnDCF"]  # #F18F01 vibrant orange
LETKF_C = COLORS["4D-LETKF"]    # #4A3AA7 violet (shared 4D- family hue)

# One marker shape per seed so a reader can trace an individual seed.
SEED_MARKERS = {2027: "o", 2028: "s", 2029: "^"}


def load_vals(results_root, cfg, m, wi):
    out = []
    for s in SEEDS:
        f = (shallow_water_root(results_root) / "runs" /
             f"idealized_inlet_5filter_energy_metric_v1_{DIR_TAG}_{cfg}"
             f"_gh_{m}_seed{s}" / "result_qpca_endcf.json")
        d = json.load(open(f))
        w = d["windows"][wi]
        out.append({
            "seed": s,
            "RMSE_E": float(w["analysis_rmse_energy_equiv_m"]),
            "et":     float(w["energy_surface_contribution_m2"]),
            "eu":     float(w["energy_u_contribution_m2"]),
            "ev":     float(w["energy_v_contribution_m2"]),
        })
    return out


def am_strict(a):
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        return np.inf
    return a.mean()


def largest_remainder_percent(fracs):
    raw = np.asarray(fracs, dtype=float) * 100.0
    floor = np.floor(raw).astype(int)
    remainder = raw - floor
    deficit = 100 - int(floor.sum())
    order = np.argsort(-remainder)
    out = floor.copy()
    for i in order[:deficit]:
        out[i] += 1
    return out


def build(results_root, save):
    apply_rcparams()

    # ---------- Gather data ----------
    summary = []
    for cfg, label, wi in CELLS:
        q = load_vals(results_root, cfg, "qpca",  wi)
        l = load_vals(results_root, cfg, "letkf", wi)
        qE = np.array([x["RMSE_E"] for x in q])
        lE = np.array([x["RMSE_E"] for x in l])

        # per-seed-then-averaged composition fractions (Panel B)
        def frac_mean(rows):
            feS, fuS, fvS = [], [], []
            for r in rows:
                tot = r["et"] + r["eu"] + r["ev"]
                if tot > 0:
                    feS.append(r["et"] / tot)
                    fuS.append(r["eu"] / tot)
                    fvS.append(r["ev"] / tot)
            return (np.mean(feS), np.mean(fuS), np.mean(fvS))

        R = am_strict(qE) / am_strict(lE)
        summary.append({
            "cfg": cfg, "label": label, "w_shelf": wi + 1,
            "qE": qE, "lE": lE,
            "qmean_RMSE_E": am_strict(qE),
            "lmean_RMSE_E": am_strict(lE),
            "R_E_rom": R,
            # seed-level tally: how often is QPCA-EnDCF the lower of the pair?
            "q_seed_wins": int(np.sum(qE < lE)),
            "fq": frac_mean(q),
            "fl": frac_mean(l),
        })

    # ---------- Figure layout ----------
    fig = plt.figure(figsize=(14.2, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ===== Panel A: per-seed magnitudes on a log axis =====
    Y_LO, Y_HI = 0.6, 150.0
    DX = 0.20           # method offset from the cell centre
    MEAN_HALFW = 0.145  # half-width of the mean dash

    for i, s in enumerate(summary):
        # Alternating band to group the two methods belonging to one cell.
        if i % 2 == 1:
            axA.axvspan(i - 0.5, i + 0.5, color="0.94", zorder=0)

        xq, xl = i - DX, i + DX

        # Matched-seed connectors: the three seeds are shared between the
        # two filters, so each pair is a within-seed comparison.
        for qv, lv in zip(s["qE"], s["lE"]):
            axA.plot([xq, xl], [qv, lv],
                     color=QPCA_C if qv < lv else LETKF_C,
                     lw=1.1, alpha=0.55, zorder=2, solid_capstyle="round")

        # Per-seed values.
        for seed, qv, lv in zip(SEEDS, s["qE"], s["lE"]):
            mk = SEED_MARKERS[seed]
            axA.scatter([xq], [qv], marker=mk, s=42, facecolor="white",
                        edgecolors=QPCA_C, linewidths=1.4, zorder=4)
            axA.scatter([xl], [lv], marker=mk, s=42, facecolor="white",
                        edgecolors=LETKF_C, linewidths=1.4, zorder=4)

        # Arithmetic seed means (the quantity R_E is built from).
        for xm, mv, cc in ((xq, s["qmean_RMSE_E"], QPCA_C),
                           (xl, s["lmean_RMSE_E"], LETKF_C)):
            axA.plot([xm - MEAN_HALFW, xm + MEAN_HALFW], [mv, mv],
                     color=cc, lw=3.0, solid_capstyle="butt", zorder=5)

        # Header: R_E and which filter has the lower mean.
        R = s["R_E_rom"]
        win_name = "QPCA-EnDCF" if R < 1 else "4D-LETKF"
        win_col = QPCA_C if R < 1 else LETKF_C
        factor = (1.0 / R) if R < 1 else R
        axA.annotate(f"$R_E = {R:.2f}$", xy=(i, 1.0),
                     xycoords=("data", "axes fraction"),
                     xytext=(0, 33), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold",
                     annotation_clip=False)
        axA.annotate(win_name, xy=(i, 1.0),
                     xycoords=("data", "axes fraction"),
                     xytext=(0, 19), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, color=win_col,
                     annotation_clip=False)
        axA.annotate(f"{factor:.2f}$\\times$ lower mean", xy=(i, 1.0),
                     xycoords=("data", "axes fraction"),
                     xytext=(0, 7), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, color=win_col,
                     annotation_clip=False)

    axA.set_yscale("log")
    axA.set_ylim(Y_LO, Y_HI)
    axA.set_xlim(-0.5, len(summary) - 0.5)
    axA.set_xticks(range(len(summary)))
    axA.set_xticklabels([s["label"] + f"\n$W_{s['w_shelf']}$" for s in summary],
                        fontsize=10, fontweight="bold", color=INK)
    axA.tick_params(axis="x", length=0, pad=6)
    axA.set_ylabel("Analysis error $\\mathrm{RMSE}_{E,w}$ (m, log scale)")
    style_axes(axA, grid_axis="y")

    axA.set_title("A  Energy-equivalent analysis error, by seed",
                  pad=52, loc="left", fontsize=14, fontweight="bold",
                  color=INK)

    _seed_keys = [
        mlines.Line2D([], [], linestyle="none", marker=SEED_MARKERS[s],
                      markerfacecolor="white", markeredgecolor="0.25",
                      markeredgewidth=1.2, markersize=6.5, label=f"seed {s}")
        for s in SEEDS
    ]
    axA.legend(
        handles=[
            mlines.Line2D([], [], color=QPCA_C, lw=3.0,
                          label="QPCA-EnDCF seed mean"),
            mlines.Line2D([], [], color=LETKF_C, lw=3.0,
                          label="4D-LETKF seed mean"),
            mlines.Line2D([], [], color=QPCA_C, lw=1.1, alpha=0.55,
                          label="matched seed, QPCA-EnDCF lower"),
            mlines.Line2D([], [], color=LETKF_C, lw=1.1, alpha=0.55,
                          label="matched seed, 4D-LETKF lower"),
        ] + _seed_keys,
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4,
        fontsize=10, frameon=False,
        handlelength=1.9, handletextpad=0.6, columnspacing=1.6,
        borderaxespad=0.0,
    )

    # ===== Panel B: fractional composition =====
    y_pos, methods, frs = [], [], []
    for i, s in enumerate(summary):
        y_pos += [i * 3.0 + 0.0, i * 3.0 + 1.1]
        methods += ["QPCA-EnDCF", "4D-LETKF"]
        frs += [s["fq"], s["fl"]]
    y_pos = np.array(y_pos)
    fes = np.array([f[0] for f in frs])
    fus = np.array([f[1] for f in frs])
    fvs = np.array([f[2] for f in frs])

    h1 = axB.barh(y_pos, fes, color=ZETA_C, edgecolor="white",
                  linewidth=0.8, height=1.0,
                  label="$f_\\zeta$ (surface elevation)")
    h2 = axB.barh(y_pos, fus, left=fes, color=U_C, edgecolor="white",
                  linewidth=0.8, height=1.0,
                  label="$f_u$ ($u$-velocity)")
    h3 = axB.barh(y_pos, fvs, left=fes + fus, color=V_C, edgecolor="white",
                  linewidth=0.8, height=1.0,
                  label="$f_v$ ($v$-velocity)")

    # Every segment is labelled.  Segments too narrow to hold their label
    # are labelled just outside the left end of the bar, in the component
    # colour, rather than silently dropped.
    MIN_INSIDE = 0.055
    for yp, fe, fu, fv, meth in zip(y_pos, fes, fus, fvs, methods):
        p = largest_remainder_percent([fe, fu, fv])
        axB.text(-0.105, yp, meth, ha="right", va="center", fontsize=10,
                 fontweight="bold", color=INK)
        left = 0.0
        for k, (frac, pct, cc) in enumerate(((fe, p[0], ZETA_C), (fu, p[1], U_C),
                                             (fv, p[2], V_C))):
            if k > 0 and frac >= MIN_INSIDE:
                axB.text(left + frac / 2.0, yp, f"{pct}%", ha="center",
                         va="center", color="white", fontsize=9,
                         fontweight="bold")
            else:
                axB.text(-0.012, yp, f"{pct}%", ha="right", va="center",
                         color=cc, fontsize=9, fontweight="bold")
            left += frac

    for i in range(len(summary) - 1):
        axB.axhline(i * 3.0 + 1.9, color="#DDDDDD", linestyle="-",
                    linewidth=0.9, zorder=0)

    for i, s in enumerate(summary):
        axB.text(1.015, i * 3.0 + 0.55,
                 s["label"] + f", $W_{s['w_shelf']}$",
                 transform=axB.get_yaxis_transform(),
                 ha="left", va="center", fontsize=10, fontweight="bold",
                 color=INK)

    axB.set_yticks([])
    axB.set_xlim(0, 1.0)
    axB.set_ylim(max(y_pos) + 1.0, -1.0)
    axB.set_xlabel("Fractional contribution to $\\mathrm{RMSE}_{E,w}^{2}$")
    axB.grid(True, which="major", axis="x",
             color="#DDDDDD", alpha=0.9, lw=0.9, zorder=0)
    axB.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axB.spines[side].set_visible(False)
    axB.spines["bottom"].set_color(INK)
    axB.spines["bottom"].set_linewidth(0.9)
    axB.tick_params(axis="x", colors=INK, labelcolor=INK)
    for lbl in axB.get_xticklabels():
        lbl.set_fontweight("bold")

    axB.set_title("B  Component composition of $\\mathrm{RMSE}_{E,w}^{2}$",
                  pad=38, loc="left", fontsize=14, fontweight="bold",
                  color=INK)

    axB.legend(
        handles=[h1, h2, h3],
        loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
        fontsize=10, frameon=False,
        handlelength=1.9, handletextpad=0.6, columnspacing=1.6,
        borderaxespad=0.0,
    )

    save(fig)

    # Printed summary
    report = Report()
    report("Panel A per-cell summary:")
    for s in summary:
        report(f"  {s['label'].replace(chr(10), ' '):<30}  W{s['w_shelf']}  "
               f"QPCA mean={s['qmean_RMSE_E']:.4g} m "
               f"(seeds {np.round(s['qE'], 2)});  "
               f"LETKF mean={s['lmean_RMSE_E']:.4g} m "
               f"(seeds {np.round(s['lE'], 2)});  "
               f"R_E={s['R_E_rom']:.3g};  QPCA lower in {s['q_seed_wins']}/3 seeds")
    report("\nPanel B mean fractions (f_zeta, f_u, f_v) of RMSE_E^2:")
    velocity = []
    for s in summary:
        for name, fr in (("QPCA-EnDCF", s["fq"]), ("4D-LETKF", s["fl"])):
            velocity.append(fr[1] + fr[2])
            report(f"  {s['cfg']:<16} {name:<11} "
                   f"{fr[0]:.3f} {fr[1]:.3f} {fr[2]:.3f}   (u+v {fr[1] + fr[2]:.1%})")
    report(f"velocity share range over the eight method-configuration pairs: "
           f"{min(velocity):.1%}-{max(velocity):.1%}")
    return report.text()
