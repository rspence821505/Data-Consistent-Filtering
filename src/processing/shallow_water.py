"""
Reduce the per-run JSON outputs of the idealized-inlet shallow-water experiment
to the tables behind paper Figures 6-8 and Table 2.

The runs themselves are not part of this repository's code: they were produced
on TACC Vista by a FEniCSx shallow-water solver (the non-public SWEMniCS-thesis
research code, an extension of the open-source SWEMniCS package), and only
their outputs are distributed, in ``results/published/shallow_water/runs/``.
The shallow-water experiment is therefore reproduced from those saved outputs
(Level 1 only).

Ported from ``scripts/analysis/build_scale_match_metrics.py`` (campaign v3) and
``scripts/analysis/build_all_filter_gamma_E.py`` of that research code; only
the input/output paths changed, and the ``source`` column now holds a path
relative to the result set instead of an absolute one.

Input layout (``<results_root>/shallow_water/runs/``)::

    idealized_inlet_5filter_energy_metric_v1_sigmaE_<cell>_gh_<method>_seed<s>/
        result_qpca_endcf.json

with cells ``capstone60`` (slow storm, 2 km), ``cap60_L10km`` (slow, 10 km),
``cap60cmp`` (fast, 2 km), ``cap60cmp_L10km`` (fast, 10 km); methods ``qpca``
(QPCA-EnDCF), ``letkf`` (4D-LETKF), ``enkf4d`` (4D-EnKF), ``seq_letkf``
(Seq-LETKF); and realizations (ensemble seeds) 2027, 2028, 2029.

Outputs (``<results_root>/shallow_water/``)::

    scale_matching_trajectories_v3.csv   per (cell, method, seed, window)
    scale_matching_regime_map_v3.csv     per (cell, method) at the shelf window
    all_filter_shelf_gamma_E_v3.csv      per-window RMSE_E, sigma_E, gamma_E
    all_filter_per_window_v3.json        per-window finite-seed means (Table 2)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

# --------------------------------------------------------------------------- #
# Campaign definition (the "v3" campaign reported in the paper)
# --------------------------------------------------------------------------- #
CELLS = {
    "capstone60":      dict(storm="slow", loc_km=2,  n_w=11, w_shelf=6, dt_min=60),
    "cap60_L10km":     dict(storm="slow", loc_km=10, n_w=11, w_shelf=6, dt_min=60),
    "cap60cmp":        dict(storm="fast", loc_km=2,  n_w=6,  w_shelf=4, dt_min=60),
    "cap60cmp_L10km":  dict(storm="fast", loc_km=10, n_w=6,  w_shelf=4, dt_min=60),
}
METHODS = ["qpca", "letkf", "enkf4d", "seq_letkf"]
METHOD_LABELS = {
    "qpca": "QPCA-EnDCF",
    "letkf": "4D-LETKF",
    "enkf4d": "4D-EnKF",
    "seq_letkf": "Seq-LETKF",
}
SEEDS = [2027, 2028, 2029]
DIR_TAG = "_sigmaE"
CAMPAIGN = "v3"
RAMP_MIN = 60.0

# JSON fields carried by every run.  The legacy raw-discrete RMSE is kept
# alongside the physical metrics for provenance.
BASE_FIELDS = [
    "analysis_rmse_energy_equiv_m",      # RMSE_E (metres)
    "analysis_rmse_eta_m",               # RMSE_zeta (metres)
    "analysis_rmse_uv_mps",              # RMSE_uv (m/s)
    "energy_surface_contribution_m2",    # (RMSE_E)^2 split: zeta
    "energy_u_contribution_m2",          #                   u
    "energy_v_contribution_m2",          #                   v
    "forecast_rmse_truth_end",           # mixed-unit forecast
    "analysis_spread_skill_truth",       # gamma_coef = sigma_coef / RMSE_coef
    "analysis_spread",                   # sigma_coef (coefficient-space)
    "n_failed_members",
    "analysis_rmse_truth_end",           # legacy raw-discrete RMSE
    "analysis_rmse_legacy_native",       # (duplicate of the above)
]
SIGMA_E_FIELDS = [
    "analysis_spread_energy_equiv_m",    # sigma_E (metres, energy norm)
    "analysis_spread_eta_m",             # sigma_zeta (metres)
    "analysis_spread_uv_mps",            # sigma_uv (m/s)
    "analysis_spread_skill_energy_equiv",  # gamma_E = sigma_E / RMSE_E
]


def run_dir_name(cell: str, method: str, seed: int) -> str:
    return f"idealized_inlet_5filter_energy_metric_v1{DIR_TAG}_{cell}_gh_{method}_seed{seed}"


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and x == x and math.isfinite(x)


def _load_windows(runs_dir: Path, cell: str, method: str, seed: int):
    """Return the per-window records list, or None if the run is missing."""
    p = Path(runs_dir) / run_dir_name(cell, method, seed) / "result_qpca_endcf.json"
    if not p.is_file():
        return None
    try:
        return json.load(open(p)).get("windows", [])
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Regime map and per-window trajectories (Figures 6-8)
# --------------------------------------------------------------------------- #
def build_scale_match_metrics(runs_dir: Path, out_dir: Path, log=print):
    """Write the trajectory and regime-map CSVs; return the regime-map rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields_physical = list(BASE_FIELDS) + SIGMA_E_FIELDS

    # ----- Per-window trajectories -----
    tri_rows: list[dict] = []
    for cell, cf in CELLS.items():
        for method in METHODS:
            for seed in SEEDS:
                windows = _load_windows(runs_dir, cell, method, seed)
                present = windows is not None
                for wi in range(cf["n_w"]):
                    row = dict(
                        cell=cell, storm=cf["storm"],
                        loc_radius_km=cf["loc_km"],
                        n_windows=cf["n_w"], w_shelf=cf["w_shelf"],
                        method=method, seed=seed,
                        window=wi + 1,
                        time_min=(wi + 1) * cf["dt_min"],
                        is_shelf_crossing=(wi + 1 == cf["w_shelf"]),
                        run_present=present,
                    )
                    if present and wi < len(windows):
                        w = windows[wi]
                        for k in fields_physical:
                            v = w.get(k)
                            if isinstance(v, (int, float)):
                                row[k] = v if math.isfinite(v) else float("nan")
                            else:
                                row[k] = v
                        row["present_in_output"] = True
                    else:
                        for k in fields_physical:
                            row[k] = None
                        row["present_in_output"] = False
                    tri_rows.append(row)
    trip = out_dir / f"scale_matching_trajectories_{CAMPAIGN}.csv"
    fields = ["cell", "storm", "loc_radius_km", "n_windows", "w_shelf",
              "method", "seed", "window", "time_min", "is_shelf_crossing",
              "run_present", "present_in_output"] + fields_physical
    with trip.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in tri_rows:
            w.writerow(r)
    log(f"wrote {trip} ({len(tri_rows)} rows)")

    # ----- Regime map at fixed shelf-crossing window -----
    rm_rows: list[dict] = []
    for cell, cf in CELLS.items():
        w_shelf = cf["w_shelf"]
        for method in METHODS:
            per_seed = []
            n_present, n_finite = 0, 0
            for seed in SEEDS:
                windows = _load_windows(runs_dir, cell, method, seed)
                if windows is None:
                    continue
                n_present += 1
                if w_shelf - 1 >= len(windows):
                    continue
                w_rec = windows[w_shelf - 1]
                rE = w_rec.get("analysis_rmse_energy_equiv_m")
                rZ = w_rec.get("analysis_rmse_eta_m")
                rUV = w_rec.get("analysis_rmse_uv_mps")
                raw = w_rec.get("analysis_rmse_truth_end")
                gam_c = w_rec.get("analysis_spread_skill_truth")
                row_seed = dict(seed=seed,
                                rmse_E=rE, rmse_zeta=rZ, rmse_uv=rUV,
                                rmse_legacy=raw, gamma_coef=gam_c)
                row_seed["sigma_E"] = w_rec.get("analysis_spread_energy_equiv_m")
                row_seed["sigma_zeta"] = w_rec.get("analysis_spread_eta_m")
                row_seed["sigma_uv"] = w_rec.get("analysis_spread_uv_mps")
                row_seed["gamma_E"] = w_rec.get("analysis_spread_skill_energy_equiv")
                if all(_finite(x) for x in (rE, rZ, rUV)):
                    n_finite += 1
                per_seed.append(row_seed)

            def mean_finite(key):
                vals = [r[key] for r in per_seed if key in r and _finite(r[key])]
                return (sum(vals) / len(vals)) if vals else float("nan")
            summary = dict(
                cell=cell, storm=cf["storm"],
                loc_radius_km=cf["loc_km"],
                w_shelf=w_shelf,
                method=method,
                n_seeds_submitted=len(SEEDS),
                n_seeds_present=n_present,
                n_seeds_finite=n_finite,
                mean_rmse_E_m=mean_finite("rmse_E"),
                mean_rmse_zeta_m=mean_finite("rmse_zeta"),
                mean_rmse_uv_mps=mean_finite("rmse_uv"),
                mean_rmse_legacy=mean_finite("rmse_legacy"),
                mean_gamma_coeffspace=mean_finite("gamma_coef"),
            )
            summary["mean_sigma_E_m"] = mean_finite("sigma_E")
            summary["mean_sigma_zeta_m"] = mean_finite("sigma_zeta")
            summary["mean_sigma_uv_mps"] = mean_finite("sigma_uv")
            summary["mean_gamma_E"] = mean_finite("gamma_E")
            for i, r in enumerate(per_seed):
                summary[f"seed{i+1}_id"] = r["seed"]
                summary[f"seed{i+1}_rmse_E_m"] = r["rmse_E"]
                summary[f"seed{i+1}_rmse_zeta_m"] = r["rmse_zeta"]
                summary[f"seed{i+1}_rmse_uv_mps"] = r["rmse_uv"]
                summary[f"seed{i+1}_gamma_coef"] = r["gamma_coef"]
                summary[f"seed{i+1}_sigma_E_m"] = r.get("sigma_E")
                summary[f"seed{i+1}_gamma_E"] = r.get("gamma_E")
            rm_rows.append(summary)
    rmp = out_dir / f"scale_matching_regime_map_{CAMPAIGN}.csv"
    header = []
    for r in rm_rows:
        for k in r:
            if k not in header:
                header.append(k)
    with rmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rm_rows:
            w.writerow(r)
    log(f"wrote {rmp} ({len(rm_rows)} rows)")

    # ----- Pretty-print the regime map (QPCA-EnDCF vs 4D-LETKF) -----
    log("")
    log(f"REGIME MAP (QPCA-EnDCF vs 4D-LETKF, shelf-crossing window)  seeds={SEEDS}")
    hdr_cols = (f"{'cell':<16} {'storm':<5} {'L_km':>5} {'w*':>3} | "
                f"{'QPCA RMSE_E':>12} {'LETKF RMSE_E':>13} {'R_E':>8}  "
                f"{'winner':<22} {'n':>5}  {'QPCA sigma_E':>13} {'QPCA gamma_E':>13}")
    log(hdr_cols)
    log("-" * len(hdr_cols))
    row_by_cm = {(r["cell"], r["method"]): r for r in rm_rows}
    for cell, cf in CELLS.items():
        q = row_by_cm.get((cell, "qpca"))
        l = row_by_cm.get((cell, "letkf"))
        if not q or not l:
            continue
        qE, lE = q["mean_rmse_E_m"], l["mean_rmse_E_m"]
        R = (qE / lE) if (_finite(qE) and _finite(lE) and lE > 0) else float("nan")
        if _finite(R):
            winner = (f"QPCA-EnDCF {1/R:4.2f}x lower" if R < 1
                      else f"4D-LETKF   {R:5.1f}x lower")
        else:
            winner = "nan"
        n = f"{q['n_seeds_finite']}/{l['n_seeds_finite']}"
        log(f"{cell:<16} {cf['storm']:<5} {cf['loc_km']:>5} "
            f"{cf['w_shelf']:>3} | {qE:>12.4f} {lE:>13.4f} {R:>8.3f}  "
            f"{winner:<22} {n:>5}  {q.get('mean_sigma_E_m', float('nan')):>13.4f} "
            f"{q.get('mean_gamma_E', float('nan')):>13.4f}")
    return rm_rows


# --------------------------------------------------------------------------- #
# All-filter per-window table (Table 2)
# --------------------------------------------------------------------------- #
def _to_float(x):
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if not math.isfinite(v):
        return float("inf") if v > 0 else float("-inf")
    return v


def _finite_mean(vals):
    """Arithmetic mean over finite entries; returns (mean, n_finite)."""
    fs = [v for v in vals if math.isfinite(v)]
    if not fs:
        return float("nan"), 0
    return sum(fs) / len(fs), len(fs)


def build_all_filter_table(runs_dir: Path, out_dir: Path, log=print):
    """Write the all-filter CSV and the per-window means behind Table 2."""
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [(cf["storm"], cf["loc_km"], cell, cf["n_w"], cf["w_shelf"])
             for cell, cf in CELLS.items()]

    rows = []
    val_table = {}
    for storm, loc_km, cell, n_win, w_cmp in cells:
        for method in METHODS:
            per_seed_gamma_shelf = []
            for s in SEEDS:
                p = runs_dir / run_dir_name(cell, method, s) / "result_qpca_endcf.json"
                source = f"runs/{run_dir_name(cell, method, s)}/result_qpca_endcf.json"
                if not p.exists():
                    rows.append(dict(
                        method=method, storm=storm, loc_km=loc_km, cell=cell,
                        seed=s, window=w_cmp,
                        t_run_min=float("nan"),
                        t_post_ramp_min=float("nan"),
                        rmse_E_m=float("nan"),
                        sigma_E_m=float("nan"),
                        gamma_E=float("nan"),
                        status="missing_file",
                        source=source,
                    ))
                    per_seed_gamma_shelf.append(float("nan"))
                    continue
                d = json.load(p.open())
                windows = d.get("windows") or []
                seed_shelf_gamma = float("nan")
                for w in windows:
                    w_idx = int(w.get("window", -1))
                    r = _to_float(w.get("analysis_rmse_energy_equiv_m"))
                    sg = _to_float(w.get("analysis_spread_energy_equiv_m"))
                    gE = _to_float(w.get("analysis_spread_skill_energy_equiv"))
                    t_run_min = _to_float(w.get("t_window_end_s", 0)) / 60.0
                    t_post_ramp_min = t_run_min - RAMP_MIN
                    status = "ok"
                    if not math.isfinite(r):
                        status = "rmse_nonfinite"
                    if not math.isfinite(sg):
                        status = status if status != "ok" else "sigma_nonfinite"
                    if not math.isfinite(gE):
                        status = status if status != "ok" else "gamma_nonfinite"
                    rows.append(dict(
                        method=method, storm=storm, loc_km=loc_km, cell=cell,
                        seed=s, window=w_idx,
                        t_run_min=t_run_min,
                        t_post_ramp_min=t_post_ramp_min,
                        rmse_E_m=r,
                        sigma_E_m=sg,
                        gamma_E=gE,
                        status=status,
                        source=source,
                    ))
                    if w_idx == w_cmp:
                        seed_shelf_gamma = gE
                per_seed_gamma_shelf.append(seed_shelf_gamma)
            val_table[(storm, loc_km, method)] = _finite_mean(per_seed_gamma_shelf)

    csv_out = out_dir / f"all_filter_shelf_gamma_E_{CAMPAIGN}.csv"
    fieldnames = ["method", "storm", "loc_km", "cell", "seed", "window",
                  "t_run_min", "t_post_ramp_min", "rmse_E_m", "sigma_E_m",
                  "gamma_E", "status", "source"]
    with csv_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    log(f"wrote {csv_out}  ({len(rows)} rows)")

    log("")
    log("Shelf-window finite-seed mean gamma_E:")
    log(f"{'cell':<12} {'method':<10} {'mean_gamma_E':>14} {'n_seed':>8}")
    log("-" * 50)
    for storm, loc_km, cell, n_win, w_cmp in cells:
        for method in METHODS:
            g, n = val_table[(storm, loc_km, method)]
            log(f"{storm}/{loc_km:>2}km   {method:<10} {g:>14.6f} {n:>8}")

    # Per-window finite-seed means: the entries of Table 2 (the paper's table
    # was typeset from these values).
    log("")
    log("Table 2: per-window finite-seed means of RMSE_E,w (m) and gamma_E,w "
        "(number of finite realizations in parentheses); t is minutes from the "
        "run start, including the one-hour ramp.")
    header = f"{'storm/loc':<12} {'W':>3} {'t':>4}  "
    for method in METHODS:
        header += f"{METHOD_LABELS[method] + ' RMSE':>18} {'gamma':>11}  "
    log(header)
    log("-" * len(header))
    per_window_summary = {}
    for storm, loc_km, cell, n_win, w_cmp in cells:
        for w_target in range(1, w_cmp + 1):
            line = f"{storm}/{loc_km:>2}km   {w_target:>3} {w_target*60 + int(RAMP_MIN):>4}  "
            for method in METHODS:
                vals_r, vals_g = [], []
                for s in SEEDS:
                    for r in rows:
                        if (r["method"] == method and r["storm"] == storm and
                                r["loc_km"] == loc_km and r["seed"] == s and
                                r["window"] == w_target):
                            vals_r.append(r["rmse_E_m"])
                            vals_g.append(r["gamma_E"])
                            break
                mr, nr = _finite_mean(vals_r)
                mg, ng = _finite_mean(vals_g)
                per_window_summary[(storm, loc_km, method, w_target)] = (mr, nr, mg, ng)
                if math.isfinite(mr):
                    r_disp = f"{mr:>10.3e}" if abs(mr) > 1e4 else f"{mr:>10.3f}"
                else:
                    r_disp = f"{'nan':>10}"
                if math.isfinite(mg):
                    g_disp = (f"{mg:>8.3e}" if (abs(mg) > 100 or (mg != 0 and abs(mg) < 1e-3))
                              else f"{mg:>8.3f}")
                else:
                    g_disp = f"{'nan':>8}"
                line += f"     {r_disp}({nr:>1}) {g_disp}({ng:>1})  "
            log(line)
        log("")

    packed = {}
    for key, (mr, nr, mg, ng) in per_window_summary.items():
        storm, loc_km, method, w = key
        packed[f"{storm}_{loc_km}km_{method}_W{w}"] = dict(
            mean_rmse_E=mr, n_finite_rmse=nr,
            mean_gamma_E=mg, n_finite_gamma=ng,
        )
    summary_out = out_dir / f"all_filter_per_window_{CAMPAIGN}.json"
    summary_out.write_text(json.dumps(
        packed, indent=2,
        default=lambda x: None if (isinstance(x, float) and not math.isfinite(x)) else x))
    log(f"wrote {summary_out}")
    return rows


def build_all(runs_dir: Path, out_dir: Path, log=print):
    """Rebuild every shallow-water table from the run JSONs in ``runs_dir``."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"No shallow-water run directories under {runs_dir}.")
    build_scale_match_metrics(runs_dir, out_dir, log=log)
    build_all_filter_table(runs_dir, out_dir, log=log)
    return Path(out_dir)
