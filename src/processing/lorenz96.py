"""
Reduce raw Lorenz-96 experiment checkpoints to the processed results behind
Figures 1-5 of the paper.

Every reduction here is the one the original analysis notebooks applied before
plotting (``01_calibration.ipynb`` for Figures 1-3, ``10_bias_variance.ipynb``
for Figures 4-5), restated without the plotting so the published and the
regenerated results pass through identical code.  Nothing here runs an
experiment: a missing checkpoint is an error, never a reason to compute it.

Raw layout expected under ``raw_root`` (the layout the drivers write, see
``src/config.py``)::

    baseline_extended/seeds/seed_<s>.npz                  Figures 1-3  (baseline.py)
    diagnostics/coverage_v1/seeds/seed_<s>.npz            Figure 2     (coverage.py)
    sensitivity/ensemble_size/N_<n>/seeds/seed_<s>.npz    Figure 3     (baseline.py)
    diagnostics/bias_variance_analysis_100x100/*.npz      Figure 4     (bias_variance_mc.py)
    diagnostics/n_kappa_bias_variance/<label>_<hash>/     Figure 5     (n_kappa_bias_variance.py)

Processed files written to ``<results_root>/lorenz96/``::

    calibration_N10.npz          per-seed, per-window RMSE and spread at N = 10
    coverage_ranks_N10.npz       per-seed, per-window rank counts at N = 10
    ensemble_size_sweep.npz      settled RMSE medians for N in {5, ..., 30}
    hierarchical_bias_variance.npz  per-truth-seed bias^2 / variance / MSE traces
    n_kappa_bootstrap.npz        hierarchical bootstrap draws over the (N, kappa) grid
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.experiments import n_kappa_bias_variance as nk
from src.experiments.baseline import (
    EXTENDED_FILTER_ORDER,
    _load_shard,
    ensemble_results_dir,
    load_config,
    resolve_seeds,
    shard_path,
)
from src.experiments.bias_variance_mc import (
    FILTER_NAMES as BIAS_VARIANCE_FILTERS,
    _hierarchical_checkpoint_matches,
    _hierarchical_checkpoint_path,
    make_truth_seeds,
)
from src.experiments.coverage import (
    coverage_shard_path,
    expected_metadata,
    load_shard as load_coverage_shard,
)
from src.config import DIAGNOSTICS_EXPERIMENT_DIR
from src.utils.config import load_experiment_config
from src.utils.observations import generate_truth

# --------------------------------------------------------------------------- #
# Settings fixed by the paper (and by the notebooks that produced its figures)
# --------------------------------------------------------------------------- #

#: Independent repetitions in the N = 10 experiment and the ensemble-size sweep.
N_SEEDS = 100
#: Windows before this are burn-in; settled statistics use windows 41-1000.
BURN_IN = 40
#: Seeds over which the climatological (no-skill) RMSE is averaged.
N_CLIMATOLOGY_SEEDS = 10
#: Ensemble sizes in the Figure 3 sweep.
ENSEMBLE_SIZES = (5, 10, 15, 20, 25, 30)

#: Hierarchical experiment (Figure 4): truth seeds drawn and used.
N_TRUTH_SEEDS_DRAWN = 117
TARGET_TRUTH_SEEDS = 100
TRUTH_SEED_BASE = 20260728
HIERARCHICAL_N_WINDOWS = 1000
HIERARCHICAL_N_MC = 100

#: (N, kappa) experiment (Figure 5).
NK_ENSEMBLE_SIZES = (5, 10, 20, 40)
NK_KAPPA_VALUES = (1, 2, 4, 8)
NK_N_BOOTSTRAP = 2000
NK_N_INNER = 250

FILENAMES = {
    "calibration": "calibration_N10.npz",
    "coverage": "coverage_ranks_N10.npz",
    "ensemble_sweep": "ensemble_size_sweep.npz",
    "hierarchical": "hierarchical_bias_variance.npz",
    "n_kappa": "n_kappa_bootstrap.npz",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _meta(**fields) -> np.ndarray:
    """Metadata as a JSON string, so the archives load with allow_pickle=False."""
    return np.asarray(json.dumps(fields, sort_keys=True, default=str))


def read_meta(archive) -> Dict:
    return json.loads(str(archive["meta"]))


def _portable_provenance(info: Dict, raw_root: Path) -> Dict:
    """Provenance with any absolute path rewritten relative to the raw root.

    The drivers record absolute paths (e.g. where a restart law was read from);
    those identify the producing machine and are meaningless after cloning.
    """
    out = dict(info)
    if "path" in out:
        path = Path(str(out["path"]))
        try:
            out["path"] = "<raw>/" + path.resolve().relative_to(Path(raw_root).resolve()).as_posix()
        except ValueError:
            out["path"] = "<external>/" + path.name
    return out


def _save(path: Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path


def window_axis(series, window_len: int, n_windows: int) -> np.ndarray:
    """Put a diagnostic series on the per-window axis shared by all filters.

    Verbatim from ``01_calibration.ipynb``.  The sequential filters report once
    per observation time and the 4D filters once per window, so a
    per-observation series is subsampled by the window length starting at
    ``window_len - 1``: ``run_filter`` scores a sequential sample k against
    ``truth[k + 1]`` and a 4D window w against ``truth[(w + 1) * window_len]``,
    so it is the *last* observation time of each window that shares a
    verification time with the 4D filters.  ``reshape(-1)`` matters: the stored
    spread shards are (K, 1) while the rmse shards are (K,).
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    return series[window_len - 1 :: window_len] if series.size > n_windows else series


def _load_baseline_runs(out_dir: Path, seeds: Sequence[int], filters: Sequence[str]):
    """``{label: [per-seed results]}`` from existing shards only.

    Same loading and the same contents check as ``baseline.run_experiment`` when
    every shard is cached, but a missing shard raises instead of being computed.
    """
    runs = {label: [] for label in filters}
    for seed in seeds:
        path = shard_path(seed, out_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing baseline shard {path}. Run the corresponding Level 2 "
                "experiment first (scripts/run_all_experiments.py)."
            )
        shard = _load_shard(path)
        missing = [label for label in filters if label not in shard]
        if missing:
            raise KeyError(f"{path} holds {sorted(shard)} but {missing} were requested.")
        for label in filters:
            runs[label].append(shard[label])
    return runs


def climatological_rmse(config: Dict, seed_list: Sequence[int], burn_in: int = BURN_IN) -> float:
    """RMSE of predicting each state's long-run mean: the no-skill reference.

    Verbatim from ``01_calibration.ipynb``.  It depends only on the truth
    trajectories, which are regenerated deterministically from the seeds.
    """
    window_len = config["assimilation"]["window_length"]
    model_cfg, obs_cfg = config["model"], config["observations"]
    values = []
    for seed in seed_list:
        np.random.seed(seed)
        x0 = np.random.randn(model_cfg["n"])
        truth = generate_truth(
            x0,
            config["derived"]["total_observations"],
            obs_cfg["steps_per_obs"],
            model_cfg["dt"],
            model_cfg["F"],
            model_cfg.get("spinup_steps", 0),
        )[burn_in * window_len:]
        values.append(np.sqrt(((truth - truth.mean(axis=0)) ** 2).mean()))
    return float(np.mean(values))


# --------------------------------------------------------------------------- #
# Figures 1-3: the N = 10 experiment, its coverage diagnostic, and the N sweep
# --------------------------------------------------------------------------- #
def extract_calibration(raw_root: Path, out_dir: Path) -> Path:
    """Window-axis RMSE and spread for every seed and filter at N = 10."""
    config = load_config()
    seeds = resolve_seeds(config, N_SEEDS)
    filters = list(EXTENDED_FILTER_ORDER)
    window_len = config["assimilation"]["window_length"]
    n_windows = config["assimilation"]["n_windows"]

    runs = _load_baseline_runs(Path(raw_root) / "baseline_extended", seeds, filters)
    rmse = np.stack([
        np.stack([window_axis(r["rmse"], window_len, n_windows) for r in runs[label]])
        for label in filters
    ])
    spread = np.stack([
        np.stack([window_axis(r["spread"], window_len, n_windows) for r in runs[label]])
        for label in filters
    ])
    assert rmse.shape == spread.shape == (len(filters), len(seeds), n_windows)

    climatology = climatological_rmse(config, seeds[:N_CLIMATOLOGY_SEEDS])
    return _save(
        Path(out_dir) / FILENAMES["calibration"],
        filters=np.asarray(filters),
        seeds=np.asarray(seeds, dtype=np.int64),
        rmse=rmse,
        spread=spread,
        climatology_rmse=np.asarray(climatology),
        meta=_meta(
            description="Window-endpoint analysis RMSE and ensemble spread, "
            "Lorenz-96, N = 10, 100 seeds x 1000 windows (Figures 1-3).",
            source="src/experiments/baseline.py (EXTENDED_FILTER_ORDER)",
            config="experiments/lorenz96/lorenz96_baseline.yaml",
            window_length=window_len,
            n_windows=n_windows,
            ensemble_size=config["assimilation"]["ensemble_size"],
            state_dim=config["model"]["n"],
            obs_dim=config["observations"]["m"],
            total_observations=config["derived"]["total_observations"],
            burn_in=BURN_IN,
            climatology_seeds=list(seeds[:N_CLIMATOLOGY_SEEDS]),
            axes="(filter, seed, window)",
        ),
    )


def extract_coverage(raw_root: Path, out_dir: Path) -> Path:
    """Common-endpoint rank counts for every seed, window and filter at N = 10."""
    config = load_config()
    seeds = resolve_seeds(config, N_SEEDS)
    filters = list(EXTENDED_FILTER_ORDER)
    meta = expected_metadata(config, filters)
    cache = Path(raw_root) / "diagnostics" / "coverage_v1"

    blocks = {label: [] for label in filters}
    for seed in seeds:
        shard = load_coverage_shard(coverage_shard_path(seed, cache), seed, meta, filters)
        if shard is None:
            raise FileNotFoundError(
                f"Coverage shard for seed {seed} is missing or does not match the "
                f"configuration under {cache}."
            )
        for label in filters:
            blocks[label].append(shard[label])
    rank_counts = np.stack([np.stack(blocks[label]) for label in filters])
    n_windows = meta["n_windows"]
    assert rank_counts.shape == (len(filters), len(seeds), n_windows, meta["ensemble_size"] + 1)
    assert (rank_counts.sum(axis=3) == meta["state_dim"]).all()

    return _save(
        Path(out_dir) / FILENAMES["coverage"],
        filters=np.asarray(filters),
        seeds=np.asarray(seeds, dtype=np.int64),
        rank_counts=rank_counts,
        meta=_meta(
            description="Rank of the truth among the ensemble members (members "
            "strictly below the truth, 0..N) counted over the 40 state components "
            "at every window endpoint; Lorenz-96, N = 10 (Figure 2).",
            source="src/experiments/coverage.py",
            config="experiments/lorenz96/lorenz96_baseline.yaml",
            axes="(filter, seed, window, rank bin)",
            **{key: value for key, value in meta.items() if key != "filters"},
        ),
    )


def extract_ensemble_sweep(raw_root: Path, out_dir: Path) -> Path:
    """Settled-median RMSE per filter and ensemble size, overall and per seed."""
    config = load_config()
    seeds = resolve_seeds(config, N_SEEDS)
    filters = list(EXTENDED_FILTER_ORDER)
    window_len = config["assimilation"]["window_length"]
    n_windows = config["assimilation"]["n_windows"]
    sweep_root = Path(raw_root) / "sensitivity"

    settled = np.empty((len(filters), len(ENSEMBLE_SIZES)))
    per_seed = np.empty((len(filters), len(ENSEMBLE_SIZES), len(seeds)))
    for j, n in enumerate(ENSEMBLE_SIZES):
        out = ensemble_results_dir(n, root=sweep_root)
        runs = _load_baseline_runs(out, seeds, filters)
        for i, label in enumerate(filters):
            matrix = np.stack([window_axis(r["rmse"], window_len, n_windows) for r in runs[label]])
            # The two reductions Figure 3 draws (01_calibration.ipynb, section 6).
            settled[i, j] = float(np.median(matrix[:, BURN_IN:]))
            per_seed[i, j] = np.median(matrix[:, BURN_IN:], axis=1)
    # The no-skill reference Figure 3's summary quotes; it depends only on the
    # truth trajectories, which do not change with N.
    climatology = climatological_rmse(config, seeds[:N_CLIMATOLOGY_SEEDS])

    return _save(
        Path(out_dir) / FILENAMES["ensemble_sweep"],
        filters=np.asarray(filters),
        ensemble_sizes=np.asarray(ENSEMBLE_SIZES, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        settled_median_rmse=settled,
        per_seed_settled_median_rmse=per_seed,
        climatology_rmse=np.asarray(climatology),
        meta=_meta(
            description="Settled analysis RMSE (windows 41-1000) for N in "
            "{5, 10, 15, 20, 25, 30}: the median over all seed-window samples, "
            "and the per-seed medians (Figure 3).",
            source="src/experiments/baseline.py with config_at_ensemble_size",
            config="experiments/lorenz96/lorenz96_baseline.yaml",
            burn_in=BURN_IN,
            window_length=window_len,
            n_windows=n_windows,
            axes="settled_median_rmse: (filter, N); per_seed: (filter, N, seed)",
        ),
    )


# --------------------------------------------------------------------------- #
# Figure 4: hierarchical 100 x 100 bias-variance decomposition
# --------------------------------------------------------------------------- #
def hierarchical_config() -> Dict:
    """The 100 x 100 configuration, built exactly as 10_bias_variance.ipynb does."""
    config = load_experiment_config(DIAGNOSTICS_EXPERIMENT_DIR / "bias_variance_analysis.yaml")
    config_100x100 = deepcopy(config)
    config_100x100["assimilation"]["n_windows"] = HIERARCHICAL_N_WINDOWS
    config_100x100["assimilation"]["total_observations"] = (
        config_100x100["assimilation"]["window_length"]
        * config_100x100["assimilation"]["n_windows"]
    )
    config_100x100["bias_variance"]["n_monte_carlo"] = HIERARCHICAL_N_MC
    config_100x100["experiment"]["seeds"] = make_truth_seeds(
        n_seeds=N_TRUTH_SEEDS_DRAWN, base_seed=TRUTH_SEED_BASE
    )
    return config_100x100


def extract_hierarchical(raw_root: Path, out_dir: Path) -> Path:
    """Per-truth-seed bias^2, variance and MSE traces for the 100 used truth seeds."""
    config = hierarchical_config()
    results_dir = Path(raw_root) / "diagnostics" / "bias_variance_analysis_100x100"
    n_mc = int(config["bias_variance"]["n_monte_carlo"])
    W = int(config["assimilation"]["n_windows"])
    n = int(config["model"]["n"])
    keys = ("bias_squared_trace", "variance_trace", "mse_trace")

    # Same completeness rule as bias_variance_mc.load_hierarchical_results and
    # the notebook: a checkpoint counts only if it matches the configuration,
    # and a truth seed counts only if every filter has one.
    loaded: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    for seed in config["experiment"]["seeds"]:
        loaded[seed] = {}
        for name in BIAS_VARIANCE_FILTERS:
            path = _hierarchical_checkpoint_path(results_dir, seed, name)
            if not path.exists() or not _hierarchical_checkpoint_matches(path, n_mc, W, n, name):
                continue
            with np.load(path) as data:
                loaded[seed][name] = {key: data[key] for key in (*keys, "bootstrap_spinup_windows")}

    drawn = list(config["experiment"]["seeds"])
    complete = [s for s in drawn if all(name in loaded[s] for name in BIAS_VARIANCE_FILTERS)]
    if len(complete) < TARGET_TRUTH_SEEDS:
        raise FileNotFoundError(
            f"Only {len(complete)} complete truth seeds under {results_dir}; "
            f"{TARGET_TRUTH_SEEDS} are needed. Run the hierarchical experiment first."
        )
    used = complete[:TARGET_TRUTH_SEEDS]
    excluded = [s for s in drawn if s not in complete]

    traces = {
        key: np.stack([
            np.stack([loaded[s][name][key] for s in used]) for name in BIAS_VARIANCE_FILTERS
        ])
        for key in keys
    }
    gram_start = int(loaded[used[0]][BIAS_VARIANCE_FILTERS[0]]["bootstrap_spinup_windows"])
    missing_by_filter = {
        name: [int(s) for s in drawn if name not in loaded[s]] for name in BIAS_VARIANCE_FILTERS
    }

    return _save(
        Path(out_dir) / FILENAMES["hierarchical"],
        filters=np.asarray(BIAS_VARIANCE_FILTERS),
        truth_seeds_drawn=np.asarray(drawn, dtype=np.int64),
        complete_truth_seeds=np.asarray(complete, dtype=np.int64),
        truth_seeds=np.asarray(used, dtype=np.int64),
        bias_squared_trace=traces["bias_squared_trace"],
        variance_trace=traces["variance_trace"],
        mse_trace=traces["mse_trace"],
        meta=_meta(
            description="Plug-in squared bias, across-realization variance and MSE "
            "of the analysis mean, averaged over the 40 state components, at every "
            "window (index 0 = initial ensemble, then windows 1-1000), for the first "
            "100 truth seeds on which all five filters completed (Figure 4).",
            source="src/experiments/bias_variance_mc.py --hierarchical",
            config="experiments/lorenz96/bias_variance_analysis.yaml "
            "(n_windows=1000, n_monte_carlo=100, 117 truth seeds from SeedSequence(20260728))",
            n_realizations=n_mc,
            n_windows=W,
            state_dim=n,
            bootstrap_spinup_windows=gram_start,
            excluded_truth_seeds=[int(s) for s in excluded],
            missing_checkpoints_by_filter=missing_by_filter,
            axes="(filter, truth seed, window 0..1000)",
        ),
    )


# --------------------------------------------------------------------------- #
# Figure 5: the (N, kappa) response surface under independent restarts
# --------------------------------------------------------------------------- #
def find_n_kappa_run(raw_root: Path) -> Path:
    """The run directory the notebook would load: the non-smoke run with most cells."""
    root = Path(raw_root) / "diagnostics" / "n_kappa_bias_variance"
    available = [run for run in nk.discover_n_kappa_runs(root) if run["label"] != "smoke"]
    if not available:
        raise FileNotFoundError(f"No (N, kappa) run found under {root}.")
    return Path(available[0]["results_dir"])


def extract_n_kappa(raw_root: Path, out_dir: Path) -> Path:
    """Hierarchical bootstrap draws over the (N, kappa) grid, restart regime."""
    run_dir = find_n_kappa_run(raw_root)
    run = nk.load_run(run_dir)
    settings = run["settings"]
    cells = run["cells"]
    report = run["seed_report"]
    sizes = sorted({int(N) for N in settings.ensemble_sizes})
    kappas = sorted({int(k) for k in settings.kappa_values})
    if sizes != sorted(NK_ENSEMBLE_SIZES) or kappas != sorted(NK_KAPPA_VALUES):
        raise ValueError(
            f"{run_dir} covers N={sizes}, kappa={kappas}, not the published grid "
            f"N={list(NK_ENSEMBLE_SIZES)}, kappa={list(NK_KAPPA_VALUES)}."
        )
    if len(cells) != report["n_required_cells"]:
        raise FileNotFoundError(
            f"{run_dir} holds {len(cells)}/{report['n_required_cells']} cells; "
            "the run is incomplete."
        )
    seeds = report["common"]

    # Exactly the notebook's call (default random_seed=20260810, stride 1).
    bootstrap = nk.bootstrap_grid(
        cells, seeds, settings, n_bootstrap=NK_N_BOOTSTRAP, n_inner=NK_N_INNER
    )
    design_points = [(int(N), int(k)) for N, k in settings.design_points]
    draws = np.stack([bootstrap["draws"][("restart", N, k)] for N, k in design_points])

    # Plug-in (no resampling) MSE and squared bias of every cell, from the same
    # error Gram the bootstrap uses with all counts equal to one.  Not used by
    # the figure; it lets a single regenerated cell be checked against the
    # published run (scripts/verify_results.py --spot-check).
    regimes = [str(g) for g in settings.regimes]
    cell_stats = np.empty((len(regimes), len(design_points), len(seeds), 2))
    for g, regime in enumerate(regimes):
        for p, (N, k) in enumerate(design_points):
            for s, seed in enumerate(seeds):
                gram = cells[(regime, N, k, seed)]["error_gram"]
                M = gram.shape[0]
                ones = np.ones(M)
                cell_stats[g, p, s, 0] = ones @ np.diag(gram) / M
                cell_stats[g, p, s, 1] = ones @ gram @ ones / M**2

    return _save(
        Path(out_dir) / FILENAMES["n_kappa"],
        ensemble_sizes=np.asarray(sizes, dtype=np.int64),
        kappa_values=np.asarray(kappas, dtype=np.int64),
        design_points=np.asarray(design_points, dtype=np.int64),
        metric_names=np.asarray(nk.METRIC_NAMES),
        truth_seeds=np.asarray(seeds, dtype=np.int64),
        restart_draws=draws,
        regimes=np.asarray(regimes),
        cell_plugin_stats=cell_stats,
        meta=_meta(
            description="Paired hierarchical bootstrap draws (2000 outer x 250 inner) "
            "of the window-averaged, state-normalized MSE, plug-in and debiased "
            "squared bias, variance and T = Bias^2_db + Variance of QPCA-EnDCF, "
            "independent-restart regime, at each valid (N, kappa) (Figure 5).",
            source="src/experiments/n_kappa_bias_variance.py --n-kappa",
            config="experiments/lorenz96/n_kappa_bias_variance.yaml",
            run_directory=run_dir.name,
            run_label=str(run["manifest"].get("label")),
            config_hash=str(run["config_hash"]),
            restart_law=_portable_provenance(run["calibration"], raw_root),
            n_mc=int(settings.n_mc),
            n_windows=int(settings.n_windows),
            window_stride=int(settings.window_stride),
            spinup_windows=int(settings.spinup_windows),
            mean_error_mode=settings.mean_error_mode,
            n_evaluation_seeds=len(settings.evaluation_seeds),
            excluded_truth_seeds={str(k): v for k, v in report["excluded"].items()},
            n_bootstrap=int(bootstrap["n_bootstrap"]),
            n_inner=int(bootstrap["n_inner"]),
            bootstrap_random_seed=int(bootstrap["random_seed"]),
            axes="restart_draws: (design point, bootstrap draw, metric); "
            "cell_plugin_stats: (regime, design point, truth seed, [mse, bias_squared])",
        ),
    )


EXTRACTORS = {
    "calibration": extract_calibration,
    "coverage": extract_coverage,
    "ensemble_sweep": extract_ensemble_sweep,
    "hierarchical": extract_hierarchical,
    "n_kappa": extract_n_kappa,
}


def extract_all(raw_root: Path, out_dir: Path, only: List[str] | None = None) -> List[Path]:
    written = []
    for name, extractor in EXTRACTORS.items():
        if only and name not in only:
            continue
        print(f"[lorenz96] {name}: reducing {raw_root} ...", flush=True)
        path = extractor(raw_root, out_dir)
        print(f"[lorenz96] {name}: wrote {path}", flush=True)
        written.append(path)
    return written
