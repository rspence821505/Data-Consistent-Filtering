"""
Baseline Lorenz-96 experiment: Seq-EnKF vs 4D-EnKF vs QPCA-EnDCF.

Each random seed is an independent experiment (its own truth trajectory,
observations and initial ensemble), so seeds are dispatched across worker
processes. Everything inside a seed is sequential in time and stays serial;
the ensemble dimension is exploited by advancing all members in a single
`l96_rhs` call rather than by adding more workers.

Each finished seed is written to its own shard under
``results/baseline/seeds/seed_<s>.npz`` and skipped on a re-run, so a long
sweep is resumable and can be extended with more seeds later. Shards hold only
the reduced diagnostics the notebooks consume (RMSE / misfit / spread
trajectories plus rank-histogram counts); the full ensemble history is ~21 MB
per seed and is never persisted.

Usage
-----
    python -m src.experiments.baseline --n-seeds 100 --n-jobs 12
    python -m src.experiments.baseline --seeds 42 123 --force
"""

from __future__ import annotations

# Each worker runs its own BLAS calls; without this, `n_jobs` processes each
# spawn a full thread pool and oversubscribe the machine. Must be set before
# numpy is imported, which is why it sits above the imports.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    BASELINE_EXPERIMENT_DIR,
    BASELINE_RESULTS_DIR,
    SENSITIVITY_RESULTS_DIR,
)
from src.filters.enkf_4d import StochasticEnKF4D
from src.filters.letkf_4d import LETKF4D
from src.filters.qpca_endcf import QPCAEnDCF
from src.filters.seq_enkf import StochasticEnKF
from src.filters.seq_letkf import SequentialLETKF
from src.models.lorenz96 import l96_rhs, rk4_step
from src.utils.linalg import sym_posdef_inverse
from src.utils.localization import build_localization_weights
from src.utils.metrics import compute_ensemble_spread, compute_rmse, rel_data_misfit
from src.utils.observations import (
    build_obs_operator,
    generate_observations,
    generate_truth,
    initialize_ensemble,
)

# Default comparison set (notebooks 00, 02, 09). Reordering it would change
# every filter's random draws, so it is fixed.
FILTER_ORDER = ["Seq-EnKF", "4D-EnKF", "QPCA-EnDCF"]

# Extended set (notebook 01). It resolves the three-way comparison into the
# separate mechanisms that distinguish the filters:
#
#   Seq-EnKF -> Seq-ETKF    replacing the stochastic gain update with the
#                           deterministic square-root transform, at the
#                           sequential cadence
#   Seq-ETKF -> 4D-ETKF     the update cadence alone: same deterministic
#                           transform, one joint update per window
#   4D-EnKF  -> 4D-ETKF     the same substitution as the first arrow, at the 4D
#                           cadence
#   4D-ETKF  -> QPCA-EnDCF  adding residual-space truncation
#
# Neither transform arrow changes one thing. Swapping a stochastic gain update
# for a square-root transform both deletes the observation perturbations and
# replaces the analysis-anomaly operator, and those two changes are individually
# large (~2x) and opposite in sign. A 4D-EnKF-det control -- the same gain with
# the perturbations simply not drawn -- used to sit between 4D-EnKF and 4D-ETKF
# and measure each separately; it has been withdrawn, so the set now reports the
# net effect of the pair and cannot decompose it.
#
# Seq-ETKF is the one filter here that is not four-dimensional. It exists to test
# the cadence itself, completing a 2x2 over {sequential, 4D} x {stochastic gain,
# deterministic transform}: the 4D arrows are only about the update, and the
# cadence is what the sequential leg varies instead.
#
# The list is also the notebook's panel order: the filters that collapse come
# first and the one that escapes comes last. Order is presentation only -- RNG
# streams are keyed by name via _STREAM_ORDER, so reordering here changes no
# result.
EXTENDED_FILTER_ORDER = [
    "Seq-EnKF",
    "Seq-ETKF",
    "4D-EnKF",
    "4D-ETKF",
    "QPCA-EnDCF",
]

# Seq-LETKF, 4D-LETKF, QPCA-EnDCF-L and 4D-EnKF-det were withdrawn from the
# comparison. What is left of them, and why:
#
#   _STREAM_ORDER still names all four, and must. It is append-only because the
#   observation stream is spawned after it (`run_seed`), so dropping names would
#   shorten the spawn and hand every remaining filter a different observation
#   realization -- every cached shard would silently become stale. Leaving them
#   costs four unused SeedSequence children and keeps the five surviving filters
#   bit-identical to the runs that produced the current cache.
#
#   `_build_filter_specs`, `FILTER_KEYS` and their YAML entries also stay, so
#   re-adding any of them is a one-line change to the list above rather than a
#   reconstruction. A shard holding more filters than were requested is fine --
#   `run_experiment` only checks that every *requested* label is present -- so
#   the existing nine-filter cache serves this five-filter set unchanged.
#
#   Withdrawing 4D-EnKF-det costs the comparison its single-variable control for
#   the observation perturbations: 4D-EnKF -> 4D-ETKF now moves two things at
#   once (the perturbations, and the analysis-anomaly operator), and nothing
#   left in the set can separate them.
WITHDRAWN_FILTERS = ["Seq-LETKF", "4D-LETKF", "QPCA-EnDCF-L", "4D-EnKF-det"]

FILTER_KEYS = {
    "Seq-EnKF": "seq_enkf",
    "Seq-ETKF": "seq_etkf",
    "Seq-LETKF": "seq_letkf",
    "4D-EnKF": "enkf_4d",
    "4D-EnKF-det": "enkf_4d_det",
    "4D-ETKF": "etkf_4d",
    "4D-LETKF": "letkf_4d",
    "QPCA-EnDCF": "qpca_endcf",
    "QPCA-EnDCF-L": "qpca_endcf_loc",
}

# Order in which per-seed RNG child streams are spawned. The original three come
# first and additions are appended, so `SeedSequence(seed).spawn(k)` yields the
# same first three children whatever the requested set -- the two stochastic
# filters therefore draw exactly the perturbations they drew before the
# extended set existed. Never reorder this; append only. (`spawn` is
# prefix-stable, so growing this list leaves every existing stream untouched.)
#
# One further child is spawned after these, for the observation noise; because
# it is last, adding it left every filter stream unchanged. See `run_seed`.
_STREAM_ORDER = [
    "Seq-EnKF",
    "4D-EnKF",
    "QPCA-EnDCF",
    "4D-ETKF",
    "4D-LETKF",
    "QPCA-EnDCF-L",
    "4D-EnKF-det",
    "Seq-ETKF",
    "Seq-LETKF",
]

METRICS = ("rmse", "misfit", "spread", "rank_hist")

SHARD_DIR_NAME = "seeds"
DEFAULT_CONFIG = BASELINE_EXPERIMENT_DIR / "lorenz96_baseline.yaml"

# The extended set writes its own shards. A shard stores whichever filters were
# requested when it was written, and run_experiment only checks that a shard
# file exists, not what is inside it -- so mixing the two sets in one directory
# would make the three-filter cache look complete to an eight-filter run.
EXTENDED_RESULTS_DIR = BASELINE_RESULTS_DIR.parent / "baseline_extended"

# Fixed so that `--n-seeds N` always yields the same seed list.
SEED_EXTENSION_ENTROPY = 20260728


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def load_config(config_path=DEFAULT_CONFIG) -> dict:
    """Load the experiment YAML and attach the derived quantities."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assim_cfg = config["assimilation"]
    config["derived"] = {
        "total_observations": assim_cfg.get(
            "total_observations", assim_cfg["n_windows"] * assim_cfg["window_length"]
        ),
        "time_between_obs": config["model"]["dt"] * config["observations"]["steps_per_obs"],
    }
    return config


def resolve_seeds(config: dict, n_seeds: int | None = None) -> list[int]:
    """
    Return the seed list to run.

    The seeds listed in the config always come first, so results computed with
    the original 5-seed list stay valid; any extra seeds are drawn from a fixed
    SeedSequence, making `--n-seeds N` reproducible and prefix-stable (asking
    for 200 seeds later reuses the first 100 shards).
    """
    base = [int(s) for s in config["experiment"]["seeds"]]
    if n_seeds is None:
        return base
    if n_seeds <= len(base):
        return base[:n_seeds]

    pool = np.random.SeedSequence(SEED_EXTENSION_ENTROPY).generate_state(
        8 * n_seeds, dtype=np.uint32
    )
    seeds, seen = list(base), set(base)
    for value in pool:
        candidate = int(value) % (2**31 - 1)
        if candidate in seen:
            continue
        seeds.append(candidate)
        seen.add(candidate)
        if len(seeds) == n_seeds:
            return seeds
    raise RuntimeError(f"Could not draw {n_seeds} distinct seeds")


def _filter_params(config: dict) -> dict:
    return {entry["name"]: entry.get("params", {}) for entry in config["filters"]}


# --------------------------------------------------------------------------
# Observation-noise sensitivity
# --------------------------------------------------------------------------
def config_at_noise(config: dict, noise_std: float) -> dict:
    """A copy of `config` with the observation noise level replaced.

    `run_seed` builds both the synthetic observations and the filters'
    ``R = sigma^2 I`` from ``observations.noise_std``, so this one substitution
    moves the whole experiment to a new noise level with R still matched to the
    noise that generated the data. Nothing else in the experiment moves with it:
    the truth trajectory, the initial ensemble and the seed list are all drawn
    independently of sigma, and `generate_observations` scales one seed-fixed
    standard normal draw by sigma rather than re-drawing at each level -- so a
    sweep over sigma is paired, and every level sees the same noise pattern at a
    different amplitude.

    The copy is deep because the filter entries are nested dicts; mutating the
    caller's config instead would silently move the *main* experiment too.
    """
    cfg = copy.deepcopy(config)
    cfg["observations"]["noise_std"] = float(noise_std)
    return cfg


NOISE_SWEEP_DIR_NAME = "observation_noise"


def noise_results_dir(noise_std: float, root=SENSITIVITY_RESULTS_DIR) -> Path:
    """Cache directory for one noise level of the sensitivity sweep.

    A shard's filename carries only its seed, and `run_experiment` reuses a
    shard on the strength of that filename, so the noise level has to be in the
    path -- otherwise a sigma = 0.5 run would silently load the shards written
    at sigma = 1.5, the same trap that keeps the three- and nine-filter sets in
    separate directories. The sweep therefore lives under `results/sensitivity/`
    and never writes into the baseline directories.

    It claims one named subtree of that root rather than the root itself:
    `SENSITIVITY_RESULTS_DIR` is the home of sensitivity studies in general, and
    observation noise is one axis among several one might sweep.

    The level is formatted with `repr`, which round-trips a float exactly, so
    two distinct sigmas can never collide on one directory name the way a
    rounded format would. It is the *exact* value that keys the cache, though,
    so pass the level as it appears in the YAML: a value that merely rounds to
    it (``np.float32(0.1)`` -> ``sigma_0.10000000149011612``) is a different
    directory, which is the safe failure but a confusing one.
    """
    return Path(root) / NOISE_SWEEP_DIR_NAME / f"sigma_{float(noise_std)!r}"


ENSEMBLE_SWEEP_DIR_NAME = "ensemble_size"


def config_at_ensemble_size(config: dict, ensemble_size: int) -> dict:
    """A copy of `config` with the ensemble size replaced.

    `run_seed` reads ``assimilation.ensemble_size`` in exactly one place, when it
    calls `initialize_ensemble`, and every filter then takes N from the shape of
    the ensemble it is handed. Changing this one value therefore moves the whole
    experiment to a new ensemble size with nothing else touched.

    Unlike the noise sweep, nothing here is paired across sizes: N changes how
    many perturbations every filter draws and how many members
    `initialize_ensemble` draws, so the sizes see different random numbers by
    construction. Only the truth trajectory and the observations are shared,
    since neither depends on N.

    The copy is deep for the same reason as `config_at_noise`: the filter
    entries are nested dicts, and mutating the caller's config would silently
    move the main experiment too.
    """
    cfg = copy.deepcopy(config)
    cfg["assimilation"]["ensemble_size"] = int(ensemble_size)
    return cfg


def ensemble_results_dir(ensemble_size: int, root=SENSITIVITY_RESULTS_DIR) -> Path:
    """Cache directory for one ensemble size of the ensemble-size sweep.

    Keyed for the same reason the noise levels are: a shard's filename carries
    only its seed, so without the size in the path an N = 50 request would load
    the N = 10 shards without complaint. It also could not be read if it did --
    the rank histogram has N + 1 bins, so the arrays are not even the same shape
    across sizes.

    Memory, measured: peak RSS per worker process is 74 MB at N = 10 and 120 MB
    at N = 50, of which 53 MB is the interpreter and numpy. Only the batched
    ETKF transform grows with the ensemble, as O(n N^2); the truth trajectory,
    the observations and the d x d residual covariance do not. Fourteen workers
    therefore hold about 1.7 GB at N = 50. `keep_ensemble=True` is the one path
    that changes this picture -- it retains K x n x N per filter, about 210 MB
    across the set at N = 50 -- and the sweep does not use it.
    """
    return Path(root) / ENSEMBLE_SWEEP_DIR_NAME / f"N_{int(ensemble_size)}"


# --------------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------------
def _rank_counts(x_truth, X, n_bins):
    """
    Counts of the truth's rank over the state components at one verification time.

    The rank is the number of members *strictly below* the truth, matching
    `src.utils.metrics.compute_rank_histogram`, so it is zero-based and runs
    0..N over `n_bins = N + 1` bins. The returned counts sum to the state
    dimension.
    """
    ranks = (X < x_truth[:, None]).sum(axis=1)
    return np.bincount(ranks, minlength=n_bins)


def _accumulate_rank_histogram(hist, x_truth, X):
    """
    Add one window's ranks to `hist` in place.

    Accumulating per window avoids holding the full ensemble history in memory.
    The per-window counts are the same numbers `rank_counts_by_window` records,
    which is why both go through `_rank_counts`: the aggregate and the
    per-window diagnostic cannot drift apart in their rank convention.
    """
    hist += _rank_counts(x_truth, X, hist.size)
    return hist


def run_filter(
    filter_obj,
    X_init,
    observations,
    truth,
    H,
    R,
    config,
    inflation=1.0,
    is_4d=False,
    rng=None,
    keep_ensemble=False,
    collect_window_ranks=False,
):
    """
    Run a data assimilation filter using src implementations.

    Parameters
    ----------
    filter_obj : object
        Filter instance from src/filters
    X_init : ndarray, shape (n, N)
        Initial ensemble
    observations : ndarray, shape (K, m)
        Observation sequence
    truth : ndarray, shape (K+1, n)
        Truth trajectory
    H : ndarray, shape (m, n)
        Observation operator
    R : ndarray, shape (m, m)
        Observation error covariance
    config : dict
        Configuration dictionary
    inflation : float, optional
        Multiplicative inflation factor applied before analysis
    is_4d : bool
        Whether filter uses 4D windows
    rng : numpy.random.Generator, optional
        Generator for observation perturbations. Pass one for reproducibility.
    keep_ensemble : bool, optional
        Also return the full truth/ensemble histories (~21 MB per seed at
        K=5000). Off by default: the notebooks only need the rank histogram,
        which is accumulated on the fly.
    collect_window_ranks : bool, optional
        Also return `rank_counts_by_window`, shape (W, N+1): the rank counts at
        each *assimilation-window endpoint*, one row per window, each row
        summing to the state dimension. This is the compact diagnostic the
        coverage analysis needs and it is ~44 kB per filter-seed at W=1000,
        N=10 -- three orders of magnitude below `keep_ensemble`.

        The endpoint convention is what makes filters comparable: a 4D filter
        already verifies once per window, while a sequential filter verifies at
        every observation time, so only every L-th of its verifications (k+1 a
        multiple of L, i.e. zero-based k = L-1, 2L-1, ..., WL-1) is recorded.
        Those are exactly the times `window_axis` keeps for the other metrics.
        Off by default, so `rank_hist` and the shard format are untouched.

    Returns
    -------
    results : dict
        RMSE, misfit and spread trajectories plus rank-histogram counts, and
        `rank_counts_by_window` when `collect_window_ranks` is set.
    """
    assim_cfg = config["assimilation"]
    model_cfg = config["model"]
    obs_cfg = config["observations"]
    derived_cfg = config["derived"]

    K = derived_cfg["total_observations"]
    L = assim_cfg["window_length"]
    W = assim_cfg["n_windows"]
    dt = model_cfg["dt"]
    F = model_cfg["F"]
    steps_per_obs = obs_cfg["steps_per_obs"]
    inflation_factor = inflation

    # Storage
    rmse_trajectory = []
    misfit_trajectory = []
    spread_trajectory = []
    truth_history = []
    ensemble_history = []
    rank_hist = np.zeros(X_init.shape[1] + 1, dtype=np.int64)
    window_ranks = [] if collect_window_ranks else None

    # Precompute R^{-1/2} for misfit calculation
    R_inv_sqrt = sym_posdef_inverse(R)
    R_inv_sqrt = la.cholesky(R_inv_sqrt)

    X = X_init.copy()
    if rng is None:
        rng = np.random.default_rng()

    if is_4d:
        # 4D filter: process by windows
        for w in range(W):
            # Get observations for this window
            start_idx = w * L
            end_idx = (w + 1) * L
            z_window = observations[start_idx:end_idx]  # (L, m)
            z_stack = z_window.flatten()  # (L*m,)

            # Forecast through window to build X_path
            X_path = []
            X_current = X.copy()

            for ell in range(L):
                # Forecast to next observation time (all members at once)
                for _ in range(steps_per_obs):
                    X_current = rk4_step(l96_rhs, X_current, dt, F=F)
                X_path.append(X_current.copy())

            # Apply inflation before update
            if inflation_factor != 1.0:
                X_mean = X_path[-1].mean(axis=1, keepdims=True)
                X_path[-1] = X_mean + inflation_factor * (X_path[-1] - X_mean)

            # Assimilate using 4D filter
            if isinstance(filter_obj, StochasticEnKF4D):
                X = filter_obj.update(X_path, z_stack, rng)
            else:
                X = filter_obj.update(X_path, z_stack)

            # Compute metrics at window end
            X_mean = X.mean(axis=1)
            x_truth = truth[end_idx]

            rmse_val = compute_rmse(X_mean, x_truth)
            spread = compute_ensemble_spread(X)

            # For misfit, use last observation in window
            y_mean = H @ X_mean
            misfit = rel_data_misfit(z_window[-1], y_mean, R_inv_sqrt)

            rmse_trajectory.append(rmse_val)
            misfit_trajectory.append(misfit)
            spread_trajectory.append(spread)
            counts = _rank_counts(x_truth, X, rank_hist.size)
            rank_hist += counts
            if collect_window_ranks:      # every 4D step is a window endpoint
                window_ranks.append(counts)
            if keep_ensemble:
                truth_history.append(x_truth.copy())
                ensemble_history.append(X.copy())

    else:
        # Sequential filter: process observation by observation
        for k in range(K):
            z = observations[k]

            # Forecast (all members at once)
            for _ in range(steps_per_obs):
                X = rk4_step(l96_rhs, X, dt, F=F)

            # Apply inflation before update
            if inflation_factor != 1.0:
                X_mean = X.mean(axis=1, keepdims=True)
                X = X_mean + inflation_factor * (X - X_mean)

            # Assimilate using sequential filter. Only the stochastic one draws
            # observation perturbations, so only it is handed the generator --
            # the deterministic transform's `update` takes no rng at all.
            if isinstance(filter_obj, StochasticEnKF):
                X = filter_obj.update(X, z, rng)
            else:
                X = filter_obj.update(X, z)

            # Compute metrics
            X_mean = X.mean(axis=1)
            x_truth = truth[k + 1]

            rmse_val = compute_rmse(X_mean, x_truth)
            spread = compute_ensemble_spread(X)

            y_mean = H @ X_mean
            misfit = rel_data_misfit(z, y_mean, R_inv_sqrt)

            rmse_trajectory.append(rmse_val)
            misfit_trajectory.append(misfit)
            spread_trajectory.append(spread)
            counts = _rank_counts(x_truth, X, rank_hist.size)
            rank_hist += counts
            # `rank_hist` keeps every observation time, as it always has; the
            # per-window diagnostic keeps only the window endpoints, so it lines
            # up with the 4D filters' single verification per window.
            if collect_window_ranks and (k + 1) % L == 0:
                window_ranks.append(counts)
            if keep_ensemble:
                truth_history.append(x_truth.copy())
                ensemble_history.append(X.copy())

    results = {
        "rmse": np.array(rmse_trajectory),
        "misfit": np.array(misfit_trajectory),
        "spread": np.array(spread_trajectory),
        "rank_hist": rank_hist.astype(float),
    }
    if collect_window_ranks:
        stacked = np.asarray(window_ranks, dtype=np.int64)
        assert stacked.shape == (W, X_init.shape[1] + 1), (
            f"window ranks are {stacked.shape}, expected {(W, X_init.shape[1] + 1)}"
        )
        results["rank_counts_by_window"] = stacked
    if keep_ensemble:
        results["truth"] = np.array(truth_history)
        results["ensemble"] = np.array(ensemble_history)
    return results


def _localization_taper(config: dict, params: dict, observe_every: int, L: int):
    """Gaspari-Cohn taper for a filter, or None when it is unlocalized.

    The radius is the cutoff in grid units (the separation at which the taper
    reaches zero). Tapers are built over the observation vector the filter
    actually sees, so a 4D filter passes ``L`` and a sequential one passes 1: a
    taper of the wrong width fails loudly at the first update rather than
    silently.
    """
    radius = params.get("localization_radius", None)
    if radius is None:
        return None
    n = config["model"]["n"]
    _H, obs_indices = build_obs_operator(n, observe_every=observe_every)
    return build_localization_weights(n, obs_indices, radius, window_len=L)


def _build_filter_specs(labels, H, R, L, filter_params, config, observe_every):
    """``(label, filter, inflation, is_4d)`` for each requested filter.

    Inflation is passed to ``run_filter``, which applies it to the prior before
    the update; the transform filters are therefore always constructed with
    ``inflation=1.0``. Their constructors can inflate the posterior themselves,
    and leaving that on as well would apply inflation twice (measured: the
    window-0 analysis spread comes out exactly lambda times too large).
    """
    seq = filter_params.get("seq_enkf", {})
    seq_etkf = filter_params.get("seq_etkf", {})
    seq_letkf = filter_params.get("seq_letkf", {})
    fourd = filter_params.get("enkf_4d", {})
    fourd_det = filter_params.get("enkf_4d_det", {})
    etkf = filter_params.get("etkf_4d", {})
    letkf = filter_params.get("letkf_4d", {})
    qpca = filter_params.get("qpca_endcf", {})
    qpca_loc = filter_params.get("qpca_endcf_loc", {})

    def build(label):
        if label == "Seq-EnKF":
            return (
                StochasticEnKF(H, R, stabilize=seq.get("stabilize", True)),
                seq.get("inflation", 1.0),
                False,
            )
        if label == "Seq-ETKF":
            # The sequential deterministic transform: same cadence as Seq-EnKF,
            # one update per observation time, so its taper spans a single
            # observation time (window_len 1) rather than the stacked window.
            return (
                SequentialLETKF(
                    H,
                    R,
                    localization=_localization_taper(
                        config, seq_etkf, observe_every, 1
                    ),
                    inflation=1.0,
                ),
                seq_etkf.get("inflation", 1.0),
                False,
            )
        if label == "Seq-LETKF":
            # Seq-ETKF plus the Gaspari-Cohn taper, at the same cadence; the
            # taper spans one observation time, not the stacked window.
            return (
                SequentialLETKF(
                    H,
                    R,
                    localization=_localization_taper(
                        config, seq_letkf, observe_every, 1
                    ),
                    inflation=1.0,
                ),
                seq_letkf.get("inflation", 1.0),
                False,
            )
        if label == "4D-EnKF":
            return StochasticEnKF4D(H, R, L), fourd.get("inflation", 1.0), True
        if label == "4D-EnKF-det":
            # The single-variable control: same gain, same everything, but the
            # memberwise observation perturbations are not drawn.
            return (
                StochasticEnKF4D(H, R, L, perturb_observations=False),
                fourd_det.get("inflation", 1.0),
                True,
            )
        if label == "4D-ETKF":
            # Deterministic, unlocalized: the control that isolates the effect
            # of dropping the observation perturbations from 4D-EnKF.
            return (
                LETKF4D(
                    H,
                    R,
                    L,
                    localization=_localization_taper(config, etkf, observe_every, L),
                    inflation=1.0,
                ),
                etkf.get("inflation", 1.0),
                True,
            )
        if label == "4D-LETKF":
            return (
                LETKF4D(
                    H,
                    R,
                    L,
                    localization=_localization_taper(config, letkf, observe_every, L),
                    inflation=1.0,
                ),
                letkf.get("inflation", 1.0),
                True,
            )
        if label == "QPCA-EnDCF":
            return (
                QPCAEnDCF(
                    H, R, L, k=qpca.get("k", 1), stabilize=qpca.get("stabilize", True)
                ),
                qpca.get("inflation", 1.0),
                True,
            )
        if label == "QPCA-EnDCF-L":
            return (
                QPCAEnDCF(
                    H,
                    R,
                    L,
                    k=qpca_loc.get("k", 1),
                    stabilize=qpca_loc.get("stabilize", True),
                    localization=_localization_taper(config, qpca_loc, observe_every, L),
                ),
                qpca_loc.get("inflation", 1.0),
                True,
            )
        raise ValueError(f"Unknown filter label: {label!r}")

    return [(label, *build(label)) for label in labels]


def run_seed(
    seed: int,
    config: dict,
    keep_ensemble: bool = False,
    filters: list[str] | None = None,
    collect_window_ranks: bool = False,
) -> dict:
    """
    Run the requested filters on one seed and return their reduced diagnostics.

    Truth, observations and the initial ensemble are shared by the filters, so
    they are built once per seed. Each filter draws its observation
    perturbations from its own child stream of `SeedSequence(seed)`, which makes
    the run reproducible and keeps the filters independent of one another.

    `filters` selects the comparison set and defaults to `FILTER_ORDER`; pass
    `EXTENDED_FILTER_ORDER` for the eight-filter mechanism decomposition. Streams
    are keyed by filter name via `_STREAM_ORDER`, so a filter draws the same
    numbers whichever set it is requested in.
    """
    labels = list(FILTER_ORDER if filters is None else filters)
    model_cfg = config["model"]
    obs_cfg = config["observations"]
    assim_cfg = config["assimilation"]
    initial_cfg = config.get("initial_ensemble", {})
    filter_params = _filter_params(config)

    observe_every = obs_cfg.get("observe_every", max(1, model_cfg["n"] // obs_cfg["m"]))
    H, _ = build_obs_operator(model_cfg["n"], observe_every=observe_every)
    R = (obs_cfg["noise_std"] ** 2) * np.eye(obs_cfg["m"])
    K = config["derived"]["total_observations"]

    np.random.seed(seed)
    x0 = np.random.randn(model_cfg["n"])
    truth = generate_truth(
        x0,
        K,
        obs_cfg["steps_per_obs"],
        model_cfg["dt"],
        model_cfg["F"],
        model_cfg.get("spinup_steps", 0),
    )

    # One child per filter, plus a last one for the observation noise. Spawning
    # the observation stream *after* the filter streams is what lets it exist at
    # all without repainting them: `spawn` is prefix-stable, so children[:k] are
    # the same k streams they were before this one was added.
    children = np.random.SeedSequence(seed).spawn(len(_STREAM_ORDER) + 1)
    streams = dict(zip(_STREAM_ORDER, children))

    # The observation noise gets its own stream rather than `seed=`. Passing the
    # seed re-starts the same global legacy stream that produced `x0` just
    # above, which made the first observation-error vectors *equal* to it --
    # with n = 40 and m = 20, the errors at k = 0 and k = 1 were exactly
    # x0[:20] and x0[20:], so the observation error was perfectly correlated
    # with the state it was observing. Only the first ceil(n/m) observation
    # times were affected and all of them sit inside the burn-in every reported
    # statistic discards, but the filters assume errors independent of the
    # truth, so the data they are fed should satisfy that.
    observations = generate_observations(
        truth, H, obs_cfg["noise_std"], rng=np.random.default_rng(children[-1])
    )

    X_init = initialize_ensemble(
        truth_0=truth[0],
        N=assim_cfg["ensemble_size"],
        common_weight=initial_cfg.get("common_perturbation_weight", 0.0),
        individual_weight=initial_cfg.get("individual_perturbation_weight", 1.0),
        seed=seed,
    )
    L = assim_cfg["window_length"]

    specs = _build_filter_specs(labels, H, R, L, filter_params, config, observe_every)

    return {
        label: run_filter(
            filter_obj,
            X_init,
            observations,
            truth,
            H,
            R,
            config,
            inflation=inflation,
            is_4d=is_4d,
            rng=np.random.default_rng(streams[label]),
            keep_ensemble=keep_ensemble,
            collect_window_ranks=collect_window_ranks,
        )
        for label, filter_obj, inflation, is_4d in specs
    }


# --------------------------------------------------------------------------
# Sharded, resumable execution
# --------------------------------------------------------------------------
def shard_path(seed: int, out_dir: Path) -> Path:
    return Path(out_dir) / SHARD_DIR_NAME / f"seed_{seed}.npz"


def _save_shard(path: Path, seed_results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        f"{label}__{metric}": seed_results[label][metric]
        for label in seed_results
        for metric in METRICS
    }
    # The temp name must also end in .npz, or savez_compressed appends it and
    # the rename below has nothing to move.
    tmp = path.parent / f"{path.stem}.tmp.npz"
    np.savez_compressed(tmp, **flat)
    tmp.replace(path)  # atomic, so an interrupted run never leaves a half shard


def _load_shard(path: Path) -> dict:
    with np.load(path) as data:
        labels = sorted({key.split("__")[0] for key in data.files})
        return {
            label: {metric: data[f"{label}__{metric}"] for metric in METRICS}
            for label in labels
        }


def _run_and_save(
    seed: int, config: dict, out_dir: Path, force: bool, filters=None
) -> str:
    """Worker entry point: compute one seed unless its shard already exists."""
    path = shard_path(seed, out_dir)
    if path.exists() and not force:
        return f"skip {seed}"

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # threadpoolctl ships with joblib, but don't hard-depend
        _save_shard(path, run_seed(seed, config, filters=filters))
    else:
        with threadpool_limits(limits=1):
            _save_shard(path, run_seed(seed, config, filters=filters))
    return f"ran {seed}"


def run_experiment(
    config: dict,
    seeds=None,
    n_jobs: int = -2,
    out_dir=BASELINE_RESULTS_DIR,
    force: bool = False,
    verbose: int = 5,
    filters=None,
) -> dict:
    """
    Run (or load) every seed and return results grouped by filter.

    Parameters
    ----------
    config : dict
        Experiment configuration from `load_config`.
    seeds : sequence of int, optional
        Seeds to run; defaults to the config's seed list.
    n_jobs : int
        Worker processes. -2 uses all cores but one; 1 runs serially, which is
        easier to debug and to interrupt.
    out_dir : Path
        Results root; shards land in ``<out_dir>/seeds/``.
    force : bool
        Recompute seeds even if their shard exists.
    filters : sequence of str, optional
        Comparison set; defaults to ``FILTER_ORDER``. Pass
        ``EXTENDED_FILTER_ORDER`` together with ``out_dir=EXTENDED_RESULTS_DIR``
        -- a shard holds only the filters that were requested when it was
        written, and a cached shard is reused on the strength of its filename
        alone, so the two sets must not share a directory.

    Returns
    -------
    dict
        ``{filter_label: [per-seed results, ...]}`` in seed order.
    """
    labels = list(FILTER_ORDER if filters is None else filters)
    out_dir = Path(out_dir)
    seeds = list(resolve_seeds(config) if seeds is None else seeds)
    pending = [s for s in seeds if force or not shard_path(s, out_dir).exists()]
    print(f"{len(seeds)} seeds requested; {len(pending)} to compute, "
          f"{len(seeds) - len(pending)} cached in {out_dir / SHARD_DIR_NAME}")

    if pending:
        if n_jobs == 1:
            for seed in pending:
                _run_and_save(seed, config, out_dir, force, labels)
        else:
            from joblib import Parallel, delayed

            Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
                delayed(_run_and_save)(seed, config, out_dir, force, labels)
                for seed in pending
            )

    all_results = {label: [] for label in labels}
    for seed in seeds:
        path = shard_path(seed, out_dir)
        shard = _load_shard(path)
        missing = [label for label in labels if label not in shard]
        if missing:
            # Reused-on-filename-alone is only safe while the contents match.
            raise KeyError(
                f"{path} holds {sorted(shard)} but {missing} were requested. "
                f"Delete that directory's shards, pass force=True, or point "
                f"`out_dir` at a directory belonging to this filter set."
            )
        for label in labels:
            all_results[label].append(shard[label])
    return all_results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Number of seeds; extends the config's list deterministically.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="Explicit seed list."
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-2, help="Worker processes (-2 = cores-1, 1 = serial)."
    )
    parser.add_argument("--out-dir", type=Path, default=BASELINE_RESULTS_DIR)
    parser.add_argument(
        "--force", action="store_true", help="Recompute seeds that already have a shard."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else resolve_seeds(config, args.n_seeds)

    run_experiment(
        config, seeds=seeds, n_jobs=args.n_jobs, out_dir=args.out_dir, force=args.force
    )
    print(f"Done. Shards in {Path(args.out_dir) / SHARD_DIR_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
