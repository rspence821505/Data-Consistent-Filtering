"""
Parallel Monte-Carlo driver for the bias-variance decomposition (notebook 15).

The notebook version advances the ensemble one member at a time through the
stateful ``Lorenz96`` wrapper, which costs a factor of ~6.6 in wall clock at
n=40, N=10.  ``l96_rhs`` is shape-agnostic, so the whole (n, N) ensemble can be
advanced in a single ``rk4_step`` call; the results are bit-identical (the
forecast consumes no random numbers, and the arithmetic per member is the same
sequence of elementwise operations).

There are two drivers.

``run_experiment`` -- the flat (seed, filter, realization) grid.  One task is
one realization; workers return only the two accumulators the decomposition
needs rather than the full trajectories, so memory stays flat as
``n_monte_carlo`` grows.  Its kernels are bit-identical to the notebook's.

``run_hierarchical_experiment`` -- the truth-seed x realization design.  One
task is a whole ``(truth seed, filter)`` pair, with realizations *batched*: all
M ensembles live in one (n, M*N) block and advance in a single ``rk4_step``
call (14.8x at M=100, bit-identical), and the gain uses an exact low-rank
identity instead of ``pinv`` (20.9x).  Truth trajectories are generated for all
seeds in one batched pass (53x).  Together these take the 100x100x1000 grid
from ~21 to ~2 core-hours.

The batched kernels reassociate floating-point operations in the gain and the
member update, so individual trajectories decorrelate from the reference path
even though both sample the same distribution; ``tests/test_batched_kernels.py``
checks agreement at short horizons and statistical agreement at long ones.

Run them as::

    python -m src.experiments.bias_variance_mc --n-jobs 10
    python -m src.experiments.bias_variance_mc --hierarchical \\
        --n-truth-seeds 100 --n-mc 100 --n-windows 1000 --n-jobs 10

and then load the ``.npz`` files under ``results/diagnostics/`` from the
notebook for plotting.  Both drivers are resumable.
"""

from __future__ import annotations

# BLAS threading must be pinned *before* numpy is imported, otherwise every
# worker process spawns its own thread pool and they fight over the cores.
import os as _os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import linalg

from ..models.lorenz96 import l96_rhs, rk4_step
from ..filters.letkf_4d import LETKF4D
from ..filters.seq_letkf import SequentialLETKF
from ..utils.config import load_experiment_config
from ..utils.localization import build_localization_weights
from ..utils.observations import (
    build_obs_operator,
    generate_observations,
    generate_truth,
)
from ..config import DIAGNOSTICS_EXPERIMENT_DIR, DIAGNOSTICS_RESULTS_DIR

# Filters included in the experiment. Seq-LETKF and 4D-LETKF (paper Algorithms
# 3 and 4) are fully implemented -- see src/filters/seq_letkf.py,
# src/filters/letkf_4d.py and the kernels below -- and are re-enabled simply by
# adding them back to this tuple; the driver, config and plots all key off it.
# Order matches baseline.py's EXTENDED_FILTER_ORDER, so notebooks 01 and 10
# present the same filters in the same sequence. Order here is presentation
# only: ensemble seeds are ENSEMBLE_SEED_BASE + realization and carry no filter
# key, so adding or reordering filters leaves every other filter's draws -- and
# every existing checkpoint -- bit-identical. (baseline.py is the opposite case;
# its _STREAM_ORDER is append-only for exactly that reason.)
FILTER_NAMES = (
    "Seq-EnKF",
    "Seq-ETKF",
    "4D-EnKF",
    "4D-ETKF",
    "QPCA-EnDCF",
)

#: Every filter the driver can run, whether or not it is in FILTER_NAMES.
AVAILABLE_FILTER_NAMES = (
    "Seq-EnKF",
    "Seq-ETKF",
    "4D-EnKF",
    "4D-ETKF",
    "Seq-LETKF",
    "4D-LETKF",
    "QPCA-EnDCF",
)

# Config key under `filters:` for each filter that reads settings from the YAML.
# The ETKF variants are the ensemble transform filters with no taper:
# build_localization_weights returns an array of ones for a null radius, which
# reduces Algorithms 3 and 4 to the global (unlocalized) ETKF. They therefore
# share the LETKF kernels and differ only in their YAML entry.
FILTER_CONFIG_KEYS = {
    "Seq-EnKF": "seq_enkf",
    "Seq-ETKF": "seq_etkf",
    "4D-EnKF": "enkf_4d",
    "4D-ETKF": "etkf_4d",
    "Seq-LETKF": "seq_letkf",
    "4D-LETKF": "letkf_4d",
    "QPCA-EnDCF": "qpca_endcf",
}

#: Which kernel implements each ensemble-transform filter, and under which name
#: it reads its settings.  Keyed so a new taper variant is one entry, not a new
#: kernel.
_TRANSFORM_FILTERS = ("Seq-ETKF", "Seq-LETKF", "4D-ETKF", "4D-LETKF")

# Base offset for per-realization ensemble seeds, matching the notebook's
# ``ensemble_seed = 10000 + i``.  The same ensemble seeds are reused across
# truth seeds and filters, which is deliberate: it is a common-random-numbers
# scheme that reduces the variance of between-filter comparisons.
ENSEMBLE_SEED_BASE = 10000
SPECTRAL_DIAGNOSTIC_NAMES = (
    "leading_eigenvalue_fraction",
    "normalized_cutoff_gap",
    "retained_mean_mismatch_fraction",
)


def make_truth_seeds(n_seeds: int, base_seed: int = 20260728) -> List[int]:
    """Return reproducible, distinct uint32 seeds for outer truth/observation runs."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive.")
    sequence = np.random.SeedSequence(base_seed)
    return [int(x) for x in sequence.generate_state(n_seeds, dtype=np.uint32)]


# --------------------------------------------------------------------------- #
# Truth / observations
# --------------------------------------------------------------------------- #

_DATA_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

# The cache is bounded because it is per worker process.  At W=1000 one entry is
# ~2.4 MB, so an unbounded cache costs 2.4 GB across 10 workers once 100 truth
# seeds have been touched.  Eight entries covers the small multi-seed runs
# completely while capping the hierarchical run at ~19 MB per worker.
_DATA_CACHE_MAXSIZE = 8


def _data_cache_key(config: Dict, seed: int) -> Tuple:
    """Cache key covering every parameter the truth and observations depend on.

    Keying on the seed alone silently returns a stale trajectory whenever the
    horizon or model parameters change inside one process -- which is exactly
    what happens when a notebook or test evaluates several configurations.
    """
    return (
        seed,
        config["model"]["n"],
        config["model"]["F"],
        config["model"]["dt"],
        config["model"].get("spinup_steps", 0),
        config["assimilation"]["window_length"],
        config["assimilation"]["n_windows"],
        config["observations"]["steps_per_obs"],
        config["observations"]["observe_every"],
        config["observations"]["noise_std"],
    )


def generate_truth_and_observations(
    config: Dict, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truth trajectory, observations and observation operator for one seed.

    Identical to the notebook's helper of the same name.  Results are cached
    per process so that a worker handling many realizations of the same seed
    integrates the truth trajectory once instead of once per task.
    """
    key = _data_cache_key(config, seed)
    cached = _DATA_CACHE.get(key)
    if cached is not None:
        return cached

    rng = np.random.default_rng(seed)

    n = config["model"]["n"]
    F = config["model"]["F"]
    dt = config["model"]["dt"]
    spinup = config["model"].get("spinup_steps", 0)

    L = config["assimilation"]["window_length"]
    W = config["assimilation"]["n_windows"]
    K = W * L
    steps_per_obs = config["observations"]["steps_per_obs"]

    x0 = rng.normal(0.0, 1.0, size=n)
    truth = generate_truth(x0, K, steps_per_obs, dt, F=F, spinup_steps=spinup)

    obs_every = config["observations"]["observe_every"]
    H, _ = build_obs_operator(n, obs_every)
    noise_std = config["observations"]["noise_std"]
    observations = generate_observations(truth, H, noise_std, seed=seed)

    while len(_DATA_CACHE) >= _DATA_CACHE_MAXSIZE:
        _DATA_CACHE.pop(next(iter(_DATA_CACHE)))
    _DATA_CACHE[key] = (truth, observations, H)
    return truth, observations, H


def generate_truth_batch(
    config: Dict, seeds: Iterable[int], chunk_size: int = 100
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Truth and observations for many seeds, integrated as a single ensemble.

    Independent truth trajectories are just independent columns of one Lorenz-96
    ensemble, so they advance together in one ``rk4_step`` sweep.  At W=1000 this
    is ~53x faster than integrating each seed separately (3.1 s versus 164 s for
    100 seeds) and, because ``l96_rhs`` is elementwise plus ``np.roll`` along the
    state axis, bit-identical column by column.

    Trajectories are produced in chunks so peak memory stays near
    ``chunk_size * (K + 1) * n * 8`` bytes (~160 MB at 100 seeds, W=1000).
    """
    seeds = list(seeds)
    n = config["model"]["n"]
    F = config["model"]["F"]
    dt = config["model"]["dt"]
    spinup = config["model"].get("spinup_steps", 0)

    L = config["assimilation"]["window_length"]
    W = config["assimilation"]["n_windows"]
    K = W * L
    steps_per_obs = config["observations"]["steps_per_obs"]

    m = config["observations"]["m"]
    noise_std = config["observations"]["noise_std"]
    H, _ = build_obs_operator(n, config["observations"]["observe_every"])

    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for start in range(0, len(seeds), max(1, chunk_size)):
        chunk = seeds[start : start + max(1, chunk_size)]

        # Same initial conditions as generate_truth_and_observations: one
        # default_rng(seed) draw per seed, stacked as ensemble columns.
        X = np.stack(
            [np.random.default_rng(s).normal(0.0, 1.0, size=n) for s in chunk],
            axis=1,
        )
        for _ in range(spinup):
            X = rk4_step(l96_rhs, X, dt, F=F)

        trajectories = np.empty((K + 1, n, len(chunk)))
        trajectories[0] = X
        for k in range(K):
            for _ in range(steps_per_obs):
                X = rk4_step(l96_rhs, X, dt, F=F)
            trajectories[k + 1] = X

        for index, seed in enumerate(chunk):
            truth = np.ascontiguousarray(trajectories[:, :, index])
            # generate_observations draws m values per observation time from the
            # legacy global RNG; randn(K, m) fills C-order, so row k is exactly
            # the k-th draw of that loop and the noise sequence is unchanged.
            np.random.seed(seed)
            noise = np.random.randn(K, m)
            observations = truth[1:] @ H.T + noise_std * noise
            out[seed] = (truth, observations, H)

    return out


# --------------------------------------------------------------------------- #
# Filter kernels
# --------------------------------------------------------------------------- #
#
# These reproduce the notebook cells exactly apart from the forecast loop.  In
# particular the legacy global RNG (``np.random.seed``) and the per-member
# perturbation draws are preserved verbatim so that the random number stream --
# and therefore every published number -- is unchanged.  Each task reseeds on
# entry, so the global RNG carries no state between tasks.


def _unpack(config: Dict):
    return (
        config["model"]["n"],
        config["observations"]["m"],
        config["observations"]["noise_std"],
        config["assimilation"]["window_length"],
        config["assimilation"]["n_windows"],
        config["assimilation"]["ensemble_size"],
        config["observations"]["steps_per_obs"],
        config["model"]["F"],
        config["model"]["dt"],
    )


def _initial_ensemble(x_init: np.ndarray, n: int, N: int) -> np.ndarray:
    """Ensemble initialization, drawing from the global RNG in notebook order."""
    xi_0 = np.random.randn(n)
    X = np.zeros((n, N))
    for j in range(N):
        xi_j = np.random.randn(n)
        X[:, j] = x_init + 0.5 * xi_0 + 0.5 * xi_j
    return X


def _advance(X: np.ndarray, steps: int, dt: float, F: float) -> np.ndarray:
    """Advance the whole ensemble ``steps`` RK4 steps in one vectorized sweep."""
    for _ in range(steps):
        X = rk4_step(l96_rhs, X, dt, F=F)
    return X


def run_seq_enkf(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    ensemble_seed: int,
) -> np.ndarray:
    """Sequential stochastic EnKF; returns the analysis mean at window endpoints."""
    np.random.seed(ensemble_seed)
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)

    X = _initial_ensemble(truth[0], n, N)
    R = noise_std**2 * np.eye(m)

    analysis = np.zeros((W + 1, n))
    analysis[0] = X.mean(axis=1)

    obs_idx = 0
    for w in range(W):
        for _ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)

            z = observations[obs_idx]
            obs_idx += 1

            x_mean = X.mean(axis=1)
            A = X - x_mean[:, np.newaxis]

            Y = H @ X
            y_mean = Y.mean(axis=1)
            A_y = Y - y_mean[:, np.newaxis]

            P_xy = (A @ A_y.T) / (N - 1)
            S = (A_y @ A_y.T) / (N - 1) + R
            S += 1e-6 * np.eye(m)

            K = P_xy @ linalg.pinv(S)

            for j in range(N):
                epsilon_j = noise_std * np.random.randn(m)
                X[:, j] = X[:, j] + K @ (z + epsilon_j - H @ X[:, j])

        analysis[w + 1] = X.mean(axis=1)

    return analysis


def run_4d_enkf(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    ensemble_seed: int,
) -> np.ndarray:
    """4D stochastic EnKF; returns the analysis mean at window endpoints."""
    np.random.seed(ensemble_seed)
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)

    X = _initial_ensemble(truth[0], n, N)
    R_L = np.kron(np.eye(L), noise_std**2 * np.eye(m))
    R_L_chol = linalg.cholesky(R_L, lower=True)
    eye_mL = np.eye(m * L)

    analysis = np.zeros((W + 1, n))
    analysis[0] = X.mean(axis=1)

    Y_window = np.zeros((L, m, N))
    for w in range(W):
        for ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)
            Y_window[ell] = H @ X

        z_stack = observations[w * L : (w + 1) * L].ravel()
        Y_stack = Y_window.reshape(m * L, N)

        X_L = X
        x_mean = X_L.mean(axis=1)
        A_x = X_L - x_mean[:, np.newaxis]

        y_mean = Y_stack.mean(axis=1)
        A_y = Y_stack - y_mean[:, np.newaxis]

        P_xy = (A_x @ A_y.T) / (N - 1)
        S = (A_y @ A_y.T) / (N - 1) + R_L
        S += 1e-6 * eye_mL

        K = P_xy @ linalg.pinv(S)

        for j in range(N):
            epsilon_j = np.random.randn(m * L)
            epsilon_j = R_L_chol @ epsilon_j
            X_L[:, j] = X_L[:, j] + K @ (z_stack + epsilon_j - Y_stack[:, j])

        X = X_L
        analysis[w + 1] = X.mean(axis=1)

    return analysis


def run_qpca_endcf(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    ensemble_seed: int,
    kappa: int = 1,
    return_diagnostics: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """QPCA-EnDCF analysis means and optional empirical spectral diagnostics.

    The established notebook path retains its original dense eigendecomposition
    for reproducibility.  The new diagnostic path uses the equivalent thin SVD
    of the ``(d, N)`` centered residual matrix, avoiding a dense ``(d, d)``
    decomposition when ``N << d``.
    """
    np.random.seed(ensemble_seed)
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)

    X = _initial_ensemble(truth[0], n, N)

    R_L = np.kron(np.eye(L), noise_std**2 * np.eye(m))
    eye_mL = np.eye(m * L)
    R_L_chol = linalg.cholesky(R_L, lower=True)
    R_L_inv_chol = linalg.solve_triangular(R_L_chol, eye_mL, lower=True)
    taper = _qpca_taper(config)

    analysis = np.zeros((W + 1, n))
    analysis[0] = X.mean(axis=1)
    spectral = np.full((W, len(SPECTRAL_DIAGNOSTIC_NAMES)), np.nan)

    Y_window = np.zeros((L, m, N))
    for w in range(W):
        for ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)
            Y_window[ell] = H @ X

        z_stack = observations[w * L : (w + 1) * L].ravel()
        Y_stack = Y_window.reshape(m * L, N)

        # Whitened residuals
        E = R_L_inv_chol.T @ (Y_stack - z_stack[:, np.newaxis])
        e_mean = E.mean(axis=1, keepdims=True)
        E_c = E - e_mean

        if return_diagnostics:
            # PCA through the thin residual SVD.  The nonzero eigenvalues of
            # C_E = E_c E_c^T/(N-1) are singular_values**2/(N-1).
            eigenvecs, singular_values, _ = linalg.svd(
                E_c, full_matrices=False, check_finite=False
            )
            eigenvals = singular_values**2 / (N - 1)
        else:
            # Preserve the original notebook computation exactly.
            C_E = (E_c @ E_c.T) / (N - 1)
            eigenvals, eigenvecs = linalg.eigh(C_E)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

        # QPCA projection
        V_kappa = eigenvecs[:, :kappa]
        Q_pca = -V_kappa @ (V_kappa.T @ E)
        Delta_obs = R_L_chol @ Q_pca

        X_L = X
        x_mean = X_L.mean(axis=1)
        A_x = X_L - x_mean[:, np.newaxis]

        y_mean = Y_stack.mean(axis=1)
        A_y = Y_stack - y_mean[:, np.newaxis]

        if taper is None:
            P_xy = (A_x @ A_y.T) / (N - 1)
            P_yy = (A_y @ A_y.T) / (N - 1)
            S = P_yy + 1e-6 * eye_mL
            K = P_xy @ linalg.pinv(S)
        else:
            K = _localized_gain(A_x, A_y, taper, N)

        X = X_L + K @ Delta_obs
        analysis[w + 1] = X.mean(axis=1)

        if return_diagnostics:
            spectrum_sum = eigenvals.sum()
            leading_fraction = (
                eigenvals[0] / spectrum_sum if spectrum_sum > 0.0 else np.nan
            )
            normalized_gap = (
                (eigenvals[0] - eigenvals[1]) / eigenvals[0]
                if eigenvals.size > 1 and eigenvals[0] > 0.0
                else np.nan
            )
            mean_vector = e_mean[:, 0]
            mean_energy = mean_vector @ mean_vector
            retained_energy = V_kappa.T @ mean_vector
            retained_fraction = (
                (retained_energy @ retained_energy) / mean_energy
                if mean_energy > 0.0
                else np.nan
            )
            spectral[w] = (
                leading_fraction,
                normalized_gap,
                retained_fraction,
            )

    if return_diagnostics:
        return analysis, spectral
    return analysis


# --------------------------------------------------------------------------- #
# Realization-batched kernels (hierarchical experiment)
# --------------------------------------------------------------------------- #
#
# All realizations of one (truth seed, filter) pair share the same truth and
# observations and differ only in their ensemble seed.  The forecast is
# per-member independent, so the M ensembles are stacked into a single
# (n, M*N) block and advanced in one rk4_step call.  At n=40, N=10, M=100 that
# is 14.8x faster than advancing them one at a time and bit-identical, because
# l96_rhs is elementwise plus np.roll along the state axis.
#
# The analysis step is per realization and stays in a Python loop: numpy's
# stacked LAPACK drivers are *slower* here than repeated 2-D calls (batched
# pinv on a (100,100,100) stack measured 0.65x).


def _make_streams(realizations: Iterable[int]) -> List[np.random.RandomState]:
    """One legacy RandomState per realization, seeded as the reference kernels.

    ``np.random.seed(s)`` seeds the global legacy RandomState, so
    ``RandomState(s)`` reproduces exactly the same stream.  Because the streams
    are independent per realization, interleaving draws across realizations
    leaves each realization's sequence untouched.
    """
    return [np.random.RandomState(ENSEMBLE_SEED_BASE + r) for r in realizations]


def _batched_initial_ensembles(
    truth_0: np.ndarray, n: int, N: int, streams: List[np.random.RandomState]
) -> np.ndarray:
    """(n, M*N) block of initial ensembles, drawn in the reference order."""
    X = np.empty((n, len(streams) * N))
    for i, stream in enumerate(streams):
        xi_0 = stream.randn(n)
        for j in range(N):
            X[:, i * N + j] = truth_0 + 0.5 * xi_0 + 0.5 * stream.randn(n)
    return X


def _check_isotropic_obs_cov(config: Dict) -> float:
    """Return the scalar observation variance, or raise if R is not sigma^2 I.

    ``_lowrank_gain`` assumes ``S = P_yy + eps*I``.  That holds here only
    because the configuration carries a single scalar ``noise_std``; a
    non-isotropic R would need the R^-1-weighted form instead.
    """
    noise_std = config["observations"]["noise_std"]
    if not np.isscalar(noise_std):
        raise ValueError(
            "The batched kernels require an isotropic observation covariance "
            "(scalar observations.noise_std); got a non-scalar value."
        )
    return float(noise_std) ** 2


def _thin_svd(A: np.ndarray, context: str = ""):
    """Thin SVD that separates filter divergence from a LAPACK driver quirk.

    ``np.linalg.svd`` uses LAPACK's ``gesdd``, which can report
    "SVD did not converge" for two very different reasons: the input contains
    non-finite values because the filter blew up, or ``gesdd`` genuinely failed
    on a finite but awkward matrix.  The first is a result to investigate and
    must not be papered over; the second is retried on the slower, more robust
    ``gesvd`` driver.
    """
    try:
        return np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        if not np.isfinite(A).all():
            raise FloatingPointError(
                f"Non-finite ensemble values{context}: the filter diverged. "
                "The SVD failure is a symptom, not a numerical quirk."
            ) from None
        return linalg.svd(A, full_matrices=False, lapack_driver="gesvd")


def _lowrank_gain(
    A_x: np.ndarray, A_y: np.ndarray, eps: float, N: int, context: str = ""
) -> np.ndarray:
    """Kalman gain ``P_xy (P_yy + eps I)^-1`` without forming a (d, d) inverse.

    ``P_yy = A_y A_y^T/(N-1)`` has rank at most ``N-1``, far below the stacked
    observation dimension.  Writing the thin SVD ``A_y = U s V^T``, the
    eigenvalues of ``S`` are ``s^2/(N-1) + eps`` on ``range(U)`` and ``eps`` on
    its complement.  ``P_xy = A_x V s U^T/(N-1)`` has its row space inside
    ``span(U)``, so the ``1/eps`` branch annihilates and never enters the gain:

        K = A_x V diag(s / (s^2/(N-1) + eps)) U^T / (N-1)

    Exact in exact arithmetic, and 20.9x faster than ``pinv`` at d=100, N=10.
    Against ``pinv`` it agrees to 5e-14 relative for the well-conditioned
    4D/sequential case (eps = sigma^2) and to 1e-8 for the QPCA case, where the
    1e-6 ridge puts cond(S) at 1.6e8 and neither form can do better.
    """
    U, s, Vt = _thin_svd(A_y, context)
    d = s / (s**2 / (N - 1) + eps)
    return (A_x @ Vt.T * d) @ U.T / (N - 1)


def run_seq_enkf_batched(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    realizations: Iterable[int],
) -> np.ndarray:
    """Sequential EnKF for many realizations at once; returns (M, W+1, n)."""
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)
    eps = _check_isotropic_obs_cov(config) + 1e-6

    streams = _make_streams(realizations)
    M = len(streams)
    X = _batched_initial_ensembles(truth[0], n, N, streams)

    analyses = np.empty((M, W + 1, n))
    analyses[:, 0, :] = X.reshape(n, M, N).mean(axis=2).T

    obs_index = 0
    for w in range(W):
        for _ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)
            z = observations[obs_index]
            obs_index += 1

            Xv = X.reshape(n, M, N)
            Yv = (H @ X).reshape(m, M, N)
            for i in range(M):
                X_i = Xv[:, i, :]
                Y_i = Yv[:, i, :]
                A_x = X_i - X_i.mean(axis=1, keepdims=True)
                A_y = Y_i - Y_i.mean(axis=1, keepdims=True)
                K = _lowrank_gain(
                    A_x, A_y, eps, N, f" [Seq-EnKF window {w}, realization {i}]"
                )
                # randn(N, m) fills C-order, so row j holds member j's draw --
                # the same sequence the reference kernel consumes.
                perturbations = noise_std * streams[i].randn(N, m).T
                Xv[:, i, :] = X_i + K @ (z[:, np.newaxis] + perturbations - Y_i)

        analyses[:, w + 1, :] = X.reshape(n, M, N).mean(axis=2).T

    return analyses


def run_4d_enkf_batched(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    realizations: Iterable[int],
) -> np.ndarray:
    """4D EnKF for many realizations at once; returns (M, W+1, n)."""
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)
    eps = _check_isotropic_obs_cov(config) + 1e-6

    streams = _make_streams(realizations)
    M = len(streams)
    X = _batched_initial_ensembles(truth[0], n, N, streams)

    mL = m * L
    R_L = np.kron(np.eye(L), noise_std**2 * np.eye(m))
    R_L_chol = linalg.cholesky(R_L, lower=True)

    analyses = np.empty((M, W + 1, n))
    analyses[:, 0, :] = X.reshape(n, M, N).mean(axis=2).T

    Y_window = np.empty((L, m, M, N))
    for w in range(W):
        for ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)
            Y_window[ell] = (H @ X).reshape(m, M, N)

        z_stack = observations[w * L : (w + 1) * L].ravel()
        Xv = X.reshape(n, M, N)
        for i in range(M):
            Y_stack = Y_window[:, :, i, :].reshape(mL, N)
            X_i = Xv[:, i, :]
            A_x = X_i - X_i.mean(axis=1, keepdims=True)
            A_y = Y_stack - Y_stack.mean(axis=1, keepdims=True)
            K = _lowrank_gain(
                A_x, A_y, eps, N, f" [4D-EnKF window {w}, realization {i}]"
            )
            perturbations = R_L_chol @ streams[i].randn(N, mL).T
            Xv[:, i, :] = X_i + K @ (
                z_stack[:, np.newaxis] + perturbations - Y_stack
            )

        analyses[:, w + 1, :] = Xv.mean(axis=2).T

    return analyses


def run_qpca_endcf_batched(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    realizations: Iterable[int],
    kappa: int = 1,
    return_diagnostics: bool = True,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """QPCA-EnDCF for many realizations at once; returns (M, W+1, n) analyses.

    QPCA draws random numbers only when the ensembles are initialized and is
    deterministic thereafter, so batching touches no random stream at all here.
    """
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)
    _check_isotropic_obs_cov(config)
    eps = 1e-6  # matches the reference kernel's S = P_yy + 1e-6 I

    streams = _make_streams(realizations)
    M = len(streams)
    X = _batched_initial_ensembles(truth[0], n, N, streams)

    mL = m * L
    R_L = np.kron(np.eye(L), noise_std**2 * np.eye(m))
    eye_mL = np.eye(mL)
    R_L_chol = linalg.cholesky(R_L, lower=True)
    R_L_inv_chol = linalg.solve_triangular(R_L_chol, eye_mL, lower=True)
    whiten = R_L_inv_chol.T
    taper = _qpca_taper(config)

    analyses = np.empty((M, W + 1, n))
    analyses[:, 0, :] = X.reshape(n, M, N).mean(axis=2).T
    spectral = (
        np.full((M, W, len(SPECTRAL_DIAGNOSTIC_NAMES)), np.nan, dtype=np.float32)
        if return_diagnostics
        else None
    )

    Y_window = np.empty((L, m, M, N))
    for w in range(W):
        for ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)
            Y_window[ell] = (H @ X).reshape(m, M, N)

        z_stack = observations[w * L : (w + 1) * L].ravel()
        Xv = X.reshape(n, M, N)
        for i in range(M):
            Y_stack = Y_window[:, :, i, :].reshape(mL, N)

            E = whiten @ (Y_stack - z_stack[:, np.newaxis])
            e_mean = E.mean(axis=1, keepdims=True)
            E_c = E - e_mean

            # Thin SVD of the (d, N) centered residual: the nonzero eigenvalues
            # of C_E = E_c E_c^T/(N-1) are singular_values**2/(N-1), already in
            # descending order.
            eigenvecs, singular_values, _ = _thin_svd(
                E_c, f" [QPCA-EnDCF residuals, window {w}, realization {i}]"
            )
            V_kappa = eigenvecs[:, :kappa]
            Delta_obs = R_L_chol @ (-V_kappa @ (V_kappa.T @ E))

            X_i = Xv[:, i, :]
            A_x = X_i - X_i.mean(axis=1, keepdims=True)
            A_y = Y_stack - Y_stack.mean(axis=1, keepdims=True)
            if taper is None:
                K = _lowrank_gain(
                    A_x, A_y, eps, N, f" [QPCA-EnDCF window {w}, realization {i}]"
                )
            else:
                K = _localized_gain(A_x, A_y, taper, N)
            Xv[:, i, :] = X_i + K @ Delta_obs

            if return_diagnostics:
                eigenvals = singular_values**2 / (N - 1)
                spectrum_sum = eigenvals.sum()
                leading_fraction = (
                    eigenvals[0] / spectrum_sum if spectrum_sum > 0.0 else np.nan
                )
                normalized_gap = (
                    (eigenvals[0] - eigenvals[1]) / eigenvals[0]
                    if eigenvals.size > 1 and eigenvals[0] > 0.0
                    else np.nan
                )
                mean_vector = e_mean[:, 0]
                mean_energy = mean_vector @ mean_vector
                retained_energy = V_kappa.T @ mean_vector
                retained_fraction = (
                    (retained_energy @ retained_energy) / mean_energy
                    if mean_energy > 0.0
                    else np.nan
                )
                spectral[i, w] = (
                    leading_fraction,
                    normalized_gap,
                    retained_fraction,
                )

        analyses[:, w + 1, :] = Xv.mean(axis=2).T

    return analyses, spectral


def run_filter_batched(
    filter_name: str,
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    realizations: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Dispatch to the realization-batched kernel for ``filter_name``."""
    if filter_name == "Seq-EnKF":
        return run_seq_enkf_batched(config, truth, observations, H, realizations), None
    if filter_name == "4D-EnKF":
        return run_4d_enkf_batched(config, truth, observations, H, realizations), None
    if filter_name in ("Seq-ETKF", "Seq-LETKF"):
        return (
            run_seq_letkf_batched(
                config, truth, observations, H, realizations, filter_name
            ),
            None,
        )
    if filter_name in ("4D-ETKF", "4D-LETKF"):
        return (
            run_4d_letkf_batched(
                config, truth, observations, H, realizations, filter_name
            ),
            None,
        )
    if filter_name == "QPCA-EnDCF":
        kappa = config.get("filters", {}).get("qpca_endcf", {}).get("k", 1)
        return run_qpca_endcf_batched(
            config,
            truth,
            observations,
            H,
            realizations,
            kappa=kappa,
            return_diagnostics=True,
        )
    raise ValueError(f"Unknown filter name: {filter_name!r}")


# --------------------------------------------------------------------------- #
# Localized ensemble transform filters (Algorithms 3 and 4)
# --------------------------------------------------------------------------- #


def _qpca_taper(config: Dict) -> np.ndarray | None:
    """Gaspari--Cohn taper for QPCA-EnDCF's cross-covariance, or None.

    QPCA-EnDCF as specified for the Lorenz-96 experiments uses no localization
    -- regularizing the whitened observation mismatch instead of imposing a
    spatial structure is the point of the method.  Setting
    ``filters.qpca_endcf.localization_radius`` enables the inlet study's
    treatment (a taper on the state--observation cross-covariance) so that
    QPCA-EnDCF can be compared against the LETKF variants on equal footing.
    """
    settings = config.get("filters", {}).get("qpca_endcf", {}) or {}
    radius = settings.get("localization_radius", None)
    if radius is None:
        return None
    n = config["model"]["n"]
    _H, obs_indices = build_obs_operator(n, config["observations"]["observe_every"])
    return build_localization_weights(
        n, obs_indices, radius, window_len=config["assimilation"]["window_length"]
    )


def letkf_settings(config: Dict, filter_name: str) -> Tuple[np.ndarray | None, float]:
    """Localization weights and inflation for one LETKF variant.

    The paper specifies Gaspari--Cohn localization and inflation for the inlet
    shallow-water study but gives no Lorenz-96 values for the LETKF variants,
    which appear only in that transfer experiment.  Both are therefore read from
    the experiment YAML, and the defaults are stated there rather than hidden
    here.
    """
    key = FILTER_CONFIG_KEYS[filter_name]
    settings = config.get("filters", {}).get(key, {}) or {}
    window_len = config["assimilation"]["window_length"] if "4D" in filter_name else 1

    n = config["model"]["n"]
    _H, obs_indices = build_obs_operator(n, config["observations"]["observe_every"])
    radius = settings.get("localization_radius", None)
    weights = build_localization_weights(n, obs_indices, radius, window_len=window_len)
    return weights, float(settings.get("inflation", 1.0))


def _localized_gain(A_x, A_y, taper, N):
    """Gain ``(rho * P_xy) P_yy^+`` for a localized cross-covariance.

    The paper localizes QPCA-EnDCF by tapering the ensemble-derived
    state--observation cross-covariance (the inlet study applies a Gaspari--Cohn
    taper to exactly that object).  Tapering destroys the property
    ``_lowrank_gain`` relies on -- that ``P_xy``'s row space lies inside the
    ensemble span -- so the 1e-6 ridge QPCA otherwise puts on ``S`` can no
    longer be used: roughly 45% of the tapered cross-covariance falls in the
    null space of ``P_yy``, where a 1e-6 ridge amplifies it by 1e6 and the gain
    blows up (max|K| goes from 0.13 to over 1e6).

    The rank-truncated Moore--Penrose pseudo-inverse maps that null space to
    zero instead, which is well posed for any taper.  With no taper it
    reproduces the ridged gain to ~1e-7 relative, the accuracy the conditioning
    of ``S`` allows in the first place.
    """
    U, s, _ = _thin_svd(A_y, " [localized QPCA gain]")
    d = np.zeros_like(s)
    keep = s > s[0] * 1e-10 if s[0] > 0 else np.zeros_like(s, dtype=bool)
    d[keep] = (N - 1) / s[keep] ** 2
    P_xy = A_x @ A_y.T / (N - 1)
    return ((taper * P_xy) @ U * d) @ U.T


def run_seq_letkf(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    ensemble_seed: int,
    filter_name: str = "Seq-LETKF",
) -> np.ndarray:
    """Sequential LETKF (Algorithm 3); analysis mean at window endpoints.

    Deterministic after initialization: the transform draws no random numbers,
    so ``ensemble_seed`` enters only through the initial ensemble.
    """
    np.random.seed(ensemble_seed)
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)

    X = _initial_ensemble(truth[0], n, N)
    localization, inflation = letkf_settings(config, filter_name)
    filt = SequentialLETKF(
        H, noise_std**2 * np.eye(m), localization=localization, inflation=inflation
    )

    analysis = np.zeros((W + 1, n))
    analysis[0] = X.mean(axis=1)

    obs_index = 0
    for w in range(W):
        for _ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)
            X = filt.update(X, observations[obs_index])
            obs_index += 1
        analysis[w + 1] = X.mean(axis=1)

    return analysis


def run_4d_letkf(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    ensemble_seed: int,
    filter_name: str = "4D-LETKF",
) -> np.ndarray:
    """Four-dimensional LETKF (Algorithm 4); analysis mean at window endpoints."""
    np.random.seed(ensemble_seed)
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)

    X = _initial_ensemble(truth[0], n, N)
    localization, inflation = letkf_settings(config, filter_name)
    filt = LETKF4D(
        H,
        noise_std**2 * np.eye(m),
        window_len=L,
        localization=localization,
        inflation=inflation,
    )

    analysis = np.zeros((W + 1, n))
    analysis[0] = X.mean(axis=1)

    for w in range(W):
        X_path = []
        for _ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)
            X_path.append(X.copy())
        z_stack = observations[w * L : (w + 1) * L].ravel()
        X = filt.update(X_path, z_stack)
        analysis[w + 1] = X.mean(axis=1)

    return analysis


def _batched_ensemble_transform(
    A_z: np.ndarray, innovation: np.ndarray, precision: np.ndarray, N: int
) -> np.ndarray:
    """Local ETKF transforms for every (realization, region) pair at once.

    Parameters
    ----------
    A_z : ndarray, shape (d, M, N)
        Predicted-observation anomalies per realization.
    innovation : ndarray, shape (d, M)
        ``z - mean(H X)`` per realization.
    precision : ndarray, shape (n_regions, d)
        Localized observation precision ``diag(rho_q) R^{-1}``, one row per
        region.  Zero weights contribute nothing, so this full-length form is
        identical to selecting a local subset while keeping every system the
        same shape.
    N : int
        Ensemble size.

    Returns
    -------
    W : ndarray, shape (M, n_regions, N, N), or (M, 1, N, N) when every region
        shares the same precision row.  ``_apply_transform`` broadcasts that
        second case over the state axis.

    Notes
    -----
    When the taper is uniform -- which is exactly what an *unlocalized* ETKF is,
    since ``build_localization_weights`` returns an array of ones for a null
    radius -- every region solves a bit-identical ``N x N`` system.  Computing
    ``n_regions`` copies of it costs ``n_regions`` times too much: at n=40 the
    Lorenz-96 ETKF spent 40x its necessary work on the eigendecompositions
    alone, which is what made Seq-ETKF pairs run ~20x slower than the
    stochastic filters.  The uniform case is therefore detected and solved once.

    The check is exact equality against the first row, not a tolerance: the
    unlocalized path produces literally ``np.ones``, and a Gaspari--Cohn taper
    that happened to be uniform to within a tolerance would still be a
    localization the caller asked for.
    """
    uniform = bool((precision == precision[0]).all())
    if uniform:
        precision = precision[:1]

    C = np.einsum("din,qd,dim->iqnm", A_z, precision, A_z, optimize=True)
    b = np.einsum("din,qd,di->iqn", A_z, precision, innovation, optimize=True)

    target = C + (N - 1) * np.eye(N)
    target = 0.5 * (target + np.swapaxes(target, -1, -2))

    # Same distinction as _thin_svd: a bare "Eigenvalues did not converge" hides
    # whether the ensemble blew up or LAPACK merely struggled.  Uninflated LETKF
    # can diverge on Lorenz-96, and that is a result to report, not to retry.
    if not np.isfinite(target).all():
        diverged = np.unique(np.argwhere(~np.isfinite(target))[:, 0])
        raise FloatingPointError(
            f"Non-finite ensemble values in {diverged.size} of {target.shape[0]} "
            f"realizations (first: {diverged[0]}): the filter diverged. Note that "
            f"Algorithms 3 and 4 include multiplicative inflation, which this "
            f"configuration sets to 1.0."
        )

    mu, V = np.linalg.eigh(target)
    mu = np.maximum(mu, 1e-12)
    Vt = np.swapaxes(V, -1, -2)

    P_tilde = np.einsum("iqnk,iqk,iqkm->iqnm", V, 1.0 / mu, Vt, optimize=True)
    W_tilde = np.einsum(
        "iqnk,iqk,iqkm->iqnm", V, np.sqrt((N - 1) / mu), Vt, optimize=True
    )
    w_bar = np.einsum("iqnm,iqm->iqn", P_tilde, b, optimize=True)
    return w_bar[..., None] + W_tilde


def _apply_transform(Xv: np.ndarray, W: np.ndarray, inflation: float) -> None:
    """Write the localized analysis back into ``Xv`` (n, M, N), in place.

    Region ``q`` is state index ``q``, so only row ``q`` of the state anomalies
    enters its transform: ``X_q^a = xbar_q 1^T + A_{x,q} W_q``.
    """
    x_mean = Xv.mean(axis=2)
    A_x = Xv - x_mean[:, :, None]
    if W.shape[1] == 1:
        # Uniform taper: one transform, broadcast over every state index.
        X_a = x_mean[:, :, None] + np.einsum(
            "qin,inm->qim", A_x, W[:, 0], optimize=True
        )
    else:
        X_a = x_mean[:, :, None] + np.einsum(
            "qin,iqnm->qim", A_x, W, optimize=True
        )

    if inflation != 1.0:
        a_mean = X_a.mean(axis=2, keepdims=True)
        X_a = a_mean + inflation * (X_a - a_mean)

    Xv[...] = X_a


def run_seq_letkf_batched(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    realizations: Iterable[int],
    filter_name: str = "Seq-LETKF",
) -> np.ndarray:
    """Sequential LETKF for many realizations at once; returns (M, W+1, n)."""
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)
    localization, inflation = letkf_settings(config, filter_name)
    precision = localization / noise_std**2

    streams = _make_streams(realizations)
    M = len(streams)
    X = _batched_initial_ensembles(truth[0], n, N, streams)

    analyses = np.empty((M, W + 1, n))
    analyses[:, 0, :] = X.reshape(n, M, N).mean(axis=2).T

    obs_index = 0
    for w in range(W):
        for _ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)
            Xv = X.reshape(n, M, N)
            Z = (H @ X).reshape(m, M, N)
            z_mean = Z.mean(axis=2)
            A_z = Z - z_mean[:, :, None]
            innovation = observations[obs_index][:, None] - z_mean
            obs_index += 1

            W_transform = _batched_ensemble_transform(A_z, innovation, precision, N)
            _apply_transform(Xv, W_transform, inflation)

        analyses[:, w + 1, :] = X.reshape(n, M, N).mean(axis=2).T

    return analyses


def run_4d_letkf_batched(
    config: Dict,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    realizations: Iterable[int],
    filter_name: str = "4D-LETKF",
) -> np.ndarray:
    """Four-dimensional LETKF for many realizations at once; returns (M, W+1, n)."""
    n, m, noise_std, L, W, N, steps_per_obs, F, dt = _unpack(config)
    localization, inflation = letkf_settings(config, filter_name)
    precision = localization / noise_std**2

    streams = _make_streams(realizations)
    M = len(streams)
    X = _batched_initial_ensembles(truth[0], n, N, streams)

    analyses = np.empty((M, W + 1, n))
    analyses[:, 0, :] = X.reshape(n, M, N).mean(axis=2).T

    Y_window = np.empty((L, m, M, N))
    for w in range(W):
        for ell in range(L):
            X = _advance(X, steps_per_obs, dt, F)
            Y_window[ell] = (H @ X).reshape(m, M, N)

        Z_stack = Y_window.reshape(m * L, M, N)
        z_mean = Z_stack.mean(axis=2)
        A_z = Z_stack - z_mean[:, :, None]
        innovation = observations[w * L : (w + 1) * L].ravel()[:, None] - z_mean

        W_transform = _batched_ensemble_transform(A_z, innovation, precision, N)
        _apply_transform(X.reshape(n, M, N), W_transform, inflation)

        analyses[:, w + 1, :] = X.reshape(n, M, N).mean(axis=2).T

    return analyses


def get_filter_kernel(name: str, config: Dict):
    """Return a ``f(config, truth, obs, H, ensemble_seed) -> analysis`` callable."""
    if name == "Seq-EnKF":
        return run_seq_enkf
    if name == "4D-EnKF":
        return run_4d_enkf
    if name in ("Seq-ETKF", "Seq-LETKF"):
        return lambda cfg, truth, obs, H, seed: run_seq_letkf(
            cfg, truth, obs, H, seed, filter_name=name
        )
    if name in ("4D-ETKF", "4D-LETKF"):
        return lambda cfg, truth, obs, H, seed: run_4d_letkf(
            cfg, truth, obs, H, seed, filter_name=name
        )
    if name == "QPCA-EnDCF":
        kappa = config.get("filters", {}).get("qpca_endcf", {}).get("k", 1)

        def _qpca(cfg, truth, obs, H, ensemble_seed):
            return run_qpca_endcf(cfg, truth, obs, H, ensemble_seed, kappa=kappa)

        return _qpca
    raise ValueError(f"Unknown filter name: {name!r}")


# --------------------------------------------------------------------------- #
# Worker task
# --------------------------------------------------------------------------- #


def _run_one(config: Dict, seed: int, filter_name: str, realization: int):
    """Run a single realization and return only what the reduction needs.

    Returning ``(analysis, squared error against truth)`` instead of the raw
    trajectory keeps the reduction associative -- partial results are combined
    by addition -- so peak memory is independent of ``n_monte_carlo``.
    """
    truth, observations, H = generate_truth_and_observations(config, seed)
    kernel = get_filter_kernel(filter_name, config)

    ensemble_seed = ENSEMBLE_SEED_BASE + realization
    analysis = kernel(config, truth, observations, H, ensemble_seed)

    L = config["assimilation"]["window_length"]
    truth_windows = truth[::L]

    return seed, filter_name, analysis, (analysis - truth_windows) ** 2


def _run_one_detailed(config: Dict, seed: int, filter_name: str, realization: int):
    """Single-realization reference path for the hierarchical quantities.

    Superseded in the driver by the realization-batched kernels, but retained
    because it is the independent implementation the batched kernels are
    validated against (see ``tests/test_batched_kernels.py``).
    """
    truth, observations, H = generate_truth_and_observations(config, seed)
    ensemble_seed = ENSEMBLE_SEED_BASE + realization

    if filter_name == "QPCA-EnDCF":
        kappa = config.get("filters", {}).get("qpca_endcf", {}).get("k", 1)
        analysis, spectral = run_qpca_endcf(
            config,
            truth,
            observations,
            H,
            ensemble_seed,
            kappa=kappa,
            return_diagnostics=True,
        )
    else:
        analysis = get_filter_kernel(filter_name, config)(
            config, truth, observations, H, ensemble_seed
        )
        spectral = None

    return realization, analysis, spectral


# --------------------------------------------------------------------------- #
# Reduction
# --------------------------------------------------------------------------- #


class _Accumulator:
    """Running sums for one (seed, filter) pair."""

    __slots__ = ("count", "sum_analysis", "sum_sq_error")

    def __init__(self, shape: Tuple[int, int]):
        self.count = 0
        self.sum_analysis = np.zeros(shape)
        self.sum_sq_error = np.zeros(shape)

    def update(self, analysis: np.ndarray, sq_error: np.ndarray) -> None:
        self.count += 1
        self.sum_analysis += analysis
        self.sum_sq_error += sq_error


def summarize(acc: _Accumulator, truth_windows: np.ndarray) -> Dict[str, np.ndarray]:
    """Derive the bias-variance decomposition from the accumulated sums.

    Uses ``Var = MSE - bias^2``, which is an exact identity for the population
    variance the notebook computes (``((r - mean) ** 2).mean(axis=0)``), not an
    approximation.

    Also reports ``bias_squared_debiased_trace``.  The plug-in estimator
    satisfies ``E[bias_hat^2] = bias^2 + sigma^2 / M``, so at small M it
    overstates the systematic error by roughly the variance divided by the
    number of realizations; subtracting ``Var_hat / (M - 1)`` removes that term
    exactly.
    """
    M = acc.count
    if M == 0:
        raise ValueError("No realizations accumulated.")

    expectation = acc.sum_analysis / M
    bias = expectation - truth_windows
    bias_squared = bias**2
    mse = acc.sum_sq_error / M
    # Nonnegative by construction; the subtraction can land a few ulp below zero
    # wherever the variance is essentially nil.
    variance = np.maximum(mse - bias_squared, 0.0)

    bias_squared_trace = bias_squared.mean(axis=1)
    variance_trace = variance.mean(axis=1)
    mse_trace = mse.mean(axis=1)

    debiased = bias_squared_trace - variance_trace / (M - 1) if M > 1 else None

    out = {
        "n_realizations": np.asarray(M),
        "expectation": expectation,
        "truth": truth_windows,
        "bias": bias,
        "bias_norm": np.sqrt(bias_squared.mean(axis=1)),
        "variance": variance,
        "variance_trace": variance_trace,
        "bias_squared": bias_squared,
        "bias_squared_trace": bias_squared_trace,
        "mse": mse,
        "mse_trace": mse_trace,
        "bias_mse_ratio": bias_squared_trace / (mse_trace + 1e-12),
    }
    if debiased is not None:
        out["bias_squared_debiased_trace"] = debiased
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _checkpoint_path(results_dir: Path, seed: int, filter_name: str) -> Path:
    slug = filter_name.lower().replace("-", "_")
    return results_dir / f"{slug}_seed_{seed}.npz"


def _checkpoint_matches(path: Path, n_mc: int, shape: Tuple[int, int]) -> bool:
    """True if an existing checkpoint was produced at the current settings.

    Files written by the pre-driver notebook lack ``n_realizations`` and store
    only the aggregate traces, so anything unreadable or shaped differently is
    treated as stale rather than reused.
    """
    try:
        with np.load(path) as data:
            return (
                "n_realizations" in data.files
                and int(data["n_realizations"]) == n_mc
                and "mse" in data.files
                and data["mse"].shape == shape
            )
    except (OSError, ValueError, KeyError):
        return False


def _iter_results(tasks: List[Tuple], config: Dict, n_jobs: int, verbose: int):
    """Yield worker results, streaming them as they complete when possible."""
    if n_jobs == 1:
        for seed, filter_name, realization in tasks:
            yield _run_one(config, seed, filter_name, realization)
        return

    import joblib
    from joblib import Parallel, delayed

    # A list, not a generator: the fallback path below would otherwise re-iterate
    # an already-partially-consumed generator and silently drop tasks.
    jobs = [
        delayed(_run_one)(config, seed, filter_name, realization)
        for seed, filter_name, realization in tasks
    ]

    # inner_max_num_threads=1 belt-and-braces: the module-level environment
    # variables only bind if this module is imported before numpy, which is not
    # true when the notebook drives the run.
    ctx = (
        joblib.parallel_config(backend="loky", inner_max_num_threads=1)
        if hasattr(joblib, "parallel_config")
        else _nullcontext()
    )
    with ctx:
        try:
            # Streaming reduction: results are consumed as they arrive, so the
            # pool never holds all the trajectories at once.
            yield from Parallel(
                n_jobs=n_jobs,
                backend="loky",
                batch_size=4,
                verbose=verbose,
                return_as="generator_unordered",
            )(jobs)
        except TypeError:
            # joblib < 1.3 has no return_as; fall back to materializing the list.
            yield from Parallel(
                n_jobs=n_jobs, backend="loky", batch_size=4, verbose=verbose
            )(jobs)


def run_experiment(
    config: Dict,
    n_jobs: int = 10,
    results_dir: Path | None = None,
    resume: bool = True,
    verbose: int = 5,
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """Run the full (seed x filter x realization) grid and save per-pair results.

    Returns ``{seed: {filter_name: summary_dict}}``.
    """
    seeds = list(config["experiment"]["seeds"])
    n_mc = config["bias_variance"]["n_monte_carlo"]
    W = config["assimilation"]["n_windows"]
    n = config["model"]["n"]

    if results_dir is None:
        results_dir = DIAGNOSTICS_RESULTS_DIR / "bias_variance_analysis"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(seed, name) for seed in seeds for name in FILTER_NAMES]
    done = set()
    if resume:
        stale = 0
        for seed, name in pairs:
            path = _checkpoint_path(results_dir, seed, name)
            if not path.exists():
                continue
            if _checkpoint_matches(path, n_mc, (W + 1, n)):
                done.add((seed, name))
            else:
                # Written by an older run (different n_monte_carlo/n_windows, or
                # the pre-driver notebook format). Recompute rather than trust it.
                stale += 1
        if done:
            print(f"Resuming: {len(done)}/{len(pairs)} (seed, filter) pairs already complete.")
        if stale:
            print(f"Ignoring {stale} checkpoint(s) that do not match the current config.")

    tasks = [
        (seed, name, r)
        for seed, name in pairs
        if (seed, name) not in done
        for r in range(n_mc)
    ]

    print(
        f"Grid: {len(seeds)} seeds x {len(FILTER_NAMES)} filters x {n_mc} realizations "
        f"= {len(seeds) * len(FILTER_NAMES) * n_mc} runs "
        f"({len(tasks)} to run), W={W}, n_jobs={n_jobs}",
        flush=True,
    )

    accumulators = {pair: _Accumulator((W + 1, n)) for pair in pairs if pair not in done}
    remaining = {pair: n_mc for pair in accumulators}

    t_start = time.perf_counter()
    completed = 0
    for seed, filter_name, analysis, sq_error in _iter_results(tasks, config, n_jobs, verbose):
        pair = (seed, filter_name)
        accumulators[pair].update(analysis, sq_error)
        remaining[pair] -= 1
        completed += 1

        if remaining[pair] == 0:
            # Whole (seed, filter) pair is in -- summarize, checkpoint, release.
            truth, _obs, _H = generate_truth_and_observations(config, seed)
            L = config["assimilation"]["window_length"]
            summary = summarize(accumulators.pop(pair), truth[::L])
            np.savez_compressed(_checkpoint_path(results_dir, seed, filter_name), **summary)
            # flush: stdout is block-buffered when redirected to a file, which
            # would otherwise hide all progress until the run ends.
            print(
                f"  [{completed}/{len(tasks)}] wrote {filter_name} seed {seed} "
                f"({time.perf_counter() - t_start:.0f}s elapsed)",
                flush=True,
            )

    elapsed = time.perf_counter() - t_start
    print(f"\nCompleted {len(tasks)} runs in {elapsed:.0f}s ({elapsed / 60:.1f} min).")

    results = load_results(config, results_dir)
    _write_multiseed_summary(results, seeds, results_dir)
    return results


# --------------------------------------------------------------------------- #
# Hierarchical 100-truth x 100-realization experiment
# --------------------------------------------------------------------------- #


def _hierarchical_checkpoint_path(
    results_dir: Path, seed: int, filter_name: str
) -> Path:
    slug = filter_name.lower().replace("-", "_")
    return results_dir / f"{slug}_truth_seed_{seed}.npz"


def _hierarchical_checkpoint_matches(
    path: Path, n_mc: int, n_windows: int, n: int, filter_name: str
) -> bool:
    try:
        with np.load(path) as data:
            base_ok = (
                int(data["n_realizations"]) == n_mc
                and data["mse"].shape == (n_windows + 1, n)
                and data["error_gram"].shape == (n_mc, n_mc)
            )
            if filter_name == "QPCA-EnDCF":
                return base_ok and data["spectral_diagnostics"].shape == (
                    n_mc,
                    n_windows,
                    len(SPECTRAL_DIAGNOSTIC_NAMES),
                )
            return base_ok
    except (OSError, ValueError, KeyError):
        return False


def _run_pair(
    config: Dict,
    seed: int,
    filter_name: str,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    results_dir: Path,
    spinup_windows: int,
):
    """Run every realization of one (truth seed, filter) pair and checkpoint it.

    The whole pair is a single task: realizations are batched inside the worker
    rather than farmed out individually.  The worker also writes its own
    checkpoint, which keeps ~300 compressed writes off the parent's critical
    path and cuts the data crossing the process boundary from ~33 MB of
    trajectories per pair to a ~3 MB summary.
    """
    started = time.perf_counter()
    n_mc = int(config["bias_variance"]["n_monte_carlo"])
    L = config["assimilation"]["window_length"]

    try:
        analyses, spectral = run_filter_batched(
            filter_name, config, truth, observations, H, range(n_mc)
        )

        summary = _summarize_detailed_pair(analyses, truth[::L], spinup_windows)
        summary["truth_seed"] = np.asarray(seed)
        summary["spectral_diagnostic_names"] = np.asarray(SPECTRAL_DIAGNOSTIC_NAMES)
        if spectral is not None:
            summary["spectral_diagnostics"] = spectral

        path = _hierarchical_checkpoint_path(results_dir, seed, filter_name)
        np.savez_compressed(path, **summary)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        # Returning the failure instead of raising keeps joblib from tearing
        # down the pool and discarding every other worker's in-flight pair.
        # The driver prints a prominent summary of whatever failed.
        return seed, filter_name, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"

    return seed, filter_name, time.perf_counter() - started, None


def _iter_pairs(tasks: List[Tuple], n_jobs: int, verbose: int):
    """Yield pair results as they complete, from a single process pool."""
    if n_jobs == 1:
        for task in tasks:
            yield _run_pair(*task)
        return

    import joblib
    from joblib import Parallel, delayed

    jobs = [delayed(_run_pair)(*task) for task in tasks]
    ctx = (
        joblib.parallel_config(backend="loky", inner_max_num_threads=1)
        if hasattr(joblib, "parallel_config")
        else _nullcontext()
    )
    with ctx:
        try:
            yield from Parallel(
                n_jobs=n_jobs,
                backend="loky",
                batch_size=1,
                verbose=verbose,
                return_as="generator_unordered",
            )(jobs)
        except TypeError:
            yield from Parallel(
                n_jobs=n_jobs,
                backend="loky",
                batch_size=1,
                verbose=verbose,
            )(jobs)


def _summarize_detailed_pair(
    analyses: np.ndarray,
    truth_windows: np.ndarray,
    spinup_windows: int,
) -> Dict[str, np.ndarray]:
    accumulator = _Accumulator(analyses.shape[1:])
    for analysis in analyses:
        accumulator.update(analysis, (analysis - truth_windows) ** 2)
    summary = summarize(accumulator, truth_windows)

    # Compact sufficient statistic for realization-level bootstrap inference.
    # If c contains multinomial bootstrap counts, then
    #   MSE*  = c^T diag(G) / M
    #   Bias²* = c^T G c / M²,
    # where G is the time/state-averaged error Gram matrix below.
    # Retain at least one evaluation window for short smoke-test or debugging
    # configurations whose requested spin-up exceeds the run length.
    start = min(max(0, int(spinup_windows)), analyses.shape[1] - 1)
    errors = (analyses[:, start:] - truth_windows[np.newaxis, start:]).reshape(
        analyses.shape[0], -1
    )
    error_gram = (errors @ errors.T) / errors.shape[1]
    summary["error_gram"] = error_gram
    summary["bootstrap_spinup_windows"] = np.asarray(start)
    return summary


def run_hierarchical_experiment(
    config: Dict,
    n_jobs: int = 10,
    results_dir: Path | None = None,
    resume: bool = True,
    verbose: int = 5,
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """Run a resumable truth-seed x filter x realization experiment.

    One ``(truth seed, filter)`` pair is one task, with its realizations batched
    inside the worker.  All pairs go through a single process pool, so a slow
    Seq-EnKF pair overlaps fast QPCA ones instead of being serialized behind a
    per-pair barrier.
    """
    seeds = list(config["experiment"]["seeds"])
    n_mc = int(config["bias_variance"]["n_monte_carlo"])
    W = int(config["assimilation"]["n_windows"])
    n = int(config["model"]["n"])
    spinup = int(config["bias_variance"].get("spinup_windows", 0))

    if results_dir is None:
        results_dir = (
            DIAGNOSTICS_RESULTS_DIR / "bias_variance_analysis_100x100"
        )
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(seed, name) for seed in seeds for name in FILTER_NAMES]
    pending = [
        (seed, name)
        for seed, name in pairs
        if not (
            resume
            and _hierarchical_checkpoint_path(results_dir, seed, name).exists()
            and _hierarchical_checkpoint_matches(
                _hierarchical_checkpoint_path(results_dir, seed, name),
                n_mc,
                W,
                n,
                name,
            )
        )
    ]

    print(
        f"Hierarchical grid: {len(seeds)} truth seeds x {len(FILTER_NAMES)} filters "
        f"x {n_mc} realizations, W={W} -- {len(pending)}/{len(pairs)} pairs to run, "
        f"n_jobs={n_jobs}",
        flush=True,
    )

    if pending:
        # One batched pass for every truth trajectory that is actually needed,
        # instead of each worker re-integrating them (which at 100 seeds and 10
        # workers would burn ~0.5 core-hours in duplicated work alone).
        needed = sorted({seed for seed, _ in pending})
        t_truth = time.perf_counter()
        truth_data = generate_truth_batch(config, needed)
        print(
            f"Generated {len(needed)} truth trajectories in "
            f"{time.perf_counter() - t_truth:.1f}s",
            flush=True,
        )

        tasks = [
            (
                config,
                seed,
                name,
                truth_data[seed][0],
                truth_data[seed][1],
                truth_data[seed][2],
                results_dir,
                spinup,
            )
            for seed, name in pending
        ]

        t_start = time.perf_counter()
        completed = 0
        failures: List[Tuple[int, str, str]] = []
        for seed, filter_name, pair_seconds, error in _iter_pairs(
            tasks, n_jobs, verbose
        ):
            completed += 1
            if error is None:
                print(
                    f"  [{completed}/{len(tasks)}] wrote {filter_name} truth seed {seed} "
                    f"({pair_seconds:.0f}s for the pair, "
                    f"{time.perf_counter() - t_start:.0f}s elapsed)",
                    flush=True,
                )
            else:
                failures.append((seed, filter_name, error))
                print(
                    f"  [{completed}/{len(tasks)}] FAILED {filter_name} truth seed "
                    f"{seed}: {error}",
                    flush=True,
                )

        elapsed = time.perf_counter() - t_start
        print(
            f"\nCompleted {len(tasks) - len(failures)}/{len(tasks)} pairs in "
            f"{elapsed:.0f}s ({elapsed / 60:.1f} min).",
            flush=True,
        )
        if failures:
            print(f"\n{len(failures)} pair(s) FAILED and have no checkpoint:", flush=True)
            for seed, filter_name, error in failures:
                print(f"  {filter_name:<12} truth seed {seed}: {error}", flush=True)
            print(
                "Rerun the same command to retry them; completed pairs are skipped.",
                flush=True,
            )

    results = load_hierarchical_results(config, results_dir)
    _write_multiseed_summary(results, seeds, results_dir)
    return results


def load_hierarchical_results(
    config: Dict, results_dir: Path | None = None
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    if results_dir is None:
        results_dir = (
            DIAGNOSTICS_RESULTS_DIR / "bias_variance_analysis_100x100"
        )
    results_dir = Path(results_dir)
    n_mc = int(config["bias_variance"]["n_monte_carlo"])
    W = int(config["assimilation"]["n_windows"])
    n = int(config["model"]["n"])

    results: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    for seed in config["experiment"]["seeds"]:
        results[seed] = {}
        for filter_name in FILTER_NAMES:
            path = _hierarchical_checkpoint_path(
                results_dir, seed, filter_name
            )
            if not path.exists() or not _hierarchical_checkpoint_matches(
                path, n_mc, W, n, filter_name
            ):
                continue
            with np.load(path) as data:
                results[seed][filter_name] = {
                    key: data[key] for key in data.files
                }
    return results


def hierarchical_bootstrap(
    results: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    seeds: Iterable[int],
    n_bootstrap: int = 2000,
    n_inner: int = 250,
    random_seed: int = 20260728,
) -> Dict[str, np.ndarray]:
    """Hierarchical paired bootstrap using compact error Gram matrices.

    Whole truth seeds are resampled at the outer level.  At the inner level,
    multinomial counts resample complete Monte Carlo realizations.  The same
    outer and inner indices are used for every filter, preserving the matched
    common-random-numbers design.
    """
    seeds = [s for s in seeds if all(f in results.get(s, {}) for f in FILTER_NAMES)]
    if not seeds:
        raise ValueError("No complete truth seeds are available.")
    rng = np.random.default_rng(random_seed)
    n_seeds = len(seeds)
    n_mc = int(results[seeds[0]][FILTER_NAMES[0]]["n_realizations"])

    metric_names = ("mse", "bias_squared", "bias_squared_debiased", "variance")
    inner = {
        name: np.empty((n_seeds, n_inner, len(metric_names)))
        for name in FILTER_NAMES
    }
    for seed_index, seed in enumerate(seeds):
        counts = rng.multinomial(
            n_mc, np.full(n_mc, 1.0 / n_mc), size=n_inner
        )
        for filter_name in FILTER_NAMES:
            gram = results[seed][filter_name]["error_gram"]
            mse = counts @ np.diag(gram) / n_mc
            bias2 = np.einsum(
                "bi,ij,bj->b", counts, gram, counts, optimize=True
            ) / (n_mc**2)
            variance = np.maximum(mse - bias2, 0.0)
            debiased = bias2 - variance / (n_mc - 1)
            inner[filter_name][seed_index] = np.column_stack(
                (mse, bias2, debiased, variance)
            )

    draws = {
        name: np.empty((n_bootstrap, len(metric_names)))
        for name in FILTER_NAMES
    }
    for b in range(n_bootstrap):
        outer = rng.integers(0, n_seeds, size=n_seeds)
        inner_index = rng.integers(0, n_inner, size=n_seeds)
        for filter_name in FILTER_NAMES:
            draws[filter_name][b] = inner[filter_name][
                outer, inner_index
            ].mean(axis=0)

    output: Dict[str, np.ndarray] = {
        "metric_names": np.asarray(metric_names),
        "truth_seeds": np.asarray(seeds),
    }
    for filter_name, values in draws.items():
        output[filter_name] = values
    # Every filter other than QPCA-EnDCF is a baseline to compare against, so
    # this picks up the LETKF variants without needing to be edited again.
    for baseline in (f for f in FILTER_NAMES if f != "QPCA-EnDCF"):
        output[f"QPCA_vs_{baseline}_mse_reduction"] = 1.0 - np.divide(
            draws["QPCA-EnDCF"][:, 0],
            draws[baseline][:, 0],
            out=np.full(n_bootstrap, np.nan),
            where=draws[baseline][:, 0] > 0.0,
        )
        output[f"QPCA_vs_{baseline}_variance_reduction"] = 1.0 - np.divide(
            draws["QPCA-EnDCF"][:, 3],
            draws[baseline][:, 3],
            out=np.full(n_bootstrap, np.nan),
            where=draws[baseline][:, 3] > 0.0,
        )
    return output


def load_results(
    config: Dict, results_dir: Path | None = None
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """Load per-(seed, filter) checkpoints into the nested dict the plots expect."""
    if results_dir is None:
        results_dir = DIAGNOSTICS_RESULTS_DIR / "bias_variance_analysis"
    results_dir = Path(results_dir)

    n_mc = config["bias_variance"]["n_monte_carlo"]
    shape = (config["assimilation"]["n_windows"] + 1, config["model"]["n"])

    results: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    skipped = []
    for seed in config["experiment"]["seeds"]:
        results[seed] = {}
        for name in FILTER_NAMES:
            path = _checkpoint_path(results_dir, seed, name)
            if not path.exists():
                continue
            if not _checkpoint_matches(path, n_mc, shape):
                # Left over from a different n_monte_carlo/n_windows; loading it
                # would quietly mix settings inside one figure.
                skipped.append(path.name)
                continue
            with np.load(path) as data:
                results[seed][name] = {k: data[k] for k in data.files}

    if skipped:
        print(
            f"Skipped {len(skipped)} checkpoint(s) not matching the current config "
            f"(n_monte_carlo={n_mc}, n_windows={shape[0] - 1}): {', '.join(skipped)}"
        )
    return results


def _write_multiseed_summary(results: Dict, seeds: Iterable[int], results_dir: Path) -> None:
    """Aggregate scalar metrics across seeds into a single npz, as the notebook did."""
    seeds = [s for s in seeds if results.get(s)]
    metric_keys = {
        "MSE": "mse_trace",
        "Bias²": "bias_squared_trace",
        "Variance": "variance_trace",
        "Bias²/MSE": "bias_mse_ratio",
        "Bias²_debiased": "bias_squared_debiased_trace",
    }

    summary_data = {}
    for metric, key in metric_keys.items():
        for name in FILTER_NAMES:
            vals = [
                results[s][name][key].mean()
                for s in seeds
                if name in results[s] and key in results[s][name]
            ]
            if vals:
                summary_data[f"{name}_{metric}"] = np.asarray(vals)

    np.savez(results_dir / "multiseed_summary.npz", **summary_data, seeds=np.asarray(seeds))
    print(f"Wrote {results_dir / 'multiseed_summary.npz'}")


def print_summary_table(results: Dict, seeds: Iterable[int]) -> None:
    """Print the mean +/- std table across seeds."""
    seeds = [s for s in seeds if results.get(s)]
    header = f"{'Filter':<12}{'MSE':>16}{'Bias²':>16}{'Bias² (debiased)':>20}{'Variance':>16}{'Bias²/MSE':>16}"
    print("\n" + header)
    print("-" * len(header))
    for name in FILTER_NAMES:
        rows = [results[s][name] for s in seeds if name in results[s]]
        if not rows:
            continue

        def stat(key):
            vals = np.array([r[key].mean() for r in rows if key in r])
            if vals.size == 0:
                return "n/a"
            return f"{vals.mean():.3f} ± {vals.std():.3f}"

        print(
            f"{name:<12}{stat('mse_trace'):>16}{stat('bias_squared_trace'):>16}"
            f"{stat('bias_squared_debiased_trace'):>20}{stat('variance_trace'):>16}"
            f"{stat('bias_mse_ratio'):>16}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=DIAGNOSTICS_EXPERIMENT_DIR / "bias_variance_analysis.yaml",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Worker processes. Default 10 -- on a 10P+5E Apple Silicon machine "
        "the efficiency cores measurably hurt throughput.",
    )
    parser.add_argument("--n-mc", type=int, default=None, help="Override n_monte_carlo.")
    parser.add_argument("--n-windows", type=int, default=None, help="Override n_windows.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seeds.")
    parser.add_argument(
        "--n-truth-seeds",
        type=int,
        default=None,
        help="Generate this many reproducible truth seeds (overrides --seeds).",
    )
    parser.add_argument(
        "--truth-seed-base",
        type=int,
        default=20260728,
        help="SeedSequence entropy used with --n-truth-seeds.",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Run the pair-wise, bootstrap-capable hierarchical experiment.",
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute (seed, filter) pairs even if a complete checkpoint exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Time one realization per filter and project the total cost.",
    )
    parser.add_argument("--verbose", type=int, default=5, help="joblib verbosity.")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    if args.n_mc is not None:
        config["bias_variance"]["n_monte_carlo"] = args.n_mc
    if args.n_windows is not None:
        config["assimilation"]["n_windows"] = args.n_windows
        config["assimilation"]["total_observations"] = (
            args.n_windows * config["assimilation"]["window_length"]
        )
    if args.n_truth_seeds is not None:
        config["experiment"]["seeds"] = make_truth_seeds(
            args.n_truth_seeds, args.truth_seed_base
        )
    elif args.seeds is not None:
        config["experiment"]["seeds"] = args.seeds

    if args.dry_run:
        return _dry_run(config, args.n_jobs, hierarchical=args.hierarchical)

    runner = run_hierarchical_experiment if args.hierarchical else run_experiment
    results = runner(
        config,
        n_jobs=args.n_jobs,
        results_dir=args.results_dir,
        resume=not args.no_resume,
        verbose=args.verbose,
    )
    print_summary_table(results, config["experiment"]["seeds"])
    return 0


def _dry_run(config: Dict, n_jobs: int, hierarchical: bool = False) -> int:
    seeds = config["experiment"]["seeds"]
    n_mc = config["bias_variance"]["n_monte_carlo"]
    n_runs = len(seeds) * len(FILTER_NAMES) * n_mc

    t0 = time.perf_counter()
    truth, obs, H = generate_truth_and_observations(config, seeds[0])
    t_truth = time.perf_counter() - t0
    print(f"truth generation: {t_truth:.1f}s per seed ({len(seeds)} seeds)")
    if hierarchical:
        print(
            f"  batched pre-pass for all {len(seeds)} seeds is one integration, "
            f"roughly {t_truth:.1f}s total rather than {t_truth * len(seeds):.0f}s"
        )

    total = 0.0
    if hierarchical:
        # Time a whole pair, since that is the unit of work the driver schedules.
        # Extrapolate along windows, not realizations: cost is linear in windows,
        # but sublinear in realizations because the batched forecast amortizes
        # over them, so shrinking n_mc would overstate the total.
        W = int(config["assimilation"]["n_windows"])
        probe_windows = min(W, 50)
        probe_config = {**config, "assimilation": {**config["assimilation"]}}
        probe_config["assimilation"]["n_windows"] = probe_windows
        for name in FILTER_NAMES:
            t0 = time.perf_counter()
            run_filter_batched(name, probe_config, truth, obs, H, range(n_mc))
            dt_ = (time.perf_counter() - t0) * (W / probe_windows)
            total += dt_
            print(f"  {name:<12} {dt_:7.2f}s per {n_mc}-realization pair (extrapolated)")
        serial = total / len(FILTER_NAMES) * len(seeds) * len(FILTER_NAMES)
        unit = f"{len(seeds) * len(FILTER_NAMES)} pairs"
    else:
        for name in FILTER_NAMES:
            kernel = get_filter_kernel(name, config)
            t0 = time.perf_counter()
            kernel(config, truth, obs, H, ENSEMBLE_SEED_BASE)
            dt_ = time.perf_counter() - t0
            total += dt_
            print(f"  {name:<12} {dt_:6.2f}s per realization")
        serial = total / len(FILTER_NAMES) * n_runs
        unit = f"{n_runs} runs"

    print(
        f"\n{unit}: {serial / 3600:.2f} core-hours serial; "
        f"at ~50% parallel efficiency on {n_jobs} workers, "
        f"~{serial / (0.5 * n_jobs) / 60:.0f} min wall clock."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
